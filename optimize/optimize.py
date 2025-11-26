"""PyTorch port of the Rust ``optimize`` routine.

This module re-implements the high-level optimization loop from ``src/lib.rs``
using PyTorch tensors/optimizers while delegating the geometric and loss
evaluations to the existing Rust bindings exposed via the ``floorplan`` module.

Notes
-----
* The Rust losses do not expose autograd information, so we wrap them in a
	custom ``torch.autograd.Function`` that reuses our finite-difference/SPSA
	gradients. This lets a small neural network learn to adjust the coordinates
	using PyTorch's standard backpropagation machinery.
* All loss weights and learning-rate stages are loaded from ``params.toml`` so
  they match the Rust build.
* The implementation keeps parity with the Rust optimizer, including jittering
  overlapping seeds and optional GIF rendering through ``CanvasGif``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import torch

import floorplan as fp

try:  # pragma: no cover - optional dependency provided by pyo3 bindings
	from pyo3_runtime import PanicException  # type: ignore
except Exception:  # pragma: no cover
	class PanicException(RuntimeError):
		pass

try:  # Python 3.11+
	import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
	import tomli as tomllib  # type: ignore


@dataclass
class LossWeights:
	each_area: float = 5.0
	total_area: float = 10.0
	wall_length: float = 1.0
	topology: float = 10.0
	fix: float = 100.0
	lloyd: float = 0.1


@dataclass
class LearningRates:
	first: float = 0.05
	second: float = 0.005
	third: float = 0.001


@dataclass
class OptimizeParams:
	loss_weights: LossWeights = field(default_factory=LossWeights)
	learning_rates: LearningRates = field(default_factory=LearningRates)


class SiteAdjustmentNet(torch.nn.Module):
	"""Two-layer perceptron that predicts coordinate offsets."""

	def __init__(self, num_coords: int, hidden_dim: int = 512) -> None:
		super().__init__()
		self.fc1 = torch.nn.Linear(num_coords, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, num_coords)
		self.activation = torch.nn.GELU()

	def forward(self, coords: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
		hidden = self.activation(self.fc1(coords))
		delta = self.fc2(hidden)
		return coords + delta


@dataclass(frozen=True)
class OptimizationContext:
	vtxl2xy: List[float]
	site2room: List[int]
	site2xy_ini: List[float]
	site2xy2flag: List[float]
	room2area_trg: List[float]
	room_connections: List[Tuple[int, int]]
	room2color: List[int]
	total_area_target: float
	bounds: Tuple[float, float, float, float]

	@property
	def num_rooms(self) -> int:
		return len(self.room2area_trg)

	@property
	def num_sites(self) -> int:
		return len(self.site2room)


def load_params(path: Optional[Path] = None) -> OptimizeParams:
	"""Load ``OptimizeParams`` from ``params.toml`` (falls back to defaults)."""

	if path is None:
		path = Path(__file__).resolve().parents[1] / "params.toml"
	if not path.exists():
		return OptimizeParams()
	data = tomllib.loads(path.read_text())
	loss_cfg = data.get("loss_weights", {})
	lr_cfg = data.get("learning_rates", {})
	default_loss = LossWeights()
	default_lr = LearningRates()
	loss_weights = LossWeights(
		each_area=float(loss_cfg.get("each_area", default_loss.each_area)),
		total_area=float(loss_cfg.get("total_area", default_loss.total_area)),
		wall_length=float(loss_cfg.get("wall_length", default_loss.wall_length)),
		topology=float(loss_cfg.get("topology", default_loss.topology)),
		fix=float(loss_cfg.get("fix", default_loss.fix)),
		lloyd=float(loss_cfg.get("lloyd", default_loss.lloyd)),
	)
	learning_rates = LearningRates(
		first=float(lr_cfg.get("first", default_lr.first)),
		second=float(lr_cfg.get("second", default_lr.second)),
		third=float(lr_cfg.get("third", default_lr.third)),
	)
	return OptimizeParams(loss_weights=loss_weights, learning_rates=learning_rates)


class _FloorplanLossFunction(torch.autograd.Function):
	"""Custom autograd bridge to the Rust loss stack."""

	@staticmethod
	def forward(
		ctx,
		site_tensor: torch.Tensor,
		opt_ctx: OptimizationContext,
		params: OptimizeParams,
		grad_method: str,
		eps: float,
		spsa_samples: int,
	):
		clamped = _clamp_tensor(site_tensor, opt_ctx.bounds)
		loss, _ = _evaluate_loss(clamped, opt_ctx, params, need_details=False)
		ctx.save_for_backward(clamped.detach())
		ctx.opt_ctx = opt_ctx
		ctx.params = params
		ctx.grad_method = grad_method
		ctx.eps = eps
		ctx.spsa_samples = max(1, int(spsa_samples))
		ctx.base_loss = loss
		return site_tensor.new_tensor(loss)

	@staticmethod
	def backward(ctx, grad_output):
		(site_snapshot,) = ctx.saved_tensors
		method = ctx.grad_method
		try:
			if method == "forward":
				grad = _forward_difference_grad(
					site_snapshot,
					ctx.opt_ctx,
					ctx.params,
					ctx.base_loss,
					ctx.eps,
				)
			elif method == "central":
				grad = _central_difference_grad(site_snapshot, ctx.opt_ctx, ctx.params, ctx.eps)
			elif method == "spsa":
				grad = _spsa_gradient(
					site_snapshot,
					ctx.opt_ctx,
					ctx.params,
					ctx.eps,
					ctx.spsa_samples,
				)
			else:  # pragma: no cover - validated upstream
				raise ValueError(f"Unknown grad_method '{method}'.")
		except (RuntimeError, PanicException) as exc:  # pragma: no cover - defensive fallback
			message = str(exc)
			if "Voronoi" in message:
				warnings.warn(
					"Voronoi panic during gradient estimation; using zero gradient for this step",
					RuntimeWarning,
				)
				grad = torch.zeros_like(site_snapshot)
			else:
				raise
		grad_factor = 1.0 if grad_output is None else grad_output
		return grad_factor * grad, None, None, None, None, None


def optimize(
	vtxl2xy: Sequence[float],
	site2xy: Sequence[float],
	site2room: Sequence[int],
	site2xy2flag: Sequence[float],
	room2area_trg: Sequence[float],
	room2color: Sequence[int],
	room_connections: Sequence[Tuple[int, int]],
	iterations: int,
	canvas: Optional[fp.CanvasGif] = None,
	render_interval: int = 1,
	grad_epsilon: float = 2.5e-4,
	params: Optional[OptimizeParams] = None,
	grad_method: str = "central",
	spsa_samples: int = 2,
	nn_hidden: int = 512,
) -> List[float]:
	"""Train a two-layer network that maps initial sites to adjusted coordinates."""

	if params is None:
		params = load_params()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		print("[optimize] Using CUDA device for tensor ops")
	input_tensor = torch.tensor(site2xy, dtype=torch.float32, device=device).reshape(1, -1)
	site2xy_ini = input_tensor.detach().to("cpu").clone().reshape(-1).tolist()
	ctx = OptimizationContext(
		vtxl2xy=list(vtxl2xy),
		site2room=[int(v) for v in site2room],
		site2xy_ini=site2xy_ini,
		site2xy2flag=list(site2xy2flag),
		room2area_trg=list(room2area_trg),
		room_connections=[(int(a), int(b)) for (a, b) in room_connections],
		room2color=list(room2color),
		total_area_target=_polygon_area(vtxl2xy),
		bounds=_polygon_bounds(vtxl2xy),
	)

	model = SiteAdjustmentNet(input_tensor.numel(), hidden_dim=nn_hidden).to(device)
	method = grad_method.lower()
	if method not in {"forward", "central", "spsa"}:
		raise ValueError("grad_method must be one of 'forward', 'central', or 'spsa'.")
	optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rates.first)
	lr_schedule = {
		150: params.learning_rates.second,
		300: params.learning_rates.third,
	}
	model.train()

	for iter_idx in range(iterations):
		need_render = canvas is not None and (iter_idx % render_interval == 0)
		if iter_idx in lr_schedule:
			for group in optimizer.param_groups:
				group["lr"] = lr_schedule[iter_idx]

		optimizer.zero_grad(set_to_none=True)
		pred_flat = model(input_tensor).reshape(-1, 2)
		loss = _FloorplanLossFunction.apply(
			pred_flat,
			ctx,
			params,
			method,
			grad_epsilon,
			max(1, spsa_samples),
		)
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			current_sites = model(input_tensor).reshape(-1, 2)
			current_sites = _clamp_tensor(current_sites, ctx.bounds)
			total_loss, payload = _evaluate_loss(current_sites, ctx, params, need_details=need_render)
		metrics = payload["metrics"]
		print(
			f"iter={iter_idx:04d} loss={total_loss:.6f} "
			f"areas={metrics['each_area']:.6f} total={metrics['total_area']:.6f} "
			f"wall={metrics['wall_length']:.6f} topo={metrics['topology']:.6f} "
			f"fix={metrics['fix']:.6f} lloyd={metrics['lloyd']:.6f}"
		)

		if need_render and canvas is not None:
			_render(canvas, ctx, payload)

	model.eval()
	with torch.no_grad():
		final_sites = model(input_tensor).reshape(-1, 2)
		final_sites = _clamp_tensor(final_sites, ctx.bounds)
	return final_sites.detach().reshape(-1).tolist()


@torch.no_grad()
def _evaluate_loss(
	site_tensor: torch.Tensor,
	ctx: OptimizationContext,
	params: OptimizeParams,
	*,
	need_details: bool,
) -> Tuple[float, Dict[str, object]]:
	flat_coords = site_tensor.detach().to("cpu").reshape(-1).tolist()
	flat_coords = _clamp_coord_list(flat_coords, ctx.bounds)
	adjusted = _jitter_overlapping_sites(flat_coords, ctx.vtxl2xy)
	try:
		voronoi = fp.VoronoiInfo(ctx.vtxl2xy, adjusted, ctx.site2room)
	except PanicException as exc:  # pragma: no cover
		raise RuntimeError(
			"Voronoi builder panicked; try reducing grad_epsilon or ensuring sites stay within bounds"
		) from exc
	vtxv2xy = voronoi.vtx_coordinates()
	room2area = fp.room2area(ctx.site2room, ctx.num_rooms, voronoi)
	loss_each_area = _sum_squared(room2area, ctx.room2area_trg)
	sum_area = sum(room2area)
	loss_total_area = abs(sum_area - ctx.total_area_target)
	edge2vtxv_wall = fp.edge2vtvx_wall(voronoi, ctx.site2room)
	loss_walllen = _wall_length(edge2vtxv_wall, vtxv2xy)
	loss_topo = fp.loss_topo_unidirectional(
		adjusted,
		ctx.num_sites,
		ctx.site2room,
		ctx.num_rooms,
		voronoi,
		ctx.room_connections,
	)
	loss_fix = _fixed_site_loss(flat_coords, ctx.site2xy_ini, ctx.site2xy2flag)
	loss_lloyd = fp.loss_lloyd_internal(
		voronoi,
		ctx.site2room,
		adjusted,
		ctx.num_sites,
		vtxv2xy,
		len(vtxv2xy) // 2,
	)

	metrics = {
		"each_area": loss_each_area,
		"total_area": loss_total_area,
		"wall_length": loss_walllen,
		"topology": loss_topo,
		"fix": loss_fix,
		"lloyd": loss_lloyd,
	}
	weights = params.loss_weights
	total = (
		metrics["each_area"] * weights.each_area
		+ metrics["total_area"] * weights.total_area
		+ metrics["wall_length"] * weights.wall_length
		+ metrics["topology"] * weights.topology
		+ metrics["fix"] * weights.fix
		+ metrics["lloyd"] * weights.lloyd
	)

	payload: Dict[str, object] = {"metrics": metrics}
	if need_details:
		payload.update(
			{
				"voronoi": voronoi,
				"site2xy_adjusted": adjusted,
				"edge2vtxv_wall": edge2vtxv_wall,
				"vtxv2xy": vtxv2xy,
			}
		)
	return float(total), payload


@torch.no_grad()
def _forward_difference_grad(
	site_tensor: torch.Tensor,
	ctx: OptimizationContext,
	params: OptimizeParams,
	base_loss: float,
	eps: float,
) -> torch.Tensor:
	grad = torch.zeros_like(site_tensor.detach())
	flat_grad = grad.reshape(-1)
	base = site_tensor.detach()
	total_elems = flat_grad.numel()
	for idx in range(total_elems):
		perturbed = base.clone().reshape(-1)
		perturbed[idx] += eps
		loss_pos, _ = _evaluate_loss(
			perturbed.reshape_as(site_tensor), ctx, params, need_details=False
		)
		flat_grad[idx] = (loss_pos - base_loss) / eps
	return grad


@torch.no_grad()
def _central_difference_grad(
	site_tensor: torch.Tensor,
	ctx: OptimizationContext,
	params: OptimizeParams,
	eps: float,
) -> torch.Tensor:
	grad = torch.zeros_like(site_tensor.detach())
	flat_grad = grad.reshape(-1)
	base = site_tensor.detach()
	total_elems = flat_grad.numel()
	for idx in range(total_elems):
		perturbed = base.clone().reshape(-1)
		perturbed[idx] += eps
		loss_pos, _ = _evaluate_loss(
			perturbed.reshape_as(site_tensor), ctx, params, need_details=False
		)
		perturbed[idx] -= 2 * eps
		loss_neg, _ = _evaluate_loss(
			perturbed.reshape_as(site_tensor), ctx, params, need_details=False
		)
		flat_grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
	return grad


@torch.no_grad()
def _spsa_gradient(
	site_tensor: torch.Tensor,
	ctx: OptimizationContext,
	params: OptimizeParams,
	eps: float,
	spsa_samples: int,
) -> torch.Tensor:
	grad_accum = torch.zeros_like(site_tensor.detach())
	base = site_tensor.detach()
	for _ in range(spsa_samples):
		delta = torch.empty_like(base).bernoulli_(0.5).mul_(2).sub_(1)
		loss_pos, _ = _evaluate_loss(base + eps * delta, ctx, params, need_details=False)
		loss_neg, _ = _evaluate_loss(base - eps * delta, ctx, params, need_details=False)
		grad_accum += ((loss_pos - loss_neg) / (2.0 * eps)) * delta
	return grad_accum / float(spsa_samples)


def _render(canvas: fp.CanvasGif, ctx: OptimizationContext, payload: Dict[str, object]) -> None:
	transform = _world_to_pixel_transform(canvas.width, canvas.height)
	canvas.clear(0)
	fp.my_paint(
		canvas,
		transform,
		ctx.vtxl2xy,
		payload["site2xy_adjusted"],
		payload["voronoi"],
		ctx.site2room,
		payload["edge2vtxv_wall"],
		vtxv2xy=payload["vtxv2xy"],
	)
	canvas.write()


def _polygon_area(loop: Sequence[float]) -> float:
	if len(loop) < 4:
		return 0.0
	area = 0.0
	for i in range(0, len(loop), 2):
		j = (i + 2) % len(loop)
		area += loop[i] * loop[j + 1] - loop[j] * loop[i + 1]
	return abs(area) * 0.5


def _polygon_bounds(loop: Sequence[float]) -> Tuple[float, float, float, float]:
	if len(loop) < 4:
		return (0.0, 1.0, 0.0, 1.0)
	xs = loop[0::2]
	ys = loop[1::2]
	min_x = min(xs)
	max_x = max(xs)
	min_y = min(ys)
	max_y = max(ys)
	if math.isclose(min_x, max_x):
		max_x = min_x + 1.0
	if math.isclose(min_y, max_y):
		max_y = min_y + 1.0
	return (min_x, max_x, min_y, max_y)


def _clamp_coord_list(coords: Sequence[float], bounds: Tuple[float, float, float, float]) -> List[float]:
	min_x, max_x, min_y, max_y = bounds
	clamped = list(coords)
	for idx in range(0, len(clamped), 2):
		x = clamped[idx]
		y = clamped[idx + 1]
		clamped[idx] = min(max(x, min_x), max_x)
		clamped[idx + 1] = min(max(y, min_y), max_y)
	return clamped


def _clamp_tensor(tensor: torch.Tensor, bounds: Tuple[float, float, float, float]) -> torch.Tensor:
	min_x, max_x, min_y, max_y = bounds
	clamped = tensor.clone()
	clamped[..., 0] = torch.clamp(clamped[..., 0], min_x, max_x)
	clamped[..., 1] = torch.clamp(clamped[..., 1], min_y, max_y)
	return clamped


def _clamp_tensor_inplace(tensor: torch.Tensor, bounds: Tuple[float, float, float, float]) -> None:
	min_x, max_x, min_y, max_y = bounds
	with torch.no_grad():
		tensor[..., 0].clamp_(min_x, max_x)
		tensor[..., 1].clamp_(min_y, max_y)


def _boundary_span(loop: Sequence[float]) -> float:
	xs = loop[0::2]
	ys = loop[1::2]
	return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def _jitter_overlapping_sites(coords: List[float], vtxl2xy: Sequence[float]) -> List[float]:
	span = _boundary_span(vtxl2xy)
	jitter_step = span * 1.0e-5
	tolerance = span * 1.0e-7 + 1.0e-9
	num_site = len(coords) // 2
	if num_site < 2 or jitter_step == 0.0:
		return list(coords)
	delta = [0.0] * len(coords)
	offsets = [
		(1.0, 0.0),
		(0.0, 1.0),
		(1.0, 1.0),
		(-1.0, 0.0),
		(0.0, -1.0),
		(-1.0, -1.0),
		(1.0, -1.0),
		(-1.0, 1.0),
	]
	quantize = 0.0 if tolerance == 0.0 else 1.0 / tolerance
	seen: Dict[Tuple[int, int], int] = {}
	for i in range(num_site):
		xi = coords[2 * i]
		yi = coords[2 * i + 1]
		if quantize == 0.0:
			key = (int(xi * 1e9), int(yi * 1e9))
		else:
			key = (int(round(xi * quantize)), int(round(yi * quantize)))
		duplicates = seen.get(key, 0)
		if duplicates > 0:
			dirx, diry = offsets[(duplicates - 1) % len(offsets)]
			shift = jitter_step * duplicates
			delta[2 * i] += shift * dirx
			delta[2 * i + 1] += shift * diry
		seen[key] = duplicates + 1
	return [c + d for c, d in zip(coords, delta)]


def _wall_length(edge2vtxv: Sequence[int], vtxv2xy: Sequence[float]) -> float:
	total = 0.0
	for idx in range(0, len(edge2vtxv), 2):
		i0 = edge2vtxv[idx] * 2
		i1 = edge2vtxv[idx + 1] * 2
		x0, y0 = vtxv2xy[i0], vtxv2xy[i0 + 1]
		x1, y1 = vtxv2xy[i1], vtxv2xy[i1 + 1]
		total += math.hypot(x1 - x0, y1 - y0)
	return total


def _sum_squared(values: Sequence[float], targets: Sequence[float]) -> float:
	return sum((v - t) ** 2 for v, t in zip(values, targets))


def _fixed_site_loss(
	current: Sequence[float],
	initial: Sequence[float],
	flags: Sequence[float],
) -> float:
	loss = 0.0
	for value, init, flag in zip(current, initial, flags):
		delta = (value - init) * flag
		loss += delta * delta
	return loss


def _world_to_pixel_transform(width: int, height: int) -> List[float]:
	return [
		width * 0.8,
		0.0,
		width * 0.1,
		0.0,
		-height * 0.8,
		height * 0.9,
		0.0,
		0.0,
		1.0,
	]


__all__ = ["optimize", "load_params", "OptimizeParams", "LossWeights", "LearningRates"]
