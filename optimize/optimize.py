"""PyTorch port of the Rust ``optimize`` routine.

This module re-implements the high-level optimization loop from ``src/lib.rs``
using PyTorch tensors/optimizers while delegating the geometric and loss
evaluations to the existing Rust bindings exposed via the ``floorplan`` module.

Notes
-----
* The Rust losses do not expose autograd information, so we approximate the
  gradient numerically (forward finite differences). PyTorch is used for tensor
  storage plus the AdamW optimizer, while the heavy geometric computations stay
  in Rust for fidelity.
* All loss weights and learning-rate stages are loaded from ``params.toml`` so
  they match the Rust build.
* The implementation keeps parity with the Rust optimizer, including jittering
  overlapping seeds and optional GIF rendering through ``CanvasGif``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import torch

import floorplan as fp

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
	grad_epsilon: float = 1.0e-3,
	params: Optional[OptimizeParams] = None,
) -> List[float]:
	"""Run the PyTorch optimizer and return the final flattened coordinates."""

	if params is None:
		params = load_params()

	device = torch.device("cpu")
	site_tensor = torch.nn.Parameter(
		torch.tensor(site2xy, dtype=torch.float32, device=device).reshape(-1, 2)
	)
	site2xy_ini = site_tensor.detach().clone().reshape(-1).tolist()
	ctx = OptimizationContext(
		vtxl2xy=list(vtxl2xy),
		site2room=[int(v) for v in site2room],
		site2xy_ini=site2xy_ini,
		site2xy2flag=list(site2xy2flag),
		room2area_trg=list(room2area_trg),
		room_connections=[(int(a), int(b)) for (a, b) in room_connections],
		room2color=list(room2color),
		total_area_target=_polygon_area(vtxl2xy),
	)

	optimizer = torch.optim.AdamW([site_tensor], lr=params.learning_rates.first)
	lr_schedule = {
		150: params.learning_rates.second,
		300: params.learning_rates.third,
	}

	for iter_idx in range(iterations):
		if iter_idx in lr_schedule:
			for group in optimizer.param_groups:
				group["lr"] = lr_schedule[iter_idx]

		total_loss, render_payload = _evaluate_loss(site_tensor, ctx, params, need_details=True)
		grad = _finite_difference_grad(site_tensor, ctx, params, total_loss, grad_epsilon)
		site_tensor.grad = grad
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)

		metrics = render_payload["metrics"]
		print(
			f"iter={iter_idx:04d} loss={total_loss:.6f} "
			f"areas={metrics['each_area']:.6f} total={metrics['total_area']:.6f} "
			f"wall={metrics['wall_length']:.6f} topo={metrics['topology']:.6f} "
			f"fix={metrics['fix']:.6f} lloyd={metrics['lloyd']:.6f}"
		)

		if canvas is not None and (iter_idx % render_interval == 0):
			_render(canvas, ctx, render_payload)

	return site_tensor.detach().reshape(-1).tolist()


def _evaluate_loss(
	site_tensor: torch.Tensor,
	ctx: OptimizationContext,
	params: OptimizeParams,
	*,
	need_details: bool,
) -> Tuple[float, Dict[str, object]]:
	flat_coords = site_tensor.detach().cpu().reshape(-1).tolist()
	adjusted = _jitter_overlapping_sites(flat_coords, ctx.vtxl2xy)
	voronoi = fp.VoronoiInfo(ctx.vtxl2xy, adjusted, ctx.site2room)
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


def _finite_difference_grad(
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


def _boundary_span(loop: Sequence[float]) -> float:
	xs = loop[0::2]
	ys = loop[1::2]
	return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def _jitter_overlapping_sites(coords: List[float], vtxl2xy: Sequence[float]) -> List[float]:
	span = _boundary_span(vtxl2xy)
	jitter_step = span * 1.0e-5
	tolerance = span * 1.0e-7 + 1.0e-9
	num_site = len(coords) // 2
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
	for i in range(num_site):
		xi = coords[2 * i]
		yi = coords[2 * i + 1]
		duplicates = 0
		for j in range(i):
			xj = coords[2 * j] + delta[2 * j]
			yj = coords[2 * j + 1] + delta[2 * j + 1]
			if abs(xi - xj) <= tolerance and abs(yi - yj) <= tolerance:
				duplicates += 1
		if duplicates > 0:
			dirx, diry = offsets[(duplicates - 1) % len(offsets)]
			shift = jitter_step * duplicates
			delta[2 * i] += shift * dirx
			delta[2 * i + 1] += shift * diry
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
