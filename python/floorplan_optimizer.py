"""Python helpers for driving the floorplan optimizer directly.

This module exposes Python-native equivalents of ``optimize_phase``, ``optimize_impl``
and ``optimize`` that orchestrate the new bindings for ``iterate_voronoi_stage`` and
``optimize_iteration``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import floorplan as fp

Matrix = Sequence[float]
RoomConnection = Tuple[int, int]


def _default_transform(width: int, height: int) -> List[float]:
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


class OptimizeDriver:
    """High level optimization driver that mirrors the Rust workflow."""

    def __init__(
        self,
        canvas: fp.CanvasGif,
        vtxl2xy: Sequence[float],
        site2xy: Sequence[float],
        site2room: Sequence[int],
        site2xy2flag: Sequence[float],
        room2area_trg: Sequence[float],
        room_connections: Sequence[RoomConnection],
        params_index: int = 0,
        transform_to_scr: Matrix | None = None,
    ) -> None:
        self.canvas = canvas
        self.vtxl2xy = list(vtxl2xy)
        self.site2room = list(site2room)
        self.room_connections = list(room_connections)
        self.transform_to_scr = (
            list(transform_to_scr)
            if transform_to_scr is not None
            else _default_transform(canvas.width, canvas.height)
        )
        self.context = fp.OptimizeContext(
            list(self.vtxl2xy),
            list(site2xy),
            list(self.site2room),
            list(site2xy2flag),
            list(room2area_trg),
            list(self.room_connections),
            params_index=params_index,
        )
        self.learning_rates = self.context.learning_rates()

    def _apply_lr_schedule(self, iteration: int) -> None:
        if iteration == 150:
            self.context.set_learning_rate(self.learning_rates[1])
        elif iteration == 300:
            self.context.set_learning_rate(self.learning_rates[2])

    def optimize_phase(self, iterations: int) -> fp.OptimizeResult | None:
        last_result: fp.OptimizeResult | None = None
        for iter_idx in range(iterations):
            self._apply_lr_schedule(iter_idx)
            stage = self.construct_stage()
            result = self.context.optimize_iteration(stage)
            last_result = result
            info = result.voronoi_info()
            self.canvas.clear(0)
            fp.my_paint(
                self.canvas,
                self.transform_to_scr,
                self.vtxl2xy,
                result.site_coordinates(),
                info,
                self.site2room,
                result.edge2vtvx_wall(),
                result.vertex_coordinates(),
            )
            self.canvas.write()
        return last_result

    def construct_stage(
        self,
        *,
        site2xy_adjusted: Optional[Sequence[float]] = None,
        voronoi_info: Optional[fp.VoronoiInfo] = None,
        vtxv2xy: Optional[Sequence[float]] = None,
        site_coords_sanitized: Optional[Sequence[float]] = None,
    ) -> fp.VoronoiStage:
        """Build a ``VoronoiStage`` from context data or caller-provided geometry."""

        provided = any(
            value is not None
            for value in (
                site2xy_adjusted,
                voronoi_info,
                vtxv2xy,
                site_coords_sanitized,
            )
        )
        if not provided:
            return self.context.iterate_voronoi_stage()

        missing = [
            name
            for name, value in (
                ("site2xy_adjusted", site2xy_adjusted),
                ("voronoi_info", voronoi_info),
                ("vtxv2xy", vtxv2xy),
                ("site_coords_sanitized", site_coords_sanitized),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "All geometry components are required when supplying external Voronoi data: "
                + ", ".join(missing)
            )

        assert site2xy_adjusted is not None
        assert voronoi_info is not None
        assert vtxv2xy is not None
        assert site_coords_sanitized is not None

        return fp.create_voronoi_stage(
            list(site2xy_adjusted),
            voronoi_info,
            list(vtxv2xy),
            list(site_coords_sanitized),
        )


def optimize_phase(
    canvas: fp.CanvasGif,
    vtxl2xy: Sequence[float],
    site2xy: Sequence[float],
    site2room: Sequence[int],
    site2xy2flag: Sequence[float],
    room2area_trg: Sequence[float],
    room_connections: Sequence[RoomConnection],
    iter_count: int,
    params_index: int = 0,
    transform_to_scr: Matrix | None = None,
) -> fp.OptimizeResult | None:
    """Run a single optimization phase from Python code."""
    driver = OptimizeDriver(
        canvas,
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room_connections,
        params_index=params_index,
        transform_to_scr=transform_to_scr,
    )
    return driver.optimize_phase(iter_count)


def optimize_impl(
    canvas: fp.CanvasGif,
    vtxl2xy: Sequence[float],
    site2xy: Sequence[float],
    site2room: Sequence[int],
    site2xy2flag: Sequence[float],
    room2area_trg: Sequence[float],
    room2color: Sequence[int],
    room_connections: Sequence[RoomConnection],
    iter_count: int,
    params_index: int = 0,
    transform_to_scr: Matrix | None = None,
) -> fp.OptimizeResult | None:
    return optimize_phase(
        canvas,
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room_connections,
        iter_count,
        params_index=params_index,
        transform_to_scr=transform_to_scr,
    )


def optimize(
    canvas: fp.CanvasGif,
    vtxl2xy: Sequence[float],
    site2xy: Sequence[float],
    site2room: Sequence[int],
    site2xy2flag: Sequence[float],
    room2area_trg: Sequence[float],
    room2color: Sequence[int],
    room_connections: Sequence[RoomConnection],
    iter_count: int,
    params_index: int = 0,
) -> fp.OptimizeResult | None:
    _ = room2color  # Palette configuration stays in the GIF canvas.
    return optimize_impl(
        canvas,
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections,
        iter_count,
        params_index=params_index,
    )


__all__ = [
    "OptimizeDriver",
    "optimize",
    "optimize_phase",
    "optimize_impl",
]
