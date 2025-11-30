"""Houdini Python SOP helper that builds a floorplan.VoronoiStage from geometry."""
from __future__ import annotations

from collections import defaultdict

import hou
import floorplan as fp

SITE_ATTR = "site_id"


def _validate_non_empty_geo(node: hou.SopNode, index: int, name: str) -> hou.Geometry:
    geo = node.inputs()[index].geometry()
    if geo is None:
        raise hou.NodeError(f"Input {index} ({name}) is not connected.")
    if name == "cells" and geo.primCount() == 0:
        raise hou.NodeError("Cell input (0) has no primitives.")
    if name == "sites" and geo.pointCount() == 0:
        raise hou.NodeError("Site input (1) has no points.")
    if name == "boundary" and geo.primCount() == 0:
        raise hou.NodeError("Boundary input (2) has no primitives.")
    return geo


def _flatten_points(points: list[hou.Point]) -> list[float]:
    data: list[float] = []
    for pt in points:
        pos = pt.position()
        data.extend([pos[0], pos[1]])
    return data


def _ordered_site_points(site_geo: hou.Geometry) -> list[hou.Point]:
    attr = site_geo.findPointAttrib(SITE_ATTR)
    if attr is None:
        raise hou.NodeError(f'Site points must carry int attribute "{SITE_ATTR}".')

    num_points = site_geo.pointCount()
    ordered: list[hou.Point | None] = [None] * num_points
    for pt in site_geo.points():
        site_id = pt.intAttribValue(attr)
        if site_id < 0:
            raise hou.NodeError(f"Point {pt.number()} has negative {SITE_ATTR}.")
        if site_id >= num_points:
            raise hou.NodeError(
                f"Point {pt.number()} has {SITE_ATTR}={site_id}, but only {num_points} sites exist."
            )
        if ordered[site_id] is not None:
            raise hou.NodeError(
                f"Duplicate point for site {site_id} (points {ordered[site_id].number()} and {pt.number()})."
            )
        ordered[site_id] = pt

    missing = [idx for idx, entry in enumerate(ordered) if entry is None]
    if missing:
        raise hou.NodeError(f"Missing site points for ids: {missing[:4]} ...")

    return [pt for pt in ordered if pt is not None]


def _build_boundary_loop(boundary_geo: hou.Geometry) -> list[float]:
    polygon = next(
        (prim for prim in boundary_geo.prims() if prim.type() == hou.primType.Polygon),
        None,
    )
    if polygon is None:
        raise hou.NodeError("Boundary input must contain a polygon primitive.")
    coords: list[float] = []
    for vtx in polygon.vertices():
        pos = vtx.point().position()
        coords.extend([pos[0], pos[1]])
    return coords


def _site_primitives(cell_geo: hou.Geometry, site_attr: hou.Attrib, total_sites: int) -> dict[int, hou.Prim]:
    mapping: dict[int, hou.Prim] = {}
    for prim in cell_geo.prims():
        site_id = prim.intAttribValue(site_attr)
        if site_id < 0:
            raise hou.NodeError(f"Primitive {prim.number()} has negative {SITE_ATTR}.")
        if site_id >= total_sites:
            raise hou.NodeError(
                f"Primitive {prim.number()} has {SITE_ATTR}={site_id}, but only {total_sites} sites exist."
            )
        if site_id in mapping:
            raise hou.NodeError(f"Duplicate cell for site {site_id}.")
        mapping[site_id] = prim
    missing = [idx for idx in range(total_sites) if idx not in mapping]
    if missing:
        raise hou.NodeError(f"Missing Voronoi cells for sites: {missing[:4]} ...")
    return mapping


def _edge_owners(site_prims: dict[int, hou.Prim]) -> dict[tuple[int, int], set[int]]:
    owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    for site_id, prim in site_prims.items():
        points = [v.point().number() for v in prim.vertices()]
        for idx, first in enumerate(points):
            second = points[(idx + 1) % len(points)]
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            owners[key].add(site_id)
    return owners


def _voronoi_indices(
    site_order: list[hou.Point],
    site_prims: dict[int, hou.Prim],
    owners: dict[tuple[int, int], set[int]],
) -> tuple[list[int], list[int], list[int], list[float]]:
    point_to_vtx: dict[int, int] = {}
    vtxv2xy: list[float] = []

    def vertex_index(hou_point: hou.Point) -> int:
        number = hou_point.number()
        existing = point_to_vtx.get(number)
        if existing is not None:
            return existing
        idx = len(vtxv2xy) // 2
        point_to_vtx[number] = idx
        pos = hou_point.position()
        vtxv2xy.extend([pos[0], pos[1]])
        return idx

    site2idx: list[int] = [0]
    idx2vtxv: list[int] = []
    idx2site: list[int] = []

    for site_id in range(len(site_order)):
        prim = site_prims[site_id]
        verts = prim.vertices()
        num_vertices = len(verts)
        if num_vertices < 3:
            site2idx.append(site2idx[-1])
            continue
        for vtx in verts:
            idx2vtxv.append(vertex_index(vtx.point()))
        for i in range(num_vertices):
            pa = verts[i].point().number()
            pb = verts[(i + 1) % num_vertices].point().number()
            key = (pa, pb) if pa < pb else (pb, pa)
            shared = owners.get(key, set())
            neighbor = next((sid for sid in shared if sid != site_id), None)
            idx2site.append(neighbor if neighbor is not None else -1)
        site2idx.append(site2idx[-1] + num_vertices)
    return site2idx, idx2vtxv, idx2site, vtxv2xy


def cook_voronoi_stage(node: hou.SopNode) -> None:
    cell_geo = _validate_non_empty_geo(node, 0, "cells")
    site_geo = _validate_non_empty_geo(node, 1, "sites")
    boundary_geo = _validate_non_empty_geo(node, 2, "boundary")

    site_attr = cell_geo.findPrimAttrib(SITE_ATTR)
    if site_attr is None:
        raise hou.NodeError(f'Cell primitives must carry int attribute "{SITE_ATTR}".')

    site_points = _ordered_site_points(site_geo)
    total_sites = len(site_points)

    context: fp.OptimizeContext = getattr(hou.session, "floorplan_context", None)
    if context is None:
        raise hou.NodeError("hou.session.floorplan_context is missing.")

    context_site2room = context.site2room()
    if len(context_site2room) != total_sites:
        raise hou.NodeError(
            "Site count mismatch: Houdini points do not match the OptimizeContext configuration."
        )

    site2room = context_site2room[:]  # Already ordered by site id

    site2xy = _flatten_points(site_points)
    site_coords_sanitized = list(site2xy)
    vtxl2xy = _build_boundary_loop(boundary_geo)

    site_prims = _site_primitives(cell_geo, site_attr, total_sites)
    owners = _edge_owners(site_prims)
    site2idx, idx2vtxv, idx2site, vtxv2xy = _voronoi_indices(site_points, site_prims, owners)

    voronoi_info = fp.voronoi_info_from_raw(site2idx, idx2vtxv, idx2site, vtxv2xy)
    stage = fp.create_voronoi_stage(site2xy, voronoi_info, vtxv2xy, site_coords_sanitized)
    result = context.optimize_iteration(stage)

    geo = node.geometry()
    geo.clear()
    new_points = [geo.createPoint() for _ in site_points]
    adjusted = result.site_coordinates()
    for idx, pt in enumerate(new_points):
        pt.setPosition(hou.Vector3(adjusted[idx * 2], adjusted[idx * 2 + 1], 0.0))

    hou.ui.setStatusMessage(
        "optimize_iteration finished",
        hou.severityType.ImportantMessage,
    )


if __name__ == "__main__":
    cook_voronoi_stage(hou.pwd())
