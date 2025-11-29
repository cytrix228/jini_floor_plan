use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};

use crate::delaunay::Triangulation;
use std::fs::File;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const EPS_POS: f32 = 1.0e-6;
const EDGE_TOLERANCE: f32 = 1.0e-6;
const EDGE_TOLERANCE_SQ: f32 = EDGE_TOLERANCE * EDGE_TOLERANCE;
const FULL_CELL_AREA_RATIO: f32 = 0.995;
const CELL_COLORS: [&str; 8] = [
    "#e6194B", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VoronoiBackend {
    Legacy,
    Delaunay,
}

impl Default for VoronoiBackend {
    fn default() -> Self {
        Self::Legacy
    }
}

pub struct VoronoiDiagram {
    pub site2idx: Vec<usize>,
    pub idx2vtxv: Vec<usize>,
    pub idx2site: Vec<usize>,
    pub vtxv2info: Vec<[usize; 4]>,
}

#[derive(Clone)]
struct Vertex {
    pos: [f32; 2],
    boundary_vertex: Option<usize>,
    boundary_edge: Option<usize>,
}

impl Vertex {
    fn new_boundary(idx: usize, pos: [f32; 2]) -> Self {
        Self {
            pos,
            boundary_vertex: Some(idx),
            boundary_edge: None,
        }
    }
}

struct BoundaryContext {
    vertices: Vec<[f32; 2]>,
}

impl BoundaryContext {
    fn len(&self) -> usize {
        self.vertices.len()
    }

    fn centroid(&self) -> [f32; 2] {
        if self.vertices.is_empty() {
            return [0.0, 0.0];
        }
        let mut sum = [0.0f32; 2];
        for v in &self.vertices {
            sum[0] += v[0];
            sum[1] += v[1];
        }
        let inv = 1.0 / (self.vertices.len() as f32);
        [sum[0] * inv, sum[1] * inv]
    }

    fn detect_vertex(&self, pos: [f32; 2]) -> Option<usize> {
        self.vertices.iter().enumerate().find_map(|(idx, v)| {
            if (v[0] - pos[0]).abs() < EDGE_TOLERANCE && (v[1] - pos[1]).abs() < EDGE_TOLERANCE {
                Some(idx)
            } else {
                None
            }
        })
    }

    fn detect_edge(&self, pos: [f32; 2]) -> Option<usize> {
        if self.vertices.len() < 2 {
            return None;
        }
        let mut best = None;
        let mut best_dist = EDGE_TOLERANCE_SQ;
        for i in 0..self.vertices.len() {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % self.vertices.len()];
            if let Some(dist_sq) = distance_sq_to_segment(pos, a, b) {
                if dist_sq <= best_dist {
                    best = Some(i);
                    best_dist = dist_sq;
                }
            }
        }
        best
    }
}

fn signed_distance_to_bisector(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    let dir = [b[0] - a[0], b[1] - a[1]];
    (p[0] - mid[0]) * dir[0] + (p[1] - mid[1]) * dir[1]
}

fn clip_polygon_by_neighbor(
    polygon: &[Vertex],
    site_pos: [f32; 2],
    neighbor_pos: [f32; 2],
    boundary: &BoundaryContext,
) -> Vec<Vertex> {
    if polygon.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut prev = polygon.last().unwrap().clone();
    let mut prev_dist = signed_distance_to_bisector(prev.pos, site_pos, neighbor_pos);
    let mut prev_inside = prev_dist <= EDGE_TOLERANCE;
    //let mut prev_inside = prev_dist == 0.0;
    for current in polygon.iter() {
        let curr_dist = signed_distance_to_bisector(current.pos, site_pos, neighbor_pos);
        let curr_inside = curr_dist <= EDGE_TOLERANCE;
        //let curr_inside = curr_dist == 0.0;
        if curr_inside {
            if !prev_inside {
                let vtx = intersect_segment(
                    prev.clone(),
                    current.clone(),
                    prev_dist,
                    curr_dist,
                    boundary,
                );
                result.push(vtx);
            }
            result.push(current.clone());
        } else if prev_inside {
            let vtx = intersect_segment(
                prev.clone(),
                current.clone(),
                prev_dist,
                curr_dist,
                boundary,
            );
            result.push(vtx);
        }
        prev = current.clone();
        prev_dist = curr_dist;
        prev_inside = curr_inside;
    }
    result
}

fn intersect_segment(
    start: Vertex,
    end: Vertex,
    start_dist: f32,
    end_dist: f32,
    boundary: &BoundaryContext,
) -> Vertex {
    let denom = start_dist - end_dist;
    let t = if denom.abs() < EPS_POS {
        0.5
    } else {
        start_dist / denom
    };
    let t = t.clamp(0.0, 1.0);
    let pos = [
        start.pos[0] + (end.pos[0] - start.pos[0]) * t,
        start.pos[1] + (end.pos[1] - start.pos[1]) * t,
    ];
    if let Some(idx) = boundary.detect_vertex(pos) {
        return Vertex::new_boundary(idx, pos);
    }
    let edge_anchor = detect_boundary_edge(&start, &end, boundary.len());
    Vertex {
        pos,
        boundary_vertex: None,
        boundary_edge: edge_anchor,
    }
}

fn detect_boundary_edge(a: &Vertex, b: &Vertex, boundary_len: usize) -> Option<usize> {
    match (a.boundary_edge, b.boundary_edge) {
        (Some(e0), Some(e1)) if e0 == e1 => return Some(e0),
        (Some(e0), None) => return Some(e0),
        (None, Some(e1)) => return Some(e1),
        _ => {}
    }
    match (a.boundary_vertex, b.boundary_vertex) {
        (Some(i0), Some(i1)) if (i0 + 1) % boundary_len == i1 => Some(i0),
        (Some(i0), Some(i1)) if (i1 + 1) % boundary_len == i0 => Some(i1),
        _ => None,
    }
}

fn distance_sq_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> Option<f32> {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if len_sq < EPS_POS {
        return None;
    }
    let ap = [p[0] - a[0], p[1] - a[1]];
    let t = (ap[0] * ab[0] + ap[1] * ab[1]) / len_sq;
    if t < -EDGE_TOLERANCE || t > 1.0 + EDGE_TOLERANCE {
        return None;
    }
    let t = t.clamp(0.0, 1.0);
    let closest = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let diff = [p[0] - closest[0], p[1] - closest[1]];
    let dist_sq = diff[0] * diff[0] + diff[1] * diff[1];
    if dist_sq <= EDGE_TOLERANCE_SQ {
        Some(dist_sq)
    } else {
        None
    }
}

fn polygon_midpoint(a: &Vertex, b: &Vertex) -> [f32; 2] {
    [(a.pos[0] + b.pos[0]) * 0.5, (a.pos[1] + b.pos[1]) * 0.5]
}

fn polygon_matches_boundary(polygon: &[Vertex], boundary_len: usize) -> bool {
    if polygon.len() != boundary_len {
        return false;
    }
    for (idx, vertex) in polygon.iter().enumerate() {
        if vertex.boundary_edge.is_some() {
            return false;
        }
        if vertex.boundary_vertex != Some(idx) {
            return false;
        }
    }
    true
}

fn polygon_area_vertices(polygon: &[Vertex]) -> f32 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        let p0 = polygon[i].pos;
        let p1 = polygon[j].pos;
        area += p0[0] * p1[1] - p1[0] * p0[1];
    }
    0.5 * area.abs()
}

fn polygon_area_points(points: &[[f32; 2]]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1];
    }
    0.5 * area.abs()
}

fn pick_neighbor_for_edge(
    site_idx: usize,
    midpoint: [f32; 2],
    site_positions: &[[f32; 2]],
    candidates: &[usize],
) -> Option<usize> {
    let site_pos = site_positions[site_idx];
    let mut best = None;
    let mut best_diff = f32::MAX;
    let site_dist = distance(midpoint, site_pos);
    for &cand in candidates {
        let d = distance(midpoint, site_positions[cand]);
        let diff = (d - site_dist).abs();
        if diff < best_diff {
            best_diff = diff;
            best = Some(cand);
        }
    }
    if best_diff < 1.0e-3 {
        best
    } else {
        None
    }
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

pub(crate) fn sanitize_site_positions(positions: &mut [[f32; 2]], fallback: [f32; 2]) -> usize {
    let mut replacements = 0;
    for pos in positions.iter_mut() {
        if pos[0].is_finite() && pos[1].is_finite() {
            continue;
        }
        *pos = fallback;
        replacements += 1;
    }
    replacements
}

pub fn compute_delaunay_voronoi(
    boundary: &[f32],
    site2xy: &[f32],
    alive: &[bool],
) -> Result<VoronoiDiagram> {
    if site2xy.len() % 2 != 0 {
        return Err(anyhow!("site2xy must contain pairs of coordinates"));
    }
    let num_sites = site2xy.len() / 2;
    if alive.len() != num_sites {
        return Err(anyhow!("alive mask must match site count"));
    }
    if boundary.len() < 6 {
        return Err(anyhow!("boundary polygon must have at least 3 vertices"));
    }
    let boundary_vertices: Vec<[f32; 2]> = boundary.chunks(2).map(|c| [c[0], c[1]]).collect();
    let boundary_ctx = BoundaryContext {
        vertices: boundary_vertices,
    };
    let boundary_area = polygon_area_points(&boundary_ctx.vertices);

    let mut site_positions: Vec<[f32; 2]> = site2xy.chunks(2).map(|c| [c[0], c[1]]).collect();
    let fallback = boundary_ctx.centroid();
    let sanitized = sanitize_site_positions(&mut site_positions, fallback);
    if sanitized > 0 {
        eprintln!(
            "[voronoi] sanitized {} site coordinates (reset to fallback)",
            sanitized
        );
    }
    let alive_count = alive.iter().copied().filter(|flag| *flag).count();

    println!("start delaunay triangulation...");
    std::io::stdout().flush().unwrap();

    let delaunay = crate::delaunay::triangulate(&site_positions, &boundary_ctx.vertices)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_millis())
        .unwrap_or_default();
    let svg_path = format!("debug_delaunay_{}.svg", timestamp);
    dump_svg_debug(&svg_path, &site_positions, &boundary_ctx, &delaunay);
    println!("wrote Delaunay debug SVG to {}", svg_path);

    println!("start building neighbor sets...");
    std::io::stdout().flush().unwrap();
    let neighbor_sets = build_neighbor_sets(&delaunay, num_sites);

    let mut site2idx = Vec::with_capacity(num_sites + 1);
    let mut idx2vtxv = Vec::new();
    let mut idx2site = Vec::new();
    let mut vtxv2info = Vec::new();
    let mut info2vtx = BTreeMap::<[usize; 4], usize>::new();
    let mut debug_cells: Vec<Vec<[f32; 2]>> = Vec::with_capacity(num_sites);
    site2idx.push(0);

    println!("clip polygon...");
    std::io::stdout().flush().unwrap();

    for site_idx in 0..num_sites {
        if !alive[site_idx] {
            debug_cells.push(Vec::new());
            site2idx.push(site2idx.last().copied().unwrap());
            continue;
        }
        let mut polygon: Vec<Vertex> = boundary_ctx
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| Vertex::new_boundary(i, *v))
            .collect();
        for &neighbor in &neighbor_sets[site_idx] {
            if neighbor == site_idx || !alive[neighbor] {
                continue;
            }
            polygon = clip_polygon_by_neighbor(
                &polygon,
                site_positions[site_idx],
                site_positions[neighbor],
                &boundary_ctx,
            );
            if polygon.is_empty() {
                break;
            }
        }
        let cell_area = polygon_area_vertices(&polygon);
        let mut polygon_is_full =
            alive_count > 1 && polygon_matches_boundary(&polygon, boundary_ctx.len());
        if !polygon_is_full && alive_count > 1 && boundary_area > EPS_POS {
            let ratio = cell_area / boundary_area;
            if ratio >= FULL_CELL_AREA_RATIO {
                polygon_is_full = true;
            }
        }
        if polygon.len() < 3 || polygon_is_full {
            debug_cells.push(Vec::new());
            site2idx.push(site2idx.last().copied().unwrap());
            continue;
        }
        let cell_positions: Vec<[f32; 2]> = polygon.iter().map(|v| v.pos).collect();
        debug_cells.push(cell_positions);
        let mut edges_neighbors = Vec::new();
        for i in 0..polygon.len() {
            let next = (i + 1) % polygon.len();
            let midpoint = polygon_midpoint(&polygon[i], &polygon[next]);
            let neighbor = pick_neighbor_for_edge(
                site_idx,
                midpoint,
                &site_positions,
                &neighbor_sets[site_idx],
            );
            edges_neighbors.push(neighbor);
        }
        let mut cell_indices = Vec::with_capacity(polygon.len());
        for (i, vertex) in polygon.iter().enumerate() {
            let prev = if i == 0 {
                edges_neighbors.len() - 1
            } else {
                i - 1
            };
            let info = vertex_info(
                site_idx,
                vertex,
                edges_neighbors[prev],
                edges_neighbors[i],
                &boundary_ctx,
            );
            if info[1] == usize::MAX && info[0] >= boundary_ctx.len() {
                return Err(anyhow!(
                    "boundary vertex info references invalid index: site={site_idx}, vertex={i}, info={info:?}, boundary_len={}",
                    boundary_ctx.len()
                ));
            }
            if info[3] == usize::MAX && info[0] >= boundary_ctx.len() {
                return Err(anyhow!(
                    "boundary edge info references invalid index: site={site_idx}, vertex={i}, info={info:?}, prev_neighbor={:?}, next_neighbor={:?}, boundary_len={}",
                    edges_neighbors[prev],
                    edges_neighbors[i],
                    boundary_ctx.len()
                ));
            }
            let key = canonicalize(info);
            let entry = info2vtx.entry(key).or_insert_with(|| {
                vtxv2info.push(key);
                vtxv2info.len() - 1
            });
            cell_indices.push(*entry);
            idx2site.push(edges_neighbors[i].unwrap_or(usize::MAX));
        }
        idx2vtxv.extend(cell_indices);
        let current_len = site2idx.last().copied().unwrap() + polygon.len();
        site2idx.push(current_len);
    }

    println!("check vtxv2info...vtxv2info len : {}", vtxv2info.len());
    std::io::stdout().flush().unwrap();

    let boundary_len = boundary_ctx.len();
    println!("boundary_len : {}", boundary_len);
    std::io::stdout().flush().unwrap();

    for (idx, info) in vtxv2info.iter().enumerate() {
        //        println!( "idx : {}, info : {:?}" , idx, info);
        //        std::io::stdout().flush().unwrap();
        if info[3] == usize::MAX && info[0] >= boundary_len {
            return Err(anyhow::anyhow!(
                "invalid boundary reference for voronoi vertex {idx}: info={info:?}, boundary_len={boundary_len}"
            ));
        }
    }

    println!(
        "return VoronoiDiagram : {}, {}, {}",
        site2idx.len(),
        idx2vtxv.len(),
        idx2site.len()
    );
    std::io::stdout().flush().unwrap();

    let diagram = VoronoiDiagram {
        site2idx,
        idx2vtxv,
        idx2site,
        vtxv2info,
    };

    let voronoi_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_millis())
        .unwrap_or_default();
    let voronoi_svg = format!("debug_voronoi_{}.svg", voronoi_timestamp);
    dump_voronoi_svg(&voronoi_svg, &boundary_ctx, &site_positions, &debug_cells);
    println!("wrote Voronoi debug SVG to {}", voronoi_svg);

    Ok(diagram)
}

fn canonicalize(info: [usize; 4]) -> [usize; 4] {
    let mut tmp = [info[1], info[2], info[3]];
    tmp.sort_unstable();
    [info[0], tmp[0], tmp[1], tmp[2]]
}

fn vertex_info(
    site_idx: usize,
    vertex: &Vertex,
    prev_neighbor: Option<usize>,
    next_neighbor: Option<usize>,
    boundary: &BoundaryContext,
) -> [usize; 4] {
    let mut vertex = vertex.clone();
    if vertex.boundary_vertex.is_none() && vertex.boundary_edge.is_none() {
        if let Some(idx) = boundary.detect_vertex(vertex.pos) {
            vertex.boundary_vertex = Some(idx);
        } else if let Some(edge_idx) = boundary.detect_edge(vertex.pos) {
            vertex.boundary_edge = Some(edge_idx);
        }
    }

    if let Some(v_idx) = vertex.boundary_vertex {
        return [v_idx, usize::MAX, usize::MAX, usize::MAX];
    }
    if let Some(edge_idx) = vertex.boundary_edge {
        let neighbor = next_neighbor.or(prev_neighbor).unwrap_or(site_idx);
        return [edge_idx % boundary.len(), site_idx, neighbor, usize::MAX];
    }

    let n0 = prev_neighbor.or(next_neighbor).unwrap_or(site_idx);
    let n1 = next_neighbor.unwrap_or(n0);
    [usize::MAX, site_idx, n0, n1]
}

fn build_neighbor_sets(tri: &Triangulation, num_sites: usize) -> Vec<Vec<usize>> {
    let mut sets = vec![BTreeSet::new(); num_sites];
    for tri_indices in &tri.triangles {
        let [a, b, c] = *tri_indices;
        if a < num_sites && b < num_sites {
            sets[a].insert(b);
            sets[b].insert(a);
        }
        if b < num_sites && c < num_sites {
            sets[b].insert(c);
            sets[c].insert(b);
        }
        if c < num_sites && a < num_sites {
            sets[c].insert(a);
            sets[a].insert(c);
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

fn dump_svg_debug(path: &str, sites: &[[f32; 2]], boundary: &BoundaryContext, tri: &Triangulation) {
    let mut all_points = Vec::with_capacity(boundary.vertices.len() + sites.len());
    all_points.extend(boundary.vertices.iter().copied());
    all_points.extend(sites.iter().copied());
    let (min_x, max_x, min_y, max_y) = match bounding_box(&all_points) {
        Some(bounds) => bounds,
        None => return,
    };
    let margin = ((max_x - min_x).max(max_y - min_y)).max(1.0e-3) * 0.05;
    let mut file = match File::create(path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[voronoi] failed to create SVG {}: {}", path, err);
            return;
        }
    };
    let width = max_x - min_x + margin * 2.0;
    let height = max_y - min_y + margin * 2.0;
    if writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\">",
        min_x - margin,
        min_y - margin,
        width,
        height
    )
    .is_err()
    {
        return;
    }

    if boundary.vertices.len() >= 2 {
        let mut poly = String::new();
        for v in &boundary.vertices {
            poly.push_str(&format!("{} {},", v[0], v[1]));
        }
        let _ = writeln!(
            file,
            "<polygon points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"{}\"/>",
            poly.trim_end_matches(','),
            width * 0.002
        );
    }

    for &[a, b, c] in &tri.triangles {
        let edges = [(a, b), (b, c), (c, a)];
        for &(i0, i1) in &edges {
            if i0 >= sites.len() || i1 >= sites.len() {
                continue;
            }
            let p0 = sites[i0];
            let p1 = sites[i1];
            let _ = writeln!(
                file,
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#888\" stroke-width=\"{}\"/>",
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                width * 0.0015
            );
        }
    }

    for (idx, p) in sites.iter().enumerate() {
        let _ = writeln!(
            file,
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"red\" fill-opacity=\"0.7\"/>",
            p[0],
            p[1],
            width * 0.003
        );
        let _ = writeln!(
            file,
            "<text x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"red\">{}</text>",
            p[0] + width * 0.002,
            p[1] - width * 0.002,
            width * 0.005,
            idx
        );
    }

    let _ = writeln!(file, "</svg>");
}

fn dump_voronoi_svg(
    path: &str,
    boundary: &BoundaryContext,
    sites: &[[f32; 2]],
    cells: &[Vec<[f32; 2]>],
) {
    let mut all_points = Vec::with_capacity(boundary.vertices.len() + sites.len());
    all_points.extend(boundary.vertices.iter().copied());
    all_points.extend(sites.iter().copied());
    for cell in cells {
        all_points.extend(cell.iter().copied());
    }
    let (min_x, max_x, min_y, max_y) = match bounding_box(&all_points) {
        Some(bounds) => bounds,
        None => return,
    };
    let margin = ((max_x - min_x).max(max_y - min_y)).max(1.0e-3) * 0.05;
    let mut file = match File::create(path) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("[voronoi] failed to create SVG {}: {}", path, err);
            return;
        }
    };
    let width = max_x - min_x + margin * 2.0;
    let height = max_y - min_y + margin * 2.0;
    if writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\">",
        min_x - margin,
        min_y - margin,
        width,
        height
    )
    .is_err()
    {
        return;
    }

    if boundary.vertices.len() >= 2 {
        let mut poly = String::new();
        for v in &boundary.vertices {
            poly.push_str(&format!("{} {},", v[0], v[1]));
        }
        let _ = writeln!(
            file,
            "<polygon points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"{}\"/>",
            poly.trim_end_matches(','),
            width * 0.002
        );
    }

    for (idx, cell) in cells.iter().enumerate() {
        if cell.len() < 2 {
            continue;
        }
        let mut poly = String::new();
        for v in cell {
            poly.push_str(&format!("{} {},", v[0], v[1]));
        }
        let color = CELL_COLORS[idx % CELL_COLORS.len()];
        let _ = writeln!(
            file,
            "<polygon points=\"{}\" fill=\"{}\" fill-opacity=\"0.12\" stroke=\"{}\" stroke-width=\"{}\"/>",
            poly.trim_end_matches(','),
            color,
            color,
            width * 0.0015
        );
    }

    for (idx, p) in sites.iter().enumerate() {
        let _ = writeln!(
            file,
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#000\" fill-opacity=\"0.7\"/>",
            p[0],
            p[1],
            width * 0.0025
        );
        let _ = writeln!(
            file,
            "<text x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"#111\">{}</text>",
            p[0] + width * 0.002,
            p[1] - width * 0.002,
            width * 0.005,
            idx
        );
    }

    let _ = writeln!(file, "</svg>");
}

fn bounding_box(points: &[[f32; 2]]) -> Option<(f32, f32, f32, f32)> {
    if points.is_empty() {
        return None;
    }
    let mut min_x = points[0][0];
    let mut max_x = points[0][0];
    let mut min_y = points[0][1];
    let mut max_y = points[0][1];
    for p in points.iter().skip(1) {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    Some((min_x, max_x, min_y, max_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_handles_nan_sites_by_sanitizing() {
        let boundary = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let site2xy = vec![0.25, 0.25, f32::NAN, 0.75];
        let alive = vec![true, true];
        let result = compute_delaunay_voronoi(&boundary, &site2xy, &alive)
            .expect("Voronoi diagram should handle NaN inputs");
        assert_eq!(result.site2idx.len(), alive.len() + 1);
    }
}
