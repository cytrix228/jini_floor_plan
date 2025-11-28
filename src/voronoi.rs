use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use delaunator::{triangulate, Triangulation};

const EPS_POS: f32 = 1.0e-6;
const EDGE_TOLERANCE: f32 = 1.0e-4;

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

    fn detect_vertex(&self, pos: [f32; 2]) -> Option<usize> {
        self.vertices.iter().enumerate().find_map(|(idx, v)| {
            if (v[0] - pos[0]).abs() < EDGE_TOLERANCE && (v[1] - pos[1]).abs() < EDGE_TOLERANCE {
                Some(idx)
            } else {
                None
            }
        })
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
    for current in polygon.iter() {
        let curr_dist = signed_distance_to_bisector(current.pos, site_pos, neighbor_pos);
        let curr_inside = curr_dist <= EDGE_TOLERANCE;
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

fn polygon_midpoint(a: &Vertex, b: &Vertex) -> [f32; 2] {
    [(a.pos[0] + b.pos[0]) * 0.5, (a.pos[1] + b.pos[1]) * 0.5]
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

    let site_positions: Vec<[f32; 2]> = site2xy.chunks(2).map(|c| [c[0], c[1]]).collect();

    let delaunay = triangulate(
        &site_positions
            .iter()
            .map(|p| delaunator::Point {
                x: p[0] as f64,
                y: p[1] as f64,
            })
            .collect::<Vec<_>>(),
    );

    let neighbor_sets = build_neighbor_sets(&delaunay, num_sites);

    let mut site2idx = Vec::with_capacity(num_sites + 1);
    let mut idx2vtxv = Vec::new();
    let mut idx2site = Vec::new();
    let mut vtxv2info = Vec::new();
    let mut info2vtx = BTreeMap::<[usize; 4], usize>::new();
    site2idx.push(0);

    for site_idx in 0..num_sites {
        if !alive[site_idx] {
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
        if polygon.len() < 3 {
            site2idx.push(site2idx.last().copied().unwrap());
            continue;
        }
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
                boundary_ctx.len(),
            );
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

    Ok(VoronoiDiagram {
        site2idx,
        idx2vtxv,
        idx2site,
        vtxv2info,
    })
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
    boundary_len: usize,
) -> [usize; 4] {
    if let Some(v_idx) = vertex.boundary_vertex {
        return [v_idx, usize::MAX, usize::MAX, usize::MAX];
    }
    if let Some(edge_idx) = vertex.boundary_edge {
        let neighbor = next_neighbor.or(prev_neighbor).unwrap_or(usize::MAX);
        return [edge_idx % boundary_len, site_idx, neighbor, usize::MAX];
    }
    let n0 = prev_neighbor.unwrap_or(usize::MAX);
    let n1 = next_neighbor.unwrap_or(usize::MAX);
    [usize::MAX, site_idx, n0, n1]
}

fn build_neighbor_sets(tri: &Triangulation, num_sites: usize) -> Vec<Vec<usize>> {
    let mut sets = vec![BTreeSet::new(); num_sites];
    for tri_indices in tri.triangles.chunks(3) {
        if tri_indices.len() < 3 {
            continue;
        }
        let a = tri_indices[0];
        let b = tri_indices[1];
        let c = tri_indices[2];
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
