use anyhow::{anyhow, Result};
use std::collections::HashMap;

const EPS_ORIENT: f64 = 1.0e-12;
const EPS_CIRCLE: f64 = 1.0e-9;
const EPS_BOUNDARY: f64 = 1.0e-9;

#[derive(Debug, Clone, Default)]
pub struct Triangulation {
    pub triangles: Vec<[usize; 3]>,
}

#[derive(Clone)]
struct Triangle {
    vertices: [usize; 3],
    circumcenter: [f64; 2],
    radius_sq: f64,
}

impl Triangle {
    fn new(vertices: [usize; 3], points: &[[f64; 2]]) -> Option<Self> {
        let mut verts = vertices;
        let a = points[verts[0]];
        let b = points[verts[1]];
        let c = points[verts[2]];
        let orient = orient2d(a, b, c);
        if orient.abs() < EPS_ORIENT {
            return None;
        }
        if orient < 0.0 {
            verts.swap(1, 2);
        }
        let (center, radius_sq) = circumcircle(a, b, c)?;
        Some(Self {
            vertices: verts,
            circumcenter: center,
            radius_sq,
        })
    }

    fn contains_point(&self, point: [f64; 2]) -> bool {
        let dx = self.circumcenter[0] - point[0];
        let dy = self.circumcenter[1] - point[1];
        (dx * dx + dy * dy) <= self.radius_sq * (1.0 + EPS_CIRCLE) + EPS_CIRCLE
    }

    fn edges(&self) -> [(usize, usize); 3] {
        let [a, b, c] = self.vertices;
        [(a, b), (b, c), (c, a)]
    }
}

pub fn triangulate(points: &[[f32; 2]], boundary: &[[f32; 2]]) -> Result<Triangulation> {
    if points.len() < 3 {
        return Ok(Triangulation {
            triangles: Vec::new(),
        });
    }
    let mut coords: Vec<[f64; 2]> = points.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let boundary64: Vec<[f64; 2]> = boundary
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64])
        .collect();
    let (min_x, max_x, min_y, max_y) = bounding_box(&coords)?;
    let span = (max_x - min_x).max(max_y - min_y);
    let delta = if span < 1.0e-6 { 1.0 } else { span };
    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    let super_points = [
        [mid_x - 20.0 * delta, mid_y - delta],
        [mid_x, mid_y + 20.0 * delta],
        [mid_x + 20.0 * delta, mid_y - delta],
    ];
    let super_start = coords.len();
    coords.extend(super_points);

    let mut triangles = Vec::<Triangle>::new();
    triangles.push(
        Triangle::new([super_start, super_start + 1, super_start + 2], &coords)
            .ok_or_else(|| anyhow!("failed to create super triangle"))?,
    );

    for point_index in 0..points.len() {
        let point = coords[point_index];
        let mut bad = vec![false; triangles.len()];
        let mut edge_map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

        for (idx, tri) in triangles.iter().enumerate() {
            if tri.contains_point(point) {
                bad[idx] = true;
                for &(a, b) in tri.edges().iter() {
                    insert_edge(&mut edge_map, a, b);
                }
            }
        }

        if edge_map.is_empty() {
            continue;
        }

        let mut next = Vec::with_capacity(triangles.len() + edge_map.len());
        for (flag, tri) in bad.iter().zip(triangles.iter()) {
            if !flag {
                next.push(tri.clone());
            }
        }
        for (_, &(a, b)) in edge_map.iter() {
            if let Some(tri) = Triangle::new([a, b, point_index], &coords) {
                next.push(tri);
            }
        }
        triangles = next;
    }

    let original_len = points.len();
    let mut final_tris = Vec::new();
    for tri in triangles {
        if tri.vertices.iter().any(|&idx| idx >= original_len) {
            continue;
        }
        if !boundary64.is_empty()
            && boundary64.len() >= 3
            && !triangle_respects_boundary(&tri.vertices, &coords[..original_len], &boundary64)
        {
            continue;
        }
        final_tris.push(tri.vertices);
    }

    Ok(Triangulation {
        triangles: final_tris,
    })
}

fn insert_edge(map: &mut HashMap<(usize, usize), (usize, usize)>, a: usize, b: usize) {
    if a == b {
        return;
    }
    let key = if a < b { (a, b) } else { (b, a) };
    if map.contains_key(&key) {
        map.remove(&key);
    } else {
        map.insert(key, (a, b));
    }
}

fn bounding_box(points: &[[f64; 2]]) -> Result<(f64, f64, f64, f64)> {
    if points.is_empty() {
        return Err(anyhow!("cannot compute bounding box for empty set"));
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
    Ok((min_x, max_x, min_y, max_y))
}

fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let ax = a[0];
    let ay = a[1];
    let bx = b[0];
    let by = b[1];
    let cx = c[0];
    let cy = c[1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < EPS_ORIENT {
        return None;
    }

    let ax2ay2 = ax * ax + ay * ay;
    let bx2by2 = bx * bx + by * by;
    let cx2cy2 = cx * cx + cy * cy;

    let ux = (ax2ay2 * (by - cy) + bx2by2 * (cy - ay) + cx2cy2 * (ay - by)) / d;
    let uy = (ax2ay2 * (cx - bx) + bx2by2 * (ax - cx) + cx2cy2 * (bx - ax)) / d;

    let dx = ux - ax;
    let dy = uy - ay;
    let radius_sq = dx * dx + dy * dy;
    Some(([ux, uy], radius_sq))
}

fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn triangle_respects_boundary(
    tri: &[usize; 3],
    points: &[[f64; 2]],
    boundary: &[[f64; 2]],
) -> bool {
    if boundary.len() < 3 {
        return true;
    }
    let verts = [points[tri[0]], points[tri[1]], points[tri[2]]];
    for vertex in &verts {
        if !point_in_polygon(*vertex, boundary) {
            return false;
        }
    }
    let edges = [(0, 1), (1, 2), (2, 0)];
    for (i0, i1) in edges {
        if segment_exits_boundary(verts[i0], verts[i1], boundary) {
            return false;
        }
    }
    true
}

fn segment_exits_boundary(a: [f64; 2], b: [f64; 2], boundary: &[[f64; 2]]) -> bool {
    if boundary.len() < 3 {
        return false;
    }
    for i in 0..boundary.len() {
        let c = boundary[i];
        let d = boundary[(i + 1) % boundary.len()];
        if segments_share_endpoint(a, b, c, d) {
            continue;
        }
        if segments_properly_intersect(a, b, c, d) {
            return true;
        }
    }
    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    !point_in_polygon(mid, boundary)
}

fn segments_share_endpoint(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> bool {
    points_close(a0, b0) || points_close(a0, b1) || points_close(a1, b0) || points_close(a1, b1)
}

fn segments_properly_intersect(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> bool {
    let o1 = orient_sign(orient2d(a0, a1, b0));
    let o2 = orient_sign(orient2d(a0, a1, b1));
    let o3 = orient_sign(orient2d(b0, b1, a0));
    let o4 = orient_sign(orient2d(b0, b1, a1));

    if o1 == 0 && on_segment(a0, a1, b0) {
        return false;
    }
    if o2 == 0 && on_segment(a0, a1, b1) {
        return false;
    }
    if o3 == 0 && on_segment(b0, b1, a0) {
        return false;
    }
    if o4 == 0 && on_segment(b0, b1, a1) {
        return false;
    }

    (o1 * o2 < 0) && (o3 * o4 < 0)
}

fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    if polygon.len() < 3 {
        return true;
    }
    if point_on_boundary(point, polygon) {
        return true;
    }
    let mut inside = false;
    let mut prev = polygon.last().copied().unwrap();
    for &curr in polygon {
        let denom = prev[1] - curr[1];
        if denom.abs() <= EPS_BOUNDARY {
            prev = curr;
            continue;
        }
        let intersects = ((curr[1] > point[1]) != (prev[1] > point[1]))
            && (point[0] < (prev[0] - curr[0]) * (point[1] - curr[1]) / denom + curr[0]);
        if intersects {
            inside = !inside;
        }
        prev = curr;
    }
    inside
}

fn point_on_boundary(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        if on_segment(a, b, point) {
            return true;
        }
    }
    false
}

fn on_segment(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    if orient2d(a, b, p).abs() > EPS_BOUNDARY {
        return false;
    }
    let min_x = a[0].min(b[0]) - EPS_BOUNDARY;
    let max_x = a[0].max(b[0]) + EPS_BOUNDARY;
    let min_y = a[1].min(b[1]) - EPS_BOUNDARY;
    let max_y = a[1].max(b[1]) + EPS_BOUNDARY;
    p[0] >= min_x && p[0] <= max_x && p[1] >= min_y && p[1] <= max_y
}

fn orient_sign(value: f64) -> i32 {
    if value > EPS_BOUNDARY {
        1
    } else if value < -EPS_BOUNDARY {
        -1
    } else {
        0
    }
}

fn points_close(a: [f64; 2], b: [f64; 2]) -> bool {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy) <= EPS_BOUNDARY * EPS_BOUNDARY
}
