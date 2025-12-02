use std::backtrace::Backtrace;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use geo::prelude::Area;
use geo::{BooleanOps, LineString, Polygon};

const DUCK_PATH: &str = "M7920 11494 c-193 -21 -251 -29 -355 -50 -540 -105 -1036 -366 -1442 \
    -758 -515 -495 -834 -1162 -904 -1891 -15 -154 -6 -563 15 -705 66 -440 220 \
    -857 442 -1203 24 -37 44 -69 44 -71 0 -2 -147 -3 -327 -4 -414 -1 -765 -23 \
    -1172 -72 -97 -12 -167 -17 -170 -11 -3 5 -33 52 -66 106 -231 372 -633 798 \
    -1040 1101 -309 229 -625 409 -936 532 -287 113 -392 130 -500 79 -65 -32 \
    -118 -81 -249 -237 -627 -745 -1009 -1563 -1170 -2505 -54 -320 -77 -574 -86 \
    -965 -28 -1207 238 -2308 785 -3242 120 -204 228 -364 270 -397 84 -67 585 \
    -319 901 -454 1197 -511 2535 -769 3865 -744 983 19 1875 166 2783 458 334 \
    108 918 340 1013 404 99 65 407 488 599 824 620 1080 835 2329 614 3561 -75 \
    415 -226 892 -401 1262 -39 82 -54 124 -47 133 5 7 42 58 82 114 41 55 77 99 \
    81 96 4 -2 68 -8 142 -14 766 -53 1474 347 1858 1051 105 192 186 439 228 693 \
    27 167 24 487 -6 660 -33 189 -64 249 -150 289 -46 21 -51 21 -846 21 -440 0 \
    -828 -3 -861 -7 l-62 -7 -32 86 c-54 143 -194 412 -289 554 -479 720 -1201 \
    1178 -2040 1295 -101 14 -496 27 -571 18z";

// ===============================================================================================
// MODULE: Random Number Generator (Dependency Free)
// ===============================================================================================

/// A robust Linear Congruential Generator (LCG) implementation.
/// We use constants derived from the PCG paper/Numerical Recipes to ensure
/// good distribution for coordinate generation and color selection.
/// State is 64-bit, sufficient for the resolution of this simulation.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Initialize with a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates a random u64 using a SplitMix64-like variant for better mixing
    /// than standard LCG.
    pub fn next_u64(&mut self) -> u64 {
        // Constants: a = 6364136223846793005 (Knuth), c = 1442695040888963407
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generates a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        const NORMALIZER: f64 = 1.0 / ((u64::MAX >> 11) as f64);
        ((self.next_u64() >> 11) as f64) * NORMALIZER
    }

    /// Generates a random f64 in [min, max).
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn distance_sq(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

fn points_equal(a: &Point, b: &Point) -> bool {
    (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9
}

fn sanitize_polygon_ring(points: &[Point]) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut deduped: Vec<Point> = Vec::with_capacity(points.len());
    for &pt in points {
        if deduped
            .last()
            .map(|last| points_equal(last, &pt))
            .unwrap_or(false)
        {
            continue;
        }
        deduped.push(pt);
    }
    if deduped.len() >= 2 {
        let first = deduped[0];
        if deduped
            .last()
            .map(|last| points_equal(&first, last))
            .unwrap_or(false)
        {
            deduped.pop();
        }
    }
    if deduped.len() < 3 {
        return deduped;
    }
    let len = deduped.len();
    let mut filtered: Vec<Point> = Vec::with_capacity(len);
    for i in 0..len {
        let prev = deduped[(i + len - 1) % len];
        let curr = deduped[i];
        let next = deduped[(i + 1) % len];
        let ax = curr.x - prev.x;
        let ay = curr.y - prev.y;
        let bx = next.x - curr.x;
        let by = next.y - curr.y;
        let cross = ax * by - ay * bx;
        let dot = ax * bx + ay * by;
        if cross.abs() <= 1.0e-12 && dot >= 0.0 {
            continue;
        }
        if filtered
            .last()
            .map(|last| points_equal(last, &curr))
            .unwrap_or(false)
        {
            continue;
        }
        filtered.push(curr);
    }
    filtered
}

fn polygon_area(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    let mut prev = *poly.last().unwrap();
    for &curr in poly {
        area += prev.x * curr.y - curr.x * prev.y;
        prev = curr;
    }
    area * 0.5
}

fn polygon_area_abs(poly: &[Point]) -> f64 {
    polygon_area(poly).abs()
}

fn ensure_ccw(poly: &mut Vec<Point>) {
    if polygon_area(poly) < 0.0 {
        poly.reverse();
    }
}

fn point_on_segment(p: Point, a: Point, b: Point) -> bool {
    let cross = ((p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)).abs();
    if cross > 1e-9 {
        return false;
    }
    let dot = (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y);
    if dot < 0.0 {
        return false;
    }
    let sq_len = (b.x - a.x).powi(2) + (b.y - a.y).powi(2);
    dot <= sq_len + 1e-9
}

fn point_in_polygon(p: &Point, polygon: &[Point]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for (i, curr) in polygon.iter().enumerate() {
        let prev = &polygon[j];
        if point_on_segment(*p, *prev, *curr) {
            return true;
        }
        let intersect = ((curr.y > p.y) != (prev.y > p.y))
            && (p.x < (prev.x - curr.x) * (p.y - curr.y) / (prev.y - curr.y + 1e-12) + curr.x);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn bounding_box(poly: &[Point]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &p in poly {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    (min_x, max_x, min_y, max_y)
}

/// Represents an edge between two points.
/// Used for identifying the boundary of the cavity in Bowyer-Watson.
#[derive(Clone, Copy, Debug)]
struct Edge {
    p1: Point,
    p2: Point,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        // Undirected equality: (A,B) == (B,A)
        (self.p1 == other.p1 && self.p2 == other.p2) || (self.p1 == other.p2 && self.p2 == other.p1)
    }
}
impl Eq for Edge {}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    p1: Point,
    p2: Point,
    p3: Point,
}

impl Triangle {
    fn new(p1: Point, p2: Point, p3: Point) -> Self {
        Triangle { p1, p2, p3 }
    }

    /// Calculates the circumcenter and squared radius of the triangle.
    /// Returns None if points are collinear (area ~ 0).
    ///
    /// The formula allows us to construct the Voronoi vertex directly.
    /// D = 2 * (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))
    fn circumcenter(&self) -> Option<(Point, f64)> {
        let d = 2.0
            * (self.p1.x * (self.p2.y - self.p3.y)
                + self.p2.x * (self.p3.y - self.p1.y)
                + self.p3.x * (self.p1.y - self.p2.y));

        // ROBUSTNESS: Handling Collinear Points
        // If D is very small, points are collinear.
        if d.abs() < 1e-9 {
            return None;
        }

        let p1_sq = self.p1.x.powi(2) + self.p1.y.powi(2);
        let p2_sq = self.p2.x.powi(2) + self.p2.y.powi(2);
        let p3_sq = self.p3.x.powi(2) + self.p3.y.powi(2);

        let ux = (p1_sq * (self.p2.y - self.p3.y)
            + p2_sq * (self.p3.y - self.p1.y)
            + p3_sq * (self.p1.y - self.p2.y))
            / d;

        let uy = (p1_sq * (self.p3.x - self.p2.x)
            + p2_sq * (self.p1.x - self.p3.x)
            + p3_sq * (self.p2.x - self.p1.x))
            / d;

        let center = Point::new(ux, uy);
        let radius_sq = center.distance_sq(&self.p1);

        Some((center, radius_sq))
    }

    /// The "InCircle" Predicate.
    /// Checks if point p lies inside the circumcircle.
    /// ROBUSTNESS: Uses an epsilon to handle floating point errors.
    fn in_circumcircle(&self, p: Point) -> bool {
        match self.circumcenter() {
            Some((center, r_sq)) => {
                let dist_sq = center.distance_sq(&p);
                // Epsilon tolerance handles the cocircular case.
                // If the point is *on* the circle, floating point noise might say inside/outside.
                // The epsilon ensures consistent behavior for near-boundary cases.
                dist_sq < r_sq - 1e-9
            }
            None => false, // Collinear triangle is degenerate; treat as not containing the point.
        }
    }
}

// ===============================================================================================
// MODULE: Delaunay Triangulation (Bowyer-Watson Algorithm)
// ===============================================================================================

/// Computes the Delaunay Triangulation of a set of points using the incremental Bowyer-Watson algorithm.
///
/// Robustness Strategy:
/// 1. Uses a Super-Triangle to bound the domain.
/// 2. Iteratively inserts points, carving out non-Delaunay triangles.
/// 3. Returns the full triangulation including Super-Triangle connections, which helps
///    in closing Voronoi cells for hull points.
fn bowyer_watson(mut points: Vec<Point>, width: f64, height: f64) -> (Vec<Triangle>, Triangle) {
    // 1. Super-Triangle Construction
    // We make it large enough so that the Voronoi vertices of hull points
    // (which will involve these super-vertices) are far outside the bounds.
    let expansion = 5000.0;
    let st_p1 = Point::new(-expansion, -expansion);
    let st_p2 = Point::new(2.0 * expansion + width, -expansion);
    let st_p3 = Point::new(width / 2.0, 2.0 * expansion + height);

    let super_triangle = Triangle::new(st_p1, st_p2, st_p3);
    let mut triangulation = vec![super_triangle];

    // 2. Incremental Insertion
    for point in points.drain(..) {
        let mut bad_triangles = Vec::new();

        // Find all triangles that are now invalid because the new point is in their circumcircle
        for (i, tri) in triangulation.iter().enumerate() {
            if tri.in_circumcircle(point) {
                bad_triangles.push(i);
            }
        }

        let mut polygon_hole_edges = Vec::new();

        // 3. Find the Boundary of the Cavity
        // The boundary consists of edges shared by exactly ONE bad triangle.
        // Edges shared by TWO bad triangles are internal to the cavity and must be removed.
        for &tri_idx in &bad_triangles {
            let tri = triangulation[tri_idx];
            let edges = [
                Edge {
                    p1: tri.p1,
                    p2: tri.p2,
                },
                Edge {
                    p1: tri.p2,
                    p2: tri.p3,
                },
                Edge {
                    p1: tri.p3,
                    p2: tri.p1,
                },
            ];

            for edge in &edges {
                let mut shared = false;
                for &other_idx in &bad_triangles {
                    if tri_idx == other_idx {
                        continue;
                    }
                    let other = triangulation[other_idx];
                    let other_edges = [
                        Edge {
                            p1: other.p1,
                            p2: other.p2,
                        },
                        Edge {
                            p1: other.p2,
                            p2: other.p3,
                        },
                        Edge {
                            p1: other.p3,
                            p2: other.p1,
                        },
                    ];
                    if other_edges.iter().any(|e| e == edge) {
                        shared = true;
                        break;
                    }
                }
                if !shared {
                    polygon_hole_edges.push(*edge);
                }
            }
        }

        // 4. Update Triangulation
        // Remove bad triangles (reverse sort to keep indices valid during removal)
        bad_triangles.sort_unstable_by(|a, b| b.cmp(a));
        for idx in bad_triangles {
            triangulation.remove(idx);
        }

        // Fill the cavity with new triangles connecting the new point to the boundary edges
        for edge in polygon_hole_edges {
            triangulation.push(Triangle::new(edge.p1, edge.p2, point));
        }
    }

    (triangulation, super_triangle)
}

// ===============================================================================================
// MODULE: Polygon Clipping (Sutherland-Hodgman)
// ===============================================================================================

/// Clips a subject polygon against an axis-aligned bounding box defined by  to [width, height].
/// Used to trim the "infinite" Voronoi cells.
fn sutherland_hodgman(subject: &[Point], width: f64, height: f64) -> Vec<Point> {
    let mut output = subject.to_vec();

    // Clip against 4 planes: Left, Right, Bottom, Top
    // We pass constraints as (is_vertical, sign_factor, limit_value)
    // Left: x >= 0
    output = clip_axis(output, true, 1.0, 0.0);
    // Right: x <= width  =>  -x >= -width
    output = clip_axis(output, true, -1.0, -width);
    // Bottom: y >= 0
    output = clip_axis(output, false, 1.0, 0.0);
    // Top: y <= height => -y >= -height
    output = clip_axis(output, false, -1.0, -height);

    output
}

/// Helper: Clips polygon against a single half-plane defined by `val * sign >= limit`.
fn clip_axis(subject: Vec<Point>, is_x: bool, sign: f64, limit: f64) -> Vec<Point> {
    let mut new_poly = Vec::new();
    if subject.is_empty() {
        return new_poly;
    }

    // Predicate: Is point inside?
    let is_inside = |p: Point| -> bool {
        let val = if is_x { p.x } else { p.y };
        val * sign >= limit
    };

    // Calculate intersection of line segment with the limit plane
    let intersection = |p1: Point, p2: Point| -> Point {
        let v1 = if is_x { p1.x } else { p1.y };
        let v2 = if is_x { p2.x } else { p2.y };
        // Avoid division by zero if line is parallel (though theoretic logic prevents this path)
        let t = (limit / sign - v1) / (v2 - v1);

        Point::new(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
    };

    let mut start = *subject.last().unwrap();
    for &end in &subject {
        let start_in = is_inside(start);
        let end_in = is_inside(end);

        if start_in && end_in {
            new_poly.push(end);
        } else if start_in && !end_in {
            new_poly.push(intersection(start, end));
        } else if !start_in && end_in {
            new_poly.push(intersection(start, end));
            new_poly.push(end);
        }
        // If both out, do nothing
        start = end;
    }

    new_poly
}

/// Clips the polygon against an arbitrary boundary polygon by explicitly
/// constructing the intersection polygon from vertices and segment intersections.
fn clip_polygon_with_boundary(subject: &[Point], boundary: &[Point]) -> Vec<Point> {
    let subject_clean = sanitize_polygon_ring(subject);
    let boundary_clean = sanitize_polygon_ring(boundary);
    if subject_clean.len() < 3 || boundary_clean.len() < 3 {
        return Vec::new();
    }

    let make_polygon = |points: &[Point]| -> Polygon<f64> {
        let mut coords: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
        if let Some(first) = coords.first().copied() {
            if coords
                .last()
                .map(|last| (last.0 - first.0).abs() > 1e-9 || (last.1 - first.1).abs() > 1e-9)
                .unwrap_or(false)
            {
                coords.push(first);
            }
        }
        Polygon::new(LineString::from(coords), vec![])
    };

    let subject_poly = make_polygon(&subject_clean);
    let boundary_poly = make_polygon(&boundary_clean);
    let mut best_poly: Option<Polygon<f64>> = None;
    let mut best_area = 0.0f64;

    let intersection = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        subject_poly.intersection(&boundary_poly)
    })) {
        Ok(result) => result,
        Err(payload) => {
            let message = crate::panic_payload_to_string(payload.as_ref());
            let backtrace = Backtrace::force_capture();
            eprintln!(
                "[fracture] geo::BooleanOps::intersection panicked (subject_vtx={}, boundary_vtx={}): {message}\n{}",
                subject_clean.len(),
                boundary_clean.len(),
                backtrace
            );
            return Vec::new();
        }
    };

    for poly in intersection {
        let area = poly.unsigned_area();
        if area > best_area {
            best_area = area;
            best_poly = Some(poly);
        }
    }

    let Some(poly) = best_poly else {
        return Vec::new();
    };

    if best_area <= 1.0e-9 {
        return Vec::new();
    }

    let mut clipped: Vec<Point> = poly
        .exterior()
        .0
        .iter()
        .map(|coord| Point::new(coord.x, coord.y))
        .collect();

    if clipped.len() >= 2 {
        if let (Some(first), Some(last)) = (clipped.first().copied(), clipped.last().copied()) {
            if points_equal(&first, &last) {
                clipped.pop();
            }
        }
    }

    if clipped.len() < 3 {
        return Vec::new();
    }

    ensure_ccw(&mut clipped);
    clipped
}

// ===============================================================================================
// MODULE: Voronoi Construction & Main Logic
// ===============================================================================================

pub struct VoronoiCell {
    pub site_index: usize,
    pub site: Point,
    pub vertices: Vec<Point>,
    pub color: String,
}

pub fn render_voronoi(
    mut points: Vec<Point>,
    width: f64,
    height: f64,
    mut rng: Rng,
    output: &str,
    boundary: Option<&[Point]>,
    show_sites: bool,
    dedup_points: bool,
    special_sites: &[Point],
) -> io::Result<()> {
    if let Some(boundary_poly) = boundary {
        let original_points = points.clone();
        points.retain(|p| point_in_polygon(p, boundary_poly));
        if points.len() < 3 {
            eprintln!(
                "[fracture] boundary filtering removed {} / {} points; keeping originals instead",
                original_points.len() - points.len(),
                original_points.len()
            );
            points = original_points;
        }
    }

    points.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    if dedup_points {
        points.dedup_by(|a, b| points_equal(a, b));
    }

    if points.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "need at least three points",
        ));
    }

    // print width and height
    println!(
        "[fracture] rendering Voronoi with width={} height={}",
        width, height
    );

    // print boundary coordinates and sites coordinates
    if let Some(boundary_poly) = boundary {
        println!("[fracture] boundary polygon:");
        for p in boundary_poly {
            println!("  ({}, {})", p.x, p.y);
        }
    }
    println!("[fracture] {} sites:", points.len());
    for (i, p) in points.iter().enumerate() {
        println!("  Site {}: ({}, {})", i, p.x, p.y);
    }

    let (cells, sites) = compute_voronoi_fracture(points, width, height, boundary, &mut rng);
    let svg_scale = 1000.0;

    write_voronoi_svg(
        output,
        width,
        height,
        &cells,
        &sites,
        boundary,
        show_sites,
        special_sites,
        svg_scale,
    )?;
    Ok(())
}

pub fn compute_voronoi_fracture(
    points: Vec<Point>,
    width: f64,
    height: f64,
    boundary: Option<&[Point]>,
    rng: &mut Rng,
) -> (Vec<VoronoiCell>, Vec<Point>) {
    let sites = points.clone();
    let (triangulation, _) = bowyer_watson(points, width, height);
    // println!(
    //     "Delaunay Triangulation complete. {} triangles generated.",
    //     triangulation.len()
    // );

    let mut cells = Vec::new();
    let mut min_cell_area = (width.abs() * height.abs()) * 1.0e-6;
    if !min_cell_area.is_finite() || min_cell_area <= 0.0 {
        min_cell_area = 1.0e-6;
    }

    for (s_idx, site) in sites.iter().enumerate() {
        let mut t_indices = Vec::new();
        for (t_idx, tri) in triangulation.iter().enumerate() {
            if points_equal(site, &tri.p1)
                || points_equal(site, &tri.p2)
                || points_equal(site, &tri.p3)
            {
                t_indices.push(t_idx);
            }
        }

        let site = sites[s_idx];
        if t_indices.is_empty() {
            continue;
        }

        let mut raw_poly = Vec::new();
        let mut valid_indices = t_indices;
        valid_indices.sort_unstable();
        valid_indices.dedup();
        valid_indices.sort_by(|&a, &b| {
            let t_a = triangulation[a];
            let t_b = triangulation[b];

            let c_a = Point::new(
                (t_a.p1.x + t_a.p2.x + t_a.p3.x) / 3.0,
                (t_a.p1.y + t_a.p2.y + t_a.p3.y) / 3.0,
            );
            let c_b = Point::new(
                (t_b.p1.x + t_b.p2.x + t_b.p3.x) / 3.0,
                (t_b.p1.y + t_b.p2.y + t_b.p3.y) / 3.0,
            );

            let ang_a = (c_a.y - site.y).atan2(c_a.x - site.x);
            let ang_b = (c_b.y - site.y).atan2(c_b.x - site.x);
            ang_a.partial_cmp(&ang_b).unwrap()
        });

        for &t_idx in &valid_indices {
            if let Some((cc, _)) = triangulation[t_idx].circumcenter() {
                raw_poly.push(cc);
            }
        }

        let mut clipped_poly = sutherland_hodgman(&raw_poly, width, height);
        if let Some(boundary_poly) = boundary {
            clipped_poly = clip_polygon_with_boundary(&clipped_poly, boundary_poly);
        }
        if clipped_poly.len() < 3 || polygon_area_abs(&clipped_poly) < min_cell_area {
            continue;
        }

        let r = (rng.next_f64() * 255.0) as u8;
        let g = (rng.next_f64() * 255.0) as u8;
        let b = (rng.next_f64() * 255.0) as u8;
        let color = format!("rgb({},{},{})", r, g, b);

        cells.push(VoronoiCell {
            site_index: s_idx,
            site,
            vertices: clipped_poly,
            color,
        });
    }

    (cells, sites)
}

fn write_voronoi_svg(
    output: &str,
    width: f64,
    height: f64,
    cells: &[VoronoiCell],
    sites: &[Point],
    boundary: Option<&[Point]>,
    show_sites: bool,
    special_sites: &[Point],
    scale: f64,
) -> io::Result<()> {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let width_scaled = width * scale;
    let height_scaled = height * scale;
    let mut file = File::create(output)?;
    writeln!(
        file,
        "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">",
        width_scaled, height_scaled
    )?;
    writeln!(
        file,
        "<rect width=\"{}\" height=\"{}\" fill=\"#eee\" />",
        width_scaled, height_scaled
    )?;

    writeln!(file, "  <g>")?;
    for cell in cells {
        write!(file, "    <polygon points=\"")?;
        for (i, p) in cell.vertices.iter().enumerate() {
            if i > 0 {
                write!(file, " ")?;
            }
            write!(file, "{},{}", p.x * scale, p.y * scale)?;
        }
        writeln!(
            file,
            "\" fill=\"{}\" stroke=\"#333\" stroke-width=\"1\" />",
            cell.color
        )?;
    }
    writeln!(file, "  </g>")?;

    if show_sites {
        writeln!(file, "  <g>")?;
        for site in sites {
            let is_special = special_sites.iter().any(|sp| points_equal(sp, site));
            let cx = site.x * scale;
            let cy = site.y * scale;
            if is_special {
                writeln!(
                    file,
                    "    <circle cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"red\" stroke=\"red\" stroke-width=\"1\" />",
                    cx,
                    cy
                )?;
                writeln!(
                    file,
                    "    <circle cx=\"{}\" cy=\"{}\" r=\"5\" fill=\"none\" stroke=\"red\" stroke-width=\"1\" />",
                    cx,
                    cy
                )?;
            } else {
                writeln!(
                    file,
                    "    <circle cx=\"{}\" cy=\"{}\" r=\"2\" fill=\"black\" />",
                    cx, cy
                )?;
            }
        }
        writeln!(file, "  </g>")?;
    }

    if let Some(boundary_poly) = boundary {
        write!(file, "  <polyline points=\"")?;
        for (i, p) in boundary_poly.iter().enumerate() {
            if i > 0 {
                write!(file, " ")?;
            }
            write!(file, "{},{}", p.x * scale, p.y * scale)?;
        }
        write!(file, " ")?;
        let first = boundary_poly[0];
        writeln!(
            file,
            "{},{}\" fill=\"none\" stroke=\"black\" stroke-width=\"2\" />",
            first.x * scale,
            first.y * scale
        )?;
    }

    writeln!(file, "</svg>")?;
    println!("Fracture pattern saved to '{output}'.");
    Ok(())
}

fn duck_outline_points(width: f64, height: f64) -> Vec<Point> {
    let outline_path = del_msh_core::io_svg::svg_outline_path_from_shape(DUCK_PATH);
    let loops = del_msh_core::io_svg::svg_loops_from_outline_path(&outline_path);
    assert!(!loops.is_empty(), "duck outline loops missing");
    let vtxl2xy =
        del_msh_core::io_svg::polybezier2polyloop(&loops[0].0, &loops[0].1, loops[0].2, 300.0);
    let mut flat: Vec<f32> = vtxl2xy.iter().flat_map(|p| [p[0], p[1]]).collect();
    flat = del_msh_core::polyloop::resample::<f32, 2>(&flat, 100);
    flat = del_msh_core::vtx2xy::normalize(&flat, &[0.5, 0.5], 1.0);

    let mut poly: Vec<Point> = flat
        .chunks_exact(2)
        .map(|chunk| Point::new(chunk[0] as f64 * width, chunk[1] as f64 * height))
        .collect();
    ensure_ccw(&mut poly);
    poly
}

fn sample_interior_point(
    rng: &mut Rng,
    boundary: &[Point],
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    max_attempts: usize,
) -> Option<Point> {
    for _ in 0..max_attempts {
        let candidate = Point::new(rng.range(min_x, max_x), rng.range(min_y, max_y));
        if point_in_polygon(&candidate, boundary) {
            return Some(candidate);
        }
    }
    None
}

fn add_cocircular_sites(
    points: &mut Vec<Point>,
    special_sites: &mut Vec<Point>,
    rng: &mut Rng,
    boundary: &[Point],
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
) {
    let bounding_span = (max_x - min_x).min(max_y - min_y);
    if let Some(center) = sample_interior_point(rng, boundary, min_x, max_x, min_y, max_y, 2_000) {
        let mut radius = bounding_span * 0.05;
        for _ in 0..6 {
            let mut circle_points = Vec::new();
            for i in 0..8 {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
                let candidate = Point::new(
                    center.x + radius * angle.cos(),
                    center.y + radius * angle.sin(),
                );
                if point_in_polygon(&candidate, boundary) {
                    circle_points.push(candidate);
                } else {
                    circle_points.clear();
                    break;
                }
            }
            if !circle_points.is_empty() {
                special_sites.extend(&circle_points);
                points.extend(circle_points);
                return;
            }
            radius *= 0.6;
        }
    }
}

fn add_colinear_and_coincident_sites(
    points: &mut Vec<Point>,
    special_sites: &mut Vec<Point>,
    rng: &mut Rng,
    boundary: &[Point],
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
) {
    let bounding_span = (max_x - min_x).min(max_y - min_y);
    if let Some(center) = sample_interior_point(rng, boundary, min_x, max_x, min_y, max_y, 2_000) {
        let mut spacing = bounding_span * 0.02;
        let offsets = [-1.5, -0.5, 0.5, 0.5];
        for _ in 0..6 {
            let mut line_points = Vec::new();
            for offset in offsets {
                let candidate = Point::new(center.x + offset * spacing, center.y);
                if point_in_polygon(&candidate, boundary) {
                    line_points.push(candidate);
                } else {
                    line_points.clear();
                    break;
                }
            }
            if !line_points.is_empty() {
                special_sites.extend(&line_points);
                points.extend(line_points);
                return;
            }
            spacing *= 0.6;
        }
    }
}

pub fn run_demo() -> io::Result<()> {
    let width = 800.0;
    let height = 800.0;
    let num_points = 50;
    let mut rng = Rng::new(123456789);

    let mut points = Vec::new();
    let special_sites: Vec<Point> = Vec::new();
    while points.len() < num_points {
        let p = Point::new(rng.range(0.0, width), rng.range(0.0, height));
        if !points.iter().any(|other| p.distance_sq(other) < 1.0) {
            points.push(p);
        }
    }

    points.push(Point::new(300.0, 300.0));
    points.push(Point::new(500.0, 300.0));
    points.push(Point::new(500.0, 500.0));
    points.push(Point::new(300.0, 500.0));
    points.push(Point::new(100.0, 100.0));
    points.push(Point::new(150.0, 150.0));
    points.push(Point::new(200.0, 200.0));

    render_voronoi(
        points,
        width,
        height,
        rng,
        "fracture.svg",
        None,
        true,
        true,
        &special_sites,
    )
}

pub fn run_duck_demo() -> io::Result<()> {
    let width = 800.0;
    let height = 800.0;
    let mut rng = Rng::new(987654321);
    let boundary = duck_outline_points(width, height);
    let (min_x, max_x, min_y, max_y) = bounding_box(&boundary);

    let mut points = Vec::new();
    let mut special_sites: Vec<Point> = Vec::new();
    let target_sites: usize = 220;
    let mut attempts = 0;
    let max_attempts = target_sites * 500;

    while points.len() < target_sites && attempts < max_attempts {
        attempts += 1;
        let candidate = Point::new(rng.range(min_x, max_x), rng.range(min_y, max_y));
        if point_in_polygon(&candidate, &boundary) {
            points.push(candidate);
        }
    }

    add_cocircular_sites(
        &mut points,
        &mut special_sites,
        &mut rng,
        &boundary,
        min_x,
        max_x,
        min_y,
        max_y,
    );
    add_colinear_and_coincident_sites(
        &mut points,
        &mut special_sites,
        &mut rng,
        &boundary,
        min_x,
        max_x,
        min_y,
        max_y,
    );

    if points.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to seed duck interior",
        ));
    }

    render_voronoi(
        points,
        width,
        height,
        rng,
        "fracture_duck.svg",
        Some(&boundary),
        true,
        false,
        &special_sites,
    )
}

fn parse_coordinate_list(line: &str) -> io::Result<Vec<Point>> {
    let start = line.find('[').ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing '[' in coordinate list")
    })? + 1;
    let end = line[start..]
        .rfind(']')
        .map(|idx| idx + start)
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing ']' in coordinate list")
        })?;
    let inner = line[start..end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let mut points = Vec::new();
    for entry in inner.split("),") {
        let trimmed = entry.trim().trim_start_matches('(').trim_end_matches(')');
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split(',');
        let x = parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing x coordinate"))?
            .trim()
            .parse::<f64>()
            .map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid x coordinate: {err}"),
                )
            })?;
        let y = parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing y coordinate"))?
            .trim()
            .parse::<f64>()
            .map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid y coordinate: {err}"),
                )
            })?;
        points.push(Point::new(x, y));
    }
    Ok(points)
}

fn load_boundary_from_diag(path: &Path) -> io::Result<Vec<Point>> {
    if !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("boundary diagnostics not found at {}", path.display()),
        ));
    }
    let content = std::fs::read_to_string(path)?;
    let mut latest: Option<Vec<Point>> = None;
    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("vertices=[") {
            let points = parse_coordinate_list(trimmed)?;
            latest = Some(points);
        }
    }
    let mut boundary = latest.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no boundary vertices found in {}", path.display()),
        )
    })?;
    if boundary.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected at least three boundary vertices in {}",
                path.display()
            ),
        ));
    }
    ensure_ccw(&mut boundary);
    Ok(boundary)
}

fn extract_key_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}=");
    line.split_whitespace()
        .find_map(|token| token.strip_prefix(&needle))
}

fn parse_position_pair(line: &str) -> io::Result<(f64, f64)> {
    let needle = "pos=(";
    let start = line
        .find(needle)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing pos=(...)"))?
        + needle.len();
    let end = line[start..]
        .find(')')
        .map(|idx| idx + start)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing closing ')' for pos"))?;
    let inner = &line[start..end];
    let mut parts = inner.split(',');
    let x = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing x in pos"))?
        .trim()
        .parse::<f64>()
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid x in pos: {err}"),
            )
        })?;
    let y = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing y in pos"))?
        .trim()
        .parse::<f64>()
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid y in pos: {err}"),
            )
        })?;
    Ok((x, y))
}

fn load_sites_from_diag(path: &Path) -> io::Result<Vec<Point>> {
    if !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("site diagnostics not found at {}", path.display()),
        ));
    }
    let content = std::fs::read_to_string(path)?;
    let mut latest: Vec<Point> = Vec::new();
    let mut current: Vec<Point> = Vec::new();
    let mut seen_iteration = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("iteration=") {
            if seen_iteration && !current.is_empty() {
                latest = current.clone();
            }
            current.clear();
            seen_iteration = true;
            continue;
        }
        if !trimmed.starts_with("site=") {
            continue;
        }
        let status = extract_key_value(trimmed, "status").unwrap_or("ok");
        if status == "inactive" {
            continue;
        }
        let (x, y) = parse_position_pair(trimmed)?;
        current.push(Point::new(x, y));
    }

    if !current.is_empty() {
        latest = current;
    }

    if latest.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "need at least three active site coordinates in {}",
                path.display()
            ),
        ));
    }

    Ok(latest)
}

pub fn run_voronoi_fracture_from_diag() -> io::Result<()> {
    println!("Loading diagnostics from disk...");

    let boundary_path = Path::new("target/boundary_diagnostics.txt");
    let site_path = Path::new("target/site_diagnostics.txt");
    let mut boundary = load_boundary_from_diag(boundary_path)?;
    // scale up boundary coordinates by 1000.0x
    // for p in &mut boundary {
    //     p.x *= 1000.0;
    //     p.y *= 1000.0;
    // }
    // print scaled boundary coordinates
    println!(
        "Loaded boundary with {} vertices from {}",
        boundary.len(),
        boundary_path.display()
    );
    // print vertex coordinates
    for (i, p) in boundary.iter().enumerate() {
        println!("  Vertex {}: ({}, {})", i, p.x, p.y);
    }

    let mut points = load_sites_from_diag(site_path)?;
    // scale up site coordinates by 1000.0x
    // for p in &mut points {
    //     p.x *= 1000.0;
    //     p.y *= 1000.0;
    // }
    // print scaled site coordinates
    println!("Loaded {} sites from {}", points.len(), site_path.display());

    // print site coordinates
    for (i, p) in points.iter().enumerate() {
        println!("  Site {}: ({}, {})", i, p.x, p.y);
    }

    if polygon_area_abs(&boundary) < 1.0e-6 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "boundary diagnostics describe a degenerate polygon",
        ));
    }

    let (min_x, max_x, min_y, max_y) = bounding_box(&boundary);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "boundary diagnostics produced non-finite coordinates",
        ));
    }
    let span_x = (max_x - min_x).abs().max(1.0);
    let span_y = (max_y - min_y).abs().max(1.0);
    let padding = (span_x.max(span_y)).max(1.0) * 0.05;
    let dx = -min_x + padding;
    let dy = -min_y + padding;
    for p in &mut boundary {
        p.x += dx;
        p.y += dy;
    }
    for p in &mut points {
        p.x += dx;
        p.y += dy;
    }

    let width = span_x + padding * 2.0;
    let height = span_y + padding * 2.0;
    let rng = Rng::new(0xD1A9_5701);

    // print width and height
    println!(
        "Computed canvas size: width = {}, height = {}",
        width, height
    );

    // print all point coordinates after translation
    for (i, p) in points.iter().enumerate() {
        println!("  Translated Site {}: ({}, {})", i, p.x, p.y);
    }

    // print all boundary vertex coordinates after translation
    for (i, p) in boundary.iter().enumerate() {
        println!("  Translated Boundary Vertex {}: ({}, {})", i, p.x, p.y);
    }

    render_voronoi(
        points,
        width,
        height,
        rng,
        "fracture_from_diag.svg",
        Some(&boundary),
        true,
        true,
        &[],
    )
}

#[test]
fn voronoi_fracture_demo_runs() -> io::Result<()> {
    run_demo()?;
    assert!(Path::new("fracture.svg").exists());
    Ok(())
}

#[test]
fn voronoi_fracture_duck_demo_runs() -> io::Result<()> {
    run_duck_demo()?;
    assert!(Path::new("fracture_duck.svg").exists());
    Ok(())
}
