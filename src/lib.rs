use anyhow::Context;
use candle_nn::Optimizer;
use del_candle::voronoi2::{Layer, VoronoiInfo};
use del_canvas_core::canvas_gif::Canvas;
use serde::Deserialize;
use std::any::Any;
use std::backtrace::Backtrace;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
mod delaunay;
pub mod fracture;
pub mod loss_topo;
mod voronoi;
pub use voronoi::VoronoiBackend;

const MY_PAINT_SVG_COLORS: [&str; 8] = [
    "#e6194B", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
];

static PROJECT_PARAMS: OnceLock<Vec<ProjectParams>> = OnceLock::new();
static ITER_TEXT_SCALE: OnceLock<usize> = OnceLock::new();

const ITER_TEXT_COLOR: u8 = 1;
const ITER_TEXT_MARGIN: usize = 4;
const FONT_HEIGHT: usize = 5;
const FONT_SPACING: usize = 1;
const ITER_TEXT_SCALE_ENV: &str = "FLOORPLAN_ITER_TEXT_SCALE";
const ITER_TEXT_SCALE_DEFAULT: f32 = 4.0;

#[derive(Debug, Deserialize, Clone)]
struct ProjectParams {
    #[serde(default)]
    loss_weights: LossWeights,
    #[serde(default)]
    learning_rates: LearningRates,
}

impl Default for ProjectParams {
    fn default() -> Self {
        Self {
            loss_weights: LossWeights::default(),
            learning_rates: LearningRates::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
struct LossWeights {
    #[serde(default = "LossWeights::default_each_area")]
    each_area: f32,
    #[serde(default = "LossWeights::default_total_area")]
    total_area: f32,
    #[serde(default = "LossWeights::default_wall_length")]
    wall_length: f32,
    #[serde(default = "LossWeights::default_wall_angle")]
    wall_angle: f32,
    #[serde(default = "LossWeights::default_topology")]
    topology: f32,
    #[serde(default = "LossWeights::default_fix")]
    fix: f32,
    #[serde(default = "LossWeights::default_lloyd")]
    lloyd: f32,
}

impl LossWeights {
    const fn default_wall_length() -> f32 {
        1.0
    }

    const fn default_wall_angle() -> f32 {
        1.0
    }

    const fn default_topology() -> f32 {
        10.0
    }

    const fn default_fix() -> f32 {
        50.0
    }

    const fn default_lloyd() -> f32 {
        0.5
    }

    const fn default_each_area() -> f32 {
        5.0
    }

    const fn default_total_area() -> f32 {
        20.0
    }
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            each_area: Self::default_each_area(),
            total_area: Self::default_total_area(),
            wall_length: Self::default_wall_length(),
            wall_angle: Self::default_wall_angle(),
            topology: Self::default_topology(),
            fix: Self::default_fix(),
            lloyd: Self::default_lloyd(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
struct LearningRates {
    #[serde(default = "LearningRates::default_first")]
    first: f32,
    #[serde(default = "LearningRates::default_second")]
    second: f32,
    #[serde(default = "LearningRates::default_third")]
    third: f32,
}

impl LearningRates {
    const fn default_first() -> f32 {
        0.05
    }
    const fn default_second() -> f32 {
        0.005
    }
    const fn default_third() -> f32 {
        0.001
    }

    fn value_for_step(&self, step: usize) -> f32 {
        if step >= 300 {
            self.third
        } else if step >= 150 {
            self.second
        } else {
            self.first
        }
    }
}

impl Default for LearningRates {
    fn default() -> Self {
        Self {
            first: Self::default_first(),
            second: Self::default_second(),
            third: Self::default_third(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LossBreakdown {
    pub each_area: f32,
    pub total_area: f32,
    pub wall_length: f32,
    pub wall_angle: f32,
    pub topology: f32,
    pub fix: f32,
    pub lloyd: f32,
    pub total: f32,
}

#[derive(Clone, Copy)]
struct Glyph {
    width: usize,
    rows: [u8; FONT_HEIGHT],
}

impl LossBreakdown {
    fn from_tensors(
        loss_each_area: &candle_core::Tensor,
        loss_total_area: &candle_core::Tensor,
        loss_walllen: &candle_core::Tensor,
        loss_wallangle: &candle_core::Tensor,
        loss_topo: &candle_core::Tensor,
        loss_fix: &candle_core::Tensor,
        loss_lloyd: &candle_core::Tensor,
        loss_total: &candle_core::Tensor,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            each_area: loss_each_area.to_scalar::<f32>()?,
            total_area: loss_total_area.to_scalar::<f32>()?,
            wall_length: loss_walllen.to_scalar::<f32>()?,
            wall_angle: loss_wallangle.to_scalar::<f32>()?,
            topology: loss_topo.to_scalar::<f32>()?,
            fix: loss_fix.to_scalar::<f32>()?,
            lloyd: loss_lloyd.to_scalar::<f32>()?,
            total: loss_total.to_scalar::<f32>()?,
        })
    }
}

fn glyph_for(ch: char) -> Option<Glyph> {
    match ch {
        '0' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b101, 0b101, 0b101, 0b111],
        }),
        '1' => Some(Glyph {
            width: 3,
            rows: [0b010, 0b110, 0b010, 0b010, 0b111],
        }),
        '2' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b001, 0b111, 0b100, 0b111],
        }),
        '3' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b001, 0b111, 0b001, 0b111],
        }),
        '4' => Some(Glyph {
            width: 3,
            rows: [0b101, 0b101, 0b111, 0b001, 0b001],
        }),
        '5' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b100, 0b111, 0b001, 0b111],
        }),
        '6' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b100, 0b111, 0b101, 0b111],
        }),
        '7' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b001, 0b010, 0b010, 0b010],
        }),
        '8' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b101, 0b111, 0b101, 0b111],
        }),
        '9' => Some(Glyph {
            width: 3,
            rows: [0b111, 0b101, 0b111, 0b001, 0b111],
        }),
        '/' => Some(Glyph {
            width: 3,
            rows: [0b001, 0b001, 0b010, 0b010, 0b100],
        }),
        _ => None,
    }
}

fn iteration_text_scale_multiplier() -> usize {
    *ITER_TEXT_SCALE.get_or_init(|| {
        let parsed = std::env::var(ITER_TEXT_SCALE_ENV)
            .ok()
            .and_then(|raw| raw.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(ITER_TEXT_SCALE_DEFAULT);
        let clamped = parsed.clamp(1.0, 64.0);
        clamped.round().max(1.0) as usize
    })
}

fn text_pixel_width(text: &str, scale: usize) -> usize {
    let mut width = 0usize;
    let mut first = true;
    for ch in text.chars() {
        if let Some(glyph) = glyph_for(ch) {
            if !first {
                width += FONT_SPACING * scale;
            }
            width += glyph.width * scale;
            first = false;
        }
    }
    width
}

fn draw_glyph(
    canvas: &mut Canvas,
    glyph: Glyph,
    origin_x: usize,
    origin_y: usize,
    color: u8,
    scale: usize,
) {
    if scale == 0 {
        return;
    }
    for (row_idx, row_bits) in glyph.rows.iter().enumerate() {
        for sy in 0..scale {
            let y = origin_y + row_idx * scale + sy;
            if y >= canvas.height {
                break;
            }
            for col in 0..glyph.width {
                let bit = 1 << (glyph.width - 1 - col);
                if row_bits & bit == 0 {
                    continue;
                }
                for sx in 0..scale {
                    let x = origin_x + col * scale + sx;
                    if x >= canvas.width {
                        break;
                    }
                    canvas.data[y * canvas.width + x] = color;
                }
            }
        }
    }
}

fn draw_text(
    canvas: &mut Canvas,
    text: &str,
    start_x: usize,
    start_y: usize,
    color: u8,
    scale: usize,
) {
    let mut cursor = start_x;
    let mut first = true;
    for ch in text.chars() {
        if let Some(glyph) = glyph_for(ch) {
            if !first {
                cursor += FONT_SPACING * scale;
            }
            draw_glyph(canvas, glyph, cursor, start_y, color, scale);
            cursor += glyph.width * scale;
            first = false;
        }
    }
}

fn overlay_iteration_counter(canvas: &mut Canvas, iter_idx: usize, total_iters: usize) {
    if canvas.width == 0 || canvas.height == 0 {
        return;
    }
    let scale = iteration_text_scale_multiplier();
    let display_total = total_iters.max(1);
    let text = format!("{:04}/{:04}", iter_idx + 1, display_total);
    let text_width = text_pixel_width(&text, scale);
    if text_width == 0 {
        return;
    }
    let text_height = FONT_HEIGHT * scale;
    if text_height >= canvas.height {
        return;
    }
    let origin_x = canvas
        .width
        .saturating_sub(text_width + ITER_TEXT_MARGIN)
        .min(canvas.width.saturating_sub(1));
    let origin_y = canvas
        .height
        .saturating_sub(text_height + ITER_TEXT_MARGIN)
        .min(canvas.height.saturating_sub(1));
    draw_text(canvas, &text, origin_x, origin_y, ITER_TEXT_COLOR, scale);
}

fn load_project_params() -> anyhow::Result<Vec<ProjectParams>> {
    let mut param_files: Vec<PathBuf> = std::fs::read_dir(Path::new("."))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_ascii_lowercase())
                    .map(|name| name.starts_with("param") && name.ends_with(".toml"))
                    .unwrap_or(false)
        })
        .collect();
    param_files.sort();

    if param_files.is_empty() {
        let legacy = Path::new("ex__params.toml");
        if legacy.exists() {
            param_files.push(legacy.to_path_buf());
        }
    }

    if param_files.is_empty() {
        return Ok(vec![ProjectParams::default()]);
    }

    let mut params_list = Vec::with_capacity(param_files.len());
    for path in param_files {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let params: ProjectParams =
            toml::from_str(&raw).with_context(|| format!("Failed to parse {}", path.display()))?;
        params_list.push(params);
    }
    Ok(params_list)
}

fn all_project_params() -> &'static Vec<ProjectParams> {
    PROJECT_PARAMS.get_or_init(|| {
        load_project_params().unwrap_or_else(|err| {
            eprintln!(
                "[floorplan] Failed to load param*.toml files ({}); using a single default",
                err
            );
            vec![ProjectParams::default()]
        })
    })
}

#[allow(dead_code)]
pub(crate) fn project_params(index: usize) -> &'static ProjectParams {
    all_project_params()
        .get(index)
        .unwrap_or_else(|| panic!("project parameters index {} out of range", index))
}

pub(crate) fn project_params_all() -> &'static [ProjectParams] {
    all_project_params()
}

pub fn my_paint(
    canvas: &mut Canvas,
    transform_to_scr: &nalgebra::Matrix3<f32>,
    vtxl2xy: &[f32],
    site2xy: &[f32],
    voronoi_info: &VoronoiInfo,
    vtxv2xy: &[f32],
    site2room: &[usize],
    edge2vtxv_wall: &[usize],
) {
    let site2idx = &voronoi_info.site2idx;
    let idx2vtxv = &voronoi_info.idx2vtxv;
    let idx2site = &voronoi_info.idx2site;
    let site_count = site2idx.len().saturating_sub(1);
    if site_count == 0 {
        return;
    }

    let transform = arrayref::array_ref![transform_to_scr.as_slice(), 0, 9];
    let max_site_vertices = site2idx
        .windows(2)
        .map(|window| window[1] - window[0])
        .max()
        .unwrap_or(0);
    let mut polygon_xy = vec![0f32; max_site_vertices.saturating_mul(2)];
    #[cfg(debug_assertions)]
    let mut skipped_sites = 0usize;

    for i_site in 0..site_count {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            #[cfg(debug_assertions)]
            {
                skipped_sites += 1;
            }
            continue;
        }
        //
        let i_color: u8 = (i_room + 2).try_into().unwrap();

        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        if num_vtx_in_site == 0 {
            continue;
        }
        let start = site2idx[i_site];
        let end = site2idx[i_site + 1];
        let site_vertices = &idx2vtxv[start..end];
        let slice_len = num_vtx_in_site * 2;
        {
            let slice = &mut polygon_xy[..slice_len];
            for (dst, &i_vtxv) in slice.chunks_exact_mut(2).zip(site_vertices.iter()) {
                dst[0] = vtxv2xy[i_vtxv * 2];
                dst[1] = vtxv2xy[i_vtxv * 2 + 1];
            }
        }
        del_canvas_core::rasterize_polygon::fill(
            &mut canvas.data,
            canvas.width,
            &polygon_xy[..slice_len],
            transform,
            i_color,
        );
        /*
        for i0_vtx in 0..num_vtx_in_site-2 {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let i2_vtx = (i0_vtx + 2) % num_vtx_in_site;
            let i0 = idx2vtxv[site2idx[i_site]];
            let i1 = idx2vtxv[site2idx[i_site] + i1_vtx];
            let i2 = idx2vtxv[site2idx[i_site] + i2_vtx];
            del_canvas_core::rasterize_triangle::fill::<usize,f32,u8>(
                &mut canvas.data,
                canvas.width,
                &[vtxv2xy[i0 * 2 + 0], vtxv2xy[i0 * 2 + 1]],
                &[vtxv2xy[i1 * 2 + 0], vtxv2xy[i1 * 2 + 1]],
                &[vtxv2xy[i2 * 2 + 0], vtxv2xy[i2 * 2 + 1]],
                arrayref::array_ref![transform_to_scr.as_slice(),0,9],
                i_color,
            );
        }
         */
    }

    #[cfg(debug_assertions)]
    if skipped_sites > 0 {
        eprintln!(
            "[floorplan] Skipped {skipped_sites} site(s) with no room assignment during my_paint"
        );
    }

    // draw points;
    for (i_site, site_xy) in site2xy.chunks_exact(2).enumerate() {
        let &[x, y] = site_xy else { unreachable!() };
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        del_canvas_core::rasterize_circle::fill(
            &mut canvas.data,
            canvas.width,
            &[x, y],
            transform,
            2.0,
            // black dot
            255, //i_color,
        );
    }

    // print check point time
    // println!("Check point time: {:?} at draw cell boundary", std::time::Instant::now());

    // draw cell boundary once per shared Voronoi edge
    for i_site in 0..site_count {
        let start = site2idx[i_site];
        let end = site2idx[i_site + 1];
        let num_vtx_in_site = end - start;
        if num_vtx_in_site == 0 {
            continue;
        }
        for (local_idx, edge_idx) in (start..end).enumerate() {
            let neighbor_site = idx2site[edge_idx];
            if neighbor_site != usize::MAX && neighbor_site < i_site {
                continue;
            }
            let next_local = (local_idx + 1) % num_vtx_in_site;
            let i0 = idx2vtxv[edge_idx];
            let i1 = idx2vtxv[start + next_local];
            let p0 = &[vtxv2xy[i0 * 2 + 0], vtxv2xy[i0 * 2 + 1]];
            let p1 = &[vtxv2xy[i1 * 2 + 0], vtxv2xy[i1 * 2 + 1]];
            draw_site_edge_line(
                canvas, transform, p0, p1, i_site, local_idx, next_local, i0, i1,
            );
        }
    }

    // println!("Check point time: {:?} at draw room boundary", std::time::Instant::now());

    // draw room boundary
    for edge in edge2vtxv_wall.chunks_exact(2) {
        let &[i0_vtxv, i1_vtxv] = edge else {
            unreachable!()
        };
        del_canvas_core::rasterize_line::draw_pixcenter(
            &mut canvas.data,
            canvas.width,
            &[vtxv2xy[i0_vtxv * 2 + 0], vtxv2xy[i0_vtxv * 2 + 1]],
            &[vtxv2xy[i1_vtxv * 2 + 0], vtxv2xy[i1_vtxv * 2 + 1]],
            transform,
            1.6,
            1,
        );
    }

    // println!("Check point time: {:?} at rasterize polygon stroke", std::time::Instant::now());

    stroke_layout_outline(canvas, vtxl2xy, transform);

    // match dump_voronoi_svg_snapshot(vtxl2xy, site2xy, voronoi_info, vtxv2xy, site2room, 1000.0) {
    //     Ok(Some(path)) => println!("[floorplan] Wrote Voronoi SVG to {}", path),
    //     Ok(None) => {}
    //     Err(err) => eprintln!("[floorplan] Failed to write Voronoi SVG: {}", err),
    // }
}

#[cfg(debug_assertions)]
fn draw_site_edge_line(
    canvas: &mut Canvas,
    transform: &[f32; 9],
    p0: &[f32; 2],
    p1: &[f32; 2],
    i_site: usize,
    i0_vtx: usize,
    i1_vtx: usize,
    i0: usize,
    i1: usize,
) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        del_canvas_core::rasterize_line::draw_dda_with_transformation(
            &mut canvas.data,
            canvas.width,
            p0,
            p1,
            transform,
            1,
        );
    })) {
        let bt = Backtrace::force_capture();
        eprintln!(
            "[floorplan] rasterize_line::draw_dda_with_transformation panicked (i_site={i_site}, i0_vtx={i0_vtx}, i1_vtx={i1_vtx}, i0={i0}, i1={i1}, p0={p0:?}, p1={p1:?})\n{}\n{}",
            panic_payload_to_string(payload.as_ref()),
            bt
        );
        std::process::exit(1);
    }
}

#[cfg(not(debug_assertions))]
fn draw_site_edge_line(
    canvas: &mut Canvas,
    transform: &[f32; 9],
    p0: &[f32; 2],
    p1: &[f32; 2],
    _i_site: usize,
    _i0_vtx: usize,
    _i1_vtx: usize,
    _i0: usize,
    _i1: usize,
) {
    del_canvas_core::rasterize_line::draw_dda_with_transformation(
        &mut canvas.data,
        canvas.width,
        p0,
        p1,
        transform,
        1,
    );
}

#[cfg(debug_assertions)]
fn stroke_layout_outline(canvas: &mut Canvas, vtxl2xy: &[f32], transform: &[f32; 9]) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        del_canvas_core::rasterize_polygon::stroke(
            &mut canvas.data,
            canvas.width,
            vtxl2xy,
            transform,
            1.6,
            1,
        );
    })) {
        let bt = Backtrace::force_capture();
        eprintln!(
            "[floorplan] rasterize_polygon::stroke panicked: {}\n{}",
            panic_payload_to_string(payload.as_ref()),
            bt
        );
        std::process::exit(1);
    }
}

#[cfg(not(debug_assertions))]
fn stroke_layout_outline(canvas: &mut Canvas, vtxl2xy: &[f32], transform: &[f32; 9]) {
    del_canvas_core::rasterize_polygon::stroke(
        &mut canvas.data,
        canvas.width,
        vtxl2xy,
        transform,
        1.6,
        1,
    );
}

fn dump_voronoi_svg_snapshot(
    vtxl2xy: &[f32],
    site2xy: &[f32],
    voronoi_info: &VoronoiInfo,
    vtxv2xy: &[f32],
    site2room: &[usize],
    scaleup: f32,
) -> std::io::Result<Option<String>> {
    if voronoi_info.site2idx.len() < 2 {
        return Ok(None);
    }
    let scale = if scaleup.is_finite() && scaleup > 0.0 {
        scaleup
    } else {
        1.0
    };
    let mut cells: Vec<Vec<[f32; 2]>> = Vec::with_capacity(voronoi_info.site2idx.len() - 1);
    let mut all_points: Vec<[f32; 2]> = Vec::new();
    for chunk in vtxl2xy.chunks(2) {
        if let [x, y] = chunk {
            all_points.push([*x, *y]);
        }
    }
    for chunk in site2xy.chunks(2) {
        if let [x, y] = chunk {
            all_points.push([*x, *y]);
        }
    }
    for i_site in 0..voronoi_info.site2idx.len() - 1 {
        let start = voronoi_info.site2idx[i_site];
        let end = voronoi_info.site2idx[i_site + 1];
        let mut cell = Vec::new();
        for idx in start..end {
            let i_vtxv = voronoi_info.idx2vtxv[idx];
            let x = vtxv2xy.get(i_vtxv * 2).copied().unwrap_or(0.0);
            let y = vtxv2xy.get(i_vtxv * 2 + 1).copied().unwrap_or(0.0);
            let coord = [x, y];
            all_points.push(coord);
            cell.push(coord);
        }
        cells.push(cell);
    }
    let bounds = match bounding_box(&all_points) {
        Some(b) => b,
        None => return Ok(None),
    };
    let (min_x, max_x, min_y, max_y) = bounds;
    let margin = ((max_x - min_x).max(max_y - min_y)).max(1.0e-3) * 0.05;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_millis())
        .unwrap_or_default();
    let path = format!("target/voronoi_runtime_{}.svg", timestamp);
    if let Some(parent) = Path::new(&path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(&path)?;
    let width_unscaled = max_x - min_x + margin * 2.0;
    let height_unscaled = max_y - min_y + margin * 2.0;
    let width = width_unscaled * scale;
    let height = height_unscaled * scale;
    let view_min_x = (min_x - margin) * scale;
    let view_min_y = (min_y - margin) * scale;
    writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\">",
        view_min_x, view_min_y, width, height
    )?;
    if vtxl2xy.len() >= 4 {
        let mut poly = String::new();
        for chunk in vtxl2xy.chunks(2) {
            if let [x, y] = chunk {
                poly.push_str(&format!("{} {},", x * scale, y * scale));
            }
        }
        writeln!(
            file,
            "<polygon points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"{}\"/>",
            poly.trim_end_matches(','),
            width * 0.002
        )?;
    }
    for (site_idx, cell) in cells.iter().enumerate() {
        if cell.len() < 2 {
            continue;
        }
        let mut poly = String::new();
        for [x, y] in cell {
            poly.push_str(&format!("{} {},", x * scale, y * scale));
        }
        let room = site2room.get(site_idx).copied().unwrap_or(usize::MAX);
        let color = if room == usize::MAX {
            "#999999"
        } else {
            MY_PAINT_SVG_COLORS[room % MY_PAINT_SVG_COLORS.len()]
        };
        writeln!(
            file,
            "<polygon points=\"{}\" fill=\"{}\" fill-opacity=\"0.15\" stroke=\"{}\" stroke-width=\"{}\"/>",
            poly.trim_end_matches(','),
            color,
            color,
            width * 0.0015
        )?;
    }
    for (idx, chunk) in site2xy.chunks(2).enumerate() {
        if let [x, y] = chunk {
            let x_scaled = x * scale;
            let y_scaled = y * scale;
            writeln!(
                file,
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#111\" fill-opacity=\"0.8\"/>",
                x_scaled,
                y_scaled,
                width * 0.0025
            )?;
            writeln!(
                file,
                "<text x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"#000\">{}</text>",
                x_scaled + width * 0.002,
                y_scaled - width * 0.002,
                width * 0.005,
                idx
            )?;
        }
    }
    writeln!(file, "</svg>")?;
    Ok(Some(path))
}

fn bounding_box(points: &[[f32; 2]]) -> Option<(f32, f32, f32, f32)> {
    if points.is_empty() {
        return None;
    }
    let mut min_x = points[0][0];
    let mut max_x = points[0][0];
    let mut min_y = points[0][1];
    let mut max_y = points[0][1];
    for [x, y] in points.iter().skip(1) {
        if *x < min_x {
            min_x = *x;
        }
        if *x > max_x {
            max_x = *x;
        }
        if *y < min_y {
            min_y = *y;
        }
        if *y > max_y {
            max_y = *y;
        }
    }
    Some((min_x, max_x, min_y, max_y))
}

pub fn draw_svg(
    file_path: String,
    transform_to_scr: &nalgebra::Matrix3<f32>,
    vtxl2xy: &[f32],
    site2xy: &[f32],
    voronoi_info: &VoronoiInfo,
    vtxv2xy: &[f32],
    site2room: &[usize],
    edge2vtxv_wall: &[usize],
    room2color: &[i32],
) {
    let mut canvas_svg = del_canvas_core::canvas_svg::Canvas::new(file_path, (300, 300));
    {
        //        let vtxv2xy = vtxv2xy.flatten_all()?.to_vec1()?;
        for i_site in 0..voronoi_info.site2idx.len() - 1 {
            let mut hoge = vec![];
            for &i_vtxv in &voronoi_info.idx2vtxv
                [voronoi_info.site2idx[i_site]..voronoi_info.site2idx[i_site + 1]]
            {
                hoge.push(vtxv2xy[i_vtxv * 2 + 0]);
                hoge.push(vtxv2xy[i_vtxv * 2 + 1]);
            }
            let i_room = site2room[i_site];
            let i_color = room2color[i_room];
            canvas_svg.polyloop(
                &hoge,
                &transform_to_scr,
                Some(0x333333),
                Some(0.1),
                Some(i_color),
            );
        }
        for i_edge in 0..edge2vtxv_wall.len() / 2 {
            let i0_vtxv = edge2vtxv_wall[i_edge * 2 + 0];
            let i1_vtxv = edge2vtxv_wall[i_edge * 2 + 1];
            let x0 = vtxv2xy[i0_vtxv * 2 + 0];
            let y0 = vtxv2xy[i0_vtxv * 2 + 1];
            let x1 = vtxv2xy[i1_vtxv * 2 + 0];
            let y1 = vtxv2xy[i1_vtxv * 2 + 1];
            canvas_svg.line(x0, y0, x1, y1, &transform_to_scr, Some(2.0));
        }
    }
    canvas_svg.polyloop(vtxl2xy, &transform_to_scr, Some(0x000000), Some(2.0), None);
    {
        //let site2xy = site2xy.flatten_all()?.to_vec1()?;
        for i_vtx in 0..site2xy.len() / 2 {
            canvas_svg.circle(
                site2xy[i_vtx * 2 + 0],
                site2xy[i_vtx * 2 + 1],
                &transform_to_scr,
                1.,
                "#FF0000",
            );
        }
    }
    canvas_svg.write();
}

pub fn random_room_color<RNG>(reng: &mut RNG) -> i32
where
    RNG: rand::Rng,
{
    let h = reng.random::<f32>();
    let s = 0.5 + 0.1 * reng.random::<f32>();
    let v = 0.9 + 0.1 * reng.random::<f32>();
    let (r, g, b) = del_canvas_core::color::rgb_from_hsv(h, s, v);
    let r = (r * 255.0) as u8;
    let g = (g * 255.0) as u8;
    let b = (b * 255.0) as u8;
    del_canvas_core::color::i32_form_u8rgb(r, g, b)
}

fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let (x, y) = point;
    let mut inside = false;
    if polygon.is_empty() {
        return inside;
    }
    for i in 0..polygon.len() {
        let (x0, y0) = polygon[i];
        let (x1, y1) = polygon[(i + 1) % polygon.len()];
        let intersects =
            ((y0 > y) != (y1 > y)) && (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-9_f32) + x0);
        if intersects {
            inside = !inside;
        }
    }
    inside
}

fn add_sample(
    pt: (f32, f32),
    samples: &mut Vec<(f32, f32)>,
    active: &mut Vec<usize>,
    grid: &mut [i32],
    min_x: f32,
    min_y: f32,
    cell: f32,
    grid_w: i32,
    grid_h: i32,
) {
    let idx = samples.len();
    samples.push(pt);
    active.push(idx);
    let gx = (((pt.0 - min_x) / cell).floor() as i32).clamp(0, grid_w - 1);
    let gy = (((pt.1 - min_y) / cell).floor() as i32).clamp(0, grid_h - 1);
    grid[(gy * grid_w + gx) as usize] = idx as i32;
}

pub fn poisson_disk_sampling<RNG>(
    polygon: &[(f32, f32)],
    radius: f32,
    k: usize,
    rng: &mut RNG,
) -> Vec<f32>
where
    RNG: rand::Rng + ?Sized,
{
    if polygon.len() < 3 || radius <= 0.0 {
        return Vec::new();
    }

    dbg!("Starting Poisson disk sampling...");
    dbg!(polygon.len());

    let mut min_x = polygon[0].0;
    let mut max_x = polygon[0].0;
    let mut min_y = polygon[0].1;
    let mut max_y = polygon[0].1;
    for &(x, y) in polygon.iter().skip(1) {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    dbg!(min_x, max_x, min_y, max_y);

    let cell = radius / std::f32::consts::SQRT_2;
    if cell <= 0.0 {
        return Vec::new();
    }
    let grid_w = (((max_x - min_x) / cell).floor() as i32 + 1).max(1);
    let grid_h = (((max_y - min_y) / cell).floor() as i32 + 1).max(1);
    let mut grid = vec![-1i32; (grid_w * grid_h) as usize];

    let mut samples: Vec<(f32, f32)> = Vec::new();
    let mut active: Vec<usize> = Vec::new();

    fn sample_range<R: rand::Rng + ?Sized>(rng: &mut R, start: f32, end: f32) -> f32 {
        if (end - start).abs() <= f32::EPSILON {
            start
        } else {
            rng.random_range(start..end)
        }
    }

    while samples.is_empty() {
        let x = sample_range(rng, min_x, max_x);
        let y = sample_range(rng, min_y, max_y);
        let candidate = (x, y);
        //dbg!( "Trying initial sample at ({}, {})", x, y );
        if point_in_polygon(candidate, polygon) {
            //dbg!("Adding initial sample at ({}, {})", x, y);
            add_sample(
                candidate,
                &mut samples,
                &mut active,
                &mut grid,
                min_x,
                min_y,
                cell,
                grid_w,
                grid_h,
            );
        }
    }

    dbg!("samples length: {}", samples.len());

    while !active.is_empty() {
        let idx = rng.random_range(0..active.len());
        let base_idx = active[idx];
        let base = samples[base_idx];
        let mut found = false;
        for _ in 0..k {
            let angle = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            let dist = rng.random_range(radius..(2.0 * radius));
            let candidate = (base.0 + angle.cos() * dist, base.1 + angle.sin() * dist);
            if candidate.0 < min_x
                || candidate.0 > max_x
                || candidate.1 < min_y
                || candidate.1 > max_y
            {
                continue;
            }
            if !point_in_polygon(candidate, polygon) {
                continue;
            }
            let gx = (((candidate.0 - min_x) / cell).floor() as i32).clamp(0, grid_w - 1);
            let gy = (((candidate.1 - min_y) / cell).floor() as i32).clamp(0, grid_h - 1);
            let mut ok = true;
            let x_start = (gx - 2).max(0);
            let x_end = (gx + 2).min(grid_w - 1);
            let y_start = (gy - 2).max(0);
            let y_end = (gy + 2).min(grid_h - 1);
            'outer: for nx in x_start..=x_end {
                for ny in y_start..=y_end {
                    let neighbor_idx = grid[(ny * grid_w + nx) as usize];
                    if neighbor_idx == -1 {
                        continue;
                    }
                    let neighbor = samples[neighbor_idx as usize];
                    if ((candidate.0 - neighbor.0).powi(2) + (candidate.1 - neighbor.1).powi(2))
                        .sqrt()
                        < radius
                    {
                        ok = false;
                        break 'outer;
                    }
                }
            }
            if ok {
                add_sample(
                    candidate,
                    &mut samples,
                    &mut active,
                    &mut grid,
                    min_x,
                    min_y,
                    cell,
                    grid_w,
                    grid_h,
                );
                found = true;
                break;
            }
        }
        if !found {
            active.swap_remove(idx);
        }
    }

    let mut flat = Vec::with_capacity(samples.len() * 2);
    for (x, y) in samples {
        flat.push(x);
        flat.push(y);
    }
    flat
}

pub fn edge2vtvx_wall(voronoi_info: &VoronoiInfo, site2room: &[usize]) -> Vec<usize> {
    let site2idx = &voronoi_info.site2idx;
    let idx2vtxv = &voronoi_info.idx2vtxv;
    let mut edge2vtxv = vec![0usize; 0];
    // get wall between rooms
    for i_site in 0..site2idx.len() - 1 {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let idx = site2idx[i_site] + i0_vtx;
            let i0_vtxv = idx2vtxv[idx];
            let i1_vtxv = idx2vtxv[site2idx[i_site] + i1_vtx];
            let j_site = voronoi_info.idx2site[idx];
            if j_site == usize::MAX {
                continue;
            }
            if i_site >= j_site {
                continue;
            }
            let j_room = site2room[j_site];
            if i_room == j_room {
                continue;
            }
            edge2vtxv.push(i0_vtxv);
            edge2vtxv.push(i1_vtxv);
        }
    }
    edge2vtxv
}

pub fn loss_lloyd_internal(
    voronoi_info: &VoronoiInfo,
    site2room: &[usize],
    site2xy: &candle_core::Var,
    vtxv2xy: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let num_site = site2room.len();
    assert_eq!(voronoi_info.site2idx.len() - 1, num_site);
    let site2idx = &voronoi_info.site2idx;
    // let idx2vtxv = &voronoi_info.idx2vtxv;
    let mut site2canmove = vec![false; num_site];
    // get wall between rooms
    for i_site in 0..site2idx.len() - 1 {
        if voronoi_info.site2idx[i_site + 1] == voronoi_info.site2idx[i_site] {
            // there is no cell
            continue;
        }
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let idx = site2idx[i_site] + i0_vtx;
            let j_site = voronoi_info.idx2site[idx];
            if j_site == usize::MAX {
                continue;
            }
            if i_site >= j_site {
                continue;
            }
            let j_room = site2room[j_site];
            if i_room == j_room {
                continue;
            }
            site2canmove[i_site] = true;
        }
    }
    // dbg!(&site2canmove);
    let mask: Vec<f32> = site2canmove
        .iter()
        .flat_map(|v| if *v { [0f32, 0f32] } else { [1f32, 1f32] })
        .collect();
    let mask = candle_core::Tensor::from_vec(mask, (num_site, 2), &candle_core::Device::Cpu)?;
    let polygonmesh2_to_cogs = del_candle::polygonmesh2_to_cogs::Layer {
        elem2idx: Vec::from(voronoi_info.site2idx.clone()),
        idx2vtx: Vec::from(voronoi_info.idx2vtxv.clone()),
    };
    let site2cogs = vtxv2xy.apply_op1(polygonmesh2_to_cogs)?;
    let diff = site2xy.sub(&site2cogs)?;
    let diffmasked = mask.mul(&diff)?;
    diffmasked.sqr().unwrap().sum_all()
}

pub fn room2area(
    site2room: &[usize],
    num_room: usize,
    site2idx: &[usize],
    idx2vtxv: &[usize],
    vtxv2xy: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let polygonmesh2_to_areas = del_candle::polygonmesh2_to_areas::Layer {
        elem2idx: Vec::<usize>::from(site2idx),
        idx2vtx: Vec::<usize>::from(idx2vtxv),
    };
    let site2areas = vtxv2xy.apply_op1(polygonmesh2_to_areas)?;
    let site2areas = site2areas.reshape((site2areas.dim(0).unwrap(), 1))?; // change shape to use .mutmul()
                                                                           //
    let num_site = site2room.len();
    let sum_sites_for_rooms = {
        let mut sum_sites_for_rooms = vec![0f32; num_site * num_room];
        for i_site in 0..num_site {
            let i_room = site2room[i_site];
            if i_room == usize::MAX {
                continue;
            }
            assert!(i_room < num_room);
            sum_sites_for_rooms[i_room * num_site + i_site] = 1f32;
        }
        candle_core::Tensor::from_slice(
            &sum_sites_for_rooms,
            candle_core::Shape::from((num_room, num_site)),
            &candle_core::Device::Cpu,
        )?
    };
    sum_sites_for_rooms.matmul(&site2areas)
}

pub fn remove_site_too_close(site2room: &mut [usize], site2xy: &candle_core::Tensor) {
    assert_eq!(site2room.len(), site2xy.dims2().unwrap().0);
    let num_site = site2room.len();
    let site2xy = site2xy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for i_site in 0..num_site {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        let p_i = site_vec(&site2xy, i_site);
        for j_site in (i_site + 1)..num_site {
            let j_room = site2room[j_site];
            if j_room == usize::MAX {
                continue;
            }
            if i_room != j_room {
                continue;
            }
            let p_j = site_vec(&site2xy, j_site);
            if (p_i - p_j).norm() < 0.02 {
                site2room[j_site] = usize::MAX;
            }
        }
    }
}

fn site_vec(site2xy: &[f32], i_site: usize) -> nalgebra::Vector2<f32> {
    let coords = del_msh_core::vtx2xy::to_vec2(site2xy, i_site);
    nalgebra::Vector2::<f32>::new(coords[0], coords[1])
}

fn enforce_min_site_distance(coords: &mut [f32], min_distance: f32) {
    if coords.len() < 4 || min_distance <= 0.0 {
        return;
    }
    let num_site = coords.len() / 2;
    let mut min_sq = min_distance * min_distance;
    if !min_sq.is_finite() {
        return;
    }
    min_sq = min_sq.max(0.0);
    for i in 0..num_site {
        let i_idx = i * 2;
        for j in (i + 1)..num_site {
            let j_idx = j * 2;
            let mut dx = coords[j_idx] - coords[i_idx];
            let mut dy = coords[j_idx + 1] - coords[i_idx + 1];
            let dist_sq = dx * dx + dy * dy;
            if dist_sq >= min_sq {
                continue;
            }
            let dist = dist_sq.sqrt();
            if dist <= 1.0e-9 {
                let angle = ((i + j) as f32 * 12.9898).sin();
                dx = angle.cos();
                dy = angle.sin();
            } else {
                dx /= dist;
                dy /= dist;
            }
            let push = (min_distance - dist.max(1.0e-9)) * 0.5;
            coords[i_idx] -= dx * push;
            coords[i_idx + 1] -= dy * push;
            coords[j_idx] += dx * push;
            coords[j_idx + 1] += dy * push;
        }
    }
}

fn find_overlapping_sites(coords: &[f32], tolerance: f32) -> Option<(usize, usize)> {
    if coords.len() < 4 {
        return None;
    }
    let num_site = coords.len() / 2;
    for i in 0..num_site {
        let xi = coords[i * 2];
        let yi = coords[i * 2 + 1];
        for j in (i + 1)..num_site {
            let xj = coords[j * 2];
            let yj = coords[j * 2 + 1];
            if (xi - xj).abs() <= tolerance && (yi - yj).abs() <= tolerance {
                return Some((i, j));
            }
        }
    }
    None
}

fn enforce_site_spacing(
    site2xy: &candle_core::Var,
    min_site_radius: f32,
) -> candle_core::Result<candle_core::Tensor> {
    let coords = site2xy.flatten_all()?.to_vec1::<f32>()?;
    let num_site = if coords.is_empty() {
        0
    } else {
        coords.len() / 2
    };
    let mut adjusted = coords.clone();
    let min_site_distance = min_site_radius.max(1.0e-6);
    enforce_min_site_distance(&mut adjusted, min_site_distance);
    let delta: Vec<f32> = adjusted
        .iter()
        .zip(coords.iter())
        .map(|(adj, orig)| adj - orig)
        .collect();
    let delta_tensor = candle_core::Tensor::from_vec(
        delta,
        candle_core::Shape::from((num_site, 2)),
        &candle_core::Device::Cpu,
    )?;
    site2xy.add(&delta_tensor)
}

pub fn site2room(num_site: usize, room2area: &[f32]) -> Vec<usize> {
    let num_room = room2area.len();
    let mut site2room: Vec<usize> = vec![usize::MAX; num_site];
    let num_site_assign = num_site - num_room;
    let area: f32 = room2area.iter().sum();
    {
        let cumsum: Vec<f32> = room2area
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        //        dbg!(&room2area);
        //        dbg!(&cumsum);
        let area_par_site = area / num_site_assign as f32;
        let mut i_site_cur = 0;
        let mut area_cur = 0.;
        for i_room in 0..num_room {
            site2room[i_site_cur] = i_room;
            i_site_cur += 1;
            loop {
                area_cur += area_par_site;
                site2room[i_site_cur] = i_room;
                i_site_cur += 1;
                if area_cur > cumsum[i_room] || i_site_cur == num_site {
                    break;
                }
            }
        }
        // dbg!(&site2room);
    }
    /*
    for iter in 0..100 {
        for i_room in 0..num_room {
            if iter * num_room + i_room >= site2room.len() {
                break;
            }
            site2room[iter * num_room + i_room] = i_room;
        }
        if (iter + 1) * num_room >= site2room.len() {
            break;
        }
    }
     */
    site2room
}

fn boundary_span(vtxl2xy: &[f32]) -> f32 {
    if vtxl2xy.len() < 2 {
        return 1.0;
    }
    let mut min_x = vtxl2xy[0];
    let mut max_x = vtxl2xy[0];
    let mut min_y = vtxl2xy[1];
    let mut max_y = vtxl2xy[1];
    for i in (0..vtxl2xy.len()).step_by(2) {
        let x = vtxl2xy[i];
        let y = vtxl2xy[i + 1];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (max_x - min_x).abs().max((max_y - min_y).abs()).max(1.0)
}

fn boundary_centroid(vtxl2xy: &[f32]) -> [f32; 2] {
    if vtxl2xy.is_empty() {
        return [0.5, 0.5];
    }
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut count = 0usize;
    for chunk in vtxl2xy.chunks(2) {
        if chunk.len() < 2 {
            continue;
        }
        sum_x += chunk[0];
        sum_y += chunk[1];
        count += 1;
    }
    if count == 0 {
        return [0.5, 0.5];
    }
    let inv = 1.0 / (count as f32);
    [sum_x * inv, sum_y * inv]
}

fn coordinate_bounds(vtxl2xy: &[f32], site_positions: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for chunk in vtxl2xy.chunks(2) {
        if chunk.len() < 2 {
            continue;
        }
        min_x = min_x.min(chunk[0]);
        max_x = max_x.max(chunk[0]);
        min_y = min_y.min(chunk[1]);
        max_y = max_y.max(chunk[1]);
    }

    for pos in site_positions {
        min_x = min_x.min(pos[0]);
        max_x = max_x.max(pos[0]);
        min_y = min_y.min(pos[1]);
        max_y = max_y.max(pos[1]);
    }

    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    if (max_x - min_x).abs() < 1.0e-6 {
        max_x = min_x + 1.0;
    }
    if (max_y - min_y).abs() < 1.0e-6 {
        max_y = min_y + 1.0;
    }

    (min_x, max_x, min_y, max_y)
}

fn merge_vertex(vertices: &mut Vec<[f32; 2]>, candidate: [f32; 2], eps: f32) -> usize {
    for (idx, existing) in vertices.iter().enumerate() {
        if (existing[0] - candidate[0]).abs() <= eps && (existing[1] - candidate[1]).abs() <= eps {
            return idx;
        }
    }
    vertices.push(candidate);
    vertices.len() - 1
}

fn build_voronoi_geometry(
    vtxl2xy: &[f32],
    site2xy: &candle_core::Tensor,
    site2room: &[usize],
    backend: VoronoiBackend,
) -> anyhow::Result<(
    candle_core::Tensor,
    candle_core::Tensor,
    VoronoiInfo,
    Vec<f32>,
)> {
    let alive: Vec<bool> = site2room.iter().map(|room| *room != usize::MAX).collect();
    let site_coords_raw = site2xy.flatten_all()?.to_vec1::<f32>()?;
    let mut site_positions: Vec<[f32; 2]> = site_coords_raw
        .chunks(2)
        .filter(|chunk| chunk.len() == 2)
        .map(|c| [c[0], c[1]])
        .collect();
    let boundary_polygon_world: Option<Vec<crate::fracture::Point>> = if vtxl2xy.len() >= 6 {
        let mut pts = Vec::with_capacity(vtxl2xy.len() / 2);
        for chunk in vtxl2xy.chunks(2) {
            pts.push(crate::fracture::Point::new(chunk[0] as f64, chunk[1] as f64));
        }
        if pts.len() >= 3 {
            Some(pts)
        } else {
            None
        }
    } else {
        None
    };
    let fallback = boundary_centroid(vtxl2xy);
    let sanitized = crate::voronoi::sanitize_site_positions(&mut site_positions, fallback);
    if sanitized > 0 {
        eprintln!(
            "[floorplan] sanitized {} site coordinates before Voronoi construction",
            sanitized
        );
    }
    if let Some(boundary_poly) = boundary_polygon_world.as_deref() {
        let mut alive_indices = Vec::new();
        let mut alive_points = Vec::new();
        for (idx, pos) in site_positions.iter().enumerate() {
            if !alive[idx] {
                continue;
            }
            alive_indices.push(idx);
            alive_points.push(crate::fracture::Point::new(pos[0] as f64, pos[1] as f64));
        }
        if !alive_points.is_empty() {
            let moved =
                crate::fracture::push_sites_inside_polygon(alive_points.as_mut_slice(), boundary_poly);
            if moved > 0 {
                for (site_idx, pushed_point) in alive_indices.into_iter().zip(alive_points.into_iter()) {
                    site_positions[site_idx][0] = pushed_point.x as f32;
                    site_positions[site_idx][1] = pushed_point.y as f32;
                }
            }
        }
    }
    let mut site_coords = Vec::with_capacity(site_positions.len() * 2);
    for pos in &site_positions {
        site_coords.push(pos[0]);
        site_coords.push(pos[1]);
    }
    let mut delta: Vec<f32> = Vec::with_capacity(site_coords.len());
    let mut has_delta = false;
    for (raw, sanitized_val) in site_coords_raw.iter().zip(site_coords.iter()) {
        let diff = sanitized_val - raw;
        if diff != 0.0 {
            has_delta = true;
        }
        delta.push(diff);
    }
    let site2xy_sanitized = if has_delta {
        let delta_tensor =
            candle_core::Tensor::from_vec(delta, site2xy.shape().clone(), site2xy.device())?;
        site2xy.add(&delta_tensor)?
    } else {
        site2xy.clone()
    };
    match backend {
        VoronoiBackend::Legacy => {
            let site2cells = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                del_msh_core::voronoi2::voronoi_cells(vtxl2xy, &site_coords, |i_site| alive[i_site])
            }))
            .map_err(|payload| {
                let message = panic_payload_to_string(payload.as_ref());
                let backtrace = Backtrace::force_capture();
                anyhow::anyhow!(
                    "voronoi_cells() panicked while building geometry: {message}\nBacktrace:\n{backtrace}"
                )
            })?;

            #[cfg(debug_assertions)]
            for (i_site, cell) in site2cells.iter().enumerate() {
                if alive[i_site] && cell.vtx2xy.is_empty() {
                    let x = site_coords[i_site * 2];
                    let y = site_coords[i_site * 2 + 1];
                    eprintln!(
                        "[floorplan] warning: site {i_site} at ({x:.10}, {y:.10}) was marked alive but produced an empty Voronoi cell"
                    );
                }
            }

            let voronoi_mesh = del_msh_core::voronoi2::indexing(&site2cells);
            let layer = Layer {
                vtxl2xy: vtxl2xy.to_vec(),
                vtxv2info: voronoi_mesh.vtxv2info.clone(),
            };
            let vtxv2xy = site2xy_sanitized.apply_op1(layer)?;
            let idx2site = del_msh_core::elem2elem::from_polygon_mesh(
                &voronoi_mesh.site2idx,
                &voronoi_mesh.idx2vtxv,
                vtxv2xy.dims2()?.0,
            );
            let info = VoronoiInfo {
                site2idx: voronoi_mesh.site2idx,
                idx2vtxv: voronoi_mesh.idx2vtxv,
                idx2site,
                vtxv2info: voronoi_mesh.vtxv2info,
            };
            Ok((site2xy_sanitized, vtxv2xy, info, site_coords))
        }
        VoronoiBackend::Fracture => {
            let (min_x, max_x, min_y, max_y) = coordinate_bounds(vtxl2xy, &site_positions);
            let width = (max_x - min_x).abs().max(1.0e-3) as f64;
            let height = (max_y - min_y).abs().max(1.0e-3) as f64;
            let offset_x = min_x as f64;
            let offset_y = min_y as f64;

            let mut fracture_sites = Vec::new();
            let mut index_map = Vec::new();
            for (idx, pos) in site_positions.iter().enumerate() {
                if !alive[idx] {
                    continue;
                }
                fracture_sites.push(crate::fracture::Point::new(
                    (pos[0] - min_x) as f64,
                    (pos[1] - min_y) as f64,
                ));
                index_map.push(idx);
            }

            if fracture_sites.len() < 3 {
                return Err(anyhow::anyhow!(
                    "Fracture backend requires at least three active sites"
                ));
            }

            let boundary_points = boundary_polygon_world.as_ref().map(|poly| {
                let mut pts = Vec::with_capacity(poly.len());
                for p in poly {
                    pts.push(crate::fracture::Point::new(p.x - offset_x, p.y - offset_y));
                }
                pts
            });

            let mut seed = index_map.len() as u64;
            seed = seed
                .wrapping_mul(636_413_622_384_679_3005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut rng = crate::fracture::Rng::new(seed);
            // let (cells, _) = crate::fracture::compute_voronoi_fracture(
            //     fracture_sites,
            //     width,
            //     height,
            //     boundary_points.as_deref(),
            //     &mut rng,
            // );
            let (cells, _) = crate::fracture::compute_voronoi_fracture2(
                fracture_sites,
                boundary_points.as_deref(),
                &mut rng,
            );

            let mut site_polys = vec![Vec::<[f32; 2]>::new(); site_positions.len()];
            for cell in cells {
                let Some(&global_idx) = index_map.get(cell.site_index) else {
                    continue;
                };
                if cell.vertices.len() < 3 {
                    continue;
                }
                let mut poly = Vec::with_capacity(cell.vertices.len());
                for vertex in cell.vertices {
                    poly.push([(vertex.x + offset_x) as f32, (vertex.y + offset_y) as f32]);
                }
                site_polys[global_idx] = poly;
            }

            if site_polys.iter().all(|poly| poly.len() < 3) {
                return Err(anyhow::anyhow!(
                    "Fracture backend did not produce any valid Voronoi cells"
                ));
            }

            const MERGE_EPS: f32 = 1.0e-4;
            let mut vertices: Vec<[f32; 2]> = Vec::new();
            let mut site2idx = Vec::with_capacity(site_polys.len() + 1);
            let mut idx2vtxv = Vec::new();
            site2idx.push(0);
            for poly in &site_polys {
                for &pt in poly {
                    let idx = merge_vertex(&mut vertices, pt, MERGE_EPS);
                    idx2vtxv.push(idx);
                }
                site2idx.push(idx2vtxv.len());
            }

            if vertices.is_empty() {
                return Err(anyhow::anyhow!(
                    "Fracture backend failed to generate Voronoi vertices"
                ));
            }

            let mut flat_vertices = Vec::with_capacity(vertices.len() * 2);
            for v in &vertices {
                flat_vertices.push(v[0]);
                flat_vertices.push(v[1]);
            }
            let vtxv2xy = candle_core::Tensor::from_vec(
                flat_vertices,
                candle_core::Shape::from((vertices.len(), 2)),
                site2xy.device(),
            )?;
            let idx2site =
                del_msh_core::elem2elem::from_polygon_mesh(&site2idx, &idx2vtxv, vertices.len());
            let info = VoronoiInfo {
                site2idx,
                idx2vtxv,
                idx2site,
                vtxv2info: vec![[usize::MAX; 4]; vertices.len()],
            };
            Ok((site2xy_sanitized, vtxv2xy, info, site_coords))
        }
        VoronoiBackend::Delaunay => {
            let diagram = crate::voronoi::compute_delaunay_voronoi(vtxl2xy, &site_coords, &alive)?;
            let layer = Layer {
                vtxl2xy: vtxl2xy.to_vec(),
                vtxv2info: diagram.vtxv2info.clone(),
            };
            let vtxv2xy = site2xy_sanitized.apply_op1(layer)?;
            let info = VoronoiInfo {
                site2idx: diagram.site2idx,
                idx2vtxv: diagram.idx2vtxv,
                idx2site: diagram.idx2site,
                vtxv2info: diagram.vtxv2info,
            };
            Ok((site2xy_sanitized, vtxv2xy, info, site_coords))
        }
    }
}

pub struct VoronoiStage {
    pub(crate) site2xy_adjusted: candle_core::Tensor,
    pub(crate) voronoi_info: del_candle::voronoi2::VoronoiInfo,
    pub(crate) vtxv2xy: candle_core::Tensor,
    pub(crate) site_coords_sanitized: Vec<f32>,
}

pub(crate) fn iterate_voronoi_stage(
    vtxl2xy: &[f32],
    site2xy: &candle_core::Var,
    site2room: &[usize],
) -> anyhow::Result<VoronoiStage> {
    let min_site_radius = boundary_span(vtxl2xy) * 1.0e-3_f32;
    let site2xy_adjusted = enforce_site_spacing(site2xy, min_site_radius)?;

    #[cfg(debug_assertions)]
    {
        let adjusted_coords = site2xy_adjusted.flatten_all()?.to_vec1::<f32>()?;
        if let Some((i0, i1)) =
            find_overlapping_sites(&adjusted_coords, min_site_radius.max(1.0e-6))
        {
            eprintln!(
                "[floorplan] overlapping sites detected after spacing ({} and {})",
                i0, i1
            );
        }
    }

    // println!(
    //     "Check point time: {:?} at start build_voronoi_geometry",
    //     std::time::Instant::now()
    // );
    // std::io::stdout().flush().unwrap();

    let (site2xy_sanitized, vtxv2xy, voronoi_info, site_coords_sanitized) = build_voronoi_geometry(
        vtxl2xy,
        &site2xy_adjusted,
        site2room,
        //        VoronoiBackend::Legacy,
        VoronoiBackend::Fracture,
    )?;

    // println!(
    //     "Check point time: {:?} at end build_voronoi_geometry",
    //     std::time::Instant::now()
    // );
    // std::io::stdout().flush().unwrap();

    Ok(VoronoiStage {
        site2xy_adjusted: site2xy_sanitized,
        voronoi_info,
        vtxv2xy,
        site_coords_sanitized,
    })
}

pub(crate) fn optimize_iteration(
    vtxl2xy: &[f32],
    site2xy: &candle_core::Var,
    site2xy_ini: &candle_core::Tensor,
    site2xy2flag: &candle_core::Var,
    site2room: &[usize],
    room2area_trg: &candle_core::Tensor,
    room_connections: &[(usize, usize)],
    optimizer: &mut candle_nn::AdamW,
    params: &ProjectParams,
    stage: VoronoiStage,
) -> anyhow::Result<(
    candle_core::Tensor,
    del_candle::voronoi2::VoronoiInfo,
    candle_core::Tensor,
    Vec<usize>,
    Vec<f32>,
    LossBreakdown,
)> {
    let VoronoiStage {
        site2xy_adjusted,
        voronoi_info,
        vtxv2xy,
        site_coords_sanitized,
    } = stage;
    let (num_rooms, _) = room2area_trg.dims2()?;
    let loss_weights = &params.loss_weights;
    let edge2vtxv_wall = crate::edge2vtvx_wall(&voronoi_info, site2room);
    let (loss_each_area, loss_total_area) = {
        let room2area = crate::room2area(
            site2room,
            num_rooms,
            &voronoi_info.site2idx,
            &voronoi_info.idx2vtxv,
            &vtxv2xy,
        )?;
        let loss_each_area = room2area.sub(room2area_trg)?.sqr()?.sum_all()?;
        let total_area_trg = del_msh_core::polyloop2::area(vtxl2xy);
        let total_area_trg = candle_core::Tensor::from_vec(
            vec![total_area_trg],
            candle_core::Shape::from(()),
            &candle_core::Device::Cpu,
        )?;
        let loss_total_area = (room2area.sum_all()? - total_area_trg)?.abs()?;
        (loss_each_area, loss_total_area)
    };
    // Use |sin(4θ)| so edges aligned to 0°, 45°, or 90° contribute zero and mid-angles approach 1.
    let (loss_walllen, loss_wallangle) = {
        let vtx2xyz_to_edgevector = del_candle::vtx2xyz_to_edgevector::Layer {
            edge2vtx: Vec::<usize>::from(edge2vtxv_wall.clone()),
        };
        let edge2xy = vtxv2xy.apply_op1(vtx2xyz_to_edgevector)?;

        let dx = edge2xy.get_on_dim(1, 0)?;
        let dy = edge2xy.get_on_dim(1, 1)?;
        let dx2 = dx.sqr()?;
        let dy2 = dy.sqr()?;
        let len_sq = (&dx2 + &dy2)?;
        let len_sq_safe = len_sq.affine(1.0, 1.0e-12)?;
        let edge_len = len_sq_safe.sqrt()?;

        let diff = dx2.sub(&dy2)?;
        let abs_ratio = diff.div(&len_sq_safe)?.abs()?.clamp(0.0, 1.0)?;
        let angle_penalty = abs_ratio.affine(-1.0, 1.0)?.sqrt()?;

        let length_weighted_angle = edge_len.mul(&angle_penalty)?;
        let loss_length = length_weighted_angle.sum_all()?;
        let loss_angle = length_weighted_angle.mul(&angle_penalty)?.sum_all()?;
        (loss_length, loss_angle)
    };
    let loss_topo = crate::loss_topo::unidirectional(
        &site2xy_adjusted,
        site2room,
        num_rooms,
        &voronoi_info,
        room_connections,
    )?;
    let loss_fix = site2xy
        .sub(site2xy_ini)?
        .mul(site2xy2flag)?
        .sqr()?
        .sum_all()?;
    let loss_lloyd = del_candle::voronoi2::loss_lloyd(
        &voronoi_info.site2idx,
        &voronoi_info.idx2vtxv,
        &site2xy_adjusted,
        &vtxv2xy,
    )?;
    let loss_each_area = loss_each_area
        .affine(loss_weights.each_area as f64, 0.0)?
        .clone();
    let loss_total_area = loss_total_area
        .affine(loss_weights.total_area as f64, 0.0)?
        .clone();
    let loss_walllen = loss_walllen.affine(loss_weights.wall_length as f64, 0.0)?;
    let loss_wallangle = loss_wallangle.affine(loss_weights.wall_angle as f64, 0.0)?;
    let loss_topo = loss_topo.affine(loss_weights.topology as f64, 0.0)?;
    let loss_fix = loss_fix.affine(loss_weights.fix as f64, 0.0)?;
    let loss_lloyd = loss_lloyd.affine(loss_weights.lloyd as f64, 0.0)?;
    let loss = (&loss_each_area
        + &loss_total_area
        + &loss_walllen
        + &loss_wallangle
        + &loss_topo
        + &loss_fix
        + &loss_lloyd)?;

    let breakdown = LossBreakdown::from_tensors(
        &loss_each_area,
        &loss_total_area,
        &loss_walllen,
        &loss_wallangle,
        &loss_topo,
        &loss_fix,
        &loss_lloyd,
        &loss,
    )?;

    if !breakdown.total.is_finite() {
        eprintln!("[floorplan] Loss is not finite: {}", breakdown.total);
        eprintln!("loss_each_area: {}", breakdown.each_area);
        eprintln!("loss_total_area: {}", breakdown.total_area);
        eprintln!("loss_walllen: {}", breakdown.wall_length);
        eprintln!("loss_wallangle: {}", breakdown.wall_angle);
        eprintln!("loss_topo: {}", breakdown.topology);
        eprintln!("loss_fix: {}", breakdown.fix);
        eprintln!("loss_lloyd: {}", breakdown.lloyd);
        return Err(anyhow::anyhow!("Loss is not finite"));
    }

    // println!(
    //     "Check point time: {:?} at start backward_step",
    //     std::time::Instant::now()
    // );
    // std::io::stdout().flush().unwrap();
    optimizer.backward_step(&loss)?;
    // println!(
    //     "Check point time: {:?} at end backward_step",
    //     std::time::Instant::now()
    // );
    // std::io::stdout().flush().unwrap();

    Ok((
        site2xy_adjusted,
        voronoi_info,
        vtxv2xy,
        edge2vtxv_wall,
        site_coords_sanitized,
        breakdown,
    ))
}
fn optimize_phase(
    canvas_gif: &mut del_canvas_core::canvas_gif::Canvas,
    transform_world2pix: &nalgebra::Matrix3<f32>,
    vtxl2xy: &[f32],
    site2xy: &candle_core::Var,
    site2xy_ini: &candle_core::Tensor,
    site2xy2flag: &candle_core::Var,
    site2room: &[usize],
    room2area_trg: &candle_core::Tensor,
    room_connections: &[(usize, usize)],
    iter: usize,
    params: &ProjectParams,
    mirror_canvas: Option<&mut del_canvas_core::canvas_gif::Canvas>,
    optimizer: &mut candle_nn::AdamW,
    lr_state: &mut Option<f32>,
    global_iter: &mut usize,
) -> anyhow::Result<()> {
    let (num_rooms, _) = room2area_trg.dims2()?;
    let diag_dir = PathBuf::from("target");
    std::fs::create_dir_all(&diag_dir)?;
    let diag_path = diag_dir.join("site_diagnostics.txt");
    let loss_diag_path = diag_dir.join("loss_diagnostics.txt");
    let boundary_diag_path = diag_dir.join("boundary_diagnostics.txt");
    let file_existed = diag_path.exists();
    {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&diag_path)?;
        if !file_existed {
            writeln!(
                file,
                "# Site diagnostics\n# sites={} rooms={} iterations={}\n",
                site2room.len(),
                num_rooms,
                iter
            )?;
        } else {
            writeln!(file)?;
            writeln!(
                file,
                "# --- New phase: sites={} rooms={} iterations={} ---",
                site2room.len(),
                num_rooms,
                iter
            )?;
        }
    }

    let loss_file_existed = loss_diag_path.exists();
    {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&loss_diag_path)?;
        if !loss_file_existed {
            writeln!(file, "# Loss diagnostics")?;
            writeln!(
                file,
                "# columns: iteration loss_total loss_each_area loss_total_area loss_wall_length loss_wall_angle loss_topology loss_fix loss_lloyd"
            )?;
        } else {
            writeln!(file)?;
            writeln!(
                file,
                "# --- New phase: sites={} rooms={} iterations={} ---",
                site2room.len(),
                num_rooms,
                iter
            )?;
        }
    }

    let learning_rates = &params.learning_rates;
    let mut mirror_canvas = mirror_canvas;
    let mut mirror_copy_enabled = true;
    if let Some(extra_canvas) = mirror_canvas.as_deref() {
        let same_dims =
            extra_canvas.width == canvas_gif.width && extra_canvas.height == canvas_gif.height;
        let same_buffer = extra_canvas.data.len() == canvas_gif.data.len();
        if !same_dims || !same_buffer {
            eprintln!(
                "[floorplan] mirror canvas dimensions ({}x{}) do not match primary canvas ({}x{}); skipping phase export",
                extra_canvas.width,
                extra_canvas.height,
                canvas_gif.width,
                canvas_gif.height
            );
            mirror_copy_enabled = false;
        }
    }

    let phase_timer = Instant::now();
    let mut persent_last = 0;
    if let Err(err) = record_boundary_diagnostics(&boundary_diag_path, vtxl2xy) {
        eprintln!("[floorplan] failed to write boundary diagnostics: {err}");
    }
    for iter_idx in 0..iter {
        let persent = ((iter_idx + 1) * 100) / iter;
        if persent != persent_last {
            persent_last = persent;
            print!("{}% ", persent);
            let mut stdout = std::io::stdout().lock();
            stdout.flush()?;
        }

        let absolute_iter = *global_iter;
        let desired_lr = learning_rates.value_for_step(absolute_iter);
        let update_lr = match lr_state {
            Some(current) => (*current - desired_lr).abs() > f32::EPSILON,
            None => true,
        };
        if update_lr {
            optimizer.set_params(candle_nn::ParamsAdamW {
                lr: desired_lr as f64,
                ..Default::default()
            });
            *lr_state = Some(desired_lr);
        }

        let voronoi_stage = iterate_voronoi_stage(vtxl2xy, &site2xy, site2room)?;

        let (
            _site2xy_adjusted,
            voronoi_info,
            vtxv2xy,
            edge2vtxv_wall,
            site_coords_sanitized,
            loss_breakdown,
        ) = optimize_iteration(
                vtxl2xy,
                &site2xy,
                site2xy_ini,
                &site2xy2flag,
                site2room,
                room2area_trg,
                room_connections,
            optimizer,
                params,
                voronoi_stage,
            )?;

        let site2xy_render = site_coords_sanitized;
        let vtxv2xy_render = vtxv2xy.flatten_all()?.to_vec1::<f32>()?;
        if let Err(err) = record_site_diagnostics(
            &diag_path,
            iter_idx,
            &site2xy_render,
            site2room,
            &voronoi_info,
            &vtxv2xy_render,
        ) {
            eprintln!("[floorplan] failed to write site diagnostics: {err}");
        }
        if let Err(err) = record_loss_diagnostics(&loss_diag_path, iter_idx, &loss_breakdown) {
            eprintln!("[floorplan] failed to write loss diagnostics: {err}");
        }
        canvas_gif.clear(0);

        // println!(
        //     "Check point time: {:?} at start my_paint",
        //     std::time::Instant::now()
        // );
        // std::io::stdout().flush().unwrap();
        crate::my_paint(
            canvas_gif,
            transform_world2pix,
            vtxl2xy,
            &site2xy_render,
            &voronoi_info,
            &vtxv2xy_render,
            site2room,
            &edge2vtxv_wall,
        );
        overlay_iteration_counter(canvas_gif, iter_idx, iter);

        // println!(
        //     "Check point time: {:?} at end my_paint",
        //     std::time::Instant::now()
        // );
        // std::io::stdout().flush().unwrap();
        if mirror_copy_enabled {
            if let Some(extra_canvas) = mirror_canvas.as_deref_mut() {
                // Mirror the drawn frame into the per-phase GIF without rerunning my_paint.
                extra_canvas.data.copy_from_slice(&canvas_gif.data);
                extra_canvas.write();
            }
        }

        canvas_gif.write();
        *global_iter += 1;
    }

    println!("Phase elapsed: {:.2?}", phase_timer.elapsed());

    fn record_boundary_diagnostics(path: &Path, boundary: &[f32]) -> std::io::Result<()> {
        use std::fs::OpenOptions;

        fn polygon_area(vertices: &[(f32, f32)]) -> f32 {
            if vertices.len() < 3 {
                return 0.0;
            }
            let mut acc = 0.0f32;
            for i in 0..vertices.len() {
                let (x0, y0) = vertices[i];
                let (x1, y1) = vertices[(i + 1) % vertices.len()];
                acc += x0 * y1 - x1 * y0;
            }
            (acc * 0.5).abs()
        }

        fn polygon_perimeter(vertices: &[(f32, f32)]) -> f32 {
            if vertices.len() < 2 {
                return 0.0;
            }
            let mut acc = 0.0f32;
            for pair in vertices.windows(2) {
                let (x0, y0) = pair[0];
                let (x1, y1) = pair[1];
                let dx = x1 - x0;
                let dy = y1 - y0;
                acc += (dx * dx + dy * dy).sqrt();
            }
            if vertices.len() > 2 {
                let (x0, y0) = vertices[0];
                let (x1, y1) = *vertices.last().unwrap();
                let dx = x0 - x1;
                let dy = y0 - y1;
                acc += (dx * dx + dy * dy).sqrt();
            }
            acc
        }

        let mut vertices = Vec::with_capacity(boundary.len() / 2);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for chunk in boundary.chunks(2) {
            if chunk.len() < 2 {
                continue;
            }
            let x = chunk[0];
            let y = chunk[1];
            vertices.push((x, y));
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        if vertices.is_empty() {
            min_x = 0.0;
            max_x = 0.0;
            min_y = 0.0;
            max_y = 0.0;
        }
        let span_x = (max_x - min_x).abs();
        let span_y = (max_y - min_y).abs();
        let diag = (span_x * span_x + span_y * span_y).sqrt();
        let perimeter = polygon_perimeter(&vertices);
        let area = polygon_area(&vertices);
        let closed = if vertices.len() >= 3 {
            let first = vertices.first().unwrap();
            let last = vertices.last().unwrap();
            (first.0 - last.0).abs() <= 1.0e-5 && (first.1 - last.1).abs() <= 1.0e-5
        } else {
            false
        };
        let geom_status = match vertices.len() {
            0 => "empty",
            1 => "point",
            2 => "segment",
            _ => "polygon",
        };

        let file_existed = path.exists();
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        if !file_existed {
            writeln!(file, "# Boundary diagnostics")?;
        }
        writeln!(file)?;
        writeln!(
            file,
            "boundary_vertices={} status={} closed={}",
            vertices.len(),
            geom_status,
            closed
        )?;
        writeln!(
            file,
            "  bbox=({:.9},{:.9})-({:.9},{:.9}) span=({:.9},{:.9}) diag={:.9}",
            min_x, min_y, max_x, max_y, span_x, span_y, diag
        )?;
        writeln!(file, "  perimeter={:.9} area={:.9}", perimeter, area)?;
        if vertices.is_empty() {
            writeln!(file, "  vertices=[]")?;
        } else {
            write!(file, "  vertices=[")?;
            for (idx, (x, y)) in vertices.iter().enumerate() {
                if idx > 0 {
                    write!(file, ", ")?;
                }
                write!(file, "({:.9},{:.9})", x, y)?;
            }
            writeln!(file, "]")?;
        }
        Ok(())
    }

    fn record_loss_diagnostics(
        path: &Path,
        iteration: usize,
        losses: &LossBreakdown,
    ) -> std::io::Result<()> {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "iteration={iteration}")?;
        writeln!(file, "  loss_total={:.9}", losses.total)?;
        writeln!(file, "  loss_each_area={:.9}", losses.each_area)?;
        writeln!(file, "  loss_total_area={:.9}", losses.total_area)?;
        writeln!(file, "  loss_wall_length={:.9}", losses.wall_length)?;
        writeln!(file, "  loss_wall_angle={:.9}", losses.wall_angle)?;
        writeln!(file, "  loss_topology={:.9}", losses.topology)?;
        writeln!(file, "  loss_fix={:.9}", losses.fix)?;
        writeln!(file, "  loss_lloyd={:.9}", losses.lloyd)?;
        Ok(())
    }

    fn record_site_diagnostics(
        path: &Path,
        iteration: usize,
        site2xy: &[f32],
        site2room: &[usize],
        voronoi_info: &VoronoiInfo,
        vtxv2xy: &[f32],
    ) -> std::io::Result<()> {
        use std::fs::OpenOptions;

        #[derive(Clone)]
        struct SiteSnapshot {
            room: usize,
            status: &'static str,
            num_vtx: usize,
            area: f32,
            pos_x: f32,
            pos_y: f32,
            vertices: Vec<(f32, f32)>,
            neighbors: Vec<usize>,
        }

        fn nearest_non_empty(sites: &[SiteSnapshot], idx: usize) -> Option<(usize, f32)> {
            let target = &sites[idx];
            if sites.len() <= 1 {
                return None;
            }
            let mut best: Option<(usize, f32)> = None;
            for (other_idx, other) in sites.iter().enumerate() {
                if other_idx == idx || other.num_vtx == 0 {
                    continue;
                }
                let dx = target.pos_x - other.pos_x;
                let dy = target.pos_y - other.pos_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if let Some((_, best_dist)) = best {
                    if dist >= best_dist {
                        continue;
                    }
                }
                best = Some((other_idx, dist));
            }
            best
        }

        fn polygon_area(vertices: &[(f32, f32)]) -> f32 {
            if vertices.len() < 3 {
                return 0.0;
            }
            let mut acc = 0.0f32;
            for i in 0..vertices.len() {
                let (x0, y0) = vertices[i];
                let (x1, y1) = vertices[(i + 1) % vertices.len()];
                acc += x0 * y1 - x1 * y0;
            }
            (acc * 0.5).abs()
        }

        let mut snapshots: Vec<SiteSnapshot> = Vec::with_capacity(site2room.len());
        for (i_site, room) in site2room.iter().enumerate() {
            let alive = *room != usize::MAX;
            let start = voronoi_info.site2idx[i_site];
            let end = voronoi_info.site2idx[i_site + 1];
            let num_vtx = end - start;
            let mut vertices = Vec::with_capacity(num_vtx);
            let mut neighbors = Vec::new();
            for idx in start..end {
                let i_vtx = voronoi_info.idx2vtxv[idx];
                vertices.push((vtxv2xy[i_vtx * 2], vtxv2xy[i_vtx * 2 + 1]));
                let neighbor_site = voronoi_info.idx2site[idx];
                if neighbor_site == usize::MAX || neighbor_site == i_site {
                    continue;
                }
                if !neighbors.contains(&neighbor_site) {
                    neighbors.push(neighbor_site);
                }
            }
            let area = polygon_area(&vertices);
            let pos_x = site2xy[i_site * 2];
            let pos_y = site2xy[i_site * 2 + 1];
            let status = if !alive {
                "inactive"
            } else if num_vtx == 0 {
                "empty-cell"
            } else if area.abs() < 1.0e-6 {
                "zero-area"
            } else {
                "ok"
            };
            snapshots.push(SiteSnapshot {
                room: *room,
                status,
                num_vtx,
                area,
                pos_x,
                pos_y,
                vertices,
                neighbors,
            });
        }

        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "iteration={iteration}")?;
        for (i_site, snapshot) in snapshots.iter().enumerate() {
            writeln!(
                file,
                "  site={i_site:04} room={} status={} num_vtx={} area={:.9} pos=({:.9},{:.9})",
                snapshot.room,
                snapshot.status,
                snapshot.num_vtx,
                snapshot.area,
                snapshot.pos_x,
                snapshot.pos_y
            )?;

            if snapshot.vertices.is_empty() {
                writeln!(file, "    vertices=[]")?;
            } else {
                write!(file, "    vertices=[")?;
                for (idx, (x, y)) in snapshot.vertices.iter().enumerate() {
                    if idx > 0 {
                        write!(file, ", ")?;
                    }
                    write!(file, "({:.9},{:.9})", x, y)?;
                }
                writeln!(file, "]")?;
            }

            if matches!(snapshot.status, "empty-cell" | "zero-area") {
                if let Some((nearest_idx, dist)) = nearest_non_empty(&snapshots, i_site) {
                    let neighbor = &snapshots[nearest_idx];
                    writeln!(
                    file,
                    "    nearest_site={:04} room={} status={} distance={:.9} num_vtx={} area={:.9}",
                    nearest_idx,
                    neighbor.room,
                    neighbor.status,
                    dist,
                    neighbor.num_vtx,
                    neighbor.area
                )?;
                    if neighbor.vertices.is_empty() {
                        writeln!(file, "      nearest_vertices=[]")?;
                    } else {
                        write!(file, "      nearest_vertices=[")?;
                        for (idx, (x, y)) in neighbor.vertices.iter().enumerate() {
                            if idx > 0 {
                                write!(file, ", ")?;
                            }
                            write!(file, "({:.9},{:.9})", x, y)?;
                        }
                        writeln!(file, "]")?;
                    }
                } else {
                    writeln!(file, "    nearest_site=none")?;
                }

                if snapshot.neighbors.is_empty() {
                    writeln!(file, "    neighbor_sites=[]")?;
                } else {
                    writeln!(
                        file,
                        "    neighbor_sites=[{}]",
                        snapshot
                            .neighbors
                            .iter()
                            .map(|idx| format!("{:04}", idx))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                    for neighbor_idx in &snapshot.neighbors {
                        let neighbor = &snapshots[*neighbor_idx];
                        writeln!(
                        file,
                        "      neighbor={:04} room={} status={} num_vtx={} area={:.9} pos=({:.9},{:.9})",
                        neighbor_idx,
                        neighbor.room,
                        neighbor.status,
                        neighbor.num_vtx,
                        neighbor.area,
                        neighbor.pos_x,
                        neighbor.pos_y
                    )?;
                        if neighbor.vertices.is_empty() {
                            writeln!(file, "        vertices=[]")?;
                        } else {
                            write!(file, "        vertices=[")?;
                            for (idx, (x, y)) in neighbor.vertices.iter().enumerate() {
                                if idx > 0 {
                                    write!(file, ", ")?;
                                }
                                write!(file, "({:.9},{:.9})", x, y)?;
                            }
                            writeln!(file, "]")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    Ok(())
}

fn optimize_impl(
    canvas_gif: &mut del_canvas_core::canvas_gif::Canvas,
    vtxl2xy: Vec<f32>,
    site2xy: Vec<f32>,
    site2room: Vec<usize>,
    site2xy2flag: Vec<f32>,
    room2area_trg: Vec<f32>,
    room2color: Vec<i32>,
    room_connections: Vec<(usize, usize)>,
    iter: usize,
    params_index: usize,
) -> anyhow::Result<()> {
    let canvas_width = canvas_gif.width;
    let canvas_height = canvas_gif.height;
    let site_positions_for_bounds: Vec<[f32; 2]> = site2xy
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect();
    let (min_x, max_x, min_y, max_y) = coordinate_bounds(&vtxl2xy, &site_positions_for_bounds);
    let span_x = (max_x - min_x).abs().max(1.0e-6);
    let span_y = (max_y - min_y).abs().max(1.0e-6);
    let padding = 0.1f32;
    let draw_fraction = 1.0 - 2.0 * padding;
    let scale_x = canvas_width as f32 * draw_fraction / span_x;
    let scale_y = -(canvas_height as f32) * draw_fraction / span_y;
    let translate_x = canvas_width as f32 * padding - min_x * scale_x;
    let translate_y = canvas_height as f32 * (1.0 - padding) - min_y * scale_y;
    let transform_world2pix = nalgebra::Matrix3::<f32>::new(
        scale_x,
        0.,
        translate_x,
        0.,
        scale_y,
        translate_y,
        0.,
        0.,
        1.,
    );
    let mut palette = vec![0x7F7F7F, 0x000000];
    palette.extend(room2color.iter().copied());

    //dbg!( canvas_gif.path );

    if site2xy.len() != site2xy2flag.len() {
        anyhow::bail!(
            "site2xy ({}) and site2xy2flag ({}) must have the same length",
            site2xy.len(),
            site2xy2flag.len()
        );
    }
    if site2xy.len() / 2 != site2room.len() {
        anyhow::bail!(
            "site2room ({}) must match number of sites ({})",
            site2room.len(),
            site2xy.len() / 2
        );
    }

    let device = &candle_core::Device::Cpu;
    let num_sites = site2xy.len() / 2;
    let site2xy_var = candle_core::Var::from_slice(
        &site2xy,
        candle_core::Shape::from((num_sites, 2)),
        device,
    )?;
    let site2xy2flag_var = candle_core::Var::from_slice(
        &site2xy2flag,
        candle_core::Shape::from((num_sites, 2)),
        device,
    )?;

    let num_rooms = room2area_trg.len();
    let room2area_trg = candle_core::Tensor::from_vec(
        room2area_trg,
        candle_core::Shape::from((num_rooms, 1)),
        device,
    )?;

    let site2xy_ini = candle_core::Tensor::from_vec(
        site2xy.clone(),
        candle_core::Shape::from((site2xy.len() / 2, 2)),
        device,
    )?;

    let params_all = project_params_all();
    if params_index >= params_all.len() {
        anyhow::bail!(
            "project parameters index {} out of range ({} files detected)",
            params_index,
            params_all.len()
        );
    }

    let mut optimizer = candle_nn::AdamW::new(
        vec![site2xy_var.clone()],
        candle_nn::ParamsAdamW {
            lr: params_all[params_index].learning_rates.first as f64,
            ..Default::default()
        },
    )?;
    let mut lr_state: Option<f32> = None;
    let mut global_iter = 0usize;

    let total_timer = Instant::now();
    for (phase_offset, params) in params_all[params_index..].iter().enumerate() {
        println!(
            "=== Starting phase {} (params index {}) ===",
            phase_offset,
            params_index + phase_offset
        );
        let phase_filename = format!("target/result{phase:02}.gif", phase = phase_offset);
        let mut phase_canvas = del_canvas_core::canvas_gif::Canvas::new(
            &phase_filename,
            (canvas_width, canvas_height),
            &palette,
        );

        
        optimize_phase(
            canvas_gif,
            &transform_world2pix,
            &vtxl2xy,
            &site2xy_var,
            &site2xy_ini,
            &site2xy2flag_var,
            &site2room,
            &room2area_trg,
            &room_connections,
            iter,
            params,
            Some(&mut phase_canvas),
            &mut optimizer,
            &mut lr_state,
            &mut global_iter,
        )?;
    }

    println!("Total optimization elapsed: {:.2?}", total_timer.elapsed());
    Ok(())
}

pub fn optimize(
    canvas_gif: &mut del_canvas_core::canvas_gif::Canvas,
    vtxl2xy: Vec<f32>,
    site2xy: Vec<f32>,
    site2room: Vec<usize>,
    site2xy2flag: Vec<f32>,
    room2area_trg: Vec<f32>,
    room2color: Vec<i32>,
    room_connections: Vec<(usize, usize)>,
    iter: usize,
    params_index: usize,
) -> anyhow::Result<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        optimize_impl(
            canvas_gif,
            vtxl2xy,
            site2xy,
            site2room,
            site2xy2flag,
            room2area_trg,
            room2color,
            room_connections,
            iter,
            params_index,
        )
    }));
    match result {
        Ok(inner) => inner,
        Err(payload) => {
            let message = panic_payload_to_string(payload.as_ref());
            let backtrace = Backtrace::force_capture();
            Err(anyhow::anyhow!(
                "optimize panicked: {message}\nBacktrace:\n{backtrace}"
            ))
        }
    }
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "optimize panicked with non-string payload".to_string()
}

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn floorplan(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bindings::register(py, m)
}

pub fn generate_voronoi_cells_robust(
    vtxl2xy: &[f32],
    site2xy: &[f32],
) -> (Vec<del_msh_core::voronoi2::Cell>, VoronoiInfo) {
    let site2cells = del_msh_core::voronoi2::voronoi_cells(vtxl2xy, site2xy, |_| true);
    let voronoi_mesh = del_msh_core::voronoi2::indexing(&site2cells);
    let idx2site = del_msh_core::elem2elem::from_polygon_mesh(
        &voronoi_mesh.site2idx,
        &voronoi_mesh.idx2vtxv,
        voronoi_mesh.vtxv2xy.len(),
    );
    let info = VoronoiInfo {
        site2idx: voronoi_mesh.site2idx,
        idx2vtxv: voronoi_mesh.idx2vtxv,
        idx2site,
        vtxv2info: voronoi_mesh.vtxv2info,
    };
    (site2cells, info)
}
