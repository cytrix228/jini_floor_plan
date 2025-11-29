use anyhow::Context;
use candle_nn::Optimizer;
use del_candle::voronoi2::{Layer, VoronoiInfo};
use del_canvas_core::canvas_gif::Canvas;
use serde::Deserialize;
use std::any::Any;
use std::backtrace::Backtrace;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
mod delaunay;
pub mod loss_topo;
mod voronoi;
pub use voronoi::VoronoiBackend;

static PROJECT_PARAMS: OnceLock<Vec<ProjectParams>> = OnceLock::new();

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
    //println!( "\n");

    let mut colors = Vec::<u8>::new();
    //
    for i_site in 0..site2idx.len() - 1 {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            println!(
                "Skipping site {} with no room assignment in my_paint",
                i_site
            );
            //flush
            std::io::stdout().flush().unwrap();
            continue;
        }
        //
        let i_color: u8 = if i_room == usize::MAX {
            println!(
                "Coloring site with 1 {} with no room assignment in my_paint",
                i_site
            );
            //flush
            std::io::stdout().flush().unwrap();
            1
        } else {
            (i_room + 2).try_into().unwrap()
        };

        colors.push(i_color);

        //        println!( "Painting site {} with color {}", i_site, i_color);
        //flush
        //        std::io::stdout().flush().unwrap();

        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        if num_vtx_in_site == 0 {
            continue;
        }
        let mut vtx2xy = vec![0f32; num_vtx_in_site * 2];
        for i_vtx in 0..num_vtx_in_site {
            let i_vtxv = idx2vtxv[site2idx[i_site] + i_vtx];
            vtx2xy[i_vtx * 2 + 0] = vtxv2xy[i_vtxv * 2 + 0];
            vtx2xy[i_vtx * 2 + 1] = vtxv2xy[i_vtxv * 2 + 1];
        }
        del_canvas_core::rasterize_polygon::fill(
            &mut canvas.data,
            canvas.width,
            &vtx2xy,
            arrayref::array_ref![transform_to_scr.as_slice(), 0, 9],
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

    //println!("Colors: {:?}", colors);
    //std::io::stdout().flush().unwrap();

    // draw points;
    for i_site in 0..site2xy.len() / 2 {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        let i_color: u8 = if i_room == usize::MAX {
            1
        } else {
            (i_room + 2).try_into().unwrap()
        };
        del_canvas_core::rasterize_circle::fill(
            &mut canvas.data,
            canvas.width,
            &[site2xy[i_site * 2 + 0], site2xy[i_site * 2 + 1]],
            arrayref::array_ref![transform_to_scr.as_slice(), 0, 9],
            2.0,
            // black dot
            255, //i_color,
        );
    }

    // print check point time
    // println!("Check point time: {:?} at draw cell boundary", std::time::Instant::now());

    // draw cell boundary
    for i_site in 0..site2idx.len() - 1 {
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let i0 = idx2vtxv[site2idx[i_site] + i0_vtx];
            let i1 = idx2vtxv[site2idx[i_site] + i1_vtx];
            let p0 = &[vtxv2xy[i0 * 2 + 0], vtxv2xy[i0 * 2 + 1]];
            let p1 = &[vtxv2xy[i1 * 2 + 0], vtxv2xy[i1 * 2 + 1]];
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                del_canvas_core::rasterize_line::draw_dda_with_transformation(
                    &mut canvas.data,
                    canvas.width,
                    p0,
                    p1,
                    arrayref::array_ref![transform_to_scr.as_slice(), 0, 9],
                    1,
                );
            })) {
                let bt = Backtrace::force_capture();
                eprintln!(
                    "[floorplan] rasterize_line::draw_dda_with_transformation panicked (i_site={i_site}, i0_vtx={i0_vtx}, i1_vtx={i1_vtx}, i0={i0}, i1={i1}, p0={p0:?}, p1={p1:?},\n)\n{}\n{}",
                    panic_payload_to_string(payload.as_ref()),
                    bt
                );
                std::process::exit(1);
            }
        }
    }

    // println!("Check point time: {:?} at draw room boundary", std::time::Instant::now());

    // draw room boundary
    for i_edge in 0..edge2vtxv_wall.len() / 2 {
        let i0_vtxv = edge2vtxv_wall[i_edge * 2 + 0];
        let i1_vtxv = edge2vtxv_wall[i_edge * 2 + 1];
        del_canvas_core::rasterize_line::draw_pixcenter(
            &mut canvas.data,
            canvas.width,
            &[vtxv2xy[i0_vtxv * 2 + 0], vtxv2xy[i0_vtxv * 2 + 1]],
            &[vtxv2xy[i1_vtxv * 2 + 0], vtxv2xy[i1_vtxv * 2 + 1]],
            arrayref::array_ref![transform_to_scr.as_slice(), 0, 9],
            1.6,
            1,
        );
    }

    // println!("Check point time: {:?} at rasterize polygon stroke", std::time::Instant::now());

    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        del_canvas_core::rasterize_polygon::stroke(
            &mut canvas.data,
            canvas.width,
            &vtxl2xy,
            arrayref::array_ref![transform_to_scr.as_slice(), 0, 9],
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

    std::io::stdout().flush().unwrap();
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

fn build_voronoi_geometry(
    vtxl2xy: &[f32],
    site2xy: &candle_core::Tensor,
    site2room: &[usize],
    backend: VoronoiBackend,
) -> anyhow::Result<(candle_core::Tensor, VoronoiInfo, Vec<f32>)> {
    let alive: Vec<bool> = site2room.iter().map(|room| *room != usize::MAX).collect();
    let site_coords_raw = site2xy.flatten_all()?.to_vec1::<f32>()?;
    let mut site_positions: Vec<[f32; 2]> = site_coords_raw
        .chunks(2)
        .filter(|chunk| chunk.len() == 2)
        .map(|c| [c[0], c[1]])
        .collect();
    let fallback = boundary_centroid(vtxl2xy);
    let sanitized = crate::voronoi::sanitize_site_positions(&mut site_positions, fallback);
    if sanitized > 0 {
        eprintln!(
            "[floorplan] sanitized {} site coordinates before Voronoi construction",
            sanitized
        );
    }
    let mut site_coords = Vec::with_capacity(site_positions.len() * 2);
    for pos in &site_positions {
        site_coords.push(pos[0]);
        site_coords.push(pos[1]);
    }
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
            let vtxv2xy = site2xy.apply_op1(layer)?;
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
            Ok((vtxv2xy, info, site_coords))
        }
        VoronoiBackend::Delaunay => {
            let diagram = crate::voronoi::compute_delaunay_voronoi(vtxl2xy, &site_coords, &alive)?;
            let layer = Layer {
                vtxl2xy: vtxl2xy.to_vec(),
                vtxv2info: diagram.vtxv2info.clone(),
            };
            let vtxv2xy = site2xy.apply_op1(layer)?;
            let info = VoronoiInfo {
                site2idx: diagram.site2idx,
                idx2vtxv: diagram.idx2vtxv,
                idx2site: diagram.idx2site,
                vtxv2info: diagram.vtxv2info,
            };
            Ok((vtxv2xy, info, site_coords))
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

    println!(
        "Check point time: {:?} at start build_voronoi_geometry",
        std::time::Instant::now()
    );
    std::io::stdout().flush().unwrap();

    let (vtxv2xy, voronoi_info, site_coords_sanitized) = build_voronoi_geometry(
        vtxl2xy,
        &site2xy_adjusted,
        site2room,
        VoronoiBackend::Delaunay,
    )?;

    println!(
        "Check point time: {:?} at end build_voronoi_geometry",
        std::time::Instant::now()
    );
    std::io::stdout().flush().unwrap();

    Ok(VoronoiStage {
        site2xy_adjusted,
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
    let loss_walllen = {
        let vtx2xyz_to_edgevector = del_candle::vtx2xyz_to_edgevector::Layer {
            edge2vtx: Vec::<usize>::from(edge2vtxv_wall.clone()),
        };
        let edge2xy = vtxv2xy.apply_op1(vtx2xyz_to_edgevector)?;
        edge2xy.abs()?.sum_all()?
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
    let loss_topo = loss_topo.affine(loss_weights.topology as f64, 0.0)?;
    let loss_fix = loss_fix.affine(loss_weights.fix as f64, 0.0)?;
    let loss_lloyd = loss_lloyd.affine(loss_weights.lloyd as f64, 0.0)?;
    let loss =
        (loss_each_area + loss_total_area + loss_walllen + loss_topo + loss_fix + loss_lloyd)?;

    println!(
        "Check point time: {:?} at start backward_step",
        std::time::Instant::now()
    );
    std::io::stdout().flush().unwrap();
    optimizer.backward_step(&loss)?;
    println!(
        "Check point time: {:?} at end backward_step",
        std::time::Instant::now()
    );
    std::io::stdout().flush().unwrap();

    Ok((
        site2xy_adjusted,
        voronoi_info,
        vtxv2xy,
        edge2vtxv_wall,
        site_coords_sanitized,
    ))
}

use std::io::Write;

fn optimize_phase(
    canvas_gif: &mut del_canvas_core::canvas_gif::Canvas,
    transform_world2pix: &nalgebra::Matrix3<f32>,
    vtxl2xy: &[f32],
    site2xy_start: &[f32],
    site2xy_ini: &candle_core::Tensor,
    site2xy2flag: &[f32],
    site2room: &[usize],
    room2area_trg: &candle_core::Tensor,
    room_connections: &[(usize, usize)],
    iter: usize,
    params: &ProjectParams,
) -> anyhow::Result<Vec<f32>> {
    let num_sites = if site2xy_start.is_empty() {
        0
    } else {
        site2xy_start.len() / 2
    };
    let device = &candle_core::Device::Cpu;
    let site2xy = candle_core::Var::from_slice(
        site2xy_start,
        candle_core::Shape::from((num_sites, 2)),
        device,
    )?;
    let site2xy2flag = candle_core::Var::from_slice(
        site2xy2flag,
        candle_core::Shape::from((num_sites, 2)),
        device,
    )?;

    let diag_path = PathBuf::from("target/site_diagnostics.txt");
    if let Some(parent) = diag_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
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
                room2area_trg.dims2()?.0,
                iter
            )?;
        } else {
            writeln!(file)?;
            writeln!(
                file,
                "# --- New phase: sites={} rooms={} iterations={} ---",
                site2room.len(),
                room2area_trg.dims2()?.0,
                iter
            )?;
        }
    }

    let learning_rates = &params.learning_rates;
    let mut optimizer = candle_nn::AdamW::new(
        vec![site2xy.clone()],
        candle_nn::ParamsAdamW {
            lr: learning_rates.first as f64,
            ..Default::default()
        },
    )?;

    let phase_timer = Instant::now();
    let mut persent_last = 0;
    for iter_idx in 0..iter {
        let persent = ((iter_idx + 1) * 100) / iter;
        if persent != persent_last {
            persent_last = persent;
            print!("{}% ", persent);
            let mut stdout = std::io::stdout().lock();
            stdout.flush()?;
        }

        if iter_idx == 150 {
            optimizer.set_params(candle_nn::ParamsAdamW {
                lr: learning_rates.second as f64,
                ..Default::default()
            });
        } else if iter_idx == 300 {
            optimizer.set_params(candle_nn::ParamsAdamW {
                lr: learning_rates.third as f64,
                ..Default::default()
            });
        }

        let voronoi_stage = iterate_voronoi_stage(vtxl2xy, &site2xy, site2room)?;

        let (_site2xy_adjusted, voronoi_info, vtxv2xy, edge2vtxv_wall, site_coords_sanitized) =
            optimize_iteration(
                vtxl2xy,
                &site2xy,
                site2xy_ini,
                &site2xy2flag,
                site2room,
                room2area_trg,
                room_connections,
                &mut optimizer,
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
        canvas_gif.clear(0);

        println!(
            "Check point time: {:?} at start my_paint",
            std::time::Instant::now()
        );
        std::io::stdout().flush().unwrap();

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

        println!(
            "Check point time: {:?} at end my_paint",
            std::time::Instant::now()
        );
        std::io::stdout().flush().unwrap();

        canvas_gif.write();
    }

    println!("Phase elapsed: {:.2?}", phase_timer.elapsed());
    let final_coords = site2xy.flatten_all()?.to_vec1::<f32>()?;

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
    Ok(final_coords)
}

fn optimize_impl(
    canvas_gif: &mut del_canvas_core::canvas_gif::Canvas,
    vtxl2xy: Vec<f32>,
    mut site2xy: Vec<f32>,
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
    let transform_world2pix = nalgebra::Matrix3::<f32>::new(
        canvas_width as f32 * 0.8,
        0.,
        canvas_width as f32 * 0.1,
        0.,
        -(canvas_height as f32) * 0.8,
        canvas_height as f32 * 0.9,
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

    let num_rooms = room2area_trg.len();
    let room2area_trg = candle_core::Tensor::from_vec(
        room2area_trg,
        candle_core::Shape::from((num_rooms, 1)),
        &candle_core::Device::Cpu,
    )?;

    let site2xy_ini = candle_core::Tensor::from_vec(
        site2xy.clone(),
        candle_core::Shape::from((site2xy.len() / 2, 2)),
        &candle_core::Device::Cpu,
    )?;

    let params_all = project_params_all();
    if params_index >= params_all.len() {
        anyhow::bail!(
            "project parameters index {} out of range ({} files detected)",
            params_index,
            params_all.len()
        );
    }

    let total_timer = Instant::now();
    for (phase_offset, params) in params_all[params_index..].iter().enumerate() {
        println!(
            "=== Starting phase {} (params index {}) ===",
            phase_offset,
            params_index + phase_offset
        );
        let phase_filename = format!("result{phase:02}.gif", phase = phase_offset);
        let mut phase_canvas = del_canvas_core::canvas_gif::Canvas::new(
            &phase_filename,
            (canvas_width, canvas_height),
            &palette,
        );
        site2xy = optimize_phase(
            &mut phase_canvas,
            &transform_world2pix,
            &vtxl2xy,
            &site2xy,
            &site2xy_ini,
            &site2xy2flag,
            &site2room,
            &room2area_trg,
            &room_connections,
            iter,
            params,
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
