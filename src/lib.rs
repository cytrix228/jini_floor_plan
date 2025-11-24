use arrayref::array_ref;
use std::any::Any;
use std::backtrace::Backtrace;
use std::iter;
use del_candle::voronoi2::VoronoiInfo;
use del_canvas_core::canvas_gif::Canvas;
pub mod loss_topo;

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
    //
    for i_site in 0..site2idx.len() - 1 {
        let i_room = site2room[i_site];
        if i_room == usize::MAX {
            continue;
        }
        //
        let i_color: u8 = if i_room == usize::MAX {
            1
        } else {
            (i_room + 2).try_into().unwrap()
        };
        //
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        if num_vtx_in_site == 0 { continue; }
        let mut vtx2xy= vec!(0f32; num_vtx_in_site * 2);
        for i_vtx in 0..num_vtx_in_site {
            let i_vtxv = idx2vtxv[site2idx[i_site] + i_vtx];
            vtx2xy[i_vtx*2+0] = vtxv2xy[i_vtxv*2+0];
            vtx2xy[i_vtx*2+1] = vtxv2xy[i_vtxv*2+1];
        }
        del_canvas_core::rasterize_polygon::fill(
            &mut canvas.data, canvas.width,
            &vtx2xy,  arrayref::array_ref![transform_to_scr.as_slice(),0,9], i_color);
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
            arrayref::array_ref![transform_to_scr.as_slice(),0,9],
            2.0,
            1,
        );
    }
    // draw cell boundary
    for i_site in 0..site2idx.len() - 1 {
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let i0 = idx2vtxv[site2idx[i_site] + i0_vtx];
            let i1 = idx2vtxv[site2idx[i_site] + i1_vtx];
            del_canvas_core::rasterize_line::draw_dda_with_transformation(
                &mut canvas.data,
                canvas.width,
                &[vtxv2xy[i0 * 2 + 0], vtxv2xy[i0 * 2 + 1]],
                &[vtxv2xy[i1 * 2 + 0], vtxv2xy[i1 * 2 + 1]],
                arrayref::array_ref![transform_to_scr.as_slice(),0,9],
                1,
            );
        }
    }
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
    del_canvas_core::rasterize_polygon::stroke(
                                           &mut canvas.data,
                                           canvas.width,
                                           &vtxl2xy,
                                           arrayref::array_ref![transform_to_scr.as_slice(),0,9],
                                           1.6,
                                           1,
    );
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
    room2color: &[i32])
{
    let mut canvas_svg = del_canvas_core::canvas_svg::Canvas::new(file_path, (300, 300));
    {
//        let vtxv2xy = vtxv2xy.flatten_all()?.to_vec1()?;
        for i_site in 0..voronoi_info.site2idx.len() - 1 {
            let mut hoge = vec!();
            for &i_vtxv in &voronoi_info.idx2vtxv[voronoi_info.site2idx[i_site]..voronoi_info.site2idx[i_site + 1]] {
                hoge.push(vtxv2xy[i_vtxv * 2 + 0]);
                hoge.push(vtxv2xy[i_vtxv * 2 + 1]);
            }
            let i_room = site2room[i_site];
            let i_color = room2color[i_room];
            canvas_svg.polyloop(&hoge, &transform_to_scr, Some(0x333333), Some(0.1), Some(i_color));
        }
        for i_edge in 0..edge2vtxv_wall.len() / 2 {
            let i0_vtxv = edge2vtxv_wall[i_edge*2+0];
            let i1_vtxv = edge2vtxv_wall[i_edge*2+1];
            let x0 = vtxv2xy[i0_vtxv*2+0];
            let y0 = vtxv2xy[i0_vtxv*2+1];
            let x1 = vtxv2xy[i1_vtxv*2+0];
            let y1 = vtxv2xy[i1_vtxv*2+1];
            canvas_svg.line(x0,y0, x1,y1, &transform_to_scr, Some(2.0));
        }
    }
    canvas_svg.polyloop(vtxl2xy, &transform_to_scr, Some(0x000000), Some(2.0), None);
    {
        //let site2xy = site2xy.flatten_all()?.to_vec1()?;
        for i_vtx in 0..site2xy.len() / 2 {
            canvas_svg.circle(
                site2xy[i_vtx * 2 + 0], site2xy[i_vtx * 2 + 1],
                &transform_to_scr, 1., "#FF0000");
        }
    }
    canvas_svg.write();
}


pub fn random_room_color<RNG>(reng: &mut RNG) -> i32
    where RNG: rand::Rng
{
    let h = reng.gen::<f32>();
    let s = reng.gen::<f32>();
    let s = 0.5 + 0.1 * s;
    let v = reng.gen::<f32>();
    let v = 0.9 + 0.1 * v;
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
        let intersects = ((y0 > y) != (y1 > y))
            && (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-9_f32) + x0);
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
    dbg!( polygon.len() );

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

    dbg!( min_x, max_x, min_y, max_y );

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
            rng.gen_range(start..end)
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

    dbg!( "samples length: {}", samples.len() );

    while !active.is_empty() {
        let idx = rng.gen_range(0..active.len());
        let base_idx = active[idx];
        let base = samples[base_idx];
        let mut found = false;
        for _ in 0..k {
            let angle = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
            let dist = rng.gen_range(radius..(2.0 * radius));
            let candidate = (
                base.0 + angle.cos() * dist,
                base.1 + angle.sin() * dist,
            );
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
                    if ((candidate.0 - neighbor.0).powi(2)
                        + (candidate.1 - neighbor.1).powi(2))
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
    vtxv2xy: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let num_site = site2room.len();
    assert_eq!(voronoi_info.site2idx.len()-1, num_site);
    let site2idx = &voronoi_info.site2idx;
    // let idx2vtxv = &voronoi_info.idx2vtxv;
    let mut site2canmove = vec![false; num_site];
    // get wall between rooms
    for i_site in 0..site2idx.len() - 1 {
        if voronoi_info.site2idx[i_site + 1] == voronoi_info.site2idx[i_site] { // there is no cell
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
    let mask: Vec<f32> = site2canmove.iter().flat_map(|v| if *v {[0f32, 0f32] }else {[1f32, 1f32]}).collect();
    let mask = candle_core::Tensor::from_vec(
        mask,
        (num_site, 2),
        &candle_core::Device::Cpu)?;
    let polygonmesh2_to_cogs = del_candle::polygonmesh2_to_cogs::Layer {
        elem2idx: Vec::from(voronoi_info.site2idx.clone()),
        idx2vtx: Vec::from(voronoi_info.idx2vtxv.clone())
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
        let p_i = del_msh_core::vtx2xy::to_navec2(&site2xy, i_site);
        for j_site in (i_site + 1)..num_site {
            let j_room = site2room[j_site];
            if j_room == usize::MAX {
                continue;
            }
            if i_room != j_room {
                continue;
            }
            let p_j = del_msh_core::vtx2xy::to_navec2(&site2xy, j_site);
            if (p_i - p_j).norm() < 0.02 {
                site2room[j_site] = usize::MAX;
            }
        }
    }
}

pub fn site2room(
    num_site: usize,
    room2area: &[f32]) -> Vec<usize>
{
    let num_room = room2area.len();
    let mut site2room: Vec<usize> = vec![usize::MAX; num_site];
    let num_site_assign = num_site - num_room;
    let area: f32 = room2area.iter().sum();
    {
        let cumsum: Vec<f32> = room2area.clone().iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            }).collect();
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

fn optimize_impl(
    canvas_gif: &mut del_canvas_core::canvas_gif::Canvas,
    vtxl2xy: Vec<f32>,
    site2xy: Vec<f32>,
    site2room: Vec<usize>,
    site2xy2flag: Vec<f32>,
    room2area_trg: Vec<f32>,
    room2color: Vec<i32>,
    room_connections: Vec<(usize, usize)>,
    iter: usize) -> anyhow::Result<()>
{

    let transform_world2pix = nalgebra::Matrix3::<f32>::new(
        canvas_gif.width as f32 * 0.8,
        0.,
        canvas_gif.width as f32 * 0.1,
        0.,
        -(canvas_gif.height as f32) * 0.8,
        canvas_gif.height as f32 * 0.9,
        0.,
        0.,
        1.,
    );
    // ---------------------
    // candle from here
    let site2xy = candle_core::Var::from_slice(
        &site2xy,
        candle_core::Shape::from((site2xy.len() / 2, 2)),
        &candle_core::Device::Cpu,
    ).unwrap();
    let site2xy2flag = candle_core::Var::from_slice(
        &site2xy2flag,
        candle_core::Shape::from((site2xy2flag.len() / 2, 2)),
        &candle_core::Device::Cpu,
    ).unwrap();
    let site2xy_ini = candle_core::Tensor::from_vec(
        site2xy.flatten_all().unwrap().to_vec1::<f32>()?,
        candle_core::Shape::from((site2xy.dims2()?.0, 2)),
        &candle_core::Device::Cpu,
    ).unwrap();
    assert_eq!(site2room.len(), site2xy.dims2()?.0);
    //
    let room2area_trg = {
        let num_room = room2area_trg.len();
        candle_core::Tensor::from_vec(
            room2area_trg,
            candle_core::Shape::from((num_room, 1)),
            &candle_core::Device::Cpu,
        )
            .unwrap()
    };
    let adamw_params = candle_nn::ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    use candle_nn::Optimizer;
    use std::time::Instant;
    dbg!(site2room.len());
    let now = Instant::now();
    let mut optimizer = candle_nn::AdamW::new(vec![site2xy.clone()], adamw_params)?;
    for _iter in 0..iter {
        if _iter == 150 {
            let adamw_params = candle_nn::ParamsAdamW {
                lr: 0.005,
                ..Default::default()
            };
            optimizer.set_params(adamw_params);
        }
        let (vtxv2xy, voronoi_info) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || del_candle::voronoi2::voronoi(&vtxl2xy, &site2xy, |i_site| {
                site2room[i_site] != usize::MAX
            }),
        ))
        .map_err(|payload| {
            let message = panic_payload_to_string(payload.as_ref());
            let backtrace = Backtrace::force_capture();
            anyhow::anyhow!(
                "voronoi() panicked while building geometry: {message}\nBacktrace:\n{backtrace}"
            )
        })?;
        let edge2vtxv_wall = crate::edge2vtvx_wall(&voronoi_info, &site2room);
        /*
        if _iter == 0 || _iter == 3 || _iter == 10 || _iter == 100 || _iter == 300 || _iter == 400 {
        //if _iter == 0 || _iter == 400 {
         */
        /*
        if _iter == 400 {
            crate::draw_svg(
                format!("target/hoge_{}.svg", _iter),
                &transform_world2pix,
                &vtxl2xy,
                &site2xy.flatten_all()?.to_vec1::<f32>()?,
                &voronoi_info,
                &vtxv2xy.flatten_all()?.to_vec1::<f32>()?,
                &site2room,
                &edge2vtxv_wall,
                &room2color);
        }
         */
        // ----------------------
        // let loss_lloyd_internal = floorplan::loss_lloyd_internal(&voronoi_info, &site2room, &site2xy, &vtxv2xy)?;
        let (loss_each_area, loss_total_area) = {
            let room2area = crate::room2area(
                &site2room,
                room2area_trg.dims2()?.0,
                &voronoi_info.site2idx,
                &voronoi_info.idx2vtxv,
                &vtxv2xy,
            )?;
            /*
            {
                let room2area = room2area.flatten_all()?.to_vec1::<f32>()?;
                let total_area = del_msh::polyloop2::area_(&vtxl2xy);
                for i_room in 0..room2area.len() {
                    println!("    room:{} area:{}", i_room, room2area[i_room]/total_area);
                }
            }
             */
            let loss_each_area = room2area.sub(&room2area_trg)?.sqr()?.sum_all()?;
            let total_area_trg = del_msh_core::polyloop2::area_(&vtxl2xy);
            let total_area_trg = candle_core::Tensor::from_vec(
                vec![total_area_trg],
                candle_core::Shape::from(()),
                &candle_core::Device::Cpu,
            )?;
            let loss_total_area = (room2area.sum_all()? - total_area_trg)?.abs()?;
            (loss_each_area, loss_total_area)
        };
        // println!("  loss each_area {}", loss_each_area.to_vec0::<f32>()?);
        // println!("  loss total_area {}", loss_total_area.to_vec0::<f32>()?);
        let loss_walllen = {
            let vtx2xyz_to_edgevector = del_candle::vtx2xyz_to_edgevector::Layer {
                edge2vtx: Vec::<usize>::from(edge2vtxv_wall.clone()),
            };
            let edge2xy = vtxv2xy.apply_op1(vtx2xyz_to_edgevector)?;
            edge2xy.abs()?.sum_all()?
            //edge2xy.sqr()?.sum_all()?
        };
        let loss_topo = crate::loss_topo::unidirectional(
            &site2xy,
            &site2room,
            room2area_trg.dims2()?.0,
            &voronoi_info,
            &room_connections,
        )?;
        // println!("  loss topo: {}", loss_topo.to_vec0::<f32>()?);
        //let loss_fix = site2xy.sub(&site2xy_ini)?.mul(&site2xy2flag)?.sum_all()?;
        //let loss_fix = site2xy.sub(&site2xy_ini)?.mul(&site2xy2flag)?.sum_all()?;
        let loss_fix = site2xy.sub(&site2xy_ini)?.mul(&site2xy2flag)?.sqr()?.sum_all()?;
        let loss_lloyd = del_candle::voronoi2::loss_lloyd(
            &voronoi_info.site2idx, &voronoi_info.idx2vtxv,
            &site2xy, &vtxv2xy)?;
        // dbg!(loss_fix.to_vec0::<f32>()?);
        // ---------
        /*
        let loss_each_area = if _iter > 150 {
            loss_each_area.affine(5.0, 0.0)?.clone()
        }
        else {
        };
         */
        let loss_each_area = loss_each_area.affine(1.0, 0.0)?.clone();
        let loss_total_area = loss_total_area.affine(10.0, 0.0)?.clone();
        let loss_walllen = loss_walllen.affine(0.02, 0.0)?;
        let loss_topo = loss_topo.affine(1., 0.0)?;
        let loss_fix = loss_fix.affine(100., 0.0)?;
        let loss_lloyd = loss_lloyd.affine(0.1, 0.0)?;
        // dbg!(loss_fix.flatten_all()?.to_vec1::<f32>());
        /*
        {
            let mut file = std::fs::OpenOptions::new().write(true).append(true).open("target/conv.csv")?;
            let mut writer = std::io::BufWriter::new(&file);
            writeln!(&mut writer, "{}, {},{},{},{},{}",
                     _iter,
                     (loss_each_area.clone() + loss_total_area.clone())?.to_vec0::<f32>()?,
                     loss_walllen.clone().to_vec0::<f32>()?,
                     loss_topo.clone().to_vec0::<f32>()?,
                     loss_fix.clone().to_vec0::<f32>()?,
                     loss_lloyd.clone().to_vec0::<f32>()?,
            );
        }
         */
        let loss = (
            loss_each_area
                + loss_total_area
                + loss_walllen
                + loss_topo
                + loss_fix
                + loss_lloyd
        )?;
        // println!("  loss: {}", loss.to_vec0::<f32>()?);
        optimizer.backward_step(&loss)?;
        // ----------------
        // visualization
        canvas_gif.clear(0);
        crate::my_paint(
            canvas_gif,
            &transform_world2pix,
            &vtxl2xy,
            &site2xy.flatten_all()?.to_vec1::<f32>()?,
            &voronoi_info,
            &vtxv2xy.flatten_all()?.to_vec1::<f32>()?,
            &site2room,
            &edge2vtxv_wall,
        );
        canvas_gif.write();
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
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
        )
    }));
    match result {
        Ok(inner) => inner,
        Err(payload) => {
            let message = panic_payload_to_string(payload.as_ref());
            let backtrace = Backtrace::force_capture();
            Err(anyhow::anyhow!("optimize panicked: {message}\nBacktrace:\n{backtrace}"))
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

