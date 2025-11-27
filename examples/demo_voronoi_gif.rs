use del_canvas_core::canvas_gif::Canvas;
use rand::SeedableRng;
use rand::Rng;

fn centroid(vtx2xy: &[f32]) -> [f32; 2] {
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut area = 0.0;
    let num_vtx = vtx2xy.len() / 2;
    for i in 0..num_vtx {
        let j = (i + 1) % num_vtx;
        let x0 = vtx2xy[i * 2];
        let y0 = vtx2xy[i * 2 + 1];
        let x1 = vtx2xy[j * 2];
        let y1 = vtx2xy[j * 2 + 1];
        let cross = x0 * y1 - x1 * y0;
        area += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
    }
    if area.abs() < 1.0e-6 {
        return [vtx2xy[0], vtx2xy[1]];
    }
    area *= 0.5;
    cx /= 6.0 * area;
    cy /= 6.0 * area;
    [cx, cy]
}

fn is_inside(vtx2xy: &[f32], p: &[f32; 2]) -> bool {
    let wn = del_msh_core::polyloop2::winding_number(vtx2xy, p);
    (wn - 1.0).abs() < 0.1
}

fn random_sampling_in_polyloop(
    vtx2xy: &[f32],
    num_samples: usize,
    rng: &mut impl Rng,
) -> Vec<f32> {
    let mut samples = Vec::with_capacity(num_samples * 2);
    let aabb = del_msh_core::vtx2xy::aabb2(vtx2xy);
    let min = [aabb[0], aabb[1]];
    let max = [aabb[2], aabb[3]];
    let w = max[0] - min[0];
    let h = max[1] - min[1];
    
    while samples.len() < num_samples * 2 {
        let x = min[0] + rng.gen::<f32>() * w;
        let y = min[1] + rng.gen::<f32>() * h;
        if is_inside(vtx2xy, &[x, y]) {
            samples.push(x);
            samples.push(y);
        }
    }
    samples
}

fn main() -> anyhow::Result<()> {
    let str_path = "M7920 11494 c-193 -21 -251 -29 -355 -50 -540 -105 -1036 -366 -1442 \
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
    
    let outline_path = del_msh_core::io_svg::svg_outline_path_from_shape(str_path);
    let loops = del_msh_core::io_svg::svg_loops_from_outline_path(&outline_path);
    let vtxl2xy_nalgebra = del_msh_core::io_svg::polybezier2polyloop(&loops[0].0, &loops[0].1, loops[0].2, 300.);
    
    let mut vtxl2xy_flat = Vec::new();
    for p in &vtxl2xy_nalgebra {
        vtxl2xy_flat.push(p[0]);
        vtxl2xy_flat.push(p[1]);
    }
    
    let vtxl2xy_flat = del_msh_core::polyloop::resample::<f32, 2>(&vtxl2xy_flat, 100);
    
    let aabb = del_msh_core::vtx2xy::aabb2(&vtxl2xy_flat);
    let min = [aabb[0], aabb[1]];
    let max = [aabb[2], aabb[3]];
    let scale = 1.0 / (max[0] - min[0]).max(max[1] - min[1]);
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let vtxl2xy_flat: Vec<f32> = vtxl2xy_flat.chunks(2).flat_map(|v| {
        [(v[0] - center[0]) * scale + 0.5, (v[1] - center[1]) * scale + 0.5]
    }).collect();

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut site2xy = random_sampling_in_polyloop(&vtxl2xy_flat, 30, &mut rng);

    let palette = vec![0xFFFFFF, 0xFF0000, 0x0000FF, 0x000000]; // 0: White, 1: Red, 2: Blue, 3: Black
    let mut canvas = Canvas::new(
        std::path::Path::new("target/demo_voronoi.gif"),
        (600, 600),
        &palette,
    );

    let transform_mat = nalgebra::Matrix3::<f32>::new(
        500.0, 0.0, 50.0,
        0.0, 500.0, 50.0,
        0.0, 0.0, 1.0,
    );
    let transform_slice = transform_mat.as_slice();
    let transform: [f32; 9] = transform_slice.try_into().unwrap();

    for _iter in 0..50 {
        let (cells, _info) = floorplan::generate_voronoi_cells_robust(&vtxl2xy_flat, &site2xy);
        
        canvas.clear(0);
        
        // Draw boundary (Black)
        del_canvas_core::rasterize_polygon::stroke(
            &mut canvas.data,
            600,
            &vtxl2xy_flat,
            &transform,
            2.0,
            3u8,
        );

        // Draw Voronoi cells (Red)
        for cell in &cells {
            if cell.vtx2xy.is_empty() { continue; }
            del_canvas_core::rasterize_polygon::stroke(
                &mut canvas.data,
                600,
                &cell.vtx2xy,
                &transform,
                1.0,
                1u8,
            );
        }

        // Draw sites (Blue)
        for site in site2xy.chunks(2) {
             let p = [site[0], site[1]];
             del_canvas_core::rasterize_circle::fill(
                 &mut canvas.data,
                 600,
                 &p,
                 &transform,
                 3.0,
                 2u8,
             );
        }

        canvas.write();

        let mut new_site2xy = vec![0f32; site2xy.len()];
        for (i_site, cell) in cells.iter().enumerate() {
            if cell.vtx2xy.is_empty() {
                new_site2xy[i_site * 2] = site2xy[i_site * 2];
                new_site2xy[i_site * 2 + 1] = site2xy[i_site * 2 + 1];
                continue;
            }
            let c = centroid(&cell.vtx2xy);
            new_site2xy[i_site * 2] = c[0];
            new_site2xy[i_site * 2 + 1] = c[1];
        }
        site2xy = new_site2xy;
    }

    Ok(())
}
