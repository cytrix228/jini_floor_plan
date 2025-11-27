use floorplan;
use del_msh_core;
use del_canvas_core;
use nalgebra;
use rand::SeedableRng;
use rand::Rng;

fn get_transform(width: usize, height: usize, vtxl2xy: &[f32], margin: f32) -> [f32; 9] {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for i in 0..vtxl2xy.len()/2 {
        let x = vtxl2xy[i*2];
        let y = vtxl2xy[i*2+1];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    let w = max_x - min_x;
    let h = max_y - min_y;
    let scale = (width as f32 / w).min(height as f32 / h) / margin;
    
    let tx = -cx * scale + width as f32 * 0.5;
    let ty = -cy * scale + height as f32 * 0.5;
    
    // Column-major: [m11, m21, m31, m12, m22, m32, m13, m23, m33]
    [scale, 0.0, 0.0, 0.0, scale, 0.0, tx, ty, 1.0]
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
    let vtxl2xy = del_msh_core::io_svg::polybezier2polyloop(&loops[0].0, &loops[0].1, loops[0].2, 300.);
    let vtxl2xy = del_msh_core::vtx2xdim::from_array_of_nalgebra(&vtxl2xy);
    let vtxl2xy = del_msh_core::polyloop::resample::<f32, 2>(&vtxl2xy, 100);
    let vtxl2xy = del_msh_core::vtx2xdim::to_array_of_nalgebra_vector(&vtxl2xy);
    let vtxl2xy = del_msh_core::vtx2vec::normalize2(
        &vtxl2xy,
        &nalgebra::Vector2::<f32>::new(0.5, 0.5),
        1.0,
    );
    let vtxl2xy_flat = del_msh_core::vtx2xdim::from_array_of_nalgebra(&vtxl2xy);

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut site2xy = del_msh_core::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.05, 30, &mut rng,
    );

    let mut canvas = del_canvas_core::canvas_gif::Canvas::new(
        "target/demo_voronoi.gif",
        (400, 400),
        vec![
            255, 255, 255, // 0: White
            0, 0, 0,       // 1: Black
            0, 0, 255,     // 2: Blue
            255, 0, 0      // 3: Red
        ],
    )?;

    let transform = get_transform(400, 400, &vtxl2xy_flat, 1.1);

    for _i_frame in 0..50 {
        // Random walk sites
        for i in 0..site2xy.len()/2 {
            site2xy[i*2] += (rng.gen::<f32>() - 0.5) * 0.01;
            site2xy[i*2+1] += (rng.gen::<f32>() - 0.5) * 0.01;
        }

        let cells = floorplan::compute_voronoi_cells_with_logging(&vtxl2xy_flat, &site2xy);

        canvas.clear(0); // Clear to background color (index 0 in palette? No, clear takes color index)
        // Wait, Canvas::new takes palette. 
        // del_canvas_core::canvas_gif::Canvas::new(path, size, palette)
        // palette is Vec<u8>. RGBRGB...
        // If I pass vec![255, 255, 255], index 0 is white.
        
        // Draw boundary
        for i in 0..vtxl2xy.len() {
            let p0 = &vtxl2xy[i];
            let p1 = &vtxl2xy[(i+1)%vtxl2xy.len()];
            del_canvas_core::rasterize_line::draw_dda_with_transformation(
                &mut canvas.data, canvas.width,
                &[p0.x, p0.y], &[p1.x, p1.y],
                &transform, 1 // Color 1? I need more colors in palette.
            );
        }

        // Draw cells
        for cell in cells {
            if cell.vtx2xy.is_empty() { continue; }
            let num_vtx = cell.vtx2xy.len() / 2;
            for i in 0..num_vtx {
                let p0 = &[cell.vtx2xy[i*2], cell.vtx2xy[i*2+1]];
                let p1 = &[cell.vtx2xy[((i+1)%num_vtx)*2], cell.vtx2xy[((i+1)%num_vtx)*2+1]];
                del_canvas_core::rasterize_line::draw_dda_with_transformation(
                    &mut canvas.data, canvas.width,
                    p0, p1,
                    &transform, 2 
                );
            }
        }

        // Draw sites
        for i in 0..site2xy.len()/2 {
            del_canvas_core::rasterize_circle::fill(
                &mut canvas.data, canvas.width,
                &[site2xy[i*2], site2xy[i*2+1]],
                &transform, 2.0, 3
            );
        }

        canvas.write();
    }

    Ok(())
}
