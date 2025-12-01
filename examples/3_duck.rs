fn problem(
    seed: u64,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<usize>,
    Vec<f32>,
    Vec<f32>,
    Vec<i32>,
    Vec<(usize, usize)>,
) {
    use rand::SeedableRng;
    let mut reng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let num_room = 6;
    let room2color: Vec<i32> = {
        let mut room2color: Vec<i32> = vec![];
        for _i_room in 0..num_room {
            let c = floorplan::random_room_color(&mut reng);
            dbg!(c, format!("fill=\"#{:06X}\"", c));
            room2color.push(c);
        }
        room2color
    };
    let mut reng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    //
    let vtxl2xy = {
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
        // dbg!(&outline_path);
        let loops = del_msh_core::io_svg::svg_loops_from_outline_path(&outline_path);
        let vtxl2xy =
            del_msh_core::io_svg::polybezier2polyloop(&loops[0].0, &loops[0].1, loops[0].2, 300.);
        let mut vtxl2xy_flat: Vec<f32> = vtxl2xy.iter().flat_map(|p| [p[0], p[1]]).collect();
        vtxl2xy_flat = del_msh_core::polyloop::resample::<f32, 2>(&vtxl2xy_flat, 100);
        vtxl2xy_flat = del_msh_core::vtx2xy::normalize(&vtxl2xy_flat, &[0.5, 0.5], 1.0);
        {
            let mut min_x = vtxl2xy_flat[0];
            let mut max_x = vtxl2xy_flat[0];
            let mut min_y = vtxl2xy_flat[1];
            let mut max_y = vtxl2xy_flat[1];
            for i in (0..vtxl2xy_flat.len()).step_by(2) {
                let x = vtxl2xy_flat[i];
                let y = vtxl2xy_flat[i + 1];
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            println!("Boundary bounds: x=[{}, {}], y=[{}, {}]", min_x, max_x, min_y, max_y);
        }
        dbg!(vtxl2xy_flat.len());
        let _ = del_msh_core::io_obj::save_vtx2xyz_as_polyloop("target/loop.obj", &vtxl2xy_flat, 2);
        vtxl2xy_flat
    };
    dbg!(&vtxl2xy);
    let area_ratio = [0.4, 0.2, 0.2, 0.2, 0.2, 0.1];
    let room2area_trg: Vec<f32> = {
        let total_area = del_msh_core::polyloop2::area(&vtxl2xy);
        let sum_ratio: f32 = area_ratio.iter().sum();
        area_ratio
            .iter()
            .map(|v| v / sum_ratio * total_area)
            .collect()
    };
    dbg!(&room2area_trg);
    //
    let (site2xy, site2xy2flag, site2room) = {
        let mut site2xy =
            del_msh_core::polyloop2::poisson_disk_sampling(&vtxl2xy, 0.03, 50, &mut reng);
        let mut site2xy2flag = vec![0f32; site2xy.len()];
        let mut site2room = floorplan::site2room(
            site2xy.len() / 2,
            &room2area_trg[0..room2area_trg.len() - 1],
        );
        site2xy.extend([0.48, 0.06]);
        site2xy2flag.extend([1., 1.]);
        site2room.push(room2area_trg.len() - 1);
        site2xy.extend([0.52, 0.06]);
        site2xy2flag.extend([1., 1.]);
        site2room.push(room2area_trg.len() - 1);
        (site2xy, site2xy2flag, site2room)
    };
    assert_eq!(site2xy.len(), site2xy2flag.len());
    let room_connections: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
    // let room_connections: Vec<(usize, usize)> = vec!();
    (
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections,
    )
}

fn main() -> anyhow::Result<()> {
    let (vtxl2xy, site2xy, site2room, site2xy2flag, room2area_trg, room2color, room_connections) =
        problem(0);
    let mut canvas_gif = {
        let num_room = room2area_trg.len();
        let mut palette = vec![0xffffff, 0x000000];
        for i_room in 0..num_room {
            palette.push(room2color[i_room]);
        }
        del_canvas_core::canvas_gif::Canvas::new("target/3_duck.gif", (300, 300), &palette)
    };
    floorplan::optimize(
        &mut canvas_gif,
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections,
        400,
        0,
    )?;
    Ok(())
}
