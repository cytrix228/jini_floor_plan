fn polygon_area(coords: &[f32]) -> f32 {
    if coords.len() < 6 {
        return 0.0;
    }
    let mut area = 0.0f32;
    let n = coords.len() / 2;
    for i in 0..n {
        let j = (i + 1) % n;
        let xi = coords[i * 2];
        let yi = coords[i * 2 + 1];
        let xj = coords[j * 2];
        let yj = coords[j * 2 + 1];
        area += xi * yj - xj * yi;
    }
    0.5 * area.abs()
}

#[test]
fn collinear_sites_still_form_cells() {
    let boundary = vec![0.0f32, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let sites = vec![0.2, 0.5, 0.5, 0.5, 0.8, 0.5, 0.5, 0.2];
    let cells = del_msh_core::voronoi2::voronoi_cells(&boundary, &sites, |_| true);
    let mut non_empty = 0;
    for cell in cells.iter() {
        if cell.vtx2xy.is_empty() {
            continue;
        }
        non_empty += 1;
        assert!(polygon_area(&cell.vtx2xy) > 1.0e-6);
    }
    assert!(
        non_empty >= 3,
        "expected at least 3 non-empty cells, got {}",
        non_empty
    );
}
