use del_msh_core::voronoi2::voronoi_cells;
use std::fs::File;
use std::io::Write;
use std::path::Path;

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

fn main() {
    // Bounding square (counter-clockwise)
    let boundary = vec![0.0f32, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    // Sites chosen so that three of them are collinear/co-circular with the center
    let sites = vec![
        0.2, 0.5, // left mid
        0.5, 0.5, // center (participates in degeneracy)
        0.8, 0.5, // right mid
        0.5, 0.2, // top of circle through the others
    ];

    let cells = voronoi_cells(&boundary, &sites, |_| true);
    let mut active_cells = vec![];
    let mut cell_vertices = vec![];
    for (i, cell) in cells.iter().enumerate() {
        if cell.vtx2xy.is_empty() {
            continue;
        }
        let area = polygon_area(&cell.vtx2xy);
        active_cells.push((i, area, cell.vtx2xy.len() / 2));
        cell_vertices.push((i, cell.vtx2xy.clone()));
    }

    assert!(
        active_cells.len() >= 3,
        "expected at least 3 non-empty cells, got {}",
        active_cells.len()
    );

    println!(
        "Voronoi demo succeeded with {} active cells",
        active_cells.len()
    );
    for (idx, area, num_vertices) in active_cells {
        println!(
            "  site #{idx}: area={area:.4}, vertices={num_vertices}",
            idx = idx,
            area = area,
            num_vertices = num_vertices
        );
    }

    let svg_path = Path::new("target/voronoi_demo.svg");
    write_svg(svg_path, &cell_vertices, &boundary, &sites).expect("failed to write svg");
    println!("SVG written to {}", svg_path.display());
}

fn write_svg(
    path: &Path,
    cells: &[(usize, Vec<f32>)],
    boundary: &[f32],
    sites: &[f32],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    let colors = [
        "#ff6b6b", "#4ecdc4", "#ffe66d", "#1a535c", "#ff9f1c", "#2ec4b6", "#b5179e",
    ];
    writeln!(
        file,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1 1\" width=\"600\" height=\"600\">"
    )?;
    // draw boundary outline
    if !boundary.is_empty() {
        let mut path_d = String::new();
        for (i, chunk) in boundary.chunks(2).enumerate() {
            let x = chunk[0];
            let y = chunk[1];
            if i == 0 {
                path_d.push_str(&format!("M{:.4},{:.4}", x, y));
            } else {
                path_d.push_str(&format!(" L{:.4},{:.4}", x, y));
            }
        }
        path_d.push('Z');
        writeln!(
            file,
            "  <path d=\"{}\" fill=\"none\" stroke=\"#333\" stroke-width=\"0.003\" opacity=\"0.8\"/>",
            path_d
        )?;
    }
    // draw cells
    for (idx, verts) in cells {
        if verts.len() < 6 {
            continue;
        }
        let color = colors[idx % colors.len()];
        let mut path_d = String::new();
        for (i, chunk) in verts.chunks(2).enumerate() {
            let x = chunk[0];
            let y = chunk[1];
            if i == 0 {
                path_d.push_str(&format!("M{:.4},{:.4}", x, y));
            } else {
                path_d.push_str(&format!(" L{:.4},{:.4}", x, y));
            }
        }
        path_d.push('Z');
        writeln!(
            file,
            "  <path d=\"{}\" fill=\"{}\" fill-opacity=\"0.35\" stroke=\"{}\" stroke-width=\"0.002\"/>",
            path_d,
            color,
            color
        )?;
    }
    // draw sites
    for chunk in sites.chunks(2) {
        writeln!(
            file,
            "  <circle cx=\"{:.4}\" cy=\"{:.4}\" r=\"0.01\" fill=\"#000\"/>",
            chunk[0], chunk[1]
        )?;
    }
    writeln!(file, "</svg>")?;
    Ok(())
}
