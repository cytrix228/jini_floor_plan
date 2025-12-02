use std::io;
use std::path::Path;

use floorplan::fracture::{run_demo, run_duck_demo, run_voronoi_fracture_from_diag};

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

#[test]
fn voronoi_fracture_from_diag_runs() -> io::Result<()> {
    match run_voronoi_fracture_from_diag() {
        Ok(()) => assert!(Path::new("fracture_from_diag.svg").exists()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!("[test] skipping voronoi_fracture_from_diag_runs: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    }
    Ok(())
}
