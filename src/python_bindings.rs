#![cfg(feature = "python-bindings")]

use crate::loss_topo;
use candle_core::{Device, Shape, Tensor, Var};
use del_candle::voronoi2::VoronoiInfo;
use nalgebra::Matrix3;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::SeedableRng;

#[pyclass(name = "CanvasGif", unsendable)]
pub struct PyCanvasGif {
    pub(crate) inner: del_canvas_core::canvas_gif::Canvas,
}

#[pymethods]
impl PyCanvasGif {
    #[new]
    #[pyo3(signature = (path, width, height, palette))]
    fn new(path: String, width: usize, height: usize, palette: Vec<i32>) -> Self {
        Self {
            inner: del_canvas_core::canvas_gif::Canvas::new(&path, (width, height), &palette),
        }
    }

    pub fn clear(&mut self, color_index: u8) {
        self.inner.clear(color_index);
    }

    pub fn write(&mut self) {
        self.inner.write();
    }

    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width
    }

    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height
    }
}

#[pyclass(name = "VoronoiInfo", unsendable)]
pub struct PyVoronoiInfo {
    inner: VoronoiInfo,
    vtxv2xy: Vec<f32>,
}

#[pymethods]
impl PyVoronoiInfo {
    #[new]
    #[pyo3(signature = (vtxl2xy, site2xy, site2room))]
    fn new(vtxl2xy: Vec<f32>, site2xy: Vec<f32>, site2room: Vec<usize>) -> PyResult<Self> {
        let num_coords = site2xy.len();
        if num_coords % 2 != 0 {
            return Err(PyValueError::new_err("site2xy must contain x/y pairs"));
        }
        let num_site = num_coords / 2;
        if site2room.len() != num_site {
            return Err(PyValueError::new_err(
                "site2room length must match the number of (x,y) pairs",
            ));
        }
        let site2xy_var = var_from_vec(site2xy, num_site, 2)?;
        let (vtxv2xy_tensor, voronoi_info) = del_candle::voronoi2::voronoi(
            &vtxl2xy,
            &site2xy_var,
            |i_site| site2room[i_site] != usize::MAX,
        );
        let vtxv2xy = tensor_to_vec(vtxv2xy_tensor)?;
        Ok(Self {
            inner: voronoi_info,
            vtxv2xy,
        })
    }

    pub fn vtx_coordinates(&self) -> Vec<f32> {
        self.vtxv2xy.clone()
    }

    pub fn site2idx(&self) -> Vec<usize> {
        self.inner.site2idx.clone()
    }

    pub fn idx2vtxv(&self) -> Vec<usize> {
        self.inner.idx2vtxv.clone()
    }

    pub fn idx2site(&self) -> Vec<usize> {
        self.inner.idx2site.clone()
    }
}

impl PyVoronoiInfo {
    fn as_ref(&self) -> &VoronoiInfo {
        &self.inner
    }

    fn default_vtxv2xy(&self) -> Vec<f32> {
        self.vtxv2xy.clone()
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCanvasGif>()?;
    m.add_class::<PyVoronoiInfo>()?;
    m.add_function(wrap_pyfunction!(py_my_paint, m)?)?;
    m.add_function(wrap_pyfunction!(py_draw_svg, m)?)?;
    m.add_function(wrap_pyfunction!(py_random_room_color, m)?)?;
    m.add_function(wrap_pyfunction!(py_poisson_disk_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(py_edge2vtvx_wall, m)?)?;
    m.add_function(wrap_pyfunction!(py_loss_lloyd_internal, m)?)?;
    m.add_function(wrap_pyfunction!(py_room2area, m)?)?;
    m.add_function(wrap_pyfunction!(py_remove_site_too_close, m)?)?;
    m.add_function(wrap_pyfunction!(py_site2room, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(py_loss_topo_inverse_map, m)?)?;
    m.add_function(wrap_pyfunction!(py_loss_topo_unidirectional, m)?)?;
    m.add_function(wrap_pyfunction!(py_loss_topo_kmean_style, m)?)?;
    Ok(())
}

#[pyfunction(name = "my_paint", signature = (canvas, transform_to_scr, vtxl2xy, site2xy, voronoi_info, site2room, edge2vtxv_wall, vtxv2xy=None))]
fn py_my_paint(
    canvas: &Bound<'_, PyCanvasGif>,
    transform_to_scr: Vec<f32>,
    vtxl2xy: Vec<f32>,
    site2xy: Vec<f32>,
    voronoi_info: &PyVoronoiInfo,
    site2room: Vec<usize>,
    edge2vtxv_wall: Vec<usize>,
    vtxv2xy: Option<Vec<f32>>,
) -> PyResult<()> {
    let transform = matrix3_from_vec(&transform_to_scr)?;
    let vertex_data = vtxv2xy.unwrap_or_else(|| voronoi_info.default_vtxv2xy());
    let mut canvas_ref = canvas.borrow_mut();
    crate::my_paint(
        &mut canvas_ref.inner,
        &transform,
        &vtxl2xy,
        &site2xy,
        voronoi_info.as_ref(),
        &vertex_data,
        &site2room,
        &edge2vtxv_wall,
    );
    Ok(())
}

#[pyfunction(name = "draw_svg", signature = (file_path, transform_to_scr, vtxl2xy, site2xy, voronoi_info, site2room, edge2vtxv_wall, room2color, vtxv2xy=None))]
fn py_draw_svg(
    file_path: String,
    transform_to_scr: Vec<f32>,
    vtxl2xy: Vec<f32>,
    site2xy: Vec<f32>,
    voronoi_info: &PyVoronoiInfo,
    site2room: Vec<usize>,
    edge2vtxv_wall: Vec<usize>,
    room2color: Vec<i32>,
    vtxv2xy: Option<Vec<f32>>,
) -> PyResult<()> {
    let transform = matrix3_from_vec(&transform_to_scr)?;
    let vertex_data = vtxv2xy.unwrap_or_else(|| voronoi_info.default_vtxv2xy());
    crate::draw_svg(
        file_path,
        &transform,
        &vtxl2xy,
        &site2xy,
        voronoi_info.as_ref(),
        &vertex_data,
        &site2room,
        &edge2vtxv_wall,
        &room2color,
    );
    Ok(())
}

#[pyfunction(name = "random_room_color", signature = (seed=None))]
fn py_random_room_color(seed: Option<u64>) -> PyResult<i32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.unwrap_or(0));
    Ok(crate::random_room_color(&mut rng))
}

#[pyfunction(name = "poisson_disk_sampling", signature = (polygon, radius, k, seed=None))]
fn py_poisson_disk_sampling(
    polygon: Vec<(f32, f32)>,
    radius: f32,
    k: usize,
    seed: Option<u64>,
) -> PyResult<Vec<f32>> {
    if polygon.len() < 3 {
        return Err(PyValueError::new_err("polygon must contain at least 3 points"));
    }
    if radius <= 0.0 {
        return Err(PyValueError::new_err("radius must be positive"));
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.unwrap_or(0));
    Ok(crate::poisson_disk_sampling(&polygon, radius, k, &mut rng))
}

#[pyfunction(name = "edge2vtvx_wall")]
fn py_edge2vtvx_wall(voronoi_info: &PyVoronoiInfo, site2room: Vec<usize>) -> PyResult<Vec<usize>> {
    Ok(crate::edge2vtvx_wall(voronoi_info.as_ref(), &site2room))
}

#[pyfunction(name = "loss_lloyd_internal", signature = (voronoi_info, site2room, site2xy, num_site, vtxv2xy=None, num_vertex=None))]
fn py_loss_lloyd_internal(
    voronoi_info: &PyVoronoiInfo,
    site2room: Vec<usize>,
    site2xy: Vec<f32>,
    num_site: usize,
    vtxv2xy: Option<Vec<f32>>,
    num_vertex: Option<usize>,
) -> PyResult<f32> {
    let site2xy_var = var_from_vec(site2xy, num_site, 2)?;
    let vertex_data = vtxv2xy.unwrap_or_else(|| voronoi_info.default_vtxv2xy());
    let num_vertex = num_vertex.unwrap_or_else(|| vertex_data.len() / 2);
    let vtx_tensor = tensor_from_vec(vertex_data, num_vertex, 2)?;
    let loss = crate::loss_lloyd_internal(
        voronoi_info.as_ref(),
        &site2room,
        &site2xy_var,
        &vtx_tensor,
    )
    .map_err(candle_err)?;
    tensor_to_scalar(loss)
}

#[pyfunction(name = "room2area", signature = (site2room, num_room, voronoi_info, vtxv2xy=None, num_vertex=None))]
fn py_room2area(
    site2room: Vec<usize>,
    num_room: usize,
    voronoi_info: &PyVoronoiInfo,
    vtxv2xy: Option<Vec<f32>>,
    num_vertex: Option<usize>,
) -> PyResult<Vec<f32>> {
    let vertex_data = vtxv2xy.unwrap_or_else(|| voronoi_info.default_vtxv2xy());
    let num_vertex = num_vertex.unwrap_or_else(|| vertex_data.len() / 2);
    let vtx_tensor = tensor_from_vec(vertex_data, num_vertex, 2)?;
    let info = voronoi_info.as_ref();
    let areas = crate::room2area(
        &site2room,
        num_room,
        &info.site2idx,
        &info.idx2vtxv,
        &vtx_tensor,
    )
    .map_err(candle_err)?;
    tensor_to_vec(areas)
}

#[pyfunction(name = "remove_site_too_close")]
fn py_remove_site_too_close(site2room: Vec<usize>, site2xy: Vec<f32>, num_site: usize) -> PyResult<Vec<usize>> {
    let mut site2room = site2room;
    let site2xy_tensor = tensor_from_vec(site2xy, num_site, 2)?;
    crate::remove_site_too_close(&mut site2room, &site2xy_tensor);
    Ok(site2room)
}

#[pyfunction(name = "site2room")]
fn py_site2room(num_site: usize, room2area: Vec<f32>) -> PyResult<Vec<usize>> {
    Ok(crate::site2room(num_site, &room2area))
}

#[pyfunction(name = "optimize")]
fn py_optimize(
    canvas: &Bound<'_, PyCanvasGif>,
    vtxl2xy: Vec<f32>,
    site2xy: Vec<f32>,
    site2room: Vec<usize>,
    site2xy2flag: Vec<f32>,
    room2area_trg: Vec<f32>,
    room2color: Vec<i32>,
    room_connections: Vec<(usize, usize)>,
) -> PyResult<()> {
    let mut canvas_ref = canvas.borrow_mut();
    crate::optimize(
        &mut canvas_ref.inner,
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections,
    )
    .map_err(anyhow_err)
}

#[pyfunction(name = "loss_topo_inverse_map")]
fn py_loss_topo_inverse_map(num_group: usize, site2group: Vec<usize>) -> PyResult<Vec<Vec<usize>>> {
    Ok(loss_topo::inverse_map(num_group, &site2group))
}

#[pyfunction(name = "loss_topo_unidirectional")]
fn py_loss_topo_unidirectional(
    site2xy: Vec<f32>,
    num_site: usize,
    site2room: Vec<usize>,
    num_room: usize,
    voronoi_info: &PyVoronoiInfo,
    room_connections: Vec<(usize, usize)>,
) -> PyResult<f32> {
    let site2xy_tensor = tensor_from_vec(site2xy, num_site, 2)?;
    let loss = loss_topo::unidirectional(
        &site2xy_tensor,
        &site2room,
        num_room,
        voronoi_info.as_ref(),
        &room_connections,
    )
    .map_err(candle_err)?;
    tensor_to_scalar(loss)
}

#[pyfunction(name = "loss_topo_kmean_style")]
fn py_loss_topo_kmean_style(
    site2xy: Vec<f32>,
    num_site: usize,
    site2room: Vec<usize>,
    num_room: usize,
    voronoi_info: &PyVoronoiInfo,
    room_connections: Vec<(usize, usize)>,
) -> PyResult<f32> {
    let site2xy_tensor = tensor_from_vec(site2xy, num_site, 2)?;
    let loss = loss_topo::kmean_style(
        &site2xy_tensor,
        &site2room,
        num_room,
        voronoi_info.as_ref(),
        &room_connections,
    )
    .map_err(candle_err)?;
    tensor_to_scalar(loss)
}

fn matrix3_from_vec(values: &[f32]) -> PyResult<Matrix3<f32>> {
    if values.len() != 9 {
        return Err(PyValueError::new_err("transform matrix must contain 9 values"));
    }
    Ok(Matrix3::from_row_slice(values))
}

fn tensor_from_vec(data: Vec<f32>, rows: usize, cols: usize) -> PyResult<Tensor> {
    ensure_len(data.len(), rows, cols)?;
    Tensor::from_vec(data, Shape::from((rows, cols)), &Device::Cpu).map_err(candle_err)
}

fn var_from_vec(data: Vec<f32>, rows: usize, cols: usize) -> PyResult<Var> {
    ensure_len(data.len(), rows, cols)?;
    Var::from_slice(&data, Shape::from((rows, cols)), &Device::Cpu).map_err(candle_err)
}

fn tensor_to_vec(tensor: Tensor) -> PyResult<Vec<f32>> {
    tensor
        .flatten_all()
        .map_err(candle_err)?
        .to_vec1::<f32>()
        .map_err(candle_err)
}

fn tensor_to_scalar(tensor: Tensor) -> PyResult<f32> {
    tensor.to_vec0::<f32>().map_err(candle_err)
}

fn ensure_len(len: usize, rows: usize, cols: usize) -> PyResult<()> {
    if len != rows * cols {
        return Err(PyValueError::new_err(format!(
            "buffer length {} does not match shape {}x{}",
            len, rows, cols
        )));
    }
    Ok(())
}

fn candle_err(err: candle_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("candle error: {err}"))
}

fn anyhow_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}
