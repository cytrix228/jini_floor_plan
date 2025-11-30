#![cfg(feature = "python-bindings")]

use crate::{loss_topo, project_params_all, ProjectParams, VoronoiStage};
use candle_core::{Device, Shape, Tensor, Var};
use candle_nn::{self, Optimizer};
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
        let (vtxv2xy_tensor, voronoi_info) =
            del_candle::voronoi2::voronoi(&vtxl2xy, &site2xy_var, |i_site| {
                site2room[i_site] != usize::MAX
            });
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

    fn from_existing(inner: VoronoiInfo, vtxv2xy: Vec<f32>) -> Self {
        Self { inner, vtxv2xy }
    }
}

#[pyclass(name = "VoronoiStage", unsendable)]
pub struct PyVoronoiStage {
    stage: Option<VoronoiStage>,
}

impl PyVoronoiStage {
    fn new(stage: VoronoiStage) -> Self {
        Self { stage: Some(stage) }
    }

    fn inner(&self) -> PyResult<&VoronoiStage> {
        self.stage
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("VoronoiStage has already been consumed"))
    }

    fn into_inner(mut self) -> VoronoiStage {
        self.stage
            .take()
            .expect("VoronoiStage already consumed")
    }

    fn take_stage(&mut self) -> PyResult<VoronoiStage> {
        self.stage
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("VoronoiStage has already been consumed"))
    }
}

#[pymethods]
impl PyVoronoiStage {
    pub fn site_positions(&self) -> PyResult<Vec<f32>> {
        tensor_to_vec(self.inner()?.site2xy_adjusted.clone())
    }

    pub fn sanitized_coordinates(&self) -> PyResult<Vec<f32>> {
        Ok(self.inner()?.site_coords_sanitized.clone())
    }

    pub fn vertex_coordinates(&self) -> PyResult<Vec<f32>> {
        tensor_to_vec(self.inner()?.vtxv2xy.clone())
    }

    pub fn voronoi_info(&self) -> PyResult<PyVoronoiInfo> {
        let info = clone_voronoi_info(&self.inner()?.voronoi_info);
        let vtxv2xy = tensor_to_vec(self.inner()?.vtxv2xy.clone())?;
        Ok(PyVoronoiInfo::from_existing(info, vtxv2xy))
    }
}

#[pyclass(name = "OptimizeResult", unsendable)]
pub struct PyOptimizeResult {
    site2xy_adjusted: Vec<f32>,
    site_coords_sanitized: Vec<f32>,
    edge2vtxv_wall: Vec<usize>,
    voronoi_info: VoronoiInfo,
    vtxv2xy: Vec<f32>,
}

impl PyOptimizeResult {
    fn from_raw(
        site2xy_adjusted: Tensor,
        voronoi_info: VoronoiInfo,
        vtxv2xy: Tensor,
        edge2vtxv_wall: Vec<usize>,
        site_coords_sanitized: Vec<f32>,
    ) -> PyResult<Self> {
        Ok(Self {
            site2xy_adjusted: tensor_to_vec(site2xy_adjusted)?,
            site_coords_sanitized,
            edge2vtxv_wall,
            vtxv2xy: tensor_to_vec(vtxv2xy)?,
            voronoi_info,
        })
    }
}

#[pymethods]
impl PyOptimizeResult {
    pub fn site2xy_adjusted(&self) -> Vec<f32> {
        self.site2xy_adjusted.clone()
    }

    #[pyo3(name = "site_coordinates")]
    pub fn site_coordinates_py(&self) -> Vec<f32> {
        self.site_coords_sanitized.clone()
    }

    pub fn edge2vtvx_wall(&self) -> Vec<usize> {
        self.edge2vtxv_wall.clone()
    }

    pub fn vertex_coordinates(&self) -> Vec<f32> {
        self.vtxv2xy.clone()
    }

    pub fn voronoi_info(&self) -> PyVoronoiInfo {
        PyVoronoiInfo::from_existing(clone_voronoi_info(&self.voronoi_info), self.vtxv2xy.clone())
    }
}

#[pyclass(name = "OptimizeContext", unsendable)]
pub struct PyOptimizeContext {
    vtxl2xy: Vec<f32>,
    site2xy: Var,
    site2xy_ini: Tensor,
    site2xy2flag: Var,
    site2room: Vec<usize>,
    room2area_trg: Tensor,
    room_connections: Vec<(usize, usize)>,
    optimizer: candle_nn::AdamW,
    params: ProjectParams,
}

#[pymethods]
impl PyOptimizeContext {
    #[new]
    #[pyo3(signature = (vtxl2xy, site2xy, site2room, site2xy2flag, room2area_trg, room_connections, params_index=None))]
    fn new(
        vtxl2xy: Vec<f32>,
        site2xy: Vec<f32>,
        site2room: Vec<usize>,
        site2xy2flag: Vec<f32>,
        room2area_trg: Vec<f32>,
        room_connections: Vec<(usize, usize)>,
        params_index: Option<usize>,
    ) -> PyResult<Self> {
        if site2xy.len() % 2 != 0 {
            return Err(PyValueError::new_err("site2xy must contain x/y pairs"));
        }
        if site2xy2flag.len() != site2xy.len() {
            return Err(PyValueError::new_err(
                "site2xy2flag length must match site2xy length",
            ));
        }
        let num_site = site2xy.len() / 2;
        if site2room.len() != num_site {
            return Err(PyValueError::new_err(
                "site2room length must match number of sites",
            ));
        }
        let num_room = room2area_trg.len();
        let params_all = project_params_all();
        let index = params_index.unwrap_or(0);
        let params = params_all
            .get(index)
            .ok_or_else(|| PyValueError::new_err("params_index out of range"))?
            .clone();

        let site2xy_ini = tensor_from_vec(site2xy.clone(), num_site, 2)?;
        let site2xy_var = var_from_vec(site2xy, num_site, 2)?;
        let site2xy2flag_var = var_from_vec(site2xy2flag, num_site, 2)?;
        let room2area_trg_tensor = Tensor::from_vec(
            room2area_trg,
            Shape::from((num_room, 1)),
            &Device::Cpu,
        )
        .map_err(candle_err)?;

        let optimizer = candle_nn::AdamW::new(
            vec![site2xy_var.clone()],
            candle_nn::ParamsAdamW {
                lr: params.learning_rates.first as f64,
                ..Default::default()
            },
        )
        .map_err(candle_err)?;

        Ok(Self {
            vtxl2xy,
            site2xy: site2xy_var,
            site2xy_ini,
            site2xy2flag: site2xy2flag_var,
            site2room,
            room2area_trg: room2area_trg_tensor,
            room_connections,
            optimizer,
            params,
        })
    }

    pub fn vtxl2xy(&self) -> Vec<f32> {
        self.vtxl2xy.clone()
    }

    pub fn set_vtxl2xy(&mut self, values: Vec<f32>) -> PyResult<()> {
        if values.len() % 2 != 0 {
            return Err(PyValueError::new_err("vtxl2xy must contain x/y coordinate pairs"));
        }
        self.vtxl2xy = values;
        Ok(())
    }

    pub fn site2xy(&self) -> PyResult<Vec<f32>> {
        tensor_to_vec(self.site2xy.as_tensor().clone())
    }

    pub fn set_site2xy(&mut self, values: Vec<f32>) -> PyResult<()> {
        let (rows, cols) = self.site2xy.dims2().map_err(candle_err)?;
        ensure_len(values.len(), rows, cols)?;
        let tensor = tensor_from_vec(values, rows, cols)?;
        self.site2xy.set(&tensor).map_err(candle_err)
    }

    pub fn site2xy_ini(&self) -> PyResult<Vec<f32>> {
        tensor_to_vec(self.site2xy_ini.clone())
    }

    pub fn set_site2xy_ini(&mut self, values: Vec<f32>) -> PyResult<()> {
        let (rows, cols) = self.site2xy_ini.dims2().map_err(candle_err)?;
        ensure_len(values.len(), rows, cols)?;
        self.site2xy_ini = tensor_from_vec(values, rows, cols)?;
        Ok(())
    }

    pub fn site2xy2flag(&self) -> PyResult<Vec<f32>> {
        tensor_to_vec(self.site2xy2flag.as_tensor().clone())
    }

    pub fn set_site2xy2flag(&mut self, values: Vec<f32>) -> PyResult<()> {
        let (rows, cols) = self.site2xy2flag.dims2().map_err(candle_err)?;
        ensure_len(values.len(), rows, cols)?;
        let tensor = tensor_from_vec(values, rows, cols)?;
        self.site2xy2flag.set(&tensor).map_err(candle_err)
    }

    pub fn site2room(&self) -> Vec<usize> {
        self.site2room.clone()
    }

    pub fn set_site2room(&mut self, values: Vec<usize>) -> PyResult<()> {
        let (rows, _) = self.site2xy.dims2().map_err(candle_err)?;
        if values.len() != rows {
            return Err(PyValueError::new_err(
                "site2room length must match the number of sites",
            ));
        }
        self.site2room = values;
        Ok(())
    }

    pub fn room2area_trg(&self) -> PyResult<Vec<f32>> {
        tensor_to_vec(self.room2area_trg.clone())
    }

    pub fn set_room2area_trg(&mut self, values: Vec<f32>) -> PyResult<()> {
        let (rows, cols) = self.room2area_trg.dims2().map_err(candle_err)?;
        ensure_len(values.len(), rows, cols)?;
        self.room2area_trg = tensor_from_vec(values, rows, cols)?;
        Ok(())
    }

    pub fn room_connections(&self) -> Vec<(usize, usize)> {
        self.room_connections.clone()
    }

    pub fn set_room_connections(&mut self, connections: Vec<(usize, usize)>) {
        self.room_connections = connections;
    }

    pub fn iterate_voronoi_stage(&self) -> PyResult<PyVoronoiStage> {
        let stage = crate::iterate_voronoi_stage(&self.vtxl2xy, &self.site2xy, &self.site2room)
            .map_err(anyhow_err)?;
        Ok(PyVoronoiStage::new(stage))
    }

    pub fn optimize_iteration(
        &mut self,
        mut stage: PyRefMut<'_, PyVoronoiStage>,
    ) -> PyResult<PyOptimizeResult> {
        let stage = stage.take_stage()?;
        let (site2xy_adjusted, voronoi_info, vtxv2xy, edge2vtxv_wall, site_coords_sanitized) =
            crate::optimize_iteration(
                &self.vtxl2xy,
                &self.site2xy,
                &self.site2xy_ini,
                &self.site2xy2flag,
                &self.site2room,
                &self.room2area_trg,
                &self.room_connections,
                &mut self.optimizer,
                &self.params,
                stage,
            )
            .map_err(anyhow_err)?;
        PyOptimizeResult::from_raw(
            site2xy_adjusted,
            voronoi_info,
            vtxv2xy,
            edge2vtxv_wall,
            site_coords_sanitized,
        )
    }

    pub fn learning_rates(&self) -> (f32, f32, f32) {
        (
            self.params.learning_rates.first,
            self.params.learning_rates.second,
            self.params.learning_rates.third,
        )
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer
            .set_params(candle_nn::ParamsAdamW {
                lr: lr as f64,
                ..Default::default()
            });
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("FLOORPLAN_VER"))?;
    m.add_class::<PyCanvasGif>()?;
    m.add_class::<PyVoronoiInfo>()?;
    m.add_class::<PyVoronoiStage>()?;
    m.add_class::<PyOptimizeResult>()?;
    m.add_class::<PyOptimizeContext>()?;
    m.add_function(wrap_pyfunction!(py_voronoi_info_from_raw, m)?)?;
    m.add_function(wrap_pyfunction!(py_create_voronoi_stage, m)?)?;
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

#[pyfunction(name = "create_voronoi_stage", signature = (site2xy_adjusted, voronoi_info, vtxv2xy, site_coords_sanitized))]
fn py_create_voronoi_stage(
    site2xy_adjusted: Vec<f32>,
    voronoi_info: &PyVoronoiInfo,
    vtxv2xy: Vec<f32>,
    site_coords_sanitized: Vec<f32>,
) -> PyResult<PyVoronoiStage> {
    if site2xy_adjusted.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "site2xy_adjusted must contain x/y coordinate pairs",
        ));
    }
    if vtxv2xy.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "vtxv2xy must contain x/y coordinate pairs",
        ));
    }
    if site_coords_sanitized.len() != site2xy_adjusted.len() {
        return Err(PyValueError::new_err(
            "site_coords_sanitized must match site2xy_adjusted length",
        ));
    }
    let num_sites = site2xy_adjusted.len() / 2;
    let num_vertices = vtxv2xy.len() / 2;
    let site_tensor = tensor_from_vec(site2xy_adjusted, num_sites, 2)?;
    let vtx_tensor = tensor_from_vec(vtxv2xy, num_vertices, 2)?;
    let stage = VoronoiStage {
        site2xy_adjusted: site_tensor,
        voronoi_info: clone_voronoi_info(voronoi_info.as_ref()),
        vtxv2xy: vtx_tensor,
        site_coords_sanitized,
    };
    Ok(PyVoronoiStage::new(stage))
}

#[pyfunction(name = "voronoi_info_from_raw", signature = (site2idx, idx2vtxv, idx2site, vtxv2xy, vtxv2info=None))]
fn py_voronoi_info_from_raw(
    site2idx: Vec<usize>,
    idx2vtxv: Vec<usize>,
    idx2site: Vec<isize>,
    vtxv2xy: Vec<f32>,
    vtxv2info: Option<Vec<usize>>,
) -> PyResult<PyVoronoiInfo> {
    if site2idx.len() < 2 {
        return Err(PyValueError::new_err(
            "site2idx must contain at least two entries (prefix array)",
        ));
    }
    if idx2vtxv.len() != idx2site.len() {
        return Err(PyValueError::new_err(
            "idx2vtxv and idx2site must have the same length",
        ));
    }
    if *site2idx.last().unwrap() != idx2vtxv.len() {
        return Err(PyValueError::new_err(
            "site2idx must terminate with idx2vtxv length",
        ));
    }
    if vtxv2xy.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "vtxv2xy must contain x/y coordinate pairs",
        ));
    }
    let num_vertices = vtxv2xy.len() / 2;
    if let Some(max_index) = idx2vtxv.iter().copied().max() {
        if max_index >= num_vertices {
            return Err(PyValueError::new_err(
                "idx2vtxv contains an index outside the vertex array",
            ));
        }
    }
    let mut idx2site_converted = Vec::with_capacity(idx2site.len());
    for value in idx2site {
        if value < 0 {
            idx2site_converted.push(usize::MAX);
        } else {
            idx2site_converted.push(value as usize);
        }
    }

    let info = match vtxv2info {
        Some(flat) => reshape_vtxv2info(flat, num_vertices)?,
        None => vec![[usize::MAX; 4]; num_vertices],
    };
    if info.len() != num_vertices {
        return Err(PyValueError::new_err(
            "vtxv2info must describe every Voronoi vertex",
        ));
    }
    let voronoi_info = VoronoiInfo {
        site2idx,
        idx2vtxv,
        idx2site: idx2site_converted,
        vtxv2info: info,
    };
    Ok(PyVoronoiInfo::from_existing(voronoi_info, vtxv2xy))
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
        return Err(PyValueError::new_err(
            "polygon must contain at least 3 points",
        ));
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
    let loss =
        crate::loss_lloyd_internal(voronoi_info.as_ref(), &site2room, &site2xy_var, &vtx_tensor)
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
fn py_remove_site_too_close(
    site2room: Vec<usize>,
    site2xy: Vec<f32>,
    num_site: usize,
) -> PyResult<Vec<usize>> {
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
    iter: usize,
    params_index: Option<usize>,
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
        iter,
        params_index.unwrap_or(0),
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
        return Err(PyValueError::new_err(
            "transform matrix must contain 9 values",
        ));
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

fn reshape_vtxv2info(flat: Vec<usize>, num_vertices: usize) -> PyResult<Vec<[usize; 4]>> {
    if flat.len() % 4 != 0 {
        return Err(PyValueError::new_err(
            "vtxv2info must be provided as groups of four indices",
        ));
    }
    let mut info = Vec::with_capacity(flat.len() / 4);
    for chunk in flat.chunks(4) {
        info.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    if info.len() != num_vertices {
        return Err(PyValueError::new_err(
            "vtxv2info length must match the number of Voronoi vertices",
        ));
    }
    Ok(info)
}

fn clone_voronoi_info(info: &VoronoiInfo) -> VoronoiInfo {
    VoronoiInfo {
        site2idx: info.site2idx.clone(),
        idx2vtxv: info.idx2vtxv.clone(),
        idx2site: info.idx2site.clone(),
        vtxv2info: info.vtxv2info.clone(),
    }
}

fn candle_err(err: candle_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("candle error: {err}"))
}

fn anyhow_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}
