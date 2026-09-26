//! Python module `hazres._engine`: thin bindings over `hazres-grid`.
//!
//! Keep this crate thin. All logic and its tests belong in `hazres-grid`.
//! Every call releases the GIL while the engine runs.

use hazres_grid::{aggregate, area, interp, GridGeom};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Geom = (f64, f64, f64, f64);
type ZonalOut<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u64>>);

fn geom(g: Geom, rows: usize, cols: usize) -> PyResult<GridGeom> {
    GridGeom::new(g.0, g.1, g.2, g.3, rows, cols).map_err(PyValueError::new_err)
}

fn same_len(a: usize, b: usize) -> PyResult<()> {
    if a == b {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "xs and ys differ in length ({a} vs {b})"
        )))
    }
}

fn to_2d<T: numpy::Element>(
    py: Python<'_>,
    v: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    v.into_pyarray(py).reshape([rows, cols])
}

/// Version of the Rust grid engine.
#[pyfunction]
fn engine_version() -> &'static str {
    hazres_grid::version()
}

macro_rules! point_method {
    ($name:ident, $func:path, $t:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            values: PyReadonlyArray2<'py, $t>,
            geometry: Geom,
            xs: PyReadonlyArray1<'py, f64>,
            ys: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<$t>>> {
            let shape = values.shape();
            let g = geom(geometry, shape[0], shape[1])?;
            let (v, x, y) = (values.as_slice()?, xs.as_slice()?, ys.as_slice()?);
            same_len(x.len(), y.len())?;
            let out = py.detach(|| $func(&g, v, x, y));
            Ok(out.into_pyarray(py))
        }
    };
}

point_method!(nearest, interp::nearest, f32);
point_method!(nearest_class, interp::nearest_class, i32);
point_method!(bilinear, interp::bilinear, f32);
point_method!(cubic, interp::cubic, f32);

macro_rules! area_method {
    ($name:ident, $func:path, $t:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            values: PyReadonlyArray2<'py, $t>,
            geometry: Geom,
            corners_x: PyReadonlyArray2<'py, f64>,
            corners_y: PyReadonlyArray2<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray2<$t>>> {
            let shape = values.shape();
            let g = geom(geometry, shape[0], shape[1])?;
            let cs = corners_x.shape();
            if cs != corners_y.shape() || cs[0] < 2 || cs[1] < 2 {
                return Err(PyValueError::new_err(
                    "corner arrays must match and be at least 2x2",
                ));
            }
            let (rows, cols) = (cs[0] - 1, cs[1] - 1);
            let (v, cx, cy) = (
                values.as_slice()?,
                corners_x.as_slice()?,
                corners_y.as_slice()?,
            );
            let out = py
                .detach(|| $func(&g, v, cx, cy, rows, cols))
                .map_err(PyValueError::new_err)?;
            to_2d(py, out, rows, cols)
        }
    };
}

area_method!(area_weighted, area::area_weighted, f32);
area_method!(area_majority, area::area_majority, i32);

/// Row-major index of the cell containing each point in a (rows, cols) grid, or -1.
#[pyfunction]
fn cell_index<'py>(
    py: Python<'py>,
    geometry: Geom,
    shape: (usize, usize),
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let g = geom(geometry, shape.0, shape.1)?;
    let (x, y) = (xs.as_slice()?, ys.as_slice()?);
    same_len(x.len(), y.len())?;
    Ok(py
        .detach(|| aggregate::cell_index(&g, x, y))
        .into_pyarray(py))
}

/// (mean, count) of valid values per zone; zones < 0 ignored.
#[pyfunction]
fn zonal_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
    zones: PyReadonlyArray1<'py, i64>,
    n_zones: usize,
) -> PyResult<ZonalOut<'py>> {
    let (v, z) = (values.as_slice()?, zones.as_slice()?);
    let (m, n) = py
        .detach(|| aggregate::zonal_mean(v, z, n_zones))
        .map_err(PyValueError::new_err)?;
    Ok((m.into_pyarray(py), n.into_pyarray(py)))
}

/// Shift values so each zone's mean equals target[zone].
#[pyfunction]
fn match_zone_means<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
    zones: PyReadonlyArray1<'py, i64>,
    target: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (v, z, t) = (values.as_slice()?, zones.as_slice()?, target.as_slice()?);
    let out = py
        .detach(|| aggregate::match_zone_means(v, z, t))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

/// Mean of each factor x factor block; blocks with fewer than min_valid valid cells are NaN.
#[pyfunction]
fn block_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f32>,
    factor: usize,
    min_valid: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let s = values.shape();
    let (rows, cols) = (s[0], s[1]);
    let v = values.as_slice()?;
    let out = py
        .detach(|| aggregate::block_mean(v, rows, cols, factor, min_valid))
        .map_err(PyValueError::new_err)?;
    to_2d(py, out, rows / factor, cols / factor)
}

/// Most common class in each factor x factor block.
#[pyfunction]
fn block_majority<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, i32>,
    factor: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let s = values.shape();
    let (rows, cols) = (s[0], s[1]);
    let v = values.as_slice()?;
    let out = py
        .detach(|| aggregate::block_majority(v, rows, cols, factor))
        .map_err(PyValueError::new_err)?;
    to_2d(py, out, rows / factor, cols / factor)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(nearest, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_class, m)?)?;
    m.add_function(wrap_pyfunction!(bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(cubic, m)?)?;
    m.add_function(wrap_pyfunction!(area_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(area_majority, m)?)?;
    m.add_function(wrap_pyfunction!(cell_index, m)?)?;
    m.add_function(wrap_pyfunction!(zonal_mean, m)?)?;
    m.add_function(wrap_pyfunction!(match_zone_means, m)?)?;
    m.add_function(wrap_pyfunction!(block_mean, m)?)?;
    m.add_function(wrap_pyfunction!(block_majority, m)?)?;
    Ok(())
}
