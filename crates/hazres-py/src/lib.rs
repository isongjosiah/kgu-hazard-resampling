//! Python module `hazres._engine`: thin bindings over `hazres-grid`.
//!
//! Keep this crate thin. All logic and its tests belong in `hazres-grid`.

use pyo3::prelude::*;

/// Version of the Rust grid engine.
#[pyfunction]
fn engine_version() -> &'static str {
    hazres_grid::version()
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    Ok(())
}
