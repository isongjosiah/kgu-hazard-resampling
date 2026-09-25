//! Grid engine for `hazres`: bringing coarse layers onto a fine grid.
//!
//! Pure Rust, no Python, tested with `cargo test`. The Python bindings live
//! in the `hazres-py` crate.
//!
//! Every function takes target positions already expressed in the source
//! grid's own CRS, so map projections are handled once, by the caller.
//!
//! - [`interp`]: point methods at target cell centres (nearest, bilinear, cubic).
//! - [`area`]: footprint methods over target cells (area-weighted mean, majority).
//! - [`aggregate`]: fine to coarse (block mean and majority, zonal means, and
//!   shifting fine values so each coarse cell's mean is kept exactly).

pub mod aggregate;
pub mod area;
pub mod geom;
pub mod interp;

pub use geom::GridGeom;

/// Version of the grid engine, reported in every run's metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::version().is_empty());
    }
}
