//! Grid engine for `hazres`.
//!
//! Pure Rust, with no Python dependency, so it can be tested with `cargo test`
//! on its own. The Python bindings live in the `hazres-py` crate.
//!
//! Planned modules:
//! - `resample`: the six ways of bringing a coarse layer onto the analysis grid.
//! - `realise`: random fine-scale versions that average back to the coarse values.

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
