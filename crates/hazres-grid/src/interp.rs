//! Point methods: the value at each target cell centre, from the source grid.
//!
//! Target centres are given in the *source* CRS (the caller transforms them),
//! so these functions never deal with map projections.
//!
//! Missing source values are NaN (continuous) or negative (classes). A method
//! that combines several source cells ignores missing ones and renormalises
//! the weights of the rest; if none is valid the result is missing.

use crate::geom::GridGeom;
use rayon::prelude::*;

/// Value of the source cell containing each point (piecewise constant).
pub fn nearest(g: &GridGeom, values: &[f32], xs: &[f64], ys: &[f64]) -> Vec<f32> {
    xs.par_iter()
        .zip(ys.par_iter())
        .map(|(&x, &y)| match g.cell_of(x, y) {
            Some((r, c)) => values[g.idx(r, c)],
            None => f32::NAN,
        })
        .collect()
}

/// Class of the source cell containing each point; -1 outside or missing.
pub fn nearest_class(g: &GridGeom, values: &[i32], xs: &[f64], ys: &[f64]) -> Vec<i32> {
    xs.par_iter()
        .zip(ys.par_iter())
        .map(|(&x, &y)| match g.cell_of(x, y) {
            Some((r, c)) => values[g.idx(r, c)].max(-1),
            None => -1,
        })
        .collect()
}

#[inline]
fn get(g: &GridGeom, values: &[f32], r: i64, c: i64) -> Option<f32> {
    if r < 0 || c < 0 || r >= g.rows as i64 || c >= g.cols as i64 {
        return None;
    }
    let v = values[g.idx(r as usize, c as usize)];
    (!v.is_nan()).then_some(v)
}

/// Continuous index measured from cell centres, so integer values sit on centres.
#[inline]
fn centre_index(g: &GridGeom, x: f64, y: f64) -> (f64, f64) {
    let (u, v) = g.frac_index(x, y);
    (u - 0.5, v - 0.5)
}

fn bilinear_one(g: &GridGeom, values: &[f32], x: f64, y: f64) -> f32 {
    let (u, v) = centre_index(g, x, y);
    if !(u.is_finite() && v.is_finite()) {
        return f32::NAN;
    }
    let (c0, r0) = (u.floor() as i64, v.floor() as i64);
    let (tx, ty) = (u - c0 as f64, v - r0 as f64);
    let mut sum = 0.0f64;
    let mut wsum = 0.0f64;
    for (dr, wy) in [(0i64, 1.0 - ty), (1, ty)] {
        for (dc, wx) in [(0i64, 1.0 - tx), (1, tx)] {
            let w = wx * wy;
            if w <= 0.0 {
                continue;
            }
            if let Some(val) = get(g, values, r0 + dr, c0 + dc) {
                sum += w * val as f64;
                wsum += w;
            }
        }
    }
    if wsum > 0.0 {
        (sum / wsum) as f32
    } else {
        f32::NAN
    }
}

/// Linear interpolation between the four nearest source cell centres.
pub fn bilinear(g: &GridGeom, values: &[f32], xs: &[f64], ys: &[f64]) -> Vec<f32> {
    xs.par_iter()
        .zip(ys.par_iter())
        .map(|(&x, &y)| bilinear_one(g, values, x, y))
        .collect()
}

/// Keys cubic convolution kernel with a = -0.5 (reproduces quadratics exactly).
#[inline]
fn keys(t: f64) -> f64 {
    const A: f64 = -0.5;
    let t = t.abs();
    if t <= 1.0 {
        (A + 2.0) * t * t * t - (A + 3.0) * t * t + 1.0
    } else if t < 2.0 {
        A * t * t * t - 5.0 * A * t * t + 8.0 * A * t - 4.0 * A
    } else {
        0.0
    }
}

fn cubic_one(g: &GridGeom, values: &[f32], x: f64, y: f64) -> f32 {
    let (u, v) = centre_index(g, x, y);
    if !(u.is_finite() && v.is_finite()) {
        return f32::NAN;
    }
    let (c1, r1) = (u.floor() as i64, v.floor() as i64);
    let (tx, ty) = (u - c1 as f64, v - r1 as f64);
    let mut sum = 0.0f64;
    for (i, dr) in (-1i64..=2).enumerate() {
        let wy = keys(ty - (i as f64 - 1.0));
        for (j, dc) in (-1i64..=2).enumerate() {
            let wx = keys(tx - (j as f64 - 1.0));
            match get(g, values, r1 + dr, c1 + dc) {
                Some(val) => sum += wx * wy * val as f64,
                // near edges and gaps the 4x4 stencil is incomplete: fall back
                None => return bilinear_one(g, values, x, y),
            }
        }
    }
    sum as f32
}

/// Bicubic (Keys) interpolation over the 4x4 nearest centres. Falls back to
/// bilinear where the stencil is incomplete. Can overshoot the source range.
pub fn cubic(g: &GridGeom, values: &[f32], xs: &[f64], ys: &[f64]) -> Vec<f32> {
    xs.par_iter()
        .zip(ys.par_iter())
        .map(|(&x, &y)| cubic_one(g, values, x, y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 5x6 grid of unit cells, value = f(centre x, centre y).
    fn grid(f: impl Fn(f64, f64) -> f64) -> (GridGeom, Vec<f32>) {
        let g = GridGeom::new(0.0, 5.0, 1.0, 1.0, 5, 6).unwrap();
        let mut v = Vec::new();
        for r in 0..5 {
            for c in 0..6 {
                v.push(f(c as f64 + 0.5, 5.0 - (r as f64 + 0.5)) as f32);
            }
        }
        (g, v)
    }

    #[test]
    fn nearest_copies_the_containing_cell() {
        let (g, v) = grid(|x, y| 10.0 * x + y);
        let out = nearest(&g, &v, &[0.1, 5.99, 6.0], &[4.9, 0.01, 1.0]);
        assert_eq!(out[0], v[0]);
        assert_eq!(out[1], v[g.idx(4, 5)]);
        assert!(out[2].is_nan());
    }

    #[test]
    fn bilinear_and_cubic_reproduce_a_plane() {
        let (g, v) = grid(|x, y| 3.0 * x - 2.0 * y + 1.0);
        let xs = [1.2, 2.5, 3.77];
        let ys = [1.9, 3.0, 2.31];
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let want = (3.0 * x - 2.0 * y + 1.0) as f32;
            assert!((bilinear(&g, &v, &xs, &ys)[i] - want).abs() < 1e-4);
            assert!((cubic(&g, &v, &xs, &ys)[i] - want).abs() < 1e-4);
        }
    }

    #[test]
    fn cubic_reproduces_a_quadratic_bilinear_does_not() {
        let (g, v) = grid(|x, _| x * x);
        let (x, y) = ([2.8], [2.5]);
        let cub = cubic(&g, &v, &x, &y)[0];
        let bil = bilinear(&g, &v, &x, &y)[0];
        assert!((cub - 7.84).abs() < 1e-3, "cubic {cub}");
        assert!((bil - 7.84).abs() > 0.01, "bilinear {bil}");
    }

    #[test]
    fn missing_cells_are_skipped_and_weights_renormalised() {
        let (g, mut v) = grid(|_, _| 1.0);
        v[g.idx(2, 2)] = f32::NAN;
        let out = bilinear(&g, &v, &[2.9], &[2.1]);
        assert!((out[0] - 1.0).abs() < 1e-6);
        // cubic falls back to bilinear next to the gap
        assert!((cubic(&g, &v, &[2.9], &[2.1])[0] - 1.0).abs() < 1e-6);
        let all_nan = vec![f32::NAN; 30];
        assert!(bilinear(&g, &all_nan, &[2.9], &[2.1])[0].is_nan());
    }
}
