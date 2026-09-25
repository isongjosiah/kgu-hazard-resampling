//! Going the other way: summarising fine cells into coarse ones.
//!
//! Used by the `coarsened_target` treatment (bring everything *up* to a coarse
//! grid) and by `downscaled` (fit at the coarse scale, then keep each coarse
//! cell's mean exact).

use crate::geom::GridGeom;
use rayon::prelude::*;

/// Index of the source cell containing each point (row-major), or -1.
pub fn cell_index(g: &GridGeom, xs: &[f64], ys: &[f64]) -> Vec<i64> {
    xs.par_iter()
        .zip(ys.par_iter())
        .map(|(&x, &y)| g.cell_of(x, y).map_or(-1, |(r, c)| g.idx(r, c) as i64))
        .collect()
}

/// Mean and count of valid (non-NaN) values per zone. Zones < 0 are ignored.
pub fn zonal_mean(
    values: &[f32],
    zones: &[i64],
    n_zones: usize,
) -> Result<(Vec<f64>, Vec<u64>), String> {
    if values.len() != zones.len() {
        return Err("values and zones differ in length".into());
    }
    let mut sum = vec![0.0f64; n_zones];
    let mut count = vec![0u64; n_zones];
    for (&v, &z) in values.iter().zip(zones) {
        if z < 0 || v.is_nan() {
            continue;
        }
        let z = z as usize;
        if z >= n_zones {
            return Err(format!("zone {z} out of range (n_zones = {n_zones})"));
        }
        sum[z] += v as f64;
        count[z] += 1;
    }
    let mean = sum
        .iter()
        .zip(&count)
        .map(|(&s, &n)| if n > 0 { s / n as f64 } else { f64::NAN })
        .collect();
    Ok((mean, count))
}

/// Shift every fine value so each zone's mean equals `target[zone]` exactly.
/// Zones with a missing target or no valid fine values are set missing.
pub fn match_zone_means(values: &[f32], zones: &[i64], target: &[f64]) -> Result<Vec<f32>, String> {
    let (mean, _) = zonal_mean(values, zones, target.len())?;
    Ok(values
        .par_iter()
        .zip(zones.par_iter())
        .map(|(&v, &z)| {
            if z < 0 || v.is_nan() {
                return f32::NAN;
            }
            let (t, m) = (target[z as usize], mean[z as usize]);
            if t.is_nan() || m.is_nan() {
                f32::NAN
            } else {
                (v as f64 + (t - m)) as f32
            }
        })
        .collect())
}

/// Mean of each `f` x `f` block of a fine grid. A block needs at least
/// `min_valid` of its cells valid, otherwise it is missing.
pub fn block_mean(
    values: &[f32],
    rows: usize,
    cols: usize,
    f: usize,
    min_valid: f64,
) -> Result<Vec<f32>, String> {
    check_blocks(values.len(), rows, cols, f)?;
    let (br, bc) = (rows / f, cols / f);
    Ok((0..br * bc)
        .into_par_iter()
        .map(|k| {
            let (r0, c0) = ((k / bc) * f, (k % bc) * f);
            let (mut s, mut n) = (0.0f64, 0usize);
            for r in r0..r0 + f {
                for c in c0..c0 + f {
                    let v = values[r * cols + c];
                    if !v.is_nan() {
                        s += v as f64;
                        n += 1;
                    }
                }
            }
            if n > 0 && n as f64 >= min_valid * (f * f) as f64 {
                (s / n as f64) as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}

/// Most common class in each `f` x `f` block (ties: lowest class; -1 if none valid).
pub fn block_majority(
    values: &[i32],
    rows: usize,
    cols: usize,
    f: usize,
) -> Result<Vec<i32>, String> {
    check_blocks(values.len(), rows, cols, f)?;
    let (br, bc) = (rows / f, cols / f);
    Ok((0..br * bc)
        .into_par_iter()
        .map(|k| {
            let (r0, c0) = ((k / bc) * f, (k % bc) * f);
            let mut classes: Vec<i32> = Vec::with_capacity(f * f);
            for r in r0..r0 + f {
                for c in c0..c0 + f {
                    let v = values[r * cols + c];
                    if v >= 0 {
                        classes.push(v);
                    }
                }
            }
            classes.sort_unstable();
            let (mut best, mut best_n, mut i) = (-1, 0usize, 0usize);
            while i < classes.len() {
                let j = classes[i..]
                    .iter()
                    .take_while(|&&x| x == classes[i])
                    .count();
                if j > best_n {
                    best = classes[i];
                    best_n = j;
                }
                i += j;
            }
            best
        })
        .collect())
}

// `%` rather than `is_multiple_of`, which needs Rust 1.87; the workspace supports 1.83.
#[allow(clippy::manual_is_multiple_of)]
fn check_blocks(n: usize, rows: usize, cols: usize, f: usize) -> Result<(), String> {
    if f == 0 || rows % f != 0 || cols % f != 0 {
        return Err(format!(
            "{rows}x{cols} grid is not divisible into {f}x{f} blocks"
        ));
    }
    if n != rows * cols {
        return Err(format!("expected {} values, got {n}", rows * cols));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_means_are_matched_exactly() {
        let values = [1.0f32, 2.0, 3.0, 10.0, f32::NAN, 30.0];
        let zones = [0i64, 0, 0, 1, 1, 1];
        let out = match_zone_means(&values, &zones, &[5.0, 0.0]).unwrap();
        let (mean, count) = zonal_mean(&out, &zones, 2).unwrap();
        assert!((mean[0] - 5.0).abs() < 1e-9 && mean[1].abs() < 1e-9);
        assert_eq!(count, vec![3, 2]);
        assert!(out[4].is_nan());
        // relative differences within a zone are kept
        assert!((out[2] - out[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn block_mean_and_majority() {
        let v = [1.0f32, 2.0, 5.0, 5.0, 3.0, f32::NAN, 5.0, 5.0];
        assert_eq!(block_mean(&v, 2, 4, 2, 0.5).unwrap(), vec![2.0, 5.0]);
        assert!(block_mean(&v, 2, 4, 2, 1.0).unwrap()[0].is_nan()); // one of four missing
        let c = [2i32, 2, 1, 3, 1, 1, 3, 1];
        assert_eq!(block_majority(&c, 2, 4, 2).unwrap(), vec![1, 1]); // tie 2-2 goes low
        assert!(block_mean(&v, 2, 4, 3, 0.5).is_err());
    }

    #[test]
    fn cell_index_marks_outside() {
        let g = GridGeom::new(0.0, 10.0, 5.0, 5.0, 2, 2).unwrap();
        assert_eq!(
            cell_index(&g, &[1.0, 9.0, 11.0], &[9.0, 1.0, 1.0]),
            vec![0, 3, -1]
        );
    }
}
