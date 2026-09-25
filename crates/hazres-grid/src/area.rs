//! Footprint methods: each target cell takes the source cells it overlaps,
//! weighted by the area of overlap.
//!
//! Target cells are given by their corner lattice in the source CRS:
//! `(rows + 1) x (cols + 1)` corner points, row-major, so neighbouring cells
//! share corners. Each cell's footprint is taken as the bounding box of its
//! four corners, which is exact when both grids are north-up in the same CRS
//! and a close approximation otherwise (target cells are much smaller than
//! source cells here).

use crate::geom::GridGeom;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// The source cells a target footprint overlaps, with the overlapping area in
/// cell units. Cells outside the source grid are skipped.
fn overlaps(g: &GridGeom, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Vec<(usize, f64)> {
    let (c0, r0) = g.frac_index(xmin, ymax);
    let (c1, r1) = g.frac_index(xmax, ymin);
    if !(c0.is_finite() && c1.is_finite() && r0.is_finite() && r1.is_finite()) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let (cs, ce) = (
        c0.floor().max(0.0) as i64,
        (c1.ceil() as i64).min(g.cols as i64),
    );
    let (rs, re) = (
        r0.floor().max(0.0) as i64,
        (r1.ceil() as i64).min(g.rows as i64),
    );
    for r in rs..re {
        let oy = (r1.min(r as f64 + 1.0) - r0.max(r as f64)).max(0.0);
        if oy <= 0.0 {
            continue;
        }
        for c in cs..ce {
            let ox = (c1.min(c as f64 + 1.0) - c0.max(c as f64)).max(0.0);
            if ox > 0.0 {
                out.push((g.idx(r as usize, c as usize), ox * oy));
            }
        }
    }
    out
}

fn footprint(cx: &[f64], cy: &[f64], cols: usize, r: usize, c: usize) -> (f64, f64, f64, f64) {
    let w = cols + 1;
    let ids = [
        r * w + c,
        r * w + c + 1,
        (r + 1) * w + c,
        (r + 1) * w + c + 1,
    ];
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for i in ids {
        xmin = xmin.min(cx[i]);
        xmax = xmax.max(cx[i]);
        ymin = ymin.min(cy[i]);
        ymax = ymax.max(cy[i]);
    }
    (xmin, xmax, ymin, ymax)
}

fn check(cx: &[f64], cy: &[f64], rows: usize, cols: usize) -> Result<(), String> {
    let n = (rows + 1) * (cols + 1);
    if cx.len() != n || cy.len() != n {
        return Err(format!(
            "expected {n} corner points for {rows}x{cols} cells"
        ));
    }
    Ok(())
}

/// Area-weighted mean of the overlapped source cells (missing ones ignored).
pub fn area_weighted(
    g: &GridGeom,
    values: &[f32],
    cx: &[f64],
    cy: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    g.check_len(values.len())?;
    check(cx, cy, rows, cols)?;
    Ok((0..rows * cols)
        .into_par_iter()
        .map(|k| {
            let (xmin, xmax, ymin, ymax) = footprint(cx, cy, cols, k / cols, k % cols);
            let (mut sum, mut wsum) = (0.0f64, 0.0f64);
            for (i, w) in overlaps(g, xmin, xmax, ymin, ymax) {
                let v = values[i];
                if !v.is_nan() {
                    sum += w * v as f64;
                    wsum += w;
                }
            }
            if wsum > 0.0 {
                (sum / wsum) as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}

/// The class covering the largest share of each footprint (ties: lowest class).
pub fn area_majority(
    g: &GridGeom,
    values: &[i32],
    cx: &[f64],
    cy: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<i32>, String> {
    g.check_len(values.len())?;
    check(cx, cy, rows, cols)?;
    Ok((0..rows * cols)
        .into_par_iter()
        .map(|k| {
            let (xmin, xmax, ymin, ymax) = footprint(cx, cy, cols, k / cols, k % cols);
            let mut area: BTreeMap<i32, f64> = BTreeMap::new();
            for (i, w) in overlaps(g, xmin, xmax, ymin, ymax) {
                if values[i] >= 0 {
                    *area.entry(values[i]).or_insert(0.0) += w;
                }
            }
            // BTreeMap iterates in class order, so the first maximum is the lowest class
            let mut best = (-1, 0.0);
            for (class, a) in area {
                if a > best.1 + 1e-12 {
                    best = (class, a);
                }
            }
            best.0
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corner lattice of a north-up target grid.
    fn corners(x0: f64, y0: f64, d: f64, rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut cx, mut cy) = (Vec::new(), Vec::new());
        for r in 0..=rows {
            for c in 0..=cols {
                cx.push(x0 + c as f64 * d);
                cy.push(y0 - r as f64 * d);
            }
        }
        (cx, cy)
    }

    #[test]
    fn cells_inside_one_source_cell_copy_it() {
        let g = GridGeom::new(0.0, 90.0, 90.0, 90.0, 1, 2).unwrap();
        let v = [1.0f32, 5.0];
        let (cx, cy) = corners(0.0, 90.0, 30.0, 3, 6);
        let out = area_weighted(&g, &v, &cx, &cy, 3, 6).unwrap();
        assert_eq!(&out[..6], &[1.0, 1.0, 1.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn straddling_cells_are_blended_by_area() {
        let g = GridGeom::new(0.0, 100.0, 100.0, 100.0, 1, 2).unwrap();
        let v = [0.0f32, 10.0];
        // a 40 m cell from x=80 to 120: 20 m in each source cell
        let (cx, cy) = corners(80.0, 100.0, 40.0, 1, 1);
        assert!((area_weighted(&g, &v, &cx, &cy, 1, 1).unwrap()[0] - 5.0).abs() < 1e-6);
        // 60 m from x=90: 10 m left, 50 m right
        let (cx, cy) = corners(90.0, 100.0, 60.0, 1, 1);
        let out = area_weighted(&g, &v, &cx, &cy, 1, 1).unwrap()[0];
        assert!((out - 10.0 * 50.0 / 60.0).abs() < 1e-5);
    }

    #[test]
    fn total_is_conserved_when_target_tiles_the_source() {
        let g = GridGeom::new(0.0, 200.0, 100.0, 100.0, 2, 2).unwrap();
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let (cx, cy) = corners(0.0, 200.0, 25.0, 8, 8);
        let out = area_weighted(&g, &v, &cx, &cy, 8, 8).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 64.0;
        assert!((mean - 2.5).abs() < 1e-6);
    }

    #[test]
    fn missing_and_outside_are_ignored() {
        let g = GridGeom::new(0.0, 100.0, 100.0, 100.0, 1, 2).unwrap();
        let v = [f32::NAN, 7.0];
        let (cx, cy) = corners(80.0, 100.0, 40.0, 1, 1);
        assert_eq!(area_weighted(&g, &v, &cx, &cy, 1, 1).unwrap()[0], 7.0);
        let (cx, cy) = corners(500.0, 100.0, 40.0, 1, 1);
        assert!(area_weighted(&g, &v, &cx, &cy, 1, 1).unwrap()[0].is_nan());
    }

    #[test]
    fn majority_takes_the_larger_share_and_ties_go_low() {
        let g = GridGeom::new(0.0, 100.0, 100.0, 100.0, 1, 2).unwrap();
        let v = [3i32, 1];
        let (cx, cy) = corners(90.0, 100.0, 60.0, 1, 1); // 10 m of class 3, 50 m of class 1
        assert_eq!(area_majority(&g, &v, &cx, &cy, 1, 1).unwrap(), vec![1]);
        let (cx, cy) = corners(80.0, 100.0, 40.0, 1, 1); // 20 m each
        assert_eq!(area_majority(&g, &v, &cx, &cy, 1, 1).unwrap(), vec![1]);
        assert!(area_majority(&g, &v, &cx[..3], &cy, 1, 1).is_err());
    }
}
