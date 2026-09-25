//! Geometry of a north-up source grid.

/// A north-up raster grid: `cols` x `rows` cells of `dx` by `dy` (both
/// positive), whose top-left corner is at (`x0`, `y0`). Row 0 is the northern
/// edge. Coordinates are in the grid's own CRS (metres or degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridGeom {
    pub x0: f64,
    pub y0: f64,
    pub dx: f64,
    pub dy: f64,
    pub rows: usize,
    pub cols: usize,
}

impl GridGeom {
    pub fn new(
        x0: f64,
        y0: f64,
        dx: f64,
        dy: f64,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        if !(dx > 0.0 && dy > 0.0) {
            return Err(format!("cell size must be positive, got {dx} x {dy}"));
        }
        if rows == 0 || cols == 0 {
            return Err("grid must have at least one cell".into());
        }
        Ok(Self {
            x0,
            y0,
            dx,
            dy,
            rows,
            cols,
        })
    }

    /// Continuous cell index of a point: cell (r, c) spans `c..c+1`, `r..r+1`.
    #[inline]
    pub fn frac_index(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.x0) / self.dx, (self.y0 - y) / self.dy)
    }

    /// The cell containing a point, if any.
    #[inline]
    pub fn cell_of(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let (u, v) = self.frac_index(x, y);
        if !(u >= 0.0 && v >= 0.0) {
            return None; // also rejects NaN
        }
        let (c, r) = (u.floor() as usize, v.floor() as usize);
        (r < self.rows && c < self.cols).then_some((r, c))
    }

    #[inline]
    pub fn idx(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    /// Checks that a values slice matches the grid.
    pub fn check_len(&self, n: usize) -> Result<(), String> {
        if n == self.rows * self.cols {
            Ok(())
        } else {
            Err(format!(
                "expected {} values for a {}x{} grid, got {n}",
                self.rows * self.cols,
                self.rows,
                self.cols
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_lookup() {
        let g = GridGeom::new(100.0, 50.0, 10.0, 5.0, 3, 4).unwrap();
        assert_eq!(g.cell_of(100.0, 50.0), Some((0, 0)));
        assert_eq!(g.cell_of(139.9, 35.1), Some((2, 3)));
        assert_eq!(g.cell_of(140.0, 40.0), None); // east edge is exclusive
        assert_eq!(g.cell_of(99.9, 40.0), None);
        assert_eq!(g.cell_of(f64::NAN, 40.0), None);
        assert!(GridGeom::new(0.0, 0.0, -1.0, 1.0, 1, 1).is_err());
    }
}
