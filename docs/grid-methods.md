# The six conversion methods

How a coarse predictor layer (for example soil at 250 m, precipitation at 1 km)
is brought onto the 30 m analysis grid. This is the experiment: the same
layer, converted six ways, with everything else held fixed.

Code: `crates/hazres-grid` (Rust engine) and `src/hazres/grid/methods.py`.

| Method | What each 30 m cell gets | Class layers? |
|---|---|---|
| `nearest` | The value of the coarse cell its centre falls in | Yes |
| `bilinear` | A linear blend of the 4 nearest coarse cell centres | No |
| `cubic` | A smooth blend of the 16 nearest centres (Keys, a = -0.5); can overshoot | No |
| `area_weighted` | The coarse cells its footprint overlaps, weighted by area (classes: the class covering most of it) | Yes |
| `downscaled` | A regression on fine layers (e.g. terrain), shifted so each coarse cell keeps its published mean | No |
| `coarsened_target` | Not a conversion: labels and fine layers go *up* to a coarse grid instead | Yes |

## Changes from the original plan (25 September 2026)

1. **`block_mean` is replaced by `cubic`.** Going from coarse to fine,
   `block_mean` gives every fine cell its coarse cell's value, which is exactly
   `nearest`. The spec already said identical methods would be dropped. `cubic`
   is the third option GIS software offers alongside nearest and bilinear, so
   it reflects real practice.
2. **Downscaling only uses covariates when the relationship is real.** A
   regression fitted on coarse-cell averages can find a small slope by chance.
   Applied to fine covariates, which vary far more than their averages, that
   slope invents detail. The fit is therefore used only if it passes an F-test
   at p < 0.05; otherwise the result is the coarse values themselves
   (`nearest`). The p-value and R² are reported for every layer.

## How it works

- Map projections are handled once, in Python: the 30 m cell centres and
  corners are transformed into the layer's own CRS (e.g. lon/lat for
  precipitation, Homolosine for SoilGrids). The Rust engine then works entirely
  in that CRS.
- Missing coarse cells are skipped and the remaining weights renormalised; a
  cell with no valid neighbour is missing.
- `cubic` falls back to `bilinear` where its 4 x 4 neighbourhood is incomplete
  (edges, gaps).
- `area_weighted` treats each target cell's footprint as the bounding box of
  its four corners in the layer's CRS: exact when both grids are north-up in
  the same CRS, a close approximation otherwise.
- `downscaled` keeps each coarse cell's mean exactly (over the part of it
  inside the area), so averaging the result back gives the published values.

## Speed

On a 50 km x 50 km tile (2.8 million 30 m cells) from a 250 m layer: each
point or footprint method takes about 0.1 s, downscaling about 0.6 s.

## Tests

- Rust (`cargo test -p hazres-grid`): exact planes and quadratics, area
  conservation, missing values, class majority and ties.
- Python (`tests/grid/test_methods.py`): planes through a map projection,
  class-safety, mass preservation of downscaling, no invented detail from an
  irrelevant covariate, and the synthetic study, where a smooth coarse layer
  survives every method and a rough one does not.
