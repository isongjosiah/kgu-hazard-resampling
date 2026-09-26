# Pilot run 2 (26 September 2026)

Three 100 km regions, GE-LUCAS gullies at their real location (Data 3),
LightGBM with conservative settings, 5 coarse layers (clay, sand, organic
carbon, bulk density at 250 m; precipitation at ~1 km), 8 fine layers (terrain,
land cover). Each region: 5 conversion methods, 30 random versions (10 at each
of x0.5, x1, x2 detail), and everything coarsened to ~1 km.

## Model quality: now meets the plan's bar

| Region | Points (gullies) | AUC | Calibration slope | Calibration gap |
|---|---|---|---|---|
| Central Sicily | 969 (168) | 0.79 | 0.88–0.90 | 0.00 |
| Trentino | 1,490 (125) | 0.83 | 0.90–0.92 | −0.01 |
| South-east Spain | 462 (133) | 0.81 | 0.84–0.85 | 0.00 |

Run 1 (Sicily, gullies at the LUCAS point, first model settings) had AUC 0.67
and calibration slope 0.35. Run 2 changed two things at once, so which one
fixed it is not yet known.

## The conversion method: no effect beyond chance

- **AUC** differs by at most 0.006 between the five methods in each region;
  random versions differ by about as much or more.
- **Top three factors** are identical under every method in Sicily
  (plan curvature, tpi, slope) and Trentino (elevation, slope, bulk density),
  and under four of five methods in Spain (downscaling swaps in bulk density).
- **The riskiest 10% of points** changes by 17–23% between methods, but by
  22–43% between random versions that are all consistent with the data.
- **Verdicts:** 2 "sensitive" out of 81 comparisons (3 regions x 3 detail
  strengths x 9 measures): Trentino at x0.5 for factor ranking (p = 0.03) and
  PR-AUC. That is fewer than the ~4 expected by chance alone, and neither is
  repeated at x1 or x2 or in another region.

By the pre-registered rule (pilot go/no-go), the conversion method **does not
matter beyond chance for these layers in these regions**.

## What does matter

- **Analysis scale.** Bringing everything up to ~1 km (coarsened target) costs a
  lot in Sicily (AUC 0.79 → 0.60) and Spain (0.81 → 0.77), little in Trentino
  (0.83 → 0.82).
- **Where the gully label sits** (and/or model settings): Sicily's AUC went
  from 0.67 to 0.79 and calibration slope from 0.35 to 0.89 between run 1 and
  run 2.
- **Unavoidable uncertainty in the riskiest places:** even random versions that
  all agree with the published coarse data move a quarter to two fifths of the
  top-10% points.

## Limits of this result

- The coarse layers here are soil (250 m) and precipitation (1 km). The layers
  the study expected to be most sensitive are missing: soil thickness and water
  table depth (1 km) and lithology (a coarse class map).
- Coarse layers were mostly not among the top factors (except bulk density in
  Trentino). A conversion method cannot change much through layers the model
  hardly uses.
- One model type (LightGBM) so far.
- The chance check depends on the assumed amount of fine detail; verdicts were
  the same at all three strengths except the two flags above.
