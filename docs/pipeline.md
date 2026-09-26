# Running an experiment

Code: `src/hazres/pipeline/` (config, cache, tables, run), `src/hazres/models/`
(spatial folds, models), `src/hazres/metrics/scores.py`, `src/hazres/compare.py`.

```bash
uv run hazres data cache --experiment configs/experiments/pilot_sicily.yaml   # once, needs internet
uv run hazres run        --experiment configs/experiments/pilot_sicily.yaml   # offline
```

Results go to `outputs/<experiment>/`: the training tables, and per model
`scores.csv`, `importance.csv`, `oof.parquet`, `comparison.csv` and `report.md`.

## What happens

1. **Cache** (`hazres data cache`): every predictor the experiment uses is read
   for its region plus 3 km, exactly as published, and saved to
   `data/raw/cache/<region>/` with a `manifest.json`. After this, nothing needs
   the internet.
2. **Points**: the labelled points inside the region (GE-LUCAS for the pilot),
   each placed in its 30 m grid cell. Optional exclusions (badlands, office
   photo-interpreted points) are switched off unless the group decides.
3. **Training tables**: one table per *variant*, same points and labels in all:
   - `method:<name>`: coarse layers converted with one method;
   - `chance:x<scale>:<i>`: coarse layers replaced by random version *i*, at
     0.5, 1 and 2 times the assumed detail;
   - `coarsened`: every layer brought up to about 1 km (reported separately,
     because it also changes the fine layers).

   Fine layers (terrain, land cover) are identical in every `method` and
   `chance` table, so exactly one thing varies between them: the coarse layers.
4. **Spatial folds**: points grouped into 10 km blocks, dealt into 5 folds with
   a similar number of gullies each; training points within 2 km of the test
   points are dropped. Built once and shared by every variant and model.
5. **Models**: LightGBM (pilot), random forest and logistic regression available.
   Fixed settings and seeds; the real mix of gully and non-gully points is kept.
6. **Measures**, all on out-of-fold predictions: AUC, PR-AUC, Brier score,
   calibration gap, slope and intercept, calibration error, Moran's I of the
   errors, which factors the model relies on (permutation importance), and the
   riskiest 10% of points.
7. **Comparison**: for each measure, the spread across the 5 methods against
   the spread across groups of 5 random versions, giving a p-value and a
   verdict ("sensitive" or "not beyond chance") at each detail strength.

## The methods compared

The comparison uses the 5 methods that change only the coarse layers (nearest,
bilinear, cubic, area-weighted, downscaled), because the random versions also
change only the coarse layers: like against like. `coarsened_target` changes
every layer, so it is reported alongside, not in the p-value.

## Speed

On a 100 km pilot region (11 million 30 m cells, about 1,000 points) on a
2-core machine: about 4 minutes to build the tables and 2 minutes to fit and
score 36 variants with LightGBM; peak memory 2.7 GB.

## Changes after pilot run 1 (26 September 2026)

Run 1 (central Sicily) found no method effect beyond chance, but the model was
weak (AUC 0.66–0.69; calibration slope about 0.35, i.e. far too extreme), so the
null was not decisive. Changes, fixed before any rerun:

1. **Gully points at their real location** (`label_options:
   presence_location: gully_location` in each pilot config). In central Sicily
   80% of gullies are more than 30 m from their LUCAS survey point.
2. **Gentler LightGBM settings** for a few thousand points: 200 trees, learning
   rate 0.03, 15 leaves, depth 4, at least 30 points per leaf, stronger
   regularisation. Random forest leaves need at least 10 points. Chosen to stop
   overfitting, not tuned on the method comparison.
3. **Two more regions** (Trentino, south-east Spain), and `hazres summary` to
   put all runs side by side. The Sicily experiment is now called
   `pilot_sicily`, so run 1's outputs (`outputs/pilot/`) are kept for comparison.
