# kgu-hazard-resampling

Tests whether a hazard susceptibility map changes with how its predictors were resampled onto a common grid, beyond what chance alone would produce.

Hazard maps combine data at very different resolutions, from 10 m satellite images to 25 km climate grids, and everything must be converted to one grid. Studies rarely say how. `hazres` converts the data six ways, fits the same model to each, and checks whether the factors the model relies on, the places it ranks riskiest, and the risk numbers change more than they would across random but equally believable fine-scale versions of the same coarse data.

**Status:** early development. Nothing here is a result yet.

## The six conversion methods

Each method is a different, reasonable way to spread one coarse value (say, one
1 km rainfall cell) over the roughly 1,100 cells of 30 m it covers. Each makes a
different assumption about what the land looks like inside the coarse cell, and
each is something researchers actually do.

| # | Method | What each 30 m cell gets | Assumption | Why we test it | Class maps? |
|---|---|---|---|---|---|
| 1 | `nearest` | The value of the coarse cell it sits in | Nothing changes inside a coarse cell; values jump at its edges | Simplest, most common; default in many GIS tools | Yes |
| 2 | `bilinear` | A linear blend of the 4 nearest coarse cell centres | Values change gradually between cell centres | The usual choice for continuous data | No |
| 3 | `cubic` | A smooth blend of the 16 nearest centres (Keys) | Values change smoothly and can curve; may overshoot | The third common GIS option; looks realistic but can invent peaks | No |
| 4 | `area_weighted` | The coarse cells it overlaps, weighted by overlap (classes: the class covering most of it) | Like nearest, softened at edges; totals kept exactly | The careful "keep the amounts right" choice | Yes |
| 5 | `downscaled` | Detail from fine layers (e.g. terrain), shifted so each coarse cell keeps its published mean | The coarse value varies with the landscape | Used in climate and soil mapping; terrain used only if the fit passes an F-test (p < 0.05) | No |
| 6 | `coarsened_target` | Nothing: labels and fine layers go *up* to the coarse grid instead | Don't claim detail you don't have | Model at the scale the data support | Yes |

**Why these six.** Methods 1–3 are what most published hazard maps use, usually
without saying which; 4 is what careful GIS users choose; 5 and 6 are what
statisticians recommend. Together they run from "flat inside each cell" to
"smooth", "follow the landscape" and "don't pretend". If all six give the same
map, the choice does not matter for that study; if they disagree, it does, and
the chance check says whether the disagreement is bigger than chance.

`block_mean`, in the first plan, is not included: going from coarse to fine it
gives exactly the same result as `nearest`. Details, conventions and tests:
[`docs/grid-methods.md`](docs/grid-methods.md).

## Layout

```
crates/hazres-grid/   Rust grid engine: resampling methods, fine-scale realisations
crates/hazres-py/     Python bindings for the engine (PyO3), built as hazres._engine
src/hazres/
  data/               1. load hazard inventories and predictor layers
  grid/               2. bring predictors onto one grid (wraps the Rust engine)
  models/             3. models and spatial cross-validation
  metrics/            4. measure each run
  compare.py          5. between-method spread vs chance
  report/             6. report, agreement map, methods paragraph
  cli.py              the `hazres` command
tests/                Python tests, one file per module
configs/              every choice, as YAML
data/, outputs/       not in git
```

## Setup

Needs [uv](https://docs.astral.sh/uv/) and a Rust toolchain ([rustup](https://rustup.rs/)).

```bash
make setup     # creates .venv and builds the Rust engine
make check     # lint + Rust tests + Python tests; must pass before every merge
uv run hazres --version
```

After changing Rust code, `uv run` rebuilds the engine automatically. `make dev` forces a rebuild.

## Data

Hazard datasets are listed in `configs/inventories.yaml`.

```bash
uv run hazres data list              # every dataset, its tier, and whether its files are present
uv run hazres data fetch ge_lucas    # download into data/raw/ and print checksums
uv run hazres data inspect ge_lucas  # load it and report what was kept and dropped
```

The Nigerian flood events come from Google Earth Engine (free for research):

```bash
uv sync --extra gee
uv run earthengine authenticate                        # once
uv run hazres data export-gfd --project YOUR_GCP_PROJECT
uv run hazres data inspect global_flood_database
```

Events are exported at 250 m (the MODIS observation size) in UTM 32N, clipped to
Nigeria. If an event is too large for direct download, rerun with
`--method drive` and move the files from Google Drive into
`data/raw/global_flood_database/nigeria/`.

### Predictor layers

Listed in `configs/predictors.yaml` (a proposed list, to be agreed by the group).
Each layer is read exactly as published, at its native resolution and in its
native CRS; converting coarse layers to the 30 m grid is the experiment and
happens in `hazres.grid`.

```bash
uv run hazres data predictors list
uv run hazres data predictors inspect elevation --bounds 3100000 1900000 3110000 1910000
uv run hazres data predictors inspect slope     --bounds 3100000 1900000 3110000 1910000
```

Bounds are in the analysis CRS (EPSG:3035, metres) unless `--bounds-crs` is given.
Elevation, land cover, soil and precipitation are read window by window from the
web; the subsurface and erosivity layers need a free account and a manual
download (see each layer's notes).

### Datasets in use

| Tier | Dataset | Hazard, region | "No hazard" labels | Access |
|---|---|---|---|---|
| 1 | GE-LUCAS v1.1 (Borrelli et al. 2025) | Gullies, EU | Observed | Open (Figshare) |
| 1 | Global Flood Database v1 (Tellman et al. 2021) | Floods, Nigeria | Partial (misses floods under cloud) | Earth Engine export |
| 1 (requested) | De Geeter et al. 2023 gully heads | Gullies, Africa | None (presence only) | Requested from the authors |
| 1 (requested) | Chen et al. 2025 site observations, African subset | Gullies, Africa | Observed (~330 m cells) | Requested from the authors |
| 2 | Kahramanmaraş 2023 (Yılmaz et al. 2026) | Landslides, Türkiye | Inside the mapped area only | Open (Zenodo) |

Tier 1 is always done. Tier 2 only if the week-1 pilot says go.

**African gullies (backup plan).** No open Africa-wide gully dataset exists. Both
African gully datasets come from the Vanmaercke group (KU Leuven) and have been
requested. Until they arrive, Africa is covered by the Nigerian flood events,
which are in Tier 1 so the paper has an African case even if the pilot says
no-go. If the gully data arrive in time, they join Tier 1. The public
repository for Chen et al. 2025 (doi:10.48804/BASVNF) holds only predicted 1 km
maps, which are model output and cannot be used as labels.

### Not used for now (to revisit)

**Deferred (Tier 3, in the registry without a loader):**

- *MODIS burned area MCD64A1* (wildfire): fire depends strongly on the weather of a given year, so it fits a susceptibility question less well, and it is another data type (500 m monthly grids).
- *Marche-Umbria 2022 landslides* (Italy): good data, but Europe is already covered by GE-LUCAS.
- *Kivu–Tanganyika Rift landslides* (East Africa): presence only, and it is unclear what is released. Worth adding for African coverage if the data become available.

The reason for all three is time: more hazards means less care for each within five weeks.

**For breadth in a follow-up (open, not in the registry):**

- Chile 2010 earthquake landslides (Serey & Sepúlveda 2024)
- Indonesia cyclone landslides benchmark (Samodra et al. 2025)
- China 2024 rainfall landslides (Fu et al. 2025)
- Global database of ~400,000 earthquake-triggered landslides (Fan et al. 2025; availability to confirm)

**Hazards ruled out:**

- *Groundwater potential*: not a hazard; the labels are productive wells.
- *Subsidence*: few open inventories, and the drivers (pumping, radar measurements) are a different kind of predictor.
- *Drought*: covers whole areas over time, not specific sites.

**Parts of included datasets not used as labels:**

- *GE-LUCAS probability map*: a model output, not an observation. Possible comparison later.
- *GE-LUCAS cross-check point sets*: checks of the survey against imagery. Could later estimate how many gullies the survey missed.

## Licence

MIT. Datasets keep their own licences; see `configs/`.
