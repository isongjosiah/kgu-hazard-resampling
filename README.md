# kgu-hazard-resampling

Tests whether a hazard susceptibility map changes with how its predictors were resampled onto a common grid, beyond what chance alone would produce.

Hazard maps combine data at very different resolutions, from 10 m satellite images to 25 km climate grids, and everything must be converted to one grid. Studies rarely say how. `hazres` converts the data six ways, fits the same model to each, and checks whether the factors the model relies on, the places it ranks riskiest, and the risk numbers change more than they would across random but equally believable fine-scale versions of the same coarse data.

**Status:** early development. Nothing here is a result yet.

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

### Datasets in use

| Tier | Dataset | Hazard, region | "No hazard" labels | Access |
|---|---|---|---|---|
| 1 | GE-LUCAS v1.1 (Borrelli et al. 2025) | Gullies, EU | Observed | Open (Figshare) |
| 1 (requested) | De Geeter et al. 2023 gully heads | Gullies, Africa | None (presence only) | Requested from the authors |
| 1 (requested) | Chen et al. 2025 site observations, African subset | Gullies, Africa | Observed (~330 m cells) | Requested from the authors |
| 2 | Kahramanmaraş 2023 (Yılmaz et al. 2026) | Landslides, Türkiye | Inside the mapped area only | Open (Zenodo) |
| 2 | Global Flood Database v1 (Tellman et al. 2021) | Floods, Nigeria | Partial (misses floods under cloud) | Earth Engine export |

Tier 1 is always done. Tier 2 only if the week-1 pilot says go.

**African gullies (backup plan).** No open Africa-wide gully dataset exists. Both
African gully datasets come from the Vanmaercke group (KU Leuven) and have been
requested. Until they arrive, Tier 1 is GE-LUCAS alone and Africa is covered by
the Nigerian flood events. If they arrive in time, they join Tier 1. The public
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
