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

## Licence

MIT. Datasets keep their own licences; see `configs/`.
