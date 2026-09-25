"""hazres: does resampling predictors onto one grid change what a hazard map says?

Pipeline, one subpackage per step:

1. ``hazres.data``    load hazard inventories and predictor layers
2. ``hazres.grid``    bring predictors onto one grid (six methods) and make the
                      random fine-scale versions for the chance check (Rust engine)
3. ``hazres.models``  fit models with spatial folds shared across all runs
4. ``hazres.metrics`` measure each run
5. ``hazres.compare`` between-method spread vs chance: p-values and a verdict
6. ``hazres.report``  one-page report, agreement map and methods paragraph
"""

from hazres._engine import engine_version

__version__ = "0.1.0"

__all__ = ["__version__", "engine_version"]
