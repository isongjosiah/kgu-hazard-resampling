"""Experiment configs (``configs/experiments/*.yaml``), validated.

An experiment names a region, the labels, which predictors are fine (handled
the same way in every run) and which are coarse (converted six ways), and the
settings for the chance check, the folds and the models. Nothing about a run
lives in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hazres.grid.methods import Method


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    crs: str = "EPSG:3035"
    bounds: tuple[float, float, float, float]
    """left, bottom, right, top in ``crs`` (metres)."""

    @field_validator("bounds")
    @classmethod
    def _non_empty(cls, b):
        if not (b[2] > b[0] and b[3] > b[1]):
            raise ValueError(f"empty bounds {b}")
        return b


class Predictors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fine: list[str]
    """At or finer than the grid: same treatment in every run."""
    coarse: list[str]
    """Coarser than the grid: converted by each method, and realised for the chance check."""

    @model_validator(mode="after")
    def _disjoint(self):
        both = set(self.fine) & set(self.coarse)
        if both:
            raise ValueError(f"layers listed as both fine and coarse: {sorted(both)}")
        if not self.coarse:
            raise ValueError("at least one coarse layer is needed")
        return self


class Chance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(10, ge=2)
    scales: list[float] = [0.5, 1.0, 2.0]
    seed: int = 20260926


class Folds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_m: float = 10_000
    k: int = Field(5, ge=2)
    buffer_m: float = 2_000
    seed: int = 1


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    region: Region
    labels: str
    predictors: Predictors
    grid_res: float = 30.0
    methods: list[Method] = [
        Method.NEAREST,
        Method.BILINEAR,
        Method.CUBIC,
        Method.AREA_WEIGHTED,
        Method.DOWNSCALED,
    ]
    """Methods compared against the chance check. Each changes only the coarse layers,
    like the random versions, so the comparison is like for like."""
    coarsened_factor: int | None = 33
    """Also run coarsened_target on a grid this many times coarser (reported separately)."""
    downscale_covariates: list[str] = ["elevation", "slope", "tpi"]
    chance: Chance = Chance()
    folds: Folds = Folds()
    models: list[Literal["lightgbm", "random_forest", "logistic"]] = ["lightgbm"]
    importance_repeats: int = Field(5, ge=1)
    top_fraction: float = Field(0.1, gt=0, lt=1)
    exclude_badlands: bool = False
    """Science decision, off unless the group decides otherwise."""
    field_visits_only: bool = False
    """Science decision, off unless the group decides otherwise."""

    @model_validator(mode="after")
    def _checks(self):
        if Method.COARSENED_TARGET in self.methods:
            raise ValueError("list coarsened_target via coarsened_factor, not in methods")
        if Method.DOWNSCALED in self.methods:
            missing = set(self.downscale_covariates) - set(self.predictors.fine)
            if missing:
                raise ValueError(f"downscale covariates must be fine layers: {sorted(missing)}")
        return self


def load_experiment(path: str | Path) -> Experiment:
    return Experiment(**yaml.safe_load(Path(path).read_text()))
