"""Training tables: the same points and labels, with predictor values under each variant.

A *variant* is one way of preparing the coarse layers:

``method:<name>``        every coarse layer converted with one of the methods
``chance:x<scale>:<i>``  every coarse layer replaced by random version ``i``,
                         with detail at ``scale`` times the estimate
``coarsened``            every layer, fine and coarse, brought up to a coarse
                         grid (about 1 km); each point takes its coarse cell's
                         values. Reported separately, because it also changes
                         the fine layers.

Fine layers (terrain, land cover) are prepared once, the same way, and are
identical in every variant except ``coarsened``. So between ``method`` and
``chance`` variants exactly one thing varies: the coarse layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hazres import _engine
from hazres.data.labels import VectorLabels
from hazres.data.predictors import Kind, Layer, PredictorRegistry, read_layer
from hazres.data.terrain import terrain_on_grid
from hazres.grid.methods import Method, downscale, resample
from hazres.grid.realise import estimate_detail, realise
from hazres.grid.spec import GridSpec
from hazres.grid.warp import warp_to_grid
from hazres.pipeline.config import Experiment

POINT_COLUMNS = ["sample_id", "label", "x", "y", "row", "col"]


@dataclass
class Tables:
    points: pd.DataFrame
    """One row per point: sample_id, label, x, y, row, col, and any kept columns."""
    variants: dict[str, pd.DataFrame]
    """Predictor values per variant, rows aligned with ``points``."""
    categorical: list[str]
    fine: list[str]
    coarse: list[str]
    info: dict = field(default_factory=dict)

    def variant_kind(self, name: str) -> str:
        return name.split(":", 1)[0]

    def names(self, kind: str, scale: float | None = None) -> list[str]:
        out = [n for n in self.variants if self.variant_kind(n) == kind]
        if scale is not None:
            out = [n for n in out if n.split(":")[1] == f"x{scale:g}"]
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.points.to_parquet(directory / "points.parquet")
        for name, df in self.variants.items():
            df.to_parquet(directory / f"{name.replace(':', '__')}.parquet")
        meta = {
            "categorical": self.categorical,
            "fine": self.fine,
            "coarse": self.coarse,
            "variants": list(self.variants),
            "info": self.info,
        }
        (directory / "tables.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: Path) -> Tables:
        meta = json.loads((directory / "tables.json").read_text())
        variants = {
            n: pd.read_parquet(directory / f"{n.replace(':', '__')}.parquet")
            for n in meta["variants"]
        }
        return cls(
            pd.read_parquet(directory / "points.parquet"),
            variants, meta["categorical"], meta["fine"], meta["coarse"], meta["info"],
        )  # fmt: skip


def region_grid(exp: Experiment) -> GridSpec:
    return GridSpec.covering(exp.region.bounds, exp.region.crs, exp.grid_res)


def select_points(labels: VectorLabels, exp: Experiment, grid: GridSpec) -> pd.DataFrame:
    """The labelled points inside the region, with their grid cell, after any exclusions."""
    f = labels.features.to_crs(exp.region.crs)
    x, y = f.geometry.x.to_numpy(), f.geometry.y.to_numpy()
    left, bottom, right, top = exp.region.bounds
    inside = (x >= left) & (x < right) & (y > bottom) & (y <= top)
    if exp.exclude_badlands and "SURVEY_GULLY_TYPE" in f.columns:
        inside &= ~(f["SURVEY_GULLY_TYPE"].to_numpy() == 3)
    if exp.field_visits_only and "SURVEY_OBS_TYPE" in f.columns:
        inside &= f["SURVEY_OBS_TYPE"].to_numpy() <= 2
    col = np.floor((x - grid.left) / grid.res).astype(np.int64)
    row = np.floor((grid.top - y) / grid.res).astype(np.int64)
    extras = [c for c in f.columns if c not in ("sample_id", "label", "geometry")]
    pts = pd.DataFrame(
        {"sample_id": f["sample_id"].to_numpy(), "label": f["label"].to_numpy().astype(np.int8),
         "x": x, "y": y, "row": row, "col": col,
         **{c: f[c].to_numpy() for c in extras}},
    )[inside].reset_index(drop=True)  # fmt: skip
    if pts.empty:
        raise ValueError(f"no labelled points inside region {exp.region.name}")
    if pts["label"].sum() == 0:
        raise ValueError(f"no hazard points inside region {exp.region.name}")
    return pts


def _sample(grid_values: np.ndarray, pts: pd.DataFrame) -> np.ndarray:
    return grid_values[pts["row"].to_numpy(), pts["col"].to_numpy()]


def _zone_majority(classes: np.ndarray, zones: np.ndarray, n_zones: int) -> np.ndarray:
    """Most common class per zone (ties: lowest class); -1 where none."""
    ok = (zones >= 0) & (classes >= 0)
    out = np.full(n_zones, -1, dtype=np.int32)
    if not ok.any():
        return out
    z, c = zones[ok], classes[ok]
    pairs, counts = np.unique(np.stack([z, c]), axis=1, return_counts=True)
    order = np.lexsort(
        (pairs[1], -counts, pairs[0])
    )  # by zone, then most common, then lowest class
    pz, pc = pairs[0][order], pairs[1][order]
    first = np.r_[True, pz[1:] != pz[:-1]]
    out[pz[first]] = pc[first]
    return out


def build_tables(
    exp: Experiment,
    registry: PredictorRegistry,
    labels: VectorLabels,
    *,
    data_root: Path,
    log=print,
) -> Tables:
    grid = region_grid(exp)
    pts = select_points(labels, exp, grid)
    log(f"{len(pts):,} points in {exp.region.name} ({int(pts['label'].sum())} hazard), "
        f"grid {grid.shape[0]:,} x {grid.shape[1]:,} of {grid.res:g} m")  # fmt: skip
    layers = registry.layers
    info: dict = {"grid": {"crs": grid.crs, "res": grid.res, "bounds": grid.bounds,
                           "shape": grid.shape}}  # fmt: skip

    # --- fine layers: once, the same for every method and chance variant -------------
    fine_grid: dict[str, np.ndarray] = {}
    categorical: list[str] = []
    if any(layers[k].access == "derived" or k == "elevation" for k in exp.predictors.fine):
        fine_grid.update(terrain_on_grid(grid, registry, data_root=data_root))
    for key in exp.predictors.fine:
        cfg = layers[key]
        if key in fine_grid:
            continue
        layer = read_layer(cfg, grid.bounds, grid.crs, data_root=data_root)
        if cfg.kind is Kind.CLASSES:
            fine_grid[key] = warp_to_grid(layer, grid, "mode")
            categorical.append(key)
        else:
            fine_grid[key] = warp_to_grid(layer, grid, "average")
    fine_values = {k: _sample(fine_grid[k], pts) for k in exp.predictors.fine}
    covariates = {
        k: fine_grid[k].astype(np.float32) for k in exp.downscale_covariates if k in fine_grid
    }

    # --- coarse layers ------------------------------------------------------------
    coarse_layers: dict[str, Layer] = {}
    for key in exp.predictors.coarse:
        layer = read_layer(layers[key], grid.bounds, grid.crs, data_root=data_root)
        if not isinstance(layer, Layer):
            raise NotImplementedError(f"{key}: vector coarse layers are not supported yet")
        coarse_layers[key] = layer
        if layer.kind is Kind.CLASSES:
            categorical.append(key)

    variants: dict[str, pd.DataFrame] = {}

    def frame(coarse_vals: dict[str, np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame({**fine_values, **coarse_vals})

    info["downscaling"] = {}
    for method in exp.methods:
        vals = {}
        for key, layer in coarse_layers.items():
            if layer.kind is Kind.CLASSES and method not in (Method.NEAREST, Method.AREA_WEIGHTED):
                vals[key] = _sample(resample(layer, grid, Method.AREA_WEIGHTED), pts)
                continue
            if method is Method.DOWNSCALED:
                d = downscale(layer, grid, covariates)
                info["downscaling"][key] = {"r2": d.r2, "p_value": d.p_value,
                                            "used_covariates": d.used_covariates,
                                            "coefficients": d.coefficients}  # fmt: skip
                vals[key] = _sample(d.values, pts)
            else:
                vals[key] = _sample(resample(layer, grid, method), pts)
        variants[f"method:{method.value}"] = frame(vals)
        log(f"  method {method.value}: done")

    # --- chance check ---------------------------------------------------------------
    info["detail"] = {}
    rows_cols = (pts["row"].to_numpy(), pts["col"].to_numpy())
    for scale in exp.chance.scales:
        per_layer: dict[str, np.ndarray] = {}
        for key, layer in coarse_layers.items():
            if layer.kind is Kind.CLASSES:
                # class realisations are not built: the class layer stays fixed
                fixed = _sample(resample(layer, grid, Method.AREA_WEIGHTED), pts)
                per_layer[key] = np.tile(fixed, (exp.chance.n, 1))
                continue
            detail = estimate_detail(layer, grid)
            r = realise(layer, grid, exp.chance.n, detail=detail, scale=scale,
                        seed=exp.chance.seed, points=rows_cols)  # fmt: skip
            info["detail"].setdefault(key, {})[f"x{scale:g}"] = {
                "sd": r.detail.sd,
                "correlation_m": r.detail.correlation_m,
            }
            per_layer[key] = r.values
        for i in range(exp.chance.n):
            variants[f"chance:x{scale:g}:{i:02d}"] = frame({k: v[i] for k, v in per_layer.items()})
        log(f"  chance x{scale:g}: {exp.chance.n} versions done")

    # --- coarsened target ------------------------------------------------------------
    if exp.coarsened_factor:
        coarse = GridSpec.covering(exp.region.bounds, grid.crs, grid.res * exp.coarsened_factor)
        rows, cols = grid.shape
        xs = grid.left + (np.arange(cols) + 0.5) * grid.res
        ys = grid.top - (np.arange(rows) + 0.5) * grid.res
        xx, yy = np.meshgrid(xs, ys)
        geom = (coarse.left, coarse.top, coarse.res, coarse.res)
        zones = _engine.cell_index(geom, coarse.shape, xx.ravel(), yy.ravel())
        n_zones = coarse.shape[0] * coarse.shape[1]
        pz = zones.reshape(rows, cols)[rows_cols]
        vals = {}
        for key in exp.predictors.fine:
            g = fine_grid[key]
            if key in categorical:
                per_zone = _zone_majority(g.ravel().astype(np.int32), zones, n_zones)
            else:
                per_zone = _engine.zonal_mean(np.ascontiguousarray(g.ravel(), dtype=np.float32),
                                              zones, n_zones)[0]  # fmt: skip
            vals[key] = per_zone[pz]
        for key, layer in coarse_layers.items():
            vals[key] = resample(layer, coarse, Method.AREA_WEIGHTED).ravel()[pz]
        variants["coarsened"] = pd.DataFrame(vals)
        info["coarsened"] = {"res": coarse.res, "shape": coarse.shape}
        log(f"  coarsened to {coarse.res:g} m: done")

    return Tables(pts, variants, categorical, list(exp.predictors.fine),
                  list(exp.predictors.coarse), info)  # fmt: skip
