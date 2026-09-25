"""The six ways of bringing a coarse layer onto the analysis grid.

This is the experiment: the same coarse layer, converted six ways, everything
else held fixed. Each method takes a :class:`~hazres.data.predictors.Layer` at
its native resolution and native CRS and returns an array on a
:class:`~hazres.grid.spec.GridSpec`.

==================  ===========================================================  ========
method              what each 30 m cell gets                                      classes
==================  ===========================================================  ========
nearest             the value of the coarse cell its centre falls in              yes
bilinear            a linear blend of the four nearest coarse cell centres        no
cubic               a smooth (Keys) blend of the 16 nearest centres; can           no
                    overshoot the coarse values
area_weighted       the coarse cells its footprint overlaps, weighted by area;    yes
                    (classes: the class covering most of it)
downscaled          a regression on fine layers (e.g. terrain), shifted so every  no
                    coarse cell keeps its original mean
coarsened_target    not a conversion: labels and fine layers go *up* to a coarse  yes
                    grid instead (see :func:`coarse_grid` and :func:`coarsen`)
==================  ===========================================================  ========

``block_mean`` from the original plan is not here: going from coarse to fine it
gives every fine cell its coarse cell's value, which is exactly ``nearest``.
``cubic`` takes its place, as the third option GIS software commonly offers.

Map projections are handled here, once: target cell centres and corners are
transformed into the layer's own CRS, and the Rust engine works in that CRS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pyproj import Transformer

from hazres import _engine
from hazres.data.predictors import Kind, Layer
from hazres.grid.spec import GridSpec


class Method(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    AREA_WEIGHTED = "area_weighted"
    DOWNSCALED = "downscaled"
    COARSENED_TARGET = "coarsened_target"


CLASS_SAFE = frozenset({Method.NEAREST, Method.AREA_WEIGHTED, Method.COARSENED_TARGET})
POINT_METHODS = {
    Method.NEAREST: _engine.nearest,
    Method.BILINEAR: _engine.bilinear,
    Method.CUBIC: _engine.cubic,
}


def methods_for(kind: Kind) -> list[Method]:
    """The methods a layer of this kind can go through."""
    return [m for m in Method if kind is Kind.CONTINUOUS or m in CLASS_SAFE]


def _geometry(layer: Layer) -> tuple[float, float, float, float]:
    t = layer.transform
    if t.b != 0 or t.d != 0 or t.e >= 0:
        raise ValueError(f"{layer.key}: only north-up grids are supported")
    return (t.c, t.f, t.a, -t.e)


def _transformer(grid: GridSpec, layer: Layer) -> Transformer:
    return Transformer.from_crs(grid.crs, layer.crs, always_xy=True)


def target_centres(grid: GridSpec, layer: Layer) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell centres, row-major, in the layer's CRS."""
    rows, cols = grid.shape
    x = grid.left + (np.arange(cols) + 0.5) * grid.res
    y = grid.top - (np.arange(rows) + 0.5) * grid.res
    xx, yy = np.meshgrid(x, y)
    tx, ty = _transformer(grid, layer).transform(xx.ravel(), yy.ravel())
    return np.ascontiguousarray(tx, dtype=np.float64), np.ascontiguousarray(ty, dtype=np.float64)


def target_corners(grid: GridSpec, layer: Layer) -> tuple[np.ndarray, np.ndarray]:
    """The (rows + 1) x (cols + 1) lattice of grid cell corners, in the layer's CRS."""
    rows, cols = grid.shape
    x = grid.left + np.arange(cols + 1) * grid.res
    y = grid.top - np.arange(rows + 1) * grid.res
    xx, yy = np.meshgrid(x, y)
    tx, ty = _transformer(grid, layer).transform(xx, yy)
    return (
        np.ascontiguousarray(tx, dtype=np.float64),
        np.ascontiguousarray(ty, dtype=np.float64),
    )


def resample(
    layer: Layer,
    grid: GridSpec,
    method: Method | str,
    *,
    covariates: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Bring ``layer`` onto ``grid`` with one method. Returns an array of ``grid.shape``."""
    method = Method(method)
    if layer.kind is Kind.CLASSES and method not in CLASS_SAFE:
        raise ValueError(f"{layer.key}: '{method.value}' would average class codes")
    if method is Method.COARSENED_TARGET:
        raise ValueError(
            "coarsened_target changes the grid, not the layer: use coarse_grid() and "
            "resample(layer, coarse, 'area_weighted'), then coarsen() the fine layers"
        )
    geometry = _geometry(layer)
    values = np.ascontiguousarray(layer.values)

    if method is Method.DOWNSCALED:
        if not covariates:
            raise ValueError("downscaled needs fine covariates on the grid (e.g. terrain)")
        return downscale(layer, grid, covariates).values

    if method is Method.AREA_WEIGHTED:
        cx, cy = target_corners(grid, layer)
        fn = _engine.area_weighted if layer.kind is Kind.CONTINUOUS else _engine.area_majority
        return fn(values, geometry, cx, cy)

    xs, ys = target_centres(grid, layer)
    if layer.kind is Kind.CLASSES:
        out = _engine.nearest_class(values, geometry, xs, ys)
    else:
        out = POINT_METHODS[method](values, geometry, xs, ys)
    return out.reshape(grid.shape)


# --- downscaling -------------------------------------------------------------------


@dataclass
class Downscaled:
    values: np.ndarray
    coefficients: dict[str, float]
    """Intercept and one slope per covariate, on standardised covariates."""
    r2: float
    """How much of the coarse-cell variation the covariates explain."""
    p_value: float
    """F-test of the regression. Above ``alpha``, the covariates are not used."""
    used_covariates: bool
    n_coarse: int


def downscale(
    layer: Layer,
    grid: GridSpec,
    covariates: Mapping[str, np.ndarray],
    *,
    min_cells: int = 10,
    alpha: float = 0.05,
) -> Downscaled:
    """Regression downscaling that keeps every coarse cell's mean exactly.

    1. Each fine cell is assigned to the coarse cell its centre falls in.
    2. Covariates are averaged over each coarse cell, and the coarse values are
       regressed on those averages (ordinary least squares).
    3. The fitted relationship predicts every fine cell from its own covariates.
    4. Predictions are shifted, coarse cell by coarse cell, so their mean
       equals the coarse value again.

    Step 4 is what makes the result consistent with the published data: averaged
    back over a coarse cell, it gives exactly what was downloaded.

    A relationship found by chance at the coarse scale is dangerous: fine
    covariates vary far more than their coarse averages, so even a small chance
    slope invents detail. So the fit is only used if it passes an F-test at
    ``alpha``; otherwise the covariates are dropped and the result is the
    coarse values themselves, i.e. ``nearest``.
    """
    from scipy import stats

    if layer.kind is not Kind.CONTINUOUS:
        raise ValueError(f"{layer.key}: cannot downscale a class layer")
    names = list(covariates)
    for name in names:
        if covariates[name].shape != grid.shape:
            raise ValueError(f"covariate {name} is {covariates[name].shape}, grid is {grid.shape}")

    rows, cols = layer.values.shape
    xs, ys = target_centres(grid, layer)
    zones = _engine.cell_index(_geometry(layer), (rows, cols), xs, ys)
    n = rows * cols
    y = layer.values.ravel().astype(np.float64)

    fine = np.column_stack([covariates[k].ravel().astype(np.float32) for k in names])
    coarse_x = np.column_stack(
        [
            _engine.zonal_mean(np.ascontiguousarray(fine[:, j]), zones, n)[0]
            for j in range(len(names))
        ]
    )
    ok = np.isfinite(y) & np.isfinite(coarse_x).all(axis=1)
    if ok.sum() < max(min_cells, len(names) + 2):
        raise ValueError(
            f"{layer.key}: only {int(ok.sum())} coarse cells with data; too few to fit "
            f"{len(names)} covariates (use a larger area)"
        )
    mu, sd = coarse_x[ok].mean(axis=0), coarse_x[ok].std(axis=0)
    sd[sd == 0] = 1.0
    design = np.column_stack([np.ones(ok.sum()), (coarse_x[ok] - mu) / sd])
    beta, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    fitted = design @ beta
    ss_tot = ((y[ok] - y[ok].mean()) ** 2).sum()
    ss_res = ((y[ok] - fitted) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    k, n_ok = len(names), int(ok.sum())
    if ss_tot > 0 and ss_res > 0:
        f_stat = ((ss_tot - ss_res) / k) / (ss_res / (n_ok - k - 1))
        p_value = float(stats.f.sf(f_stat, k, n_ok - k - 1))
    else:
        p_value = 0.0 if ss_tot > 0 else 1.0
    used = p_value < alpha
    if not used:
        beta = np.array([y[ok].mean(), *np.zeros(k)])

    pred = (beta[0] + ((fine - mu) / sd) @ beta[1:]).astype(np.float32)
    out = _engine.match_zone_means(np.ascontiguousarray(pred), zones, np.ascontiguousarray(y))
    coefficients = {
        "intercept": float(beta[0]),
        **{k: float(b) for k, b in zip(names, beta[1:], strict=True)},
    }
    return Downscaled(out.reshape(grid.shape), coefficients, r2, p_value, used, n_ok)


# --- coarsened target -----------------------------------------------------------------


def coarse_grid(grid: GridSpec, factor: int) -> GridSpec:
    """The grid with cells ``factor`` times larger, covering exactly the same area."""
    rows, cols = grid.shape
    if factor < 1 or rows % factor or cols % factor:
        raise ValueError(f"a {rows}x{cols} grid is not divisible into {factor}x{factor} blocks")
    return GridSpec(grid.crs, grid.res * factor, grid.left, grid.bottom, grid.right, grid.top)


def coarsen(values: np.ndarray, factor: int, kind: Kind, *, min_valid: float = 0.5) -> np.ndarray:
    """Bring a layer already on the fine grid up to the coarse grid (mean, or majority class)."""
    if kind is Kind.CONTINUOUS:
        return _engine.block_mean(np.ascontiguousarray(values, dtype=np.float32), factor, min_valid)
    return _engine.block_majority(np.ascontiguousarray(values, dtype=np.int32), factor)
