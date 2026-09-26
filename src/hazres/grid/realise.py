"""The chance check: random but equally believable fine-scale versions of a coarse layer.

A difference between conversion methods only counts if it is bigger than the
differences between versions of the data that the coarse layer cannot tell
apart. Each version made here:

1. starts from a smooth base that already gives every coarse cell its published
   average (bilinear, or downscaled when fine covariates are given, then shifted
   cell by cell so the averages are exact);
2. adds random fine-scale detail: a smooth random field with a chosen strength
   (``sd``) and grain (``correlation_m``), with its own average removed inside
   every coarse cell, then rescaled so its spread inside the cells is exactly
   ``sd``;
3. is shifted once more so every coarse cell's average is exactly the published
   value.

So every version agrees with the downloaded data. They differ only in detail
the coarse data cannot see.

**How much detail to add is an assumption, not something the data can tell
us.** For a rough layer, averaging to 1 km destroys the fine detail, and the
coarse numbers do not say how much there was. On the synthetic study the
automatic estimate below is about half the truth for the rough layer and
nearly double for the smooth one. So the strength is always reported, and the
check is always run at ``scales`` 0.5, 1 and 2 of it, to see whether the
verdict depends on it.

Class layers (e.g. rock type) are not supported yet.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from hazres import _engine
from hazres.data.predictors import Kind, Layer
from hazres.grid.fields import gaussian_field
from hazres.grid.methods import _geometry, downscale, resample, target_centres
from hazres.grid.spec import GridSpec

DEFAULT_SCALES = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Detail:
    """How much random fine-scale detail to add inside each coarse cell."""

    sd: float
    """Spread of the added detail inside a coarse cell, in the layer's units."""
    correlation_m: float
    """Distance over which the added detail stays similar, in metres."""
    source: str
    """How it was chosen: 'auto' or 'given'."""

    def scaled(self, factor: float) -> Detail:
        return Detail(self.sd * factor, self.correlation_m, f"{self.source} x{factor:g}")


def coarse_cell_size_m(layer: Layer, grid: GridSpec) -> float:
    """Approximate size of the layer's cells on the grid, in metres."""
    xs, ys = target_centres(grid, layer)
    zones = _engine.cell_index(_geometry(layer), layer.values.shape, xs, ys)
    covered = np.unique(zones[zones >= 0]).size
    if covered == 0:
        raise ValueError(f"{layer.key}: the layer does not cover the grid")
    area = (zones >= 0).sum() * grid.res**2 / covered
    return float(np.sqrt(area))


def estimate_detail(layer: Layer, grid: GridSpec) -> Detail:
    """A transparent default: detail as strong as the difference between neighbouring cells.

    ``sd`` is the spread of differences between adjacent coarse cells divided by
    sqrt(2), i.e. the variation one coarse cell away. ``correlation_m`` is half a
    coarse cell. Both are assumptions; see the module notes.
    """
    if layer.kind is not Kind.CONTINUOUS:
        raise NotImplementedError(f"{layer.key}: realisations for class layers are not built yet")
    v = layer.values
    diffs = np.concatenate([np.diff(v, axis=0).ravel(), np.diff(v, axis=1).ravel()])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 4:
        raise ValueError(f"{layer.key}: too few neighbouring coarse cells to estimate detail")
    sd = float(diffs.std() / np.sqrt(2))
    return Detail(sd, 0.5 * coarse_cell_size_m(layer, grid), "auto")


@dataclass
class Realisations:
    """Random fine-scale versions of one coarse layer, all consistent with it."""

    key: str
    values: np.ndarray
    """Shape (n, rows, cols) on the grid, or (n, n_points) if points were given."""
    base: np.ndarray
    detail: Detail
    seed: int
    zones: np.ndarray = field(repr=False)
    """For each grid cell (row-major), the coarse cell containing its centre, or -1."""

    @property
    def n(self) -> int:
        return self.values.shape[0]


def _seed(seed: int, key: str, i: int) -> np.random.Generator:
    # stable across machines and runs: same seed, layer and index, same field
    return np.random.default_rng([seed, zlib.crc32(key.encode()), i])


def realise(
    layer: Layer,
    grid: GridSpec,
    n: int = 10,
    *,
    detail: Detail | None = None,
    scale: float = 1.0,
    covariates: dict[str, np.ndarray] | None = None,
    seed: int = 20260926,
    points: tuple[np.ndarray, np.ndarray] | None = None,
) -> Realisations:
    """Make ``n`` fine-scale versions of ``layer`` on ``grid``.

    ``detail`` defaults to :func:`estimate_detail`; ``scale`` multiplies its
    strength (use 0.5, 1 and 2 to test the assumption). With ``covariates`` the
    base is the downscaled field instead of bilinear.

    With ``points=(rows, cols)`` only the values at those grid cells are kept,
    so ``values`` has shape (n, n_points): large grids never hold all versions
    in memory at once.
    """
    if layer.kind is not Kind.CONTINUOUS:
        raise NotImplementedError(f"{layer.key}: realisations for class layers are not built yet")
    if n < 1:
        raise ValueError("n must be at least 1")
    if scale < 0:
        raise ValueError("scale must be >= 0")
    detail = (detail or estimate_detail(layer, grid)).scaled(scale)

    rows, cols = grid.shape
    xs, ys = target_centres(grid, layer)
    zones = _engine.cell_index(_geometry(layer), layer.values.shape, xs, ys)
    target = np.ascontiguousarray(layer.values.ravel(), dtype=np.float64)
    n_zones = target.size

    if covariates:
        smooth = downscale(layer, grid, covariates).values
    else:
        smooth = resample(layer, grid, "bilinear")
    base = _engine.match_zone_means(
        np.ascontiguousarray(smooth.ravel(), dtype=np.float32), zones, target
    )

    corr_cells = detail.correlation_m / grid.res
    if points is not None:
        pr, pc = (np.asarray(a, dtype=np.int64) for a in points)
        out = np.empty((n, pr.size), dtype=np.float32)
    else:
        out = np.empty((n, rows, cols), dtype=np.float32)
    for i in range(n):
        noise = gaussian_field((rows, cols), corr_cells, _seed(seed, layer.key, i), pad=True)
        noise = np.ascontiguousarray(noise.ravel(), dtype=np.float32)
        # remove the noise's own average inside each coarse cell ...
        zero = np.zeros(n_zones, dtype=np.float64)
        within = _engine.match_zone_means(noise, zones, zero)
        spread = np.nanstd(within)
        if spread > 0 and detail.sd > 0:
            within = within * np.float32(detail.sd / spread)  # ... and set its spread exactly
        else:
            within = np.where(np.isnan(within), np.nan, 0.0).astype(np.float32)
        # the base already has the right averages and the detail averages zero,
        # but shift once more so float rounding cannot move them
        version = _engine.match_zone_means(
            np.ascontiguousarray(base + within, dtype=np.float32), zones, target
        )
        version = version.reshape(rows, cols)
        out[i] = version[pr, pc] if points is not None else version

    return Realisations(layer.key, out, base.reshape(rows, cols), detail, seed, zones)


def zone_means(values: np.ndarray, zones: np.ndarray, n_zones: int) -> np.ndarray:
    """Average of a grid field inside each coarse cell (for checking consistency)."""
    flat = np.ascontiguousarray(values.ravel(), dtype=np.float32)
    return _engine.zonal_mean(flat, zones, n_zones)[0]
