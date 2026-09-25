"""Terrain layers computed from the 30 m elevation, on the analysis grid.

Elevation is the one layer that defines the grid, so it is brought onto the
grid once, the same way for every run (bilinear), and everything here is
computed from that. None of these layers is part of the resampling test.

Conventions (rows run north to south, columns west to east):

* ``slope``            degrees, Horn (1981).
* ``northness``        cos(aspect): +1 facing north, -1 facing south.
* ``eastness``         sin(aspect): +1 facing east, -1 facing west.
* ``profile_curvature`` along the slope, Zevenbergen & Thorne (1987), 1/m.
  Negative where the slope gets steeper downhill (convex), positive where it
  flattens (concave), which is where runoff slows and deposits.
* ``plan_curvature``   across the slope, 1/m. Negative where flow converges
  (hollows), positive where it spreads (spurs).
* ``tpi``              elevation minus the mean of a surrounding square, m.
  Negative in valleys and hollows.

Cells within one cell of the edge, or next to missing elevation, are NaN.

Flow accumulation and the topographic wetness index need a flow-routing
algorithm; they are left for the Rust engine.
"""

from __future__ import annotations

import numpy as np


def _neighbours(z: np.ndarray) -> tuple[np.ndarray, ...]:
    """z1..z9 of each interior cell, numbered row by row from the north-west."""
    return (
        z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:],
        z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:],
        z[2:, :-2], z[2:, 1:-1], z[2:, 2:],
    )  # fmt: skip


def _pad(inner: np.ndarray) -> np.ndarray:
    out = np.full((inner.shape[0] + 2, inner.shape[1] + 2), np.nan, dtype=np.float32)
    out[1:-1, 1:-1] = inner
    return out


def _gradients(dem: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn's dz/dx (east) and dz/dy (north)."""
    z1, z2, z3, z4, _, z6, z7, z8, z9 = _neighbours(dem.astype(np.float64))
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell)
    dzdy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8 * cell)
    return dzdx, dzdy


def slope(dem: np.ndarray, cell: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, cell)
    return _pad(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))


def aspect_components(dem: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """(northness, eastness). Flat cells get 0 for both."""
    dzdx, dzdy = _gradients(dem, cell)
    # the slope faces downhill, i.e. along minus the gradient
    fx, fy = -dzdx, -dzdy
    norm = np.hypot(fx, fy)
    flat = norm == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        north = np.where(flat, 0.0, fy / norm)
        east = np.where(flat, 0.0, fx / norm)
    return _pad(north), _pad(east)


def curvatures(dem: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """(profile, plan) curvature, Zevenbergen & Thorne. Flat cells get 0."""
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = _neighbours(dem.astype(np.float64))
    L2 = cell * cell  # noqa: N806
    d = ((z4 + z6) / 2 - z5) / L2
    e = ((z2 + z8) / 2 - z5) / L2
    f = (-z1 + z3 + z7 - z9) / (4 * L2)
    g = (z6 - z4) / (2 * cell)  # dz/dx
    h = (z2 - z8) / (2 * cell)  # dz/dy
    gh2 = g * g + h * h
    flat = gh2 == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        # second derivative along the slope line: > 0 where the slope flattens
        profile = np.where(flat, 0.0, 2 * (d * g * g + e * h * h + f * g * h) / gh2)
        # minus the second derivative across the slope: < 0 where flow converges
        plan = np.where(flat, 0.0, -2 * (d * h * h + e * g * g - f * g * h) / gh2)
    return _pad(profile), _pad(plan)


def tpi(dem: np.ndarray, radius_cells: int = 5) -> np.ndarray:
    """Elevation minus the mean of the (2r+1)^2 square around each cell."""
    if radius_cells < 1:
        raise ValueError("radius_cells must be at least 1")
    z = dem.astype(np.float64)
    k = 2 * radius_cells + 1
    padded = np.pad(z, radius_cells, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    mean = windows.mean(axis=(-1, -2))  # NaN wherever the square is incomplete
    return (z - mean).astype(np.float32)


def terrain_layers(
    dem: np.ndarray, cell: float, tpi_radius_cells: int = 5
) -> dict[str, np.ndarray]:
    """Every terrain layer from an elevation grid (metres, north-up, square cells)."""
    if dem.ndim != 2 or min(dem.shape) < 3:
        raise ValueError("elevation must be a 2-D grid at least 3 x 3")
    north, east = aspect_components(dem, cell)
    prof, plan = curvatures(dem, cell)
    return {
        "slope": slope(dem, cell),
        "northness": north,
        "eastness": east,
        "profile_curvature": prof,
        "plan_curvature": plan,
        "tpi": tpi(dem, tpi_radius_cells),
    }


def terrain_on_grid(grid, registry, *, data_root, tpi_radius_cells: int = 5):
    """Read elevation for a grid (plus a margin), put it on the grid, compute every terrain layer.

    Elevation goes onto the grid by bilinear interpolation, the same in every
    run. Returns ``{"elevation": ..., "slope": ..., ...}``, all on ``grid``.
    """
    from hazres.data.predictors import read_layer
    from hazres.grid.spec import GridSpec
    from hazres.grid.warp import warp_to_grid

    margin = (tpi_radius_cells + 1) * grid.res
    wide = GridSpec(
        grid.crs, grid.res,
        grid.left - margin, grid.bottom - margin, grid.right + margin, grid.top + margin,
    )  # fmt: skip
    dem = read_layer(registry.layers["elevation"], wide.bounds, grid.crs, data_root=data_root)
    z = warp_to_grid(dem, wide, "bilinear")
    layers = {"elevation": z, **terrain_layers(z, grid.res, tpi_radius_cells)}
    m = tpi_radius_cells + 1
    return {k: v[m:-m, m:-m] for k, v in layers.items()}
