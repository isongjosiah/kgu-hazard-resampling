"""Bring a layer that is already at or finer than the grid onto the analysis grid.

Only for the *reference* treatment of fine layers (elevation at 30 m, land
cover at 10 m), which is the same in every run so that exactly one thing
varies: how the *coarse* layers are converted. Coarse layers go through the
six methods in the Rust engine instead.
"""

from __future__ import annotations

import numpy as np
from rasterio.warp import Resampling, reproject

from hazres.data.predictors import Kind, Layer
from hazres.grid.spec import GridSpec

FINE_METHODS = {
    "bilinear": Resampling.bilinear,  # continuous, e.g. elevation
    "average": Resampling.average,  # continuous, finer than the grid
    "mode": Resampling.mode,  # classes, finer than the grid (majority)
    "nearest": Resampling.nearest,
}


def warp_to_grid(layer: Layer, grid: GridSpec, method: str) -> np.ndarray:
    if method not in FINE_METHODS:
        raise ValueError(f"method must be one of {sorted(FINE_METHODS)}")
    if layer.kind is Kind.CLASSES and method in ("bilinear", "average"):
        raise ValueError(f"{layer.key}: class layers cannot be averaged; use mode or nearest")
    if layer.kind is Kind.CONTINUOUS:
        dst = np.full(grid.shape, np.nan, dtype=np.float32)
        src_nodata = dst_nodata = np.nan
    else:
        dst = np.full(grid.shape, -1, dtype=np.int32)
        src_nodata = dst_nodata = -1
    reproject(
        source=layer.values,
        destination=dst,
        src_transform=layer.transform,
        src_crs=layer.crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=dst_nodata,
        resampling=FINE_METHODS[method],
    )
    return dst
