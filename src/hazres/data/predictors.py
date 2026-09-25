"""Predictor layers, read exactly as published: native resolution, native CRS.

**This module never converts a layer to the analysis grid.** Converting coarse
layers to the grid is what the study tests, six different ways, and that
happens in ``hazres.grid``. Here we only read the part of each layer that
covers an area, plus a margin of whole native cells so that methods which look
at neighbouring cells (bilinear, area-weighted, downscaling) have them.

Layers are described in ``configs/predictors.yaml``. Each one is reached in
one of four ways:

``file``   a raster under the data root (downloaded by hand if it needs a login)
``url``    a single raster on the web, read window by window (no full download)
``tiles``  a set of web or local tiles named by latitude and longitude
           (Copernicus DEM, ESA WorldCover)
``vector`` a polygon map of classes (e.g. lithology), clipped to the area

A raster comes back as a :class:`Layer`: continuous layers as float32 with NaN
for missing values, class layers as int32 with -1 for missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import yaml
from affine import Affine
from pydantic import BaseModel, ConfigDict, model_validator
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from hazres.data.registry import DEFAULT_DATA_ROOT

DEFAULT_PREDICTORS = Path("configs/predictors.yaml")
Bounds = tuple[float, float, float, float]


class Kind(str, Enum):
    CONTINUOUS = "continuous"
    CLASSES = "classes"


class TileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme: Literal["copernicus_dem", "worldcover"]
    base: str
    """URL or path (relative to the data root) that tile names are appended to."""


class PredictorConfig(BaseModel):
    """One predictor layer in ``configs/predictors.yaml``."""

    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    kind: Kind
    units: str
    native_res: str
    """As published, for reporting (e.g. "30 m", "1 km", "30 arcsec")."""
    group: str
    """What it stands for: terrain, land_cover, soil, subsurface, geology, climate, human."""
    mechanism: str
    """Why it can matter for gullies. Reviewed by the group; no mechanism, no layer."""
    citation: str
    licence: str
    status: Literal["open", "login", "manual", "planned"]
    access: Literal["file", "url", "tiles", "vector", "derived"]
    path: str | None = None
    url: str | None = None
    tiles: TileSpec | None = None
    derived_from: str | None = None
    band: int = 1
    scale: float = 1.0
    offset: float = 0.0
    nodata: float | None = None
    class_column: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _access_has_location(self) -> PredictorConfig:
        need = {
            "file": self.path,
            "url": self.url,
            "tiles": self.tiles,
            "vector": self.path,
            "derived": self.derived_from,
        }[self.access]
        if need is None and self.status != "planned":
            raise ValueError(f"{self.key}: access '{self.access}' needs its location set")
        if self.access == "vector" and self.kind is Kind.CLASSES and not self.class_column:
            raise ValueError(f"{self.key}: vector class layers need class_column")
        return self


class AnalysisGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crs: str = "EPSG:3035"
    res: float = 30.0


class PredictorRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    analysis_grid: AnalysisGrid
    layers: dict[str, PredictorConfig]


def load_predictors(path: str | Path = DEFAULT_PREDICTORS) -> PredictorRegistry:
    raw = yaml.safe_load(Path(path).read_text())
    raw["layers"] = {k: {"key": k, **v} for k, v in (raw.get("layers") or {}).items()}
    reg = PredictorRegistry(**raw)
    for cfg in reg.layers.values():
        if cfg.derived_from and cfg.derived_from not in reg.layers:
            raise ValueError(f"{cfg.key}: derived_from {cfg.derived_from!r} is not a layer")
    return reg


# --- the layer object --------------------------------------------------------------


@dataclass
class Layer:
    """A window of one predictor, at its native resolution and in its native CRS."""

    key: str
    kind: Kind
    values: np.ndarray
    transform: Affine
    crs: Any
    units: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"{self.key}: values must be 2-D")
        want = np.float32 if self.kind is Kind.CONTINUOUS else np.int32
        if self.values.dtype != want:
            raise ValueError(f"{self.key}: {self.kind.value} values must be {want.__name__}")

    @property
    def res(self) -> tuple[float, float]:
        """Native cell size in the layer's own CRS units (metres or degrees)."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> Bounds:
        h, w = self.values.shape
        left, top = self.transform.c, self.transform.f
        return (left, top - h * self.res[1], left + w * self.res[0], top)

    @property
    def missing(self) -> np.ndarray:
        if self.kind is Kind.CONTINUOUS:
            return np.isnan(self.values)
        return self.values < 0

    def describe(self) -> str:
        h, w = self.values.shape
        crs = _crs_label(self.crs)
        lines = [
            f"{self.key}: {h:,} x {w:,} native cells, cell {self.res[0]:.6g} x {self.res[1]:.6g} "
            f"({crs}), {self.missing.mean():.1%} missing"
        ]
        valid = self.values[~self.missing]
        if valid.size and self.kind is Kind.CONTINUOUS:
            q = np.percentile(valid, [0, 50, 100])
            lines.append(f"  {self.units}: min {q[0]:.4g}, median {q[1]:.4g}, max {q[2]:.4g}")
        elif valid.size:
            classes, counts = np.unique(valid, return_counts=True)
            top = sorted(zip(counts, classes, strict=True), reverse=True)[:8]
            lines.append("  classes: " + ", ".join(f"{c} ({n / valid.size:.0%})" for n, c in top))
        return "\n".join(lines)


def _crs_label(crs) -> str:
    """Short CRS name for reports: the EPSG code if there is one, else the CRS name."""
    from pyproj import CRS

    try:
        c = CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    except Exception:  # an unusual CRS should not break a report
        return str(crs)[:60]
    code = c.to_epsg()
    return f"EPSG:{code}" if code else c.name


@dataclass
class VectorLayer:
    """A polygon class map (e.g. lithology), clipped to an area. Rasterised in the grid step."""

    key: str
    features: Any  # GeoDataFrame
    class_column: str
    units: str
    meta: dict[str, Any] = field(default_factory=dict)


# --- reading -----------------------------------------------------------------------


def _source(location: str, data_root: Path) -> str:
    if location.startswith(("http://", "https://", "s3://", "/vsi")):
        return location
    return str(data_root / location)


def _tile_names(scheme: str, lonlat_bounds: Bounds) -> list[str]:
    """Names of the tiles covering a lon/lat box, by each product's naming scheme."""
    west, south, east, north = lonlat_bounds
    step = {"copernicus_dem": 1, "worldcover": 3}[scheme]
    lats = range(math.floor(south / step) * step, math.ceil(north / step) * step, step)
    lons = range(math.floor(west / step) * step, math.ceil(east / step) * step, step)
    names = []
    for lat in lats:
        for lon in lons:
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            if scheme == "copernicus_dem":
                stem = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
                names.append(f"{stem}/{stem}.tif")
            else:
                names.append(
                    f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map.tif"
                )
    return names


def _padded_window(src, bounds: Bounds, pad_cells: int) -> Window | None:
    """Window covering ``bounds`` (in the source CRS), grown by whole cells and clipped."""
    win = from_bounds(*bounds, transform=src.transform)
    col0 = math.floor(win.col_off) - pad_cells
    row0 = math.floor(win.row_off) - pad_cells
    col1 = math.ceil(win.col_off + win.width) + pad_cells
    row1 = math.ceil(win.row_off + win.height) + pad_cells
    col0, row0 = max(col0, 0), max(row0, 0)
    col1, row1 = min(col1, src.width), min(row1, src.height)
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def _finish(cfg: PredictorConfig, raw: np.ndarray, nodata, transform, crs, meta) -> Layer:
    nodata = cfg.nodata if cfg.nodata is not None else nodata
    missing = np.zeros(raw.shape, dtype=bool) if nodata is None else raw == nodata
    if np.issubdtype(raw.dtype, np.floating):
        missing |= np.isnan(raw)
    if cfg.kind is Kind.CONTINUOUS:
        values = raw.astype(np.float32) * np.float32(cfg.scale) + np.float32(cfg.offset)
        values[missing] = np.nan
    else:
        values = raw.astype(np.int32)
        values[missing] = -1
    return Layer(cfg.key, cfg.kind, values, transform, crs, cfg.units, meta)


def read_layer(
    cfg: PredictorConfig,
    bounds: Bounds,
    bounds_crs: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    pad_cells: int = 2,
) -> Layer | VectorLayer:
    """Read the part of a layer covering ``bounds`` (given in ``bounds_crs``)."""
    data_root = Path(data_root)
    if cfg.status == "planned":
        raise NotImplementedError(f"{cfg.key}: planned layer, not available yet")
    if cfg.access == "derived":
        raise ValueError(f"{cfg.key}: derived layers are computed in hazres.data.terrain")
    if cfg.access == "vector":
        return _read_vector(cfg, bounds, bounds_crs, data_root)

    if cfg.access == "tiles":
        lonlat = transform_bounds(bounds_crs, "EPSG:4326", *bounds, densify_pts=21)
        names = _tile_names(cfg.tiles.scheme, lonlat)
        sources = [_source(f"{cfg.tiles.base.rstrip('/')}/{n}", data_root) for n in names]
    else:
        sources = [_source(cfg.path if cfg.access == "file" else cfg.url, data_root)]

    datasets, missing_tiles = [], []
    try:
        for s in sources:
            try:
                datasets.append(rasterio.open(s))
            except rasterio.errors.RasterioIOError:
                if cfg.access != "tiles":
                    raise FileNotFoundError(_missing(cfg, s)) from None
                missing_tiles.append(s)  # tiles over sea do not exist
        if not datasets:
            raise FileNotFoundError(f"{cfg.key}: no tiles found for this area")

        first = datasets[0]
        src_bounds = transform_bounds(bounds_crs, first.crs, *bounds, densify_pts=21)
        meta = {
            "sources": [d.name for d in datasets],
            "missing_tiles": len(missing_tiles),
            "native_res": cfg.native_res,
        }
        if len(datasets) == 1:
            win = _padded_window(first, src_bounds, pad_cells)
            if win is None:
                raise ValueError(f"{cfg.key}: the area is outside the layer")
            raw = first.read(cfg.band, window=win)
            transform = first.window_transform(win)
        else:
            # grow by whole cells, snapped to the tiles' own cell edges
            rx, ry = first.res
            x0, y0 = first.transform.c, first.transform.f

            def snap(v, origin, step, up):
                k = (v - origin) / step
                return origin + (math.ceil(k - 1e-9) if up else math.floor(k + 1e-9)) * step

            padded = (
                snap(src_bounds[0], x0, rx, False) - pad_cells * rx,
                snap(src_bounds[1], y0, ry, False) - pad_cells * ry,
                snap(src_bounds[2], x0, rx, True) + pad_cells * rx,
                snap(src_bounds[3], y0, ry, True) + pad_cells * ry,
            )
            stack, transform = merge(
                datasets, bounds=padded, indexes=[cfg.band], nodata=first.nodata
            )
            raw = stack[0]
        return _finish(cfg, raw, first.nodata, transform, first.crs, meta)
    finally:
        for d in datasets:
            d.close()


def _read_vector(cfg: PredictorConfig, bounds, bounds_crs, data_root: Path) -> VectorLayer:
    import geopandas as gpd

    path = data_root / cfg.path
    if not path.exists():
        raise FileNotFoundError(_missing(cfg, str(path)))
    file_crs = gpd.read_file(path, rows=0).crs
    if file_crs is None:
        raise ValueError(f"{cfg.key}: {path.name} has no CRS")
    bbox = transform_bounds(bounds_crs, file_crs, *bounds, densify_pts=21)
    gdf = gpd.read_file(path, bbox=bbox)
    if cfg.class_column not in gdf.columns:
        raise KeyError(f"{cfg.key}: class_column {cfg.class_column!r} not in {path.name}")
    return VectorLayer(cfg.key, gdf, cfg.class_column, cfg.units, {"source": path.name})


def _missing(cfg: PredictorConfig, where: str) -> str:
    how = {
        "login": "needs a free account; download by hand (see notes in configs/predictors.yaml)",
        "manual": "download by hand (see notes in configs/predictors.yaml)",
        "open": "check the URL or path in configs/predictors.yaml",
    }.get(cfg.status, "")
    return f"{cfg.key}: cannot open {where}\n  {how}"
