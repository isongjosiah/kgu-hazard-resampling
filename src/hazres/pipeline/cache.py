"""Save the predictor layers for a region, once, so runs never need the internet.

``hazres data cache --experiment configs/experiments/pilot.yaml`` reads every
layer the experiment uses for its region (plus a margin) exactly as published
and writes it to ``data/raw/cache/<region>/<layer>.tif``, at native resolution
and in native CRS, with a ``manifest.json`` recording where each came from.
Runs then read the cached files instead of the web.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from hazres.data.predictors import Kind, Layer, PredictorRegistry, VectorLayer, read_layer
from hazres.pipeline.config import Experiment

DEFAULT_CACHE = Path("data/raw/cache")
MARGIN_M = 3_000.0


@dataclass
class CacheResult:
    key: str
    status: str  # "cached", "present", "skipped"
    path: Path | None
    note: str = ""


def layers_needed(exp: Experiment, reg: PredictorRegistry) -> list[str]:
    """Every layer to read: the listed ones, plus the sources of derived layers."""
    keys: list[str] = []
    for key in [*exp.predictors.fine, *exp.predictors.coarse]:
        if key not in reg.layers:
            raise KeyError(f"layer {key!r} is not in the predictor registry")
        cfg = reg.layers[key]
        src = cfg.derived_from if cfg.access == "derived" else key
        if src not in keys:
            keys.append(src)
    return keys


def region_dir(exp: Experiment, cache_root: Path) -> Path:
    return cache_root / exp.region.name


def padded_bounds(exp: Experiment, margin: float = MARGIN_M) -> tuple[float, float, float, float]:
    b = exp.region.bounds
    return (b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)


def write_layer(layer: Layer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = np.nan if layer.kind is Kind.CONTINUOUS else -1
    crs = layer.crs if not hasattr(layer.crs, "to_wkt") else layer.crs.to_wkt()
    with rasterio.open(
        path, "w", driver="GTiff", height=layer.values.shape[0], width=layer.values.shape[1],
        count=1, dtype=layer.values.dtype, crs=crs, transform=layer.transform, nodata=nodata,
        compress="deflate", tiled=True,
    ) as dst:  # fmt: skip
        dst.write(layer.values, 1)


def cache_region(
    exp: Experiment,
    reg: PredictorRegistry,
    *,
    data_root: Path,
    cache_root: Path = DEFAULT_CACHE,
    force: bool = False,
) -> list[CacheResult]:
    out_dir = region_dir(exp, cache_root)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    bounds = padded_bounds(exp)
    results = []
    for key in layers_needed(exp, reg):
        cfg = reg.layers[key]
        suffix = ".gpkg" if cfg.access == "vector" else ".tif"
        path = out_dir / f"{key}{suffix}"
        if path.exists() and not force:
            results.append(CacheResult(key, "present", path))
            continue
        try:
            layer = read_layer(cfg, bounds, exp.region.crs, data_root=data_root)
        except (FileNotFoundError, NotImplementedError) as exc:
            results.append(CacheResult(key, "skipped", None, str(exc).splitlines()[0]))
            continue
        if isinstance(layer, VectorLayer):
            path.parent.mkdir(parents=True, exist_ok=True)
            layer.features.to_file(path)
            sources = [layer.meta.get("source")]
        else:
            write_layer(layer, path)
            sources = layer.meta.get("sources", [])
        manifest[key] = {
            "sources": sources,
            "native_res": cfg.native_res,
            "units": cfg.units,
            "kind": cfg.kind.value,
            "bounds": list(bounds),
            "bounds_crs": exp.region.crs,
            "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        results.append(CacheResult(key, "cached", path))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return results


def cached_registry(exp: Experiment, reg: PredictorRegistry, cache_root: Path = DEFAULT_CACHE):
    """A copy of the registry that reads each cached layer from disk.

    Cached rasters already hold final values (scaled, missing as NaN or -1), so
    scale, offset and nodata are reset. Layers with no cached file are left as
    they are.
    """
    out_dir = region_dir(exp, cache_root)
    layers = {}
    for key, cfg in reg.layers.items():
        tif, gpkg = out_dir / f"{key}.tif", out_dir / f"{key}.gpkg"
        if tif.exists():
            cfg = cfg.model_copy(update={
                "access": "file", "path": str(tif.resolve()), "url": None, "tiles": None,
                "scale": 1.0, "offset": 0.0, "nodata": None, "status": "open",
            })  # fmt: skip
        elif gpkg.exists():
            cfg = cfg.model_copy(update={"path": str(gpkg.resolve()), "status": "open"})
        layers[key] = cfg
    return reg.model_copy(update={"layers": layers})
