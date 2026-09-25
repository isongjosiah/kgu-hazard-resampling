"""Global Flood Database v1 flood events, exported from Earth Engine as GeoTIFFs.

Tellman et al. 2021, *Nature*, doi:10.1038/s41586-021-03695-w (CC BY-NC 4.0).
Earth Engine collection ``GLOBAL_FLOOD_DB/MODIS_EVENTS/V1``. One GeoTIFF per
event, with the bands ``flooded``, ``clear_perc`` and ``jrc_perm_water``.

Each cell becomes:

* ``1`` flooded during the event, and not permanent water;
* ``0`` not flooded, not permanent water, and seen clearly often enough
  (``clear_perc`` at least ``options.min_clear_perc``);
* ``-1`` unknown: permanent water, too cloudy to tell, or no data.

The underlying observations are MODIS at 250 m, even though Earth Engine serves
them on a 30 m grid. So flood labels are coarse too, and how they are brought
to the analysis grid is part of the resampling question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from hazres.data.labels import (
    ABSENCE,
    PRESENCE,
    UNKNOWN,
    LoadReport,
    RasterEvents,
    RasterLabel,
)
from hazres.data.registry import InventoryConfig, register, resolve

BANDS = ("flooded", "clear_perc", "jrc_perm_water")


def _band_index(src: rasterio.DatasetReader, name: str, fallback: list[str] | None) -> int:
    descriptions = [d or "" for d in src.descriptions]
    if name in descriptions:
        return descriptions.index(name) + 1
    if fallback and name in fallback:
        return fallback.index(name) + 1
    raise KeyError(
        f"{src.name}: no band named {name!r} (band names: {descriptions}). "
        "Set options.band_order if the export lost the band names."
    )


def read_event(path: Path, min_clear_perc: float, band_order: list[str] | None = None):
    with rasterio.open(path) as src:
        bands = {name: src.read(_band_index(src, name, band_order), masked=True) for name in BANDS}
        transform, crs = src.transform, src.crs
        tags = src.tags()

    no_data = np.zeros(bands["flooded"].shape, dtype=bool)
    for arr in bands.values():
        no_data |= np.ma.getmaskarray(arr)
    flooded = bands["flooded"].filled(0) == 1
    permanent = bands["jrc_perm_water"].filled(0) == 1
    clear = bands["clear_perc"].filled(0) >= min_clear_perc

    label = np.full(flooded.shape, UNKNOWN, dtype=np.int8)
    label[flooded & ~permanent] = PRESENCE
    label[~flooded & ~permanent & clear] = ABSENCE
    label[no_data] = UNKNOWN

    counts = {
        "no data": int(no_data.sum()),
        "permanent water": int((permanent & ~no_data).sum()),
        "too cloudy to tell": int((~flooded & ~permanent & ~clear & ~no_data).sum()),
    }
    return label, transform, crs, tags, counts


@register("flood_raster")
def load(cfg: InventoryConfig, data_root: Path) -> RasterEvents:
    opts = cfg.options
    min_clear = float(opts.get("min_clear_perc", 50))
    spec = cfg.files["events"]
    paths = resolve(spec, data_root)
    if not paths:
        raise FileNotFoundError(
            f"{cfg.key}: no event files match {data_root / spec.path}\n  {spec.note or ''}"
        )

    events: list[RasterLabel] = []
    dropped: dict[str, int] = {}
    warnings: list[str] = []
    n_pos = n_neg = n_read = 0
    for path in paths:
        label, transform, crs, tags, counts = read_event(path, min_clear, opts.get("band_order"))
        if crs is None or crs.is_geographic:
            warnings.append(
                f"{path.name}: CRS is {'missing' if crs is None else 'geographic'}; "
                "reproject in the grid step before measuring distances or areas"
            )
        n_read += label.size
        n_pos += int((label == PRESENCE).sum())
        n_neg += int((label == ABSENCE).sum())
        for reason, n in counts.items():
            dropped[f"cells: {reason}"] = dropped.get(f"cells: {reason}", 0) + n
        if not (label == PRESENCE).any():
            warnings.append(f"{path.name}: no flooded cells")
        events.append(
            RasterLabel(
                event_id=tags.get("id", path.stem),
                label=label,
                transform=transform,
                crs=crs,
                meta={"file": path.name, **tags},
            )
        )

    report = LoadReport(cfg.key, n_read, n_pos, n_neg, dropped, warnings)
    return RasterEvents(
        dataset=cfg.key,
        events=events,
        absence_kind=cfg.real_absences,
        report=report,
        meta={"min_clear_perc": min_clear, "n_events": len(events)},
    )
