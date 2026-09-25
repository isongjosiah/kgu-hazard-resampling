"""Export Global Flood Database events for one country from Google Earth Engine.

The Global Flood Database (Tellman et al. 2021) is only published through Earth
Engine, collection ``GLOBAL_FLOOD_DB/MODIS_EVENTS/V1``. This module exports one
GeoTIFF per flood event that touches a country, clipped to that country, with
three bands in a fixed order: ``flooded``, ``clear_perc``, ``jrc_perm_water``.
The ``flood_raster`` loader reads these files.

Defaults:

* **250 m cells.** The floods were observed by MODIS at 250 m. Earth Engine
  serves them on a 30 m grid, but exporting at 30 m would only copy each
  250 m value about 70 times. How labels reach the 30 m analysis grid is
  decided later, in the grid step, as part of the resampling question.
* **UTM zone 32N (EPSG:32632)**, a projected CRS covering most of Nigeria.

One-time setup on your machine::

    uv sync --extra gee
    uv run earthengine authenticate
    # and a Google Cloud project registered for Earth Engine (free for research)

Two ways to export:

* ``download`` (default): straight into ``data/raw``. Works when an event is
  small enough for Earth Engine's direct-download limit.
* ``drive``: starts one export task per event into a Google Drive folder. Use
  it if direct download fails for size; then move the files into ``data/raw``.

Earth Engine calls are kept thin and take the ``ee`` module as an argument, so
the rest of this module can be tested without an Earth Engine account.
"""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

COLLECTION = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
BANDS = ("flooded", "clear_perc", "jrc_perm_water")
BOUNDARIES = "USDOS/LSIB_SIMPLE/2017"
EVENT_FIELDS = (
    "id",
    "began",
    "ended",
    "cc",
    "dfo_main_cause",
    "dfo_severity",
    "dfo_dead",
    "dfo_displaced",
)
Method = Literal["download", "drive"]


@dataclass(frozen=True)
class ExportJob:
    event_id: str
    filename: str
    path: Path


def event_filename(event_id: Any) -> str:
    """``DFO_4321.tif``. The loader uses the file name as the event ID."""
    text = str(event_id).strip()
    if text.endswith(".0"):  # Earth Engine returns numeric IDs as floats
        text = text[:-2]
    if not text or not text.replace("_", "").isalnum():
        raise ValueError(f"unusable event id {event_id!r}")
    return f"DFO_{text}.tif"


def plan_exports(events: list[dict[str, Any]], out_dir: Path) -> list[ExportJob]:
    jobs = []
    seen: set[str] = set()
    for ev in events:
        name = event_filename(ev["id"])
        if name in seen:
            raise ValueError(f"event {ev['id']} listed twice")
        seen.add(name)
        jobs.append(ExportJob(name.removesuffix(".tif"), name, out_dir / name))
    return jobs


def write_manifest(events: list[dict[str, Any]], path: Path) -> Path:
    """One row per event with its dates, cause and impact, from the Earth Engine metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", *EVENT_FIELDS])
        writer.writeheader()
        for ev in sorted(events, key=lambda e: str(e.get("began", ""))):
            writer.writerow(
                {"file": event_filename(ev["id"]), **{k: ev.get(k) for k in EVENT_FIELDS}}
            )
    return path


# --- Earth Engine calls --------------------------------------------------------


def initialise(project: str | None):
    try:
        import ee
    except ImportError as exc:
        raise ImportError(
            "Earth Engine support is not installed. Run: uv sync --extra gee"
        ) from exc
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # the ee client raises its own exception types
        raise RuntimeError(
            f"could not start Earth Engine ({exc}). Run `uv run earthengine authenticate` "
            "once, and pass --project with a Cloud project registered for Earth Engine."
        ) from exc
    return ee


def country_region(ee, country_name: str):
    fc = ee.FeatureCollection(BOUNDARIES).filter(ee.Filter.eq("country_na", country_name))
    if fc.size().getInfo() == 0:
        raise ValueError(f"no country named {country_name!r} in {BOUNDARIES}")
    return fc.geometry()


def list_events(ee, iso3: str) -> list[dict[str, Any]]:
    """Metadata for every event whose country list includes ``iso3``."""
    coll = ee.ImageCollection(COLLECTION).filter(ee.Filter.stringContains("cc", iso3))
    columns = {k: coll.aggregate_array(k) for k in EVENT_FIELDS}
    values = ee.Dictionary(columns).getInfo()
    n = len(values["id"])
    return [{k: values[k][i] for k in EVENT_FIELDS} for i in range(n)]


def _event_image(ee, event_id: Any, region):
    img = (
        ee.ImageCollection(COLLECTION)
        .filter(ee.Filter.eq("id", event_id))
        .first()
        .select(list(BANDS))
        .toUint8()
        .clip(region)
    )
    return img


def export_event(
    ee,
    raw_id: Any,
    job: ExportJob,
    region,
    *,
    scale: float,
    crs: str,
    method: Method,
    drive_folder: str,
) -> str:
    img = _event_image(ee, raw_id, region)
    if method == "drive":
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=job.event_id,
            folder=drive_folder,
            fileNamePrefix=job.event_id,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=10**10,
            fileFormat="GeoTIFF",
        )
        task.start()
        return f"drive task started ({drive_folder}/{job.filename})"

    url = img.getDownloadURL(
        {"region": region, "scale": scale, "crs": crs, "format": "GEO_TIFF", "filePerBand": False}
    )
    job.path.parent.mkdir(parents=True, exist_ok=True)
    part = job.path.with_name(job.path.name + ".part")
    with urllib.request.urlopen(url) as resp, part.open("wb") as out:  # noqa: S310
        out.write(resp.read())
    part.replace(job.path)
    return "downloaded"


def export_gfd(
    out_dir: Path,
    *,
    country_name: str = "Nigeria",
    iso3: str = "NGA",
    scale: float = 250.0,
    crs: str = "EPSG:32632",
    method: Method = "download",
    drive_folder: str = "hazres_gfd",
    project: str | None = None,
    skip_existing: bool = True,
    ee_module=None,
) -> list[tuple[ExportJob, str]]:
    """Export every flood event touching a country. Returns (job, status) per event."""
    ee = ee_module or initialise(project)
    region = country_region(ee, country_name)
    events = list_events(ee, iso3)
    if not events:
        raise ValueError(f"no Global Flood Database events list {iso3}")
    write_manifest(events, out_dir / "events.csv")

    results = []
    for ev, job in zip(events, plan_exports(events, out_dir), strict=True):
        if skip_existing and job.path.exists():
            results.append((job, "present"))
            continue
        try:
            status = export_event(
                ee,
                ev["id"],
                job,
                region,
                scale=scale,
                crs=crs,
                method=method,
                drive_folder=drive_folder,
            )
        except Exception as exc:  # report per event and carry on
            status = f"failed: {exc}"
            if method == "download":
                status += " (if it is too large, rerun with --method drive)"
        results.append((job, status))
    return results
