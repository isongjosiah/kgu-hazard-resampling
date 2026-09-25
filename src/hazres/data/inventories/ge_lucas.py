"""GE-LUCAS v1.1: gully presence and absence at LUCAS 2022 survey points.

Borrelli et al. 2025, *Scientific Data*, doi:10.1038/s41597-025-05074-w (CC BY 4.0).

Reads ``Data 1 - LUCAS2022 original`` (a CSV of 399,591 points, 307 columns,
zipped or unzipped). Each point was assessed for gully channels visible from
the point, so a 0 is an *observed* absence, which is why this dataset can
support calibration results.

The CSV's record descriptor (``LUCAS-2022-record-descriptor.ods``, inside
the zip) defines the gully question as ``SURVEY_GULLY_SIGNS``, "Can you see
signs of gully erosion?", 1 = Yes, 2 = No. The registry sets this explicitly.
Six columns start with ``SURVEY_GULLY_``, so the column is never guessed from
a pattern when that name is present.

The loader still refuses to run until ``options.presence_values`` and
``options.absence_values`` are set, and prints the codes it found, so a change
in a future release of the file cannot slip through. Values in neither list
are dropped and counted.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from hazres.data.labels import ABSENCE, PRESENCE, LoadReport, VectorLabels
from hazres.data.registry import InventoryConfig, register

ID_COL = "POINT_ID"
LAT_COL = "POINT_LAT"
LON_COL = "POINT_LONG"
PRESENCE_COLUMN = "SURVEY_GULLY_SIGNS"
PRESENCE_PATTERN = re.compile(r"^SURVEY_GUL", re.IGNORECASE)


class CodingNotConfirmedError(ValueError):
    """Raised until someone has confirmed what the presence codes mean."""


def _open_csv(path: Path) -> io.BufferedIOBase | Path:
    if path.suffix.lower() != ".zip":
        return path
    zf = zipfile.ZipFile(path)
    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise ValueError(f"{path}: expected one CSV inside the zip, found {csvs}")
    return zf.open(csvs[0])


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(_open_csv(path), encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: could not decode as UTF-8 or Latin-1")


def _presence_column(columns: list[str], override: str | None) -> str:
    if override:
        if override not in columns:
            raise KeyError(f"presence_column {override!r} not in CSV columns")
        return override
    if PRESENCE_COLUMN in columns:
        return PRESENCE_COLUMN
    hits = [c for c in columns if PRESENCE_PATTERN.match(c)]
    if len(hits) != 1:
        raise KeyError(
            f"could not identify the gully presence column (candidates: {hits or 'none'}). "
            "Set options.presence_column in configs/inventories.yaml."
        )
    return hits[0]


def _norm(values) -> set[str]:
    return {str(v).strip() for v in values}


@register("ge_lucas")
def load(cfg: InventoryConfig, data_root: Path) -> VectorLabels:
    opts = cfg.options
    path = cfg.file("points", data_root)

    header = _read_csv(path, nrows=0).columns.tolist()
    for col in (ID_COL, LAT_COL, LON_COL):
        if col not in header:
            raise KeyError(f"{cfg.key}: column {col} missing from {path.name}")
    presence_col = _presence_column(header, opts.get("presence_column"))
    keep_cols = [c for c in opts.get("keep_columns", []) if c in header]
    warnings = [
        f"keep_columns not in CSV: {c}" for c in opts.get("keep_columns", []) if c not in header
    ]

    df = _read_csv(
        path,
        usecols=[ID_COL, LAT_COL, LON_COL, presence_col, *keep_cols],
        dtype={presence_col: str, ID_COL: str},
    )
    n_read = len(df)
    codes = df[presence_col].fillna("<missing>").str.strip()

    yes = _norm(opts.get("presence_values", []))
    no = _norm(opts.get("absence_values", []))
    if not yes or not no:
        counts = codes.value_counts().to_string()
        raise CodingNotConfirmedError(
            f"{cfg.key}: confirm how {presence_col!r} is coded before loading.\n"
            f"Codes found:\n{counts}\n"
            "Set options.presence_values and options.absence_values in "
            "configs/inventories.yaml. Anything in neither list is dropped as not assessed."
        )
    if yes & no:
        raise ValueError(f"{cfg.key}: codes {sorted(yes & no)} listed as both presence and absence")

    dropped: dict[str, int] = {}
    label = pd.Series(np.nan, index=df.index)
    label[codes.isin(yes)] = PRESENCE
    label[codes.isin(no)] = ABSENCE
    for code, n in codes[label.isna()].value_counts().items():
        dropped[f"gully code {code!r} (not presence or absence)"] = int(n)

    lat = pd.to_numeric(df[LAT_COL], errors="coerce")
    lon = pd.to_numeric(df[LON_COL], errors="coerce")
    bad_xy = lat.isna() | lon.isna() | ~lat.between(-90, 90) | ~lon.between(-180, 180)
    n_bad_xy = int((bad_xy & label.notna()).sum())
    if n_bad_xy:
        dropped["missing or invalid coordinates"] = n_bad_xy

    dup = df[ID_COL].duplicated(keep="first")
    n_dup = int((dup & label.notna() & ~bad_xy).sum())
    if n_dup:
        dropped["duplicate POINT_ID (first kept)"] = n_dup

    keep = label.notna() & ~bad_xy & ~dup
    gdf = gpd.GeoDataFrame(
        {
            "sample_id": df.loc[keep, ID_COL].astype(str).to_numpy(),
            "label": label[keep].astype("int8").to_numpy(),
            **{c: df.loc[keep, c].to_numpy() for c in keep_cols},
        },
        geometry=gpd.points_from_xy(lon[keep], lat[keep]),
        crs="EPSG:4326",
    ).to_crs(cfg.crs or "EPSG:3035")

    n_pos = int((gdf["label"] == PRESENCE).sum())
    n_neg = int((gdf["label"] == ABSENCE).sum())
    expected = opts.get("expected_presences")
    if expected is not None and n_pos != expected:
        warnings.append(
            f"{n_pos:,} gully points, but the paper reports {expected:,}. "
            "Check the presence coding before using these labels."
        )

    report = LoadReport(cfg.key, n_read, n_pos, n_neg, dropped, warnings)
    return VectorLabels(
        dataset=cfg.key,
        features=gdf.reset_index(drop=True),
        absence_kind=cfg.real_absences,
        report=report,
        meta={"presence_column": presence_col, "source_file": path.name},
    )
