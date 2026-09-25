"""Generic loader for point or polygon inventories in any GIS format.

Used for the Kahramanmaraş landslide polygons, the African gully heads
(De Geeter et al. 2023) and later Tier 3 inventories. Everything specific to a
dataset goes in its ``options`` in ``configs/inventories.yaml``:

``layer``
    Layer name, if the file has several.
``label_column`` / ``presence_values`` / ``absence_values``
    If the file marks presence and absence. Without ``label_column`` every
    feature is a presence.
``id_column``
    Feature ID. Without it, IDs are ``<dataset>-<row number>``.
``keep_columns``
    Extra attributes to carry through (e.g. landslide movement type).
``geometry``
    ``point``, ``polygon`` or ``any``; features of another type are dropped.

A ``footprint`` file, if the registry lists one, is the systematically mapped
area. For ``within_footprint`` datasets it is the only place absences may be
drawn, so the loader warns if it is missing.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from hazres.data.labels import ABSENCE, PRESENCE, AbsenceKind, LoadReport, VectorLabels
from hazres.data.registry import InventoryConfig, register

GEOMETRY_KINDS = {
    "point": {"Point"},
    "polygon": {"Polygon", "MultiPolygon"},
    "any": {"Point", "MultiPoint", "Polygon", "MultiPolygon"},
}


def read_vector(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        raise ValueError(
            f"{path.name} has no CRS. Refusing to guess: a file silently assumed to be in "
            "the wrong CRS puts every feature in the wrong place."
        )
    return gdf


def _to_crs(gdf: gpd.GeoDataFrame, cfg: InventoryConfig) -> gpd.GeoDataFrame:
    if cfg.crs:
        return gdf.to_crs(cfg.crs)
    if gdf.crs.is_geographic:
        raise ValueError(f"{cfg.key}: data are in a geographic CRS; set `crs` in the registry")
    return gdf


@register("vector")
def load(cfg: InventoryConfig, data_root: Path) -> VectorLabels:
    opts = cfg.options
    path = cfg.file("features", data_root)
    raw = read_vector(path, opts.get("layer"))
    n_read = len(raw)
    dropped: dict[str, int] = {}
    warnings: list[str] = []

    empty = raw.geometry.isna() | raw.geometry.is_empty
    if empty.any():
        dropped["empty geometry"] = int(empty.sum())
    gdf = raw[~empty].copy()

    allowed = GEOMETRY_KINDS[opts.get("geometry", "any")]
    wrong = ~gdf.geom_type.isin(allowed)
    if wrong.any():
        dropped[f"geometry not {opts.get('geometry', 'any')}"] = int(wrong.sum())
        gdf = gdf[~wrong]

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = shapely.make_valid(gdf.geometry[invalid].to_numpy())
        warnings.append(f"repaired {int(invalid.sum()):,} invalid geometries")

    label_col = opts.get("label_column")
    if label_col:
        if label_col not in gdf.columns:
            raise KeyError(f"{cfg.key}: label_column {label_col!r} not in {path.name}")
        codes = gdf[label_col].astype(str).str.strip()
        yes = {str(v) for v in opts.get("presence_values", [])}
        no = {str(v) for v in opts.get("absence_values", [])}
        if not yes:
            raise ValueError(f"{cfg.key}: label_column set but presence_values is empty")
        label = pd.Series(pd.NA, index=gdf.index, dtype="Int8")
        label[codes.isin(yes)] = PRESENCE
        label[codes.isin(no)] = ABSENCE
        for code, n in codes[label.isna()].value_counts().items():
            dropped[f"{label_col} code {code!r} (not presence or absence)"] = int(n)
        gdf = gdf[label.notna()]
        gdf["label"] = label[label.notna()].astype("int8")
    else:
        gdf["label"] = pd.Series(PRESENCE, index=gdf.index, dtype="int8")

    id_col = opts.get("id_column")
    if id_col:
        if id_col not in gdf.columns:
            raise KeyError(f"{cfg.key}: id_column {id_col!r} not in {path.name}")
        ids = gdf[id_col].astype(str)
        dup = ids.duplicated()
        if dup.any():
            dropped[f"duplicate {id_col} (first kept)"] = int(dup.sum())
            gdf, ids = gdf[~dup], ids[~dup]
        gdf["sample_id"] = ids
    else:
        gdf["sample_id"] = [f"{cfg.key}-{i}" for i in gdf.index]

    keep_cols = [c for c in opts.get("keep_columns", []) if c in gdf.columns]
    warnings += [
        f"keep_columns not in file: {c}"
        for c in opts.get("keep_columns", [])
        if c not in gdf.columns
    ]
    gdf = _to_crs(gdf[["sample_id", "label", *keep_cols, "geometry"]], cfg).reset_index(drop=True)

    footprint = None
    fp_path = cfg.optional_file("footprint", data_root)
    if fp_path is not None:
        fp = _to_crs(read_vector(fp_path), cfg)
        footprint = shapely.union_all(fp.geometry.to_numpy())
        outside = ~gdf.geometry.intersects(footprint)
        if outside.any():
            warnings.append(f"{int(outside.sum()):,} features lie outside the mapped footprint")
    elif cfg.real_absences is AbsenceKind.WITHIN_FOOTPRINT:
        warnings.append(
            "no mapped-area footprint found: absences cannot be placed until it is provided "
            "(files.footprint in the registry)"
        )

    report = LoadReport(
        cfg.key,
        n_read,
        int((gdf["label"] == PRESENCE).sum()),
        int((gdf["label"] == ABSENCE).sum()),
        dropped,
        warnings,
    )
    return VectorLabels(
        dataset=cfg.key,
        features=gdf,
        absence_kind=cfg.real_absences,
        report=report,
        footprint=footprint,
        meta={"source_file": path.name, "geometry_types": sorted(set(gdf.geom_type))},
    )
