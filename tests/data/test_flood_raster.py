from pathlib import Path

import numpy as np
import pytest
import rasterio
from conftest import make_cfg
from rasterio.transform import from_origin

from hazres.data.inventories.flood_raster import load as load_floods

FLOODED = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
CLEAR = np.array([[90, 90, 90, 10], [90, 90, 90, 90]], dtype=np.uint8)
PERM = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
EXPECTED = np.array([[1, -1, 0, -1], [1, 0, -1, -1]], dtype=np.int8)  # last cell: no data


def _write(path: Path, names=("flooded", "clear_perc", "jrc_perm_water"), event_id="DFO_4321"):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=2,
        width=4,
        count=3,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(300_000, 800_000, 30, 30),
        nodata=255,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, arr in enumerate((FLOODED, CLEAR, PERM), start=1):
            arr = arr.copy()
            arr[1, 3] = 255
            dst.write(arr, i)
            dst.set_band_description(i, names[i - 1])
        dst.update_tags(id=event_id)


def _cfg(**options):
    return make_cfg(
        "global_flood_database",
        hazard="flood",
        loader="flood_raster",
        crs=None,
        real_absences="partial",
        files={"events": {"path": "gfd/*.tif"}},
        options={"min_clear_perc": 50, **options},
    )


def test_labels_follow_the_rules(tmp_path):
    _write(tmp_path / "gfd/e1.tif")
    out = load_floods(_cfg(), tmp_path)
    assert len(out.events) == 1
    ev = out.events[0]
    np.testing.assert_array_equal(ev.label, EXPECTED)
    assert ev.event_id == "DFO_4321"
    assert ev.cell_size == (30.0, 30.0)
    r = out.report
    assert (r.n_read, r.n_presence, r.n_absence) == (8, 2, 2)
    assert r.dropped == {
        "cells: no data": 1,
        "cells: permanent water": 2,
        "cells: too cloudy to tell": 1,
    }


def test_band_order_fallback(tmp_path):
    _write(tmp_path / "gfd/e1.tif", names=("", "", ""))
    with pytest.raises(KeyError, match="band_order"):
        load_floods(_cfg(), tmp_path)
    out = load_floods(_cfg(band_order=["flooded", "clear_perc", "jrc_perm_water"]), tmp_path)
    np.testing.assert_array_equal(out.events[0].label, EXPECTED)


def test_no_event_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_floods(_cfg(), tmp_path)
