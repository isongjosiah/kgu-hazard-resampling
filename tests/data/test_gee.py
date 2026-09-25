"""Earth Engine export, tested against a stand-in for the ``ee`` module."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from conftest import make_cfg
from rasterio.transform import from_origin

from hazres.data.gee import event_filename, export_gfd, plan_exports, write_manifest
from hazres.data.inventories.flood_raster import load as load_floods

EVENTS = [
    {
        "id": 4321.0,
        "began": "2012-09-01",
        "ended": "2012-10-20",
        "cc": "NGA, CMR",
        "dfo_main_cause": "Heavy rain",
        "dfo_severity": 2,
        "dfo_dead": 363,
        "dfo_displaced": 2100000,
    },
    {
        "id": 3100.0,
        "began": "2007-08-01",
        "ended": "2007-08-30",
        "cc": "NGA",
        "dfo_main_cause": "Heavy rain",
        "dfo_severity": 1,
        "dfo_dead": 12,
        "dfo_displaced": 5000,
    },
]


def _tif(path: Path) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(300_000, 800_000, 250, 250),
    ) as dst:
        dst.write(np.array([[1, 0], [0, 0]], dtype=np.uint8), 1)
        dst.write(np.full((2, 2), 90, dtype=np.uint8), 2)
        dst.write(np.zeros((2, 2), dtype=np.uint8), 3)
    return path


class FakeEE:
    """Just enough of the ``ee`` API for export_gfd."""

    def __init__(self, source_tif: Path, fail_ids=()):
        self.source = source_tif
        self.fail_ids = set(fail_ids)
        self.drive_tasks: list[dict] = []
        fake = self

        class Filter:
            eq = staticmethod(lambda k, v: ("eq", k, v))
            stringContains = staticmethod(lambda k, v: ("contains", k, v))  # noqa: N815

        class Info:
            def __init__(self, v):
                self.v = v

            def getInfo(self):  # noqa: N802
                return self.v

        class FeatureCollection:
            def __init__(self, name):
                self.name = name

            def filter(self, f):
                self.country = f[2]
                return self

            def size(self):
                return Info(1 if self.country == "Nigeria" else 0)

            def geometry(self):
                return "NIGERIA"

        class Image:
            def __init__(self, event_id):
                self.event_id = event_id

            def select(self, bands):
                assert bands == ["flooded", "clear_perc", "jrc_perm_water"]
                return self

            def toUint8(self):  # noqa: N802
                return self

            def clip(self, region):
                assert region == "NIGERIA"
                return self

            def getDownloadURL(self, params):  # noqa: N802
                if self.event_id in fake.fail_ids:
                    raise RuntimeError("User memory limit exceeded")
                assert params["scale"] == 250.0 and params["format"] == "GEO_TIFF"
                return fake.source.as_uri()

        class ImageCollection:
            def __init__(self, name):
                self.filters = []

            def filter(self, f):
                self.filters.append(f)
                return self

            def aggregate_array(self, k):
                return [e[k] for e in EVENTS]

            def first(self):
                return Image(self.filters[-1][2])

        class Export:
            class image:  # noqa: N801
                @staticmethod
                def toDrive(**kw):  # noqa: N802
                    class Task:
                        def start(self):
                            fake.drive_tasks.append(kw)

                    return Task()

        class batch:  # noqa: N801
            pass

        batch.Export = Export
        self.Filter, self.FeatureCollection = Filter, FeatureCollection
        self.ImageCollection, self.batch = ImageCollection, batch
        self.Dictionary = lambda cols: Info(cols)


def test_event_filenames():
    assert event_filename(4321.0) == "DFO_4321.tif"
    assert event_filename("4321") == "DFO_4321.tif"
    with pytest.raises(ValueError):
        event_filename("../x")
    with pytest.raises(ValueError, match="twice"):
        plan_exports([{"id": 1}, {"id": 1.0}], Path("."))


def test_manifest_sorted_by_date(tmp_path):
    path = write_manifest(EVENTS, tmp_path / "events.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == ["DFO_3100.tif", "DFO_4321.tif"]
    assert rows[1]["dfo_dead"] == "363"


def test_download_export_feeds_the_loader(tmp_path):
    ee = FakeEE(_tif(tmp_path / "src.tif"))
    out = tmp_path / "raw/global_flood_database/nigeria"
    results = export_gfd(out, ee_module=ee)
    assert [(j.filename, s) for j, s in results] == [
        ("DFO_4321.tif", "downloaded"),
        ("DFO_3100.tif", "downloaded"),
    ]
    assert (out / "events.csv").exists()

    # a second run skips what is already there
    assert {s for _, s in export_gfd(out, ee_module=ee)} == {"present"}

    # and the loader reads the exported files
    cfg = make_cfg(
        "global_flood_database",
        hazard="flood",
        loader="flood_raster",
        crs=None,
        real_absences="partial",
        files={"events": {"path": "global_flood_database/nigeria/DFO_*.tif"}},
        options={"band_order": ["flooded", "clear_perc", "jrc_perm_water"]},
    )
    floods = load_floods(cfg, tmp_path / "raw")
    assert sorted(e.event_id for e in floods.events) == ["DFO_3100", "DFO_4321"]
    assert floods.report.n_presence == 2


def test_failures_are_reported_per_event(tmp_path):
    ee = FakeEE(_tif(tmp_path / "src.tif"), fail_ids={4321.0})
    results = dict((j.filename, s) for j, s in export_gfd(tmp_path / "out", ee_module=ee))
    assert results["DFO_3100.tif"] == "downloaded"
    assert (
        results["DFO_4321.tif"].startswith("failed") and "--method drive" in results["DFO_4321.tif"]
    )


def test_drive_method_starts_tasks(tmp_path):
    ee = FakeEE(_tif(tmp_path / "src.tif"))
    results = export_gfd(tmp_path / "out", ee_module=ee, method="drive", drive_folder="f")
    assert all(s.startswith("drive task started") for _, s in results)
    assert [t["fileNamePrefix"] for t in ee.drive_tasks] == ["DFO_4321", "DFO_3100"]
    assert all(t["crs"] == "EPSG:32632" and t["scale"] == 250.0 for t in ee.drive_tasks)


def test_unknown_country(tmp_path):
    with pytest.raises(ValueError, match="no country"):
        export_gfd(tmp_path, country_name="Narnia", ee_module=FakeEE(tmp_path / "x.tif"))
