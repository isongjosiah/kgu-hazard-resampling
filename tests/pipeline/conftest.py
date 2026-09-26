"""A small, hand-made study area for pipeline tests (checks the plumbing, not results)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from affine import Affine
from pyproj import Transformer

from hazres.data.labels import AbsenceKind, LoadReport, VectorLabels
from hazres.data.predictors import load_predictors
from hazres.pipeline.config import Experiment

LEFT, BOTTOM = 4_000_000.0, 3_000_000.0
SIZE = 6_000.0  # a 6 km square


def _write(path: Path, data, transform, crs, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:  # fmt: skip
        dst.write(data, 1)


@pytest.fixture(scope="session")
def area(tmp_path_factory):
    root = tmp_path_factory.mktemp("area")
    rng = np.random.default_rng(7)
    m = 3_000  # margin around the square, in metres
    # 30 m elevation: rolling hills
    n = int((SIZE + 2 * m) / 30)
    xx, yy = np.meshgrid(np.arange(n) * 30.0, np.arange(n) * 30.0)
    dem = (80 * np.sin(xx / 700) * np.cos(yy / 900) + 0.02 * xx + 200).astype(np.float32)
    _write(root / "dem.tif", dem, Affine(30, 0, LEFT - m, 0, -30, BOTTOM + SIZE + m), "EPSG:3035")
    # 10 m land cover, 3 classes in bands
    n10 = int((SIZE + 2 * m) / 10)
    lc = (10 * (1 + (np.arange(n10)[None, :] // 400 % 3)) * np.ones((n10, 1))).astype(np.uint8)
    _write(root / "lc.tif", lc, Affine(10, 0, LEFT - m, 0, -10, BOTTOM + SIZE + m), "EPSG:3035", 0)
    # 250 m soil
    n250 = int((SIZE + 2 * m) / 250)
    soil = (rng.normal(300, 60, (n250, n250))).astype(np.int16)
    _write(root / "soil.tif", soil, Affine(250, 0, LEFT - m, 0, -250, BOTTOM + SIZE + m),
           "EPSG:3035", -32768)  # fmt: skip
    # ~1 km rainfall in lon/lat
    t = Transformer.from_crs(3035, 4326, always_xy=True)
    lon0, lat0 = t.transform(LEFT - m, BOTTOM + SIZE + m)
    lon1, lat1 = t.transform(LEFT + SIZE + m, BOTTOM - m)
    res = 1 / 120
    x0, y0 = np.floor(lon0 / res) * res - 2 * res, np.ceil(lat0 / res) * res + 2 * res
    nx, ny = int((lon1 - x0) / res) + 4, int((y0 - lat1) / res) + 4
    rain = (rng.normal(6000, 400, (ny, nx))).astype(np.int16)
    _write(root / "rain.tif", rain, Affine(res, 0, x0, 0, -res, y0), "EPSG:4326")

    reg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/predictors.yaml").read_text()
    )
    layers = reg["layers"]
    for key, path in (("elevation", "dem.tif"), ("land_cover", "lc.tif"), ("clay_0_5", "soil.tif"),
                      ("annual_precipitation", "rain.tif")):  # fmt: skip
        layers[key].update(access="file", path=path, status="open")
        layers[key].pop("tiles", None)
        layers[key].pop("url", None)
    (root / "predictors.yaml").write_text(yaml.safe_dump(reg))

    # labels: points every ~250 m, hazard more likely on steep ground
    px, py = np.meshgrid(LEFT + 125 + 250 * np.arange(24), BOTTOM + 125 + 250 * np.arange(24))
    px, py = px.ravel(), py.ravel()
    col = ((px - (LEFT - m)) / 30).astype(int)
    row = ((BOTTOM + SIZE + m - py) / 30).astype(int)
    gy, gx = np.gradient(dem, 30)
    steep = np.hypot(gx, gy)[row, col]
    p = 1 / (1 + np.exp(-(steep - np.quantile(steep, 0.8)) * 30))
    label = (rng.random(px.size) < p).astype(np.int8)
    feats = gpd.GeoDataFrame(
        {"sample_id": [f"p{i}" for i in range(px.size)], "label": label},
        geometry=gpd.points_from_xy(px, py), crs="EPSG:3035",
    )  # fmt: skip
    report = LoadReport("test", px.size, int(label.sum()), int((1 - label).sum()))
    labels = VectorLabels("test", feats, AbsenceKind.OBSERVED, report)
    return {"root": root, "labels": labels, "registry": load_predictors(root / "predictors.yaml")}


@pytest.fixture
def experiment():
    return Experiment(
        name="t",
        region={"name": "square", "bounds": [LEFT, BOTTOM, LEFT + SIZE, BOTTOM + SIZE]},
        labels="test",
        predictors={
            "fine": ["elevation", "slope", "tpi", "land_cover"],
            "coarse": ["clay_0_5", "annual_precipitation"],
        },
        coarsened_factor=33,
        downscale_covariates=["elevation", "slope"],
        chance={"n": 5, "scales": [1.0, 2.0]},
        folds={"block_m": 1500, "k": 3, "buffer_m": 100},
        importance_repeats=1,
    )
