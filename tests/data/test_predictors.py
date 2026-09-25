from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from hazres.data.predictors import (
    Kind,
    Layer,
    PredictorConfig,
    _tile_names,
    load_predictors,
    read_layer,
)
from hazres.grid.spec import GridSpec
from hazres.grid.warp import warp_to_grid

REPO = Path(__file__).resolve().parents[2]


def _cfg(**over) -> PredictorConfig:
    base = dict(
        key="x",
        name="x",
        kind="continuous",
        units="m",
        native_res="1 km",
        group="soil",
        mechanism="test",
        citation="test",
        licence="test",
        status="open",
        access="file",
        path="x.tif",
    )
    base.update(over)
    return PredictorConfig(**base)


def _write(path: Path, data: np.ndarray, transform, crs="EPSG:3035", nodata=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


# --- registry ---------------------------------------------------------------------


def test_repo_predictor_registry_is_valid():
    reg = load_predictors(REPO / "configs/predictors.yaml")
    assert reg.analysis_grid.crs == "EPSG:3035" and reg.analysis_grid.res == 30
    layers = reg.layers
    for key in ("elevation", "slope", "land_cover", "clay_0_5", "soil_thickness", "lithology"):
        assert key in layers
    assert all(c.mechanism.strip() for c in layers.values()), "every layer needs a mechanism"
    assert {c.derived_from for c in layers.values() if c.access == "derived"} == {"elevation"}
    # the study needs both kinds among the coarse layers
    coarse = [c for c in layers.values() if c.group in ("soil", "subsurface", "geology", "climate")]
    assert {c.kind for c in coarse} == {Kind.CONTINUOUS, Kind.CLASSES}


def test_location_is_required_unless_planned():
    with pytest.raises(ValueError, match="needs its location"):
        _cfg(path=None)
    assert _cfg(path=None, status="planned").status == "planned"


# --- tiles ------------------------------------------------------------------------


def test_copernicus_dem_tile_names():
    names = _tile_names("copernicus_dem", (-3.5, 40.2, -2.5, 40.8))
    assert names == [
        "Copernicus_DSM_COG_10_N40_00_W004_00_DEM/Copernicus_DSM_COG_10_N40_00_W004_00_DEM.tif",
        "Copernicus_DSM_COG_10_N40_00_W003_00_DEM/Copernicus_DSM_COG_10_N40_00_W003_00_DEM.tif",
    ]


def test_worldcover_tile_names():
    names = _tile_names("worldcover", (-5.0, 38.0, -1.0, 40.0))
    assert names == [
        "ESA_WorldCover_10m_2021_v200_N36W006_Map.tif",
        "ESA_WorldCover_10m_2021_v200_N36W003_Map.tif",
        "ESA_WorldCover_10m_2021_v200_N39W006_Map.tif",
        "ESA_WorldCover_10m_2021_v200_N39W003_Map.tif",
    ]


def test_tiles_are_merged_and_missing_tiles_skipped(tmp_path):
    # two 1°x1° tiles side by side (a third, over "sea", does not exist)
    for lon, value in ((-4, 100), (-3, 200)):
        stem = f"Copernicus_DSM_COG_10_N40_00_W{abs(lon):03d}_00_DEM"
        _write(
            tmp_path / "dem" / stem / f"{stem}.tif",
            np.full((10, 10), value, dtype=np.float32),
            from_origin(lon, 41, 0.1, 0.1),
            crs="EPSG:4326",
        )
    cfg = _cfg(
        key="elevation",
        access="tiles",
        path=None,
        tiles={"scheme": "copernicus_dem", "base": "dem"},
    )
    layer = read_layer(
        cfg, (-3.35, 40.3, -2.75, 40.7), "EPSG:4326", data_root=tmp_path, pad_cells=1
    )
    assert layer.meta["missing_tiles"] == 0
    left, _, right, _ = layer.bounds
    # snapped to the tiles' 0.1° cells, then one extra cell each side
    assert left == pytest.approx(-3.5) and right == pytest.approx(-2.6)
    assert set(np.unique(layer.values)) == {100.0, 200.0}

    layer = read_layer(cfg, (-3.35, 40.3, -1.5, 40.7), "EPSG:4326", data_root=tmp_path)
    assert layer.meta["missing_tiles"] == 1  # the W002 tile does not exist


# --- single files -------------------------------------------------------------------


def test_window_is_native_padded_scaled_and_masked(tmp_path):
    data = np.arange(100, dtype=np.int16).reshape(10, 10)
    data[5, 5] = -32768
    _write(tmp_path / "x.tif", data, from_origin(4_000_000, 3_010_000, 1000, 1000), nodata=-32768)
    cfg = _cfg(scale=0.1)
    # ask for 2 x 2 native cells, in the middle
    layer = read_layer(
        cfg,
        (4_004_000, 3_004_000, 4_006_000, 3_006_000),
        "EPSG:3035",
        data_root=tmp_path,
        pad_cells=2,
    )
    assert isinstance(layer, Layer)
    assert layer.values.shape == (6, 6)  # 2 cells + 2 on each side
    assert layer.res == (1000.0, 1000.0)  # still native: never resampled
    assert layer.values.dtype == np.float32
    assert np.isnan(layer.values[3, 3])  # the nodata cell
    assert layer.values[0, 0] == pytest.approx(0.1 * data[2, 2])
    assert "missing" in layer.describe()


def test_window_is_clipped_at_the_layer_edge(tmp_path):
    _write(
        tmp_path / "x.tif",
        np.ones((10, 10), dtype=np.float32),
        from_origin(4_000_000, 3_010_000, 1000, 1000),
    )
    layer = read_layer(
        _cfg(),
        (4_000_000, 3_009_000, 4_001_000, 3_010_000),
        "EPSG:3035",
        data_root=tmp_path,
        pad_cells=2,
    )
    assert layer.values.shape == (3, 3)


def test_classes_come_back_as_int_with_minus_one(tmp_path):
    data = np.array([[10, 20], [0, 40]], dtype=np.uint8)
    _write(tmp_path / "x.tif", data, from_origin(4_000_000, 3_000_020, 10, 10))
    layer = read_layer(
        _cfg(kind="classes", nodata=0),
        (4_000_000, 3_000_000, 4_000_020, 3_000_020),
        "EPSG:3035",
        data_root=tmp_path,
        pad_cells=0,
    )
    assert layer.values.dtype == np.int32
    assert layer.values.tolist() == [[10, 20], [-1, 40]]


def test_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="download by hand"):
        read_layer(_cfg(status="login"), (0, 0, 1, 1), "EPSG:3035", data_root=tmp_path)


def test_bounds_in_another_crs_are_transformed(tmp_path):
    # a lon/lat layer read with bounds given in LAEA metres
    _write(
        tmp_path / "x.tif",
        np.ones((100, 100), dtype=np.float32),
        from_origin(-5, 42, 0.01, 0.01),
        crs="EPSG:4326",
    )
    from rasterio.warp import transform_bounds

    b = transform_bounds("EPSG:4326", "EPSG:3035", -4.6, 41.6, -4.4, 41.8)
    layer = read_layer(_cfg(), b, "EPSG:3035", data_root=tmp_path, pad_cells=0)
    left, bottom, right, top = layer.bounds
    assert left <= -4.6 and right >= -4.4 and bottom <= 41.6 and top >= 41.8
    assert layer.crs.to_epsg() == 4326  # native CRS kept


def test_vector_layer(tmp_path):
    (tmp_path / "g").mkdir()
    gpd.GeoDataFrame(
        {"xx": ["sc", "mt"]},
        geometry=[
            box(4_000_000, 3_000_000, 4_010_000, 3_010_000),
            box(4_500_000, 3_500_000, 4_510_000, 3_510_000),
        ],
        crs="EPSG:3035",
    ).to_file(tmp_path / "g/glim.gpkg")
    cfg = _cfg(kind="classes", access="vector", path="g/glim.gpkg", class_column="xx")
    layer = read_layer(
        cfg, (4_001_000, 3_001_000, 4_002_000, 3_002_000), "EPSG:3035", data_root=tmp_path
    )
    assert list(layer.features["xx"]) == ["sc"]


# --- the grid and fine layers ----------------------------------------------------------


def test_grid_is_aligned_to_the_cell_size():
    g = GridSpec.covering((4_000_010, 3_000_005, 4_000_100, 3_000_061), "EPSG:3035", 30)
    assert g.bounds == (3_999_990, 3_000_000, 4_000_110, 3_000_090)
    assert g.shape == (3, 4)
    assert g.shape == (round((g.top - g.bottom) / 30), round((g.right - g.left) / 30))


def test_fine_layer_average_onto_grid_is_block_mean():
    fine = np.arange(36, dtype=np.float32).reshape(6, 6)
    layer = Layer(
        "f", Kind.CONTINUOUS, fine, from_origin(4_000_020, 3_000_070, 10, 10), "EPSG:3035", "m"
    )
    grid = GridSpec("EPSG:3035", 30, 4_000_020, 3_000_010, 4_000_080, 3_000_070)
    out = warp_to_grid(layer, grid, "average")
    expected = fine.reshape(2, 3, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(out, expected)


def test_class_layers_cannot_be_averaged():
    layer = Layer(
        "c",
        Kind.CLASSES,
        np.zeros((3, 3), dtype=np.int32),
        from_origin(0, 30, 10, 10),
        "EPSG:3035",
        "class",
    )
    grid = GridSpec("EPSG:3035", 30, 0, 0, 30, 30)
    with pytest.raises(ValueError, match="cannot be averaged"):
        warp_to_grid(layer, grid, "average")
    assert warp_to_grid(layer, grid, "mode").tolist() == [[0]]


def test_describe_uses_a_short_crs_name():
    from rasterio.crs import CRS

    homolosine = CRS.from_wkt(
        'PROJCS["Interrupted_Goode_Homolosine",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["Degree",0.0174532925199433]],PROJECTION["Interrupted_Goode_Homolosine"],'
        'UNIT["metre",1]]'
    )
    layer = Layer(
        "c",
        Kind.CONTINUOUS,
        np.ones((2, 2), dtype=np.float32),
        from_origin(0, 500, 250, 250),
        homolosine,
        "%",
    )
    assert "(Interrupted_Goode_Homolosine)" in layer.describe()
    layer = Layer(
        "e",
        Kind.CONTINUOUS,
        np.ones((2, 2), dtype=np.float32),
        from_origin(0, 1, 0.1, 0.1),
        CRS.from_epsg(4326),
        "m",
    )
    assert "(EPSG:4326)" in layer.describe()
