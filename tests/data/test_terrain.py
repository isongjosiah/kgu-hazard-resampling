from pathlib import Path

import numpy as np
import pytest

from hazres.data.terrain import aspect_components, curvatures, slope, terrain_layers, tpi

CELL = 30.0


def _xy(n=9):
    """Coordinates of cell centres: x east, y north (row 0 is the northern edge)."""
    idx = np.arange(n) * CELL
    x, y_down = np.meshgrid(idx, idx)
    return x, -y_down


def test_plane_slope_and_aspect():
    x, y = _xy()
    dem = 0.1 * x  # rises to the east, so the slope faces west
    s = slope(dem, CELL)
    assert np.isnan(s[0]).all() and np.isnan(s[:, -1]).all()  # edges
    np.testing.assert_allclose(s[1:-1, 1:-1], np.degrees(np.arctan(0.1)), rtol=1e-6)
    north, east = aspect_components(dem, CELL)
    np.testing.assert_allclose(east[1:-1, 1:-1], -1.0, atol=1e-12)
    np.testing.assert_allclose(north[1:-1, 1:-1], 0.0, atol=1e-12)

    north, _ = aspect_components(-0.1 * y, CELL)  # lower to the north: faces north
    np.testing.assert_allclose(north[1:-1, 1:-1], 1.0, atol=1e-12)


def test_plane_has_no_curvature_and_no_tpi():
    x, y = _xy()
    dem = 0.05 * x + 0.02 * y + 100
    prof, plan = curvatures(dem, CELL)
    np.testing.assert_allclose(prof[1:-1, 1:-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(plan[1:-1, 1:-1], 0.0, atol=1e-12)
    t = tpi(dem, 2)
    np.testing.assert_allclose(t[2:-2, 2:-2], 0.0, atol=1e-4)
    assert np.isnan(t[0, 0])


def test_bowl_curvature_signs():
    # z = (x^2 + y^2) / 1000: a bowl. Along any slope line it flattens downhill
    # (concave, positive profile) and flow converges towards the centre
    # (negative plan curvature).
    x, y = _xy(11)
    cx, cy = x - x.mean(), y - y.mean()
    dem = (cx**2 + cy**2) / 1000.0
    prof, plan = curvatures(dem, CELL)
    inner = (slice(1, -1), slice(1, -1))
    off_centre = (cx[inner] ** 2 + cy[inner] ** 2) > 0
    # second derivative of the bowl in any direction is 2/1000
    np.testing.assert_allclose(prof[inner][off_centre], 2 / 1000.0, rtol=1e-6)
    np.testing.assert_allclose(plan[inner][off_centre], -2 / 1000.0, rtol=1e-6)

    # a dome is the opposite: steepening downhill, flow spreading
    prof, plan = curvatures(-dem, CELL)
    np.testing.assert_allclose(prof[inner][off_centre], -2 / 1000.0, rtol=1e-6)
    np.testing.assert_allclose(plan[inner][off_centre], 2 / 1000.0, rtol=1e-6)
    assert (tpi(dem, 2)[3:-3, 3:-3][off_centre[2:-2, 2:-2]] < 0).all()


def test_terrain_layers_bundle():
    x, _ = _xy()
    out = terrain_layers(0.1 * x, CELL, tpi_radius_cells=1)
    assert set(out) == {
        "slope",
        "northness",
        "eastness",
        "profile_curvature",
        "plan_curvature",
        "tpi",
    }
    assert all(v.shape == (9, 9) and v.dtype == np.float32 for v in out.values())
    with pytest.raises(ValueError):
        terrain_layers(np.zeros((2, 2)), CELL)


def test_terrain_on_grid_from_a_local_dem(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from hazres.data.predictors import load_predictors
    from hazres.data.terrain import terrain_on_grid
    from hazres.grid.spec import GridSpec

    # a 30 m DEM in the analysis CRS, rising 0.1 m per m to the east
    x, _ = _xy(60)
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=60,
        width=60,
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_origin(4_000_000, 3_001_800, 30, 30),
    ) as dst:
        dst.write((0.1 * x).astype("float32"), 1)
    reg = load_predictors(Path(__file__).resolve().parents[2] / "configs/predictors.yaml")
    elev = reg.layers["elevation"].model_copy(
        update={"access": "file", "path": "dem.tif", "tiles": None}
    )
    reg.layers["elevation"] = elev
    grid = GridSpec("EPSG:3035", 30, 4_000_300, 3_000_300, 4_000_900, 3_000_900)
    out = terrain_on_grid(grid, reg, data_root=tmp_path, tpi_radius_cells=2)
    assert out["slope"].shape == grid.shape == (20, 20)
    np.testing.assert_allclose(out["slope"], np.degrees(np.arctan(0.1)), rtol=1e-4)
    np.testing.assert_allclose(out["eastness"], -1.0, atol=1e-5)
