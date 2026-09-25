import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from hazres.data.predictors import Kind, Layer
from hazres.data.synthetic import make_synthetic
from hazres.grid.methods import (
    Method,
    coarse_grid,
    coarsen,
    downscale,
    methods_for,
    resample,
    target_centres,
)
from hazres.grid.spec import GridSpec


def _layer(values, transform, crs="EPSG:3035", kind=Kind.CONTINUOUS):
    dtype = np.float32 if kind is Kind.CONTINUOUS else np.int32
    return Layer("t", kind, np.asarray(values, dtype=dtype), transform, crs, "u")


# a 30 m grid 3 km across, and a 250 m coarse layer covering it with a margin
GRID = GridSpec("EPSG:3035", 30, 4_000_000, 3_000_000, 4_003_000, 3_003_000)


def _coarse(f, res=250.0, n=16, x0=3_999_500.0, y0=3_003_500.0, crs="EPSG:3035"):
    cx = x0 + (np.arange(n) + 0.5) * res
    cy = y0 - (np.arange(n) + 0.5) * res
    xx, yy = np.meshgrid(cx, cy)
    return _layer(f(xx, yy), Affine(res, 0, x0, 0, -res, y0), crs)


def test_every_method_returns_the_grid_shape():
    layer = _coarse(lambda x, y: (x - 4e6) / 100 + (y - 3e6) / 300)
    cov = {"c": np.random.default_rng(0).normal(size=GRID.shape).astype(np.float32)}
    for m in (
        Method.NEAREST,
        Method.BILINEAR,
        Method.CUBIC,
        Method.AREA_WEIGHTED,
        Method.DOWNSCALED,
    ):
        out = resample(layer, GRID, m, covariates=cov)
        assert out.shape == GRID.shape and out.dtype == np.float32
        assert np.isfinite(out).all(), m


def test_bilinear_and_cubic_reproduce_a_plane_nearest_steps():
    plane = lambda x, y: 0.01 * (x - 4e6) - 0.02 * (y - 3e6) + 5  # noqa: E731
    layer = _coarse(plane)
    rows, cols = GRID.shape
    x = GRID.left + (np.arange(cols) + 0.5) * 30
    y = GRID.top - (np.arange(rows) + 0.5) * 30
    truth = plane(*np.meshgrid(x, y))
    for m in (Method.BILINEAR, Method.CUBIC):
        np.testing.assert_allclose(resample(layer, GRID, m), truth, atol=1e-2)
    near = resample(layer, GRID, Method.NEAREST)
    assert np.abs(near - truth).max() > 1.0  # steps of up to half a coarse cell
    assert len(np.unique(near)) <= 13 * 13  # at most one value per coarse cell


def test_nearest_and_area_weighted_differ_only_at_coarse_cell_edges():
    rng = np.random.default_rng(1)
    layer = _coarse(lambda x, y: rng.normal(size=x.shape))
    near = resample(layer, GRID, Method.NEAREST)
    area = resample(layer, GRID, Method.AREA_WEIGHTED)
    differs = ~np.isclose(near, area)
    # 250 m cells, 30 m grid: a 30 m cell straddles an edge in about 1 of 8 rows/cols
    assert 0.05 < differs.mean() < 0.35
    assert not np.array_equal(near, area)


def test_layer_in_another_crs_goes_through_the_projection():
    # a lon/lat layer, linear in lon and lat: bilinear must reproduce it at the
    # lon/lat of each 30 m cell centre
    t = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(GRID.left, GRID.top)
    res = 0.01
    x0, y0 = np.floor(lon0 / res) * res - 5 * res, np.ceil(lat0 / res) * res + 5 * res
    f = lambda lon, lat: 3 * lon + 7 * lat  # noqa: E731
    layer = _coarse(f, res=res, n=20, x0=x0, y0=y0, crs="EPSG:4326")
    xs, ys = target_centres(GRID, layer)
    out = resample(layer, GRID, Method.BILINEAR)
    np.testing.assert_allclose(out.ravel(), f(xs, ys), atol=1e-4)


def test_class_layers_only_take_class_safe_methods():
    classes = _layer(
        np.arange(256).reshape(16, 16) % 5,
        Affine(250, 0, 3_999_500, 0, -250, 3_003_500),
        kind=Kind.CLASSES,
    )
    assert methods_for(Kind.CLASSES) == [
        Method.NEAREST,
        Method.AREA_WEIGHTED,
        Method.COARSENED_TARGET,
    ]
    for m in (Method.BILINEAR, Method.CUBIC, Method.DOWNSCALED):
        with pytest.raises(ValueError, match="average class codes"):
            resample(classes, GRID, m)
    for m in (Method.NEAREST, Method.AREA_WEIGHTED):
        out = resample(classes, GRID, m)
        assert out.dtype == np.int32 and set(np.unique(out)) <= {0, 1, 2, 3, 4}


def test_coarsened_target_is_a_grid_change():
    layer = _coarse(lambda x, y: x * 0 + 1)
    with pytest.raises(ValueError, match="changes the grid"):
        resample(layer, GRID, Method.COARSENED_TARGET)
    coarse = coarse_grid(GRID, 10)
    assert coarse.res == 300 and coarse.shape == (10, 10)
    fine = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    np.testing.assert_allclose(
        coarsen(fine, 10, Kind.CONTINUOUS), fine.reshape(10, 10, 10, 10).mean(axis=(1, 3))
    )
    with pytest.raises(ValueError):
        coarse_grid(GRID, 7)


# --- downscaling ----------------------------------------------------------------------


def test_downscaling_recovers_detail_the_covariate_explains_and_keeps_coarse_means():
    # the truth is 2 x a fine covariate + a smooth trend; the coarse layer is its
    # exact 300 m block mean, aligned with the 30 m grid
    rng = np.random.default_rng(2)
    rows, cols = GRID.shape
    cov = rng.normal(size=(rows, cols)).astype(np.float32)
    trend = np.add.outer(np.linspace(0, 3, rows), np.linspace(0, 1, cols)).astype(np.float32)
    truth = 2 * cov + trend
    coarse_vals = truth.reshape(10, 10, 10, 10).mean(axis=(1, 3))
    layer = _layer(coarse_vals, Affine(300, 0, GRID.left, 0, -300, GRID.top))

    res = downscale(layer, GRID, {"cov": cov})
    near = resample(layer, GRID, Method.NEAREST)
    corr = lambda a: np.corrcoef(a.ravel(), truth.ravel())[0, 1]  # noqa: E731
    assert corr(res.values) > 0.95 > corr(near)
    # averaged back over each coarse cell, it gives exactly the published values
    np.testing.assert_allclose(
        res.values.reshape(10, 10, 10, 10).mean(axis=(1, 3)), coarse_vals, atol=1e-4
    )
    assert set(res.coefficients) == {"intercept", "cov"}
    assert res.used_covariates and res.p_value < 0.05


def test_downscaling_with_an_irrelevant_covariate_stays_close_to_nearest():
    rng = np.random.default_rng(3)
    coarse_vals = rng.normal(size=(10, 10)).astype(np.float32)
    layer = _layer(coarse_vals, Affine(300, 0, GRID.left, 0, -300, GRID.top))
    res = downscale(layer, GRID, {"noise": rng.normal(size=GRID.shape).astype(np.float32)})
    near = resample(layer, GRID, Method.NEAREST)
    assert res.r2 < 0.2 and res.p_value > 0.05 and not res.used_covariates
    # no invented detail: exactly the coarse values
    np.testing.assert_allclose(res.values, near, atol=1e-5)


def test_downscaling_refuses_tiny_areas():
    layer = _layer(np.ones((2, 2)), Affine(1500, 0, GRID.left, 0, -1500, GRID.top))
    with pytest.raises(ValueError, match="too few"):
        downscale(layer, GRID, {"c": np.ones(GRID.shape, dtype=np.float32)})


# --- on the synthetic study, where the truth is known -------------------------------------


def test_synthetic_study_smooth_layer_survives_rough_layer_does_not():
    study = make_synthetic(shape=(256, 256), n_samples=500)
    grid = study.grid
    corr = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]  # noqa: E731
    for m in (Method.NEAREST, Method.BILINEAR, Method.CUBIC, Method.AREA_WEIGHTED):
        rain = resample(study.layer("rainfall"), grid, m)
        rock = resample(study.layer("bedrock_depth"), grid, m)
        assert corr(rain, study.truth["rainfall"]) > 0.95, m
        assert corr(rock, study.truth["bedrock_depth"]) < 0.6, m
    # and the methods give different fields for the rough layer
    a = resample(study.layer("bedrock_depth"), grid, Method.NEAREST)
    b = resample(study.layer("bedrock_depth"), grid, Method.CUBIC)
    assert np.abs(a - b).mean() > 0.05
    lith = resample(study.layer("lithology"), grid, Method.AREA_WEIGHTED)
    assert (lith == study.truth["lithology"]).mean() > 0.5
