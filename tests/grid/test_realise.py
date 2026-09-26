import numpy as np
import pytest
from affine import Affine

from hazres.data.predictors import Kind, Layer
from hazres.data.synthetic import make_synthetic
from hazres.grid.realise import (
    DEFAULT_SCALES,
    Detail,
    coarse_cell_size_m,
    estimate_detail,
    realise,
    zone_means,
)
from hazres.grid.spec import GridSpec


@pytest.fixture(scope="module")
def study():
    return make_synthetic(shape=(256, 256), n_samples=200)


def _up(study, name):
    f = study.spec(name).factor
    return np.kron(study.observed[name], np.ones((f, f)))


def test_every_version_gives_back_the_published_coarse_values(study):
    layer = study.layer("bedrock_depth")
    r = realise(layer, study.grid, n=4)
    n_zones = layer.values.size
    for v in r.values:
        np.testing.assert_allclose(zone_means(v, r.zones, n_zones), layer.values.ravel(), atol=1e-4)
    np.testing.assert_allclose(
        zone_means(r.base, r.zones, n_zones), layer.values.ravel(), atol=1e-4
    )


def test_versions_differ_only_by_the_stated_amount_of_detail(study):
    layer = study.layer("bedrock_depth")
    detail = Detail(0.7, 150.0, "given")
    r = realise(layer, study.grid, n=5, detail=detail)
    assert r.values.shape == (5, *study.grid.shape)
    for v in r.values:
        assert (v - r.base).std() == pytest.approx(0.7, rel=1e-3)
    assert not np.allclose(r.values[0], r.values[1])  # genuinely different versions


def test_given_the_true_detail_the_versions_look_like_the_truth(study):
    # the generator must be right when the assumption is right
    name = "bedrock_depth"
    up = _up(study, name)
    truth_within = (study.truth[name] - up).std()
    spec = study.spec(name)
    r = realise(
        study.layer(name), study.grid, n=5,
        detail=Detail(truth_within, spec.correlation_cells * study.cell_size, "given"),
    )  # fmt: skip
    realised = np.mean([(v - up).std() for v in r.values])
    assert realised == pytest.approx(truth_within, rel=0.1)
    # the truth itself is one of the believable versions: it gives back the coarse data
    np.testing.assert_allclose(
        zone_means(study.truth[name].astype(np.float32), r.zones, study.observed[name].size),
        study.observed[name].ravel(),
        atol=1e-5,
    )


def test_the_automatic_estimate_is_an_assumption_not_the_truth(study):
    # documented behaviour: for a rough layer the coarse data hide most of the
    # detail, so the estimate is well below the truth. That is why the check
    # always runs at several scales.
    name = "bedrock_depth"
    truth_within = (study.truth[name] - _up(study, name)).std()
    est = estimate_detail(study.layer(name), study.grid)
    assert 0 < est.sd < truth_within
    assert est.source == "auto"
    assert est.correlation_m == pytest.approx(0.5 * 32 * 30, rel=0.02)
    assert DEFAULT_SCALES == (0.5, 1.0, 2.0)
    assert est.scaled(2.0).sd == pytest.approx(2 * est.sd)


def test_same_seed_same_versions_different_seed_different(study):
    layer = study.layer("rainfall")
    a = realise(layer, study.grid, n=2, seed=1)
    b = realise(layer, study.grid, n=2, seed=1)
    c = realise(layer, study.grid, n=2, seed=2)
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.allclose(a.values, c.values)


def test_scale_zero_gives_the_base_every_time(study):
    r = realise(study.layer("rainfall"), study.grid, n=3, scale=0.0)
    for v in r.values:
        np.testing.assert_allclose(v, r.base, atol=1e-5)


def test_downscaled_base_with_covariates(study):
    layer = study.layer("rainfall")
    cov = {"slope": study.truth["slope"].astype(np.float32)}
    r = realise(layer, study.grid, n=2, covariates=cov)
    np.testing.assert_allclose(
        zone_means(r.values[0], r.zones, layer.values.size), layer.values.ravel(), atol=1e-4
    )


def test_class_layers_are_not_supported_yet(study):
    with pytest.raises(NotImplementedError):
        realise(study.layer("lithology"), study.grid)


def test_coarse_cell_size_in_metres_for_a_lon_lat_layer():
    grid = GridSpec("EPSG:3035", 30, 3_030_000, 1_755_000, 3_036_000, 1_761_000)
    layer = Layer(
        "p", Kind.CONTINUOUS, np.ones((60, 60), dtype=np.float32),
        Affine(1 / 120, 0, -5.0, 0, -1 / 120, 38.2), "EPSG:4326", "mm",
    )  # fmt: skip
    # 30 arc-seconds near 38°N: about 0.73 km east-west by 0.93 km north-south
    assert 700 < coarse_cell_size_m(layer, grid) < 950
