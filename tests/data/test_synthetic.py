import numpy as np
import pytest

from hazres.data.synthetic import LayerSpec, block_majority, block_mean, make_synthetic


@pytest.fixture(scope="module")
def study():
    return make_synthetic(shape=(256, 256), n_samples=3000, prevalence=0.05)


def test_same_seed_same_study(study):
    again = make_synthetic(shape=(256, 256), n_samples=3000, prevalence=0.05)
    np.testing.assert_array_equal(study.probability, again.probability)
    assert (study.labels.features["label"] == again.labels.features["label"]).all()


def test_prevalence_is_hit(study):
    assert study.probability.mean() == pytest.approx(0.05, abs=1e-6)
    assert study.labels.features["label"].mean() == pytest.approx(0.05, abs=0.015)


def test_observed_layers_are_exact_aggregates_of_the_truth(study):
    for s in study.layers:
        obs = study.observed[s.name]
        assert obs.shape == (256 // s.factor, 256 // s.factor)
        if s.factor == 1:
            np.testing.assert_array_equal(obs, study.truth[s.name])
        elif s.kind == "continuous":
            np.testing.assert_allclose(obs, block_mean(study.truth[s.name], s.factor))
        else:
            np.testing.assert_array_equal(
                obs, block_majority(study.truth[s.name], s.factor, s.n_classes)
            )


def test_rough_coarse_layer_loses_more_detail_than_smooth_one(study):
    # the design premise: bedrock_depth (rough) loses most of its detail when
    # coarsened, rainfall (smooth) very little
    def kept(name):
        s = study.spec(name)
        up = np.kron(study.observed[name], np.ones((s.factor, s.factor)))
        return np.corrcoef(up.ravel(), study.truth[name].ravel())[0, 1]

    assert kept("rainfall") > 0.9
    assert kept("bedrock_depth") < 0.5


def test_labels_sit_on_cell_centres(study):
    f = study.labels.features
    x, y = f.geometry.x.to_numpy(), f.geometry.y.to_numpy()
    col, row = ~study.transform @ (x, y)
    np.testing.assert_allclose(col, f["col"] + 0.5)
    np.testing.assert_allclose(row, f["row"] + 0.5)
    assert study.labels.crs.to_epsg() == 3035


def test_bad_factor_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        make_synthetic(shape=(100, 100), layers=(LayerSpec("a", "continuous", 32, 5.0, 1.0),))


def test_observed_transform_matches_native_cell_size(study):
    t = study.observed_transform("bedrock_depth")
    assert (t.a, -t.e) == (30.0 * 32, 30.0 * 32)
    assert (t.c, t.f) == (study.transform.c, study.transform.f)
