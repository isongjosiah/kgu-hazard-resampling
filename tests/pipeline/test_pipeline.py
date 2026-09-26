import numpy as np
import pandas as pd
import pytest

from hazres.compare import compare
from hazres.models.spatial_cv import spatial_folds
from hazres.pipeline.cache import cache_region, cached_registry, layers_needed
from hazres.pipeline.config import Experiment, load_experiment
from hazres.pipeline.run import run_models
from hazres.pipeline.tables import Tables, build_tables


def test_pilot_config_is_valid():
    from pathlib import Path

    exp = load_experiment(Path(__file__).resolve().parents[2] / "configs/experiments/pilot.yaml")
    assert exp.region.name == "central_sicily" and exp.labels == "ge_lucas"
    assert [m.value for m in exp.methods] == [
        "nearest",
        "bilinear",
        "cubic",
        "area_weighted",
        "downscaled",
    ]
    assert not exp.exclude_badlands and not exp.field_visits_only


def test_config_rejects_overlaps_and_bad_covariates(experiment):
    d = experiment.model_dump()
    d["predictors"]["coarse"].append("slope")
    with pytest.raises(ValueError, match="both fine and coarse"):
        Experiment(**d)
    d = experiment.model_dump()
    d["downscale_covariates"] = ["clay_0_5"]
    with pytest.raises(ValueError, match="must be fine layers"):
        Experiment(**d)


def test_cache_writes_native_layers_and_runs_read_them(area, experiment, tmp_path):
    reg = area["registry"]
    assert layers_needed(experiment, reg) == [
        "elevation",
        "land_cover",
        "clay_0_5",
        "annual_precipitation",
    ]
    res = cache_region(experiment, reg, data_root=area["root"], cache_root=tmp_path)
    assert [r.status for r in res] == ["cached"] * 4
    assert (tmp_path / "square" / "manifest.json").exists()
    again = cache_region(experiment, reg, data_root=area["root"], cache_root=tmp_path)
    assert {r.status for r in again} == {"present"}
    cached = cached_registry(experiment, reg, tmp_path)
    assert cached.layers["clay_0_5"].path.endswith("clay_0_5.tif")
    assert cached.layers["clay_0_5"].scale == 1.0


@pytest.fixture
def built(area, experiment):
    return build_tables(
        experiment, area["registry"], area["labels"], data_root=area["root"], log=lambda *_: None
    )


def test_tables_hold_one_thing_fixed_and_vary_the_other(built, experiment):
    t = built
    n_chance = experiment.chance.n * len(experiment.chance.scales)
    assert len(t.names("method")) == 5 and len(t.names("chance")) == n_chance
    assert "coarsened" in t.variants
    assert set(t.names("chance", 1.0)) == {f"chance:x1:{i:02d}" for i in range(5)}
    ref = t.variants["method:nearest"]
    for name, df in t.variants.items():
        assert len(df) == len(t.points)
        if name != "coarsened":
            # fine layers are identical in every method and chance variant
            pd.testing.assert_frame_equal(df[t.fine], ref[t.fine])
    # coarse layers differ between methods and between random versions
    assert not np.allclose(t.variants["method:bilinear"]["clay_0_5"], ref["clay_0_5"])
    assert not np.allclose(
        t.variants["chance:x1:00"]["clay_0_5"], t.variants["chance:x1:01"]["clay_0_5"]
    )
    # coarsened changes the fine layers too
    assert not np.allclose(t.variants["coarsened"]["slope"], ref["slope"])
    assert t.categorical == ["land_cover"]
    assert set(t.info["downscaling"]) == {"clay_0_5", "annual_precipitation"}
    assert t.info["detail"]["clay_0_5"]["x2"]["sd"] == pytest.approx(
        2 * t.info["detail"]["clay_0_5"]["x1"]["sd"]
    )


def test_tables_round_trip(built, tmp_path):
    built.save(tmp_path / "tables")
    back = Tables.load(tmp_path / "tables")
    assert list(back.variants) == list(built.variants)
    pd.testing.assert_frame_equal(back.variants["method:cubic"], built.variants["method:cubic"])


def test_spatial_folds_test_every_point_once_and_buffer():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 20_000, 800), rng.uniform(0, 20_000, 800)
    lab = (rng.random(800) < 0.1).astype(int)
    folds = spatial_folds(x, y, lab, block_m=2_000, k=5, buffer_m=500)
    tested = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(tested), np.arange(800))
    for f in folds:
        assert not set(f.train) & set(f.test)
        d = np.hypot(x[f.train][:, None] - x[f.test][None], y[f.train][:, None] - y[f.test][None])
        assert d.min() >= 500
        assert lab[f.test].sum() > 0
    again = spatial_folds(x, y, lab, block_m=2_000, k=5, buffer_m=500)
    assert all(np.array_equal(a.test, b.test) for a, b in zip(folds, again, strict=True))


def _fake(auc, imp, oof):
    return {"scores": {"auc": auc, "pr_auc": auc, "brier": 0.1, "calibration_gap": 0.0,
                       "calibration_slope": 1.0, "ece": 0.0, "morans_i": 0.0},
            "importance": pd.Series(imp, index=list("abcd")), "oof": np.asarray(oof)}  # fmt: skip


def test_compare_flags_real_differences_and_not_noise():
    rng = np.random.default_rng(1)
    base = rng.random(200)
    res = {}
    methods = []
    for i in range(4):  # methods disagree a lot
        res[f"m{i}"] = _fake(0.7 + 0.05 * i, rng.permutation([4, 3, 2, 1]), rng.random(200))
        methods.append(f"m{i}")
    pool = []
    for i in range(8):  # random versions barely differ
        res[f"c{i}"] = _fake(0.75 + 0.001 * i, [4, 3, 2, 1], base + 0.001 * rng.random(200))
        pool.append(f"c{i}")
    out = compare(res, methods, {"x1": pool})
    v = out.set_index("measure").verdict
    assert (
        v["auc"] == "sensitive"
        and v["top_set"] == "sensitive"
        and v["importance_ranking"] == "sensitive"
    )
    assert v["brier"] == "not computable" or v["brier"] == "not beyond chance"
    # identical groups: nothing is beyond chance
    same = compare(res, pool[:4], {"x1": pool})
    assert (same.verdict != "sensitive").all()
    with pytest.raises(ValueError, match="at least"):
        compare(res, methods, {"x1": pool[:3]})


def test_end_to_end_run_writes_the_report(built, experiment, tmp_path):
    out = run_models(experiment, built, tmp_path, log=lambda *_: None)
    mdir = out / "lightgbm"
    for f in ("scores.csv", "importance.csv", "oof.parquet", "comparison.csv", "report.md"):
        assert (mdir / f).exists(), f
    comp = pd.read_csv(mdir / "comparison.csv")
    assert set(comp.scale) == {"x1", "x2"}
    assert {"auc", "importance_ranking", "top_set"} <= set(comp.measure)
    scores = pd.read_csv(mdir / "scores.csv", index_col=0)
    assert scores.loc["method:nearest", "auc"] > 0.5
    assert (
        "Is the conversion method's effect bigger than chance?" in (mdir / "report.md").read_text()
    )
