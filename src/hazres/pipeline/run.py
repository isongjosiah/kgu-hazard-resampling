"""Run an experiment: tables, folds, every model on every variant, comparison, report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hazres import __version__, engine_version
from hazres.compare import compare
from hazres.data.predictors import load_predictors
from hazres.data.registry import load_inventory
from hazres.metrics.scores import scores
from hazres.models.run import fit_variant
from hazres.models.spatial_cv import spatial_folds
from hazres.pipeline.cache import DEFAULT_CACHE, cached_registry
from hazres.pipeline.config import Experiment
from hazres.pipeline.tables import Tables, build_tables


@dataclass
class RunPaths:
    predictors: Path = Path("configs/predictors.yaml")
    inventories: Path = Path("configs/inventories.yaml")
    data_root: Path = Path("data/raw")
    cache_root: Path = DEFAULT_CACHE
    out_root: Path = Path("outputs")


def run_experiment(exp: Experiment, paths: RunPaths | None = None, *, log=print) -> Path:
    paths = paths or RunPaths()
    out = paths.out_root / exp.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.json").write_text(exp.model_dump_json(indent=2))

    registry = cached_registry(exp, load_predictors(paths.predictors), paths.cache_root)
    labels = load_inventory(exp.labels, registry=paths.inventories, data_root=paths.data_root)
    log(labels.report.describe())

    tables = build_tables(exp, registry, labels, data_root=paths.data_root, log=log)
    tables.save(out / "tables")
    return run_models(exp, tables, out, log=log)


def run_models(exp: Experiment, tables: Tables, out: Path, *, log=print) -> Path:
    pts = tables.points
    y = pts["label"].to_numpy().astype(int)
    x, yc = pts["x"].to_numpy(), pts["y"].to_numpy()
    folds = spatial_folds(x, yc, y, block_m=exp.folds.block_m, k=exp.folds.k,
                          buffer_m=exp.folds.buffer_m, seed=exp.folds.seed)  # fmt: skip
    removed = sum(f.buffered_out for f in folds)
    log(f"{len(folds)} spatial folds; test sizes {[len(f.test) for f in folds]}; "
        f"buffer removed {removed:,} training points in total")  # fmt: skip

    method_names = tables.names("method")
    chance = {f"x{s:g}": tables.names("chance", s) for s in exp.chance.scales}
    summary = {"version": __version__, "engine": engine_version(), "models": {}}

    for model in exp.models:
        mdir = out / model
        mdir.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict] = {}
        for i, (name, X) in enumerate(tables.variants.items(), start=1):
            r = fit_variant(X, y, folds, model=model, categorical=tables.categorical,
                            importance_repeats=exp.importance_repeats)  # fmt: skip
            results[name] = {"scores": scores(y, r.oof, x, yc), "importance": r.importance,
                             "oof": r.oof}  # fmt: skip
            if i % 5 == 0 or i == len(tables.variants):
                log(f"  {model}: {i}/{len(tables.variants)} variants fitted")

        score_df = pd.DataFrame({n: r["scores"] for n, r in results.items()}).T
        score_df.index.name = "variant"
        score_df.to_csv(mdir / "scores.csv")
        imp_df = pd.DataFrame({n: r["importance"] for n, r in results.items()}).T
        imp_df.index.name = "variant"
        imp_df.to_csv(mdir / "importance.csv")
        pd.DataFrame({n: r["oof"] for n, r in results.items()}).assign(
            sample_id=pts["sample_id"].to_numpy(), label=y
        ).to_parquet(mdir / "oof.parquet")

        comparison = compare(results, method_names, chance, top_fraction=exp.top_fraction,
                             seed=exp.chance.seed)  # fmt: skip
        comparison.to_csv(mdir / "comparison.csv", index=False)
        summary["models"][model] = {
            "sensitive": comparison.loc[
                comparison.verdict == "sensitive", ["measure", "scale"]
            ].to_dict("records"),
        }
        (mdir / "report.md").write_text(report(exp, tables, score_df, imp_df, comparison, model))
        log(f"  {model}: report written to {mdir / 'report.md'}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return "–" if not np.isfinite(v) else f"{v:.3f}"
    return str(v)


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def report(exp, tables: Tables, score_df, imp_df, comparison, model: str) -> str:
    pts = tables.points
    methods = [n for n in score_df.index if n.startswith("method:")]
    head = score_df.loc[methods + (["coarsened"] if "coarsened" in score_df.index else [])]
    head = head[["auc", "pr_auc", "brier", "calibration_gap", "calibration_slope", "morans_i"]]
    head = head.reset_index()
    chance_rows = [n for n in score_df.index if n.startswith("chance:x1:")]
    top_factors = imp_df.loc[methods].apply(lambda s: ", ".join(s.nlargest(3).index), axis=1)
    rows, cols = tables.info["grid"]["shape"]
    scales = ", ".join(f"x{s:g}" for s in exp.chance.scales)
    lines = [
        f"# {exp.name}: {exp.region.name} ({model})",
        "",
        f"- Points: {len(pts):,} ({int(pts['label'].sum())} with the hazard)",
        f"- Grid: {rows:,} x {cols:,} cells of {tables.info['grid']['res']:g} m",
        f"- Fine layers (same in every run): {', '.join(tables.fine)}",
        f"- Coarse layers (converted each way): {', '.join(tables.coarse)}",
        f"- Random versions: {exp.chance.n} per strength, at {scales}",
        "",
        "## Is the conversion method's effect bigger than chance?",
        "",
        _table(
            comparison[
                [
                    "measure",
                    "scale",
                    "between_methods",
                    "chance_median",
                    "chance_95th",
                    "p_value",
                    "verdict",
                ]
            ]
        ),  # fmt: skip
        "",
        "## Scores under each method",
        "",
        _table(head),
        "",
        f"AUC across random versions at x1: {score_df.loc[chance_rows, 'auc'].min():.3f} to "
        f"{score_df.loc[chance_rows, 'auc'].max():.3f}."
        if chance_rows
        else "",
        "",
        "## Top three factors under each method",
        "",
        *[f"- {n}: {f}" for n, f in top_factors.items()],
        "",
        "## Downscaling fits",
        "",
        *[
            f"- {k}: R² {v['r2']:.2f}, p {v['p_value']:.3g}, terrain used: {v['used_covariates']}"
            for k, v in tables.info.get("downscaling", {}).items()
        ],  # fmt: skip
        "",
        "## Assumed fine detail (chance check)",
        "",
        *[
            f"- {k}: " + "; ".join(f"{s} sd {d['sd']:.3g}" for s, d in v.items())
            for k, v in tables.info.get("detail", {}).items()
        ],  # fmt: skip
        "",
    ]
    return "\n".join(lines)
