"""Step 5: is the spread between conversion methods bigger than chance?

For each measure we compute one *spread* over a group of variants:

- a single number (AUC, PR-AUC, Brier, calibration, Moran's I): its range,
  largest minus smallest;
- which factors the model relies on: the average disagreement between pairs of
  importance rankings, 1 - Kendall's tau;
- the riskiest points: the average disagreement between pairs of top sets,
  1 - Jaccard overlap.

The spread across the methods is then compared with the spread across groups of
the same size drawn from the random versions (every such group if there are few
enough, otherwise a random sample). The p-value is the share of those groups
that spread at least as much as the methods do. A small p-value means the
conversion method changes the result by more than the natural uncertainty of
not knowing the fine detail.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from hazres.metrics.scores import jaccard, top_set

SCALAR_MEASURES = (
    "auc",
    "pr_auc",
    "brier",
    "calibration_gap",
    "calibration_slope",
    "ece",
    "morans_i",
)


def _range(values: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size >= 2 else float("nan")


def _importance_disagreement(imps: list[pd.Series]) -> float:
    taus = []
    for a, b in itertools.combinations(imps, 2):
        tau = kendalltau(a.to_numpy(), b.reindex(a.index).to_numpy()).statistic
        taus.append(1.0 - (tau if np.isfinite(tau) else 0.0))
    return float(np.mean(taus))


def _top_disagreement(preds: list[np.ndarray], fraction: float) -> float:
    tops = [top_set(p, fraction) for p in preds]
    return float(np.mean([1.0 - jaccard(a, b) for a, b in itertools.combinations(tops, 2)]))


def spreads(results: dict[str, dict], names: list[str], top_fraction: float) -> dict[str, float]:
    """The spread of every measure across the variants ``names``."""
    out = {m: _range([results[n]["scores"][m] for n in names]) for m in SCALAR_MEASURES}
    out["importance_ranking"] = _importance_disagreement([results[n]["importance"] for n in names])
    out["top_set"] = _top_disagreement([results[n]["oof"] for n in names], top_fraction)
    return out


def _groups(pool: list[str], size: int, max_groups: int, seed: int) -> list[tuple[str, ...]]:
    if size > len(pool):
        raise ValueError(f"need at least {size} random versions to compare with {size} methods")
    if math.comb(len(pool), size) <= max_groups:
        return list(itertools.combinations(pool, size))
    rng = np.random.default_rng(seed)
    return [tuple(rng.choice(pool, size=size, replace=False)) for _ in range(max_groups)]


def compare(
    results: dict[str, dict],
    methods: list[str],
    chance: dict[str, list[str]],
    *,
    top_fraction: float = 0.1,
    alpha: float = 0.05,
    max_groups: int = 2000,
    seed: int = 20260926,
) -> pd.DataFrame:
    """One row per (measure, scale): methods' spread, chance spreads, p-value, verdict.

    ``results[name]`` holds ``scores`` (dict), ``importance`` (Series) and ``oof``
    (array) for each variant. ``chance`` maps a scale label (e.g. "x1") to the
    names of its random versions.
    """
    observed = spreads(results, methods, top_fraction)
    rows = []
    for scale, pool in chance.items():
        null = [
            spreads(results, list(g), top_fraction)
            for g in _groups(pool, len(methods), max_groups, seed)
        ]
        for measure, obs in observed.items():
            dist = np.array([d[measure] for d in null], dtype=float)
            dist = dist[np.isfinite(dist)]
            if not np.isfinite(obs) or dist.size == 0:
                p, verdict = float("nan"), "not computable"
            else:
                p = float((dist >= obs).mean())
                verdict = "sensitive" if p < alpha else "not beyond chance"
            rows.append({
                "measure": measure,
                "scale": scale,
                "between_methods": obs,
                "chance_median": float(np.median(dist)) if dist.size else float("nan"),
                "chance_95th": float(np.percentile(dist, 95)) if dist.size else float("nan"),
                "p_value": p,
                "verdict": verdict,
                "n_groups": int(dist.size),
            })  # fmt: skip
    return pd.DataFrame(rows)
