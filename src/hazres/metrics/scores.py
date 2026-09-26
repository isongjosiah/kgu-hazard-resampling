"""Scores for one set of out-of-fold predictions.

Every score is computed on predictions for points the model did not train on.

- ``auc``: does the model rank hazard points above the rest? (What almost every
  study reports.)
- ``pr_auc``: the same question, more honest when the hazard is rare.
- ``brier``: average squared gap between predicted risk and what happened.
- ``calibration_gap``: mean predicted risk minus the observed rate (too high > 0).
- ``calibration_slope`` / ``calibration_intercept``: logistic recalibration of
  the outcome on logit(prediction); slope 1 and intercept 0 are perfect. A slope
  below 1 means predictions are too extreme.
- ``ece``: expected calibration error over 10 equal-width risk bins.
- ``morans_i``: clustering of errors in space (residual y - p, 8 nearest
  neighbours). Near 0: errors are scattered; positive: they cluster.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EPS = 1e-6


def calibration_slope_intercept(
    y: np.ndarray, p: np.ndarray, max_iter: int = 50
) -> tuple[float, float]:
    """Fit logit P(y=1) = a + b * logit(p) by Newton-Raphson (no penalty)."""
    p = np.clip(p, EPS, 1 - EPS)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])  # noqa: N806
    beta = np.zeros(2)
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        hess = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(hess, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    if np.abs(beta).max() > 25:  # separation: not interpretable
        return float("nan"), float("nan")
    return float(beta[1]), float(beta[0])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def morans_i(values: np.ndarray, x: np.ndarray, y: np.ndarray, k: int = 8) -> float:
    """Moran's I with row-standardised k-nearest-neighbour weights."""
    n = values.size
    if n <= k:
        return float("nan")
    z = values - values.mean()
    denom = (z**2).sum()
    if denom == 0:
        return 0.0
    _, nbr = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=k + 1)
    lag = z[nbr[:, 1:]].mean(axis=1)
    return float((z * lag).sum() / denom)


def scores(y: np.ndarray, p: np.ndarray, x: np.ndarray, yc: np.ndarray) -> dict[str, float]:
    """All scalar scores. ``x``/``yc`` are point coordinates, for Moran's I."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_gap": float(p.mean() - y.mean()),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece": expected_calibration_error(y, p),
        "morans_i": morans_i(y - p, x, yc),
    }


def top_set(p: np.ndarray, fraction: float) -> np.ndarray:
    """Indices of the riskiest ``fraction`` of points (ties broken by index, stable)."""
    n = max(1, int(round(fraction * p.size)))
    return np.sort(np.argsort(-p, kind="stable")[:n])


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    union = len(sa | sb)
    return len(sa & sb) / union if union else 1.0
