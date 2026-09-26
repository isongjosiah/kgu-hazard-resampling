"""Fit one model on one variant: out-of-fold predictions and factor importance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from hazres.models.registry import make_model
from hazres.models.spatial_cv import Fold


@dataclass
class VariantResult:
    oof: np.ndarray
    """Out-of-fold predicted risk for every point."""
    importance: pd.Series
    """Out-of-fold permutation importance: rise in log loss when a factor is shuffled."""


def fit_variant(
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    folds: list[Fold],
    *,
    model: str,
    categorical: list[str],
    seed: int = 20260926,
    importance_repeats: int = 5,
) -> VariantResult:
    features = list(X.columns)
    oof = np.full(len(y), np.nan)
    imp_sum = pd.Series(0.0, index=features)
    weight = 0
    for f, fold in enumerate(folds):
        m = make_model(model, features, categorical, seed=seed)
        m.fit(X.iloc[fold.train], y[fold.train])
        Xt, yt = X.iloc[fold.test], y[fold.test]  # noqa: N806
        p = m.predict_proba(Xt)[:, 1]
        oof[fold.test] = p
        base = log_loss(yt, p, labels=[0, 1])
        rng = np.random.default_rng([seed, f])
        for col in features:
            rises = []
            for _ in range(importance_repeats):
                shuffled = Xt.copy()
                shuffled[col] = rng.permutation(shuffled[col].to_numpy())
                rises.append(log_loss(yt, m.predict_proba(shuffled)[:, 1], labels=[0, 1]) - base)
            imp_sum[col] += np.mean(rises) * len(fold.test)
        weight += len(fold.test)
    if np.isnan(oof).any():
        raise RuntimeError("some points were never in a test fold")
    return VariantResult(oof, imp_sum / weight)
