"""The models, with fixed settings and seeds.

Standard, widely used models on purpose: the study is about how maps are
checked, not about building a better model. Settings are fixed in advance and
identical for every variant. No re-weighting of the rare class: the real mix of
hazard and no-hazard points is kept, because re-balancing is what inflates the
risk numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = ("lightgbm", "random_forest", "logistic")


class LightGBMModel:
    """LightGBM with native categorical handling and fixed settings."""

    def __init__(self, seed: int, categorical: list[str]):
        import lightgbm as lgb

        self.categorical = categorical
        # Conservative settings for samples of a few thousand points. Changed after
        # pilot run 1 (26 Sep 2026), where the first settings (300 trees, 31 leaves)
        # overfitted about 750 training points: calibration slope 0.35. Fixed in
        # advance for every variant; not tuned on the method comparison.
        self.model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=5.0, random_state=seed, n_jobs=1, verbose=-1, deterministic=True,
            force_row_wise=True,
        )  # fmt: skip

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X

    def fit(self, X: pd.DataFrame, y) -> LightGBMModel:  # noqa: N803
        self.categories_ = {c: sorted(pd.unique(X[c].dropna())) for c in self.categorical}
        self.model.fit(self._prep(X), y, categorical_feature=self.categorical or "auto")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self.model.predict_proba(self._prep(X))


def _tabular(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])  # fmt: skip


def make_model(name: str, features: list[str], categorical: list[str], seed: int = 20260926):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    numeric = [f for f in features if f not in categorical]
    if name == "lightgbm":
        return LightGBMModel(seed, categorical)
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=10, max_features="sqrt", n_jobs=1, random_state=seed
        )
        return Pipeline([("prep", _tabular(numeric, categorical, scale=False)), ("model", clf)])
    clf = LogisticRegression(C=1.0, max_iter=2000)
    return Pipeline([("prep", _tabular(numeric, categorical, scale=True)), ("model", clf)])
