"""Spatial folds: test on blocks of land the model has not seen.

Points are grouped into square blocks. Blocks are dealt to ``k`` folds so each
fold gets a similar number of hazard points. When a fold is tested, training
points closer than ``buffer_m`` to any of its test points are left out, so
neighbours cannot leak the answer across the boundary.

The folds are built once per experiment and shared by every variant and model,
so differences between variants are never differences in how the data were
split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    buffered_out: int
    """Training points dropped because they were within the buffer of the test set."""


def block_ids(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    bx = np.floor(x / block_m).astype(np.int64)
    by = np.floor(y / block_m).astype(np.int64)
    _, ids = np.unique(np.stack([bx, by], axis=1), axis=0, return_inverse=True)
    return ids.ravel()


def spatial_folds(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    *,
    block_m: float = 10_000,
    k: int = 5,
    buffer_m: float = 2_000,
    seed: int = 1,
) -> list[Fold]:
    x, y, labels = np.asarray(x, float), np.asarray(y, float), np.asarray(labels)
    blocks = block_ids(x, y, block_m)
    n_blocks = blocks.max() + 1
    if n_blocks < k:
        raise ValueError(
            f"only {n_blocks} blocks of {block_m:g} m for {k} folds; use smaller blocks"
        )
    pos = np.bincount(blocks, weights=labels == 1, minlength=n_blocks)
    size = np.bincount(blocks, minlength=n_blocks)

    # deal blocks to folds: most hazard points first, each to the fold with fewest so far;
    # a seeded shuffle breaks ties so the split is reproducible but not grid-aligned
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    order = order[np.argsort(-pos[order], kind="stable")]
    fold_pos, fold_size = np.zeros(k), np.zeros(k)
    assign = np.empty(n_blocks, dtype=np.int64)
    for b in order:
        f = int(np.lexsort((fold_size, fold_pos))[0])
        assign[b] = f
        fold_pos[f] += pos[b]
        fold_size[f] += size[b]
    point_fold = assign[blocks]

    xy = np.column_stack([x, y])
    folds = []
    for f in range(k):
        test = np.flatnonzero(point_fold == f)
        train = np.flatnonzero(point_fold != f)
        dropped = 0
        if buffer_m > 0 and test.size and train.size:
            dist, _ = cKDTree(xy[test]).query(xy[train], k=1)
            keep = dist >= buffer_m
            dropped = int((~keep).sum())
            train = train[keep]
        if labels[test].sum() == 0 or labels[train].sum() == 0:
            raise ValueError(f"fold {f} has no hazard points in its train or test set")
        folds.append(Fold(train, test, dropped))
    return folds
