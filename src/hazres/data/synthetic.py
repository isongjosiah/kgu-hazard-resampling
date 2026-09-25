"""Synthetic study areas where the true answer is known.

Used for the pilot and for every test that checks a diagnostic can tell a real
effect from none. The generator makes:

* **true layers** at the 30 m analysis resolution;
* **observed layers**: the same layers as a user would get them, at their
  native resolution. Each coarse cell holds the exact average (or majority
  class) of the true cells inside it;
* the **true hazard probability** of every cell, from a known formula;
* **labels** at random survey points, with observed absences (like GE-LUCAS).

The default layers are chosen so that resampling *should* matter for one of
them and not for another:

``slope``          fine (30 m), moderately smooth. Not resampled.
``rainfall``       coarse (x32, about 1 km) but very smooth, so little is lost.
``bedrock_depth``  coarse (x32) and rough, so most of its detail is lost.
                   This is where the resampling choice should show.
``lithology``      classes, coarse (x16, about 500 m).

Set ``beta`` to zero for a layer to make a case where it has no effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import numpy as np
from affine import Affine

from hazres.data.labels import AbsenceKind, LoadReport, VectorLabels

Kind = Literal["continuous", "classes"]


@dataclass(frozen=True)
class LayerSpec:
    name: str
    kind: Kind
    factor: int
    """Native cell size as a multiple of the analysis cell (1 = already on the grid)."""
    correlation_cells: float
    """How far (in analysis cells) values stay similar. Larger = smoother."""
    beta: float | tuple[float, ...]
    """Effect on the log-odds: one number for continuous layers, one per class."""
    n_classes: int = 0


DEFAULT_LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("slope", "continuous", 1, 6.0, 1.2),
    LayerSpec("rainfall", "continuous", 32, 120.0, 0.6),
    LayerSpec("bedrock_depth", "continuous", 32, 5.0, -1.0),
    LayerSpec("lithology", "classes", 16, 30.0, (0.0, 0.5, -0.4, 0.9), n_classes=4),
)


@dataclass
class SyntheticStudy:
    layers: tuple[LayerSpec, ...]
    cell_size: float
    crs: str
    transform: Affine
    truth: dict[str, np.ndarray]
    """Every layer at the analysis resolution, as it really is."""
    observed: dict[str, np.ndarray]
    """Every layer at its native resolution, as a user would download it."""
    probability: np.ndarray
    labels: VectorLabels
    intercept: float
    seed: int
    extra: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.probability.shape

    def observed_transform(self, name: str) -> Affine:
        f = self.spec(name).factor
        return self.transform @ Affine.scale(f)

    def spec(self, name: str) -> LayerSpec:
        return next(s for s in self.layers if s.name == name)


def gaussian_field(shape: tuple[int, int], correlation_cells: float, rng) -> np.ndarray:
    """A smooth random field with mean 0 and standard deviation 1 (FFT, periodic edges)."""
    noise = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    kernel = np.exp(-2.0 * (np.pi * correlation_cells) ** 2 * (kx**2 + ky**2))
    field_ = np.fft.irfft2(np.fft.rfft2(noise) * kernel, s=shape)
    return (field_ - field_.mean()) / field_.std()


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def block_majority(a: np.ndarray, f: int, n_classes: int) -> np.ndarray:
    h, w = a.shape
    blocks = a.reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, -1)
    counts = np.stack([(blocks == c).sum(axis=-1) for c in range(n_classes)], axis=-1)
    return counts.argmax(axis=-1).astype(np.int16)  # ties go to the lowest class


def _classes(field_: np.ndarray, n: int) -> np.ndarray:
    edges = np.quantile(field_, np.linspace(0, 1, n + 1)[1:-1])
    return np.digitize(field_, edges).astype(np.int16)


def _intercept_for(logit_wo_b0: np.ndarray, prevalence: float) -> float:
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit_wo_b0 + mid)))).mean() < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_synthetic(
    *,
    seed: int = 20260925,
    shape: tuple[int, int] = (512, 512),
    cell_size: float = 30.0,
    layers: tuple[LayerSpec, ...] = DEFAULT_LAYERS,
    prevalence: float = 0.02,
    n_samples: int = 5000,
    crs: str = "EPSG:3035",
    origin: tuple[float, float] = (4_000_000.0, 3_000_000.0),
) -> SyntheticStudy:
    """Build a synthetic study area. Same seed, same study, on every machine."""
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be between 0 and 1")
    for s in layers:
        if shape[0] % s.factor or shape[1] % s.factor:
            raise ValueError(f"{s.name}: shape {shape} is not divisible by factor {s.factor}")
        if s.kind == "classes" and (s.n_classes < 2 or len(s.beta) != s.n_classes):
            raise ValueError(f"{s.name}: class layers need n_classes >= 2 and one beta per class")
    if n_samples > shape[0] * shape[1]:
        raise ValueError("n_samples exceeds the number of cells")

    rng = np.random.default_rng(seed)
    truth: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    logit = np.zeros(shape)
    for s in layers:
        raw = gaussian_field(shape, s.correlation_cells, rng)
        if s.kind == "classes":
            truth[s.name] = _classes(raw, s.n_classes)
            observed[s.name] = (
                truth[s.name]
                if s.factor == 1
                else block_majority(truth[s.name], s.factor, s.n_classes)
            )
            logit += np.asarray(s.beta)[truth[s.name]]
        else:
            truth[s.name] = raw
            observed[s.name] = raw if s.factor == 1 else block_mean(raw, s.factor)
            logit += s.beta * raw

    b0 = _intercept_for(logit, prevalence)
    probability = 1 / (1 + np.exp(-(logit + b0)))

    transform = Affine(cell_size, 0, origin[0], 0, -cell_size, origin[1] + shape[0] * cell_size)
    flat = rng.choice(shape[0] * shape[1], size=n_samples, replace=False)
    rows, cols = np.divmod(flat, shape[1])
    label = (rng.random(n_samples) < probability[rows, cols]).astype(np.int8)
    xs, ys = transform @ (cols + 0.5, rows + 0.5)
    features = gpd.GeoDataFrame(
        {
            "sample_id": [f"synthetic-{i}" for i in range(n_samples)],
            "label": label,
            "row": rows,
            "col": cols,
            "true_probability": probability[rows, cols],
        },
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )
    n_pos = int(label.sum())
    labels = VectorLabels(
        dataset="synthetic",
        features=features,
        absence_kind=AbsenceKind.OBSERVED,
        report=LoadReport("synthetic", n_samples, n_pos, n_samples - n_pos),
        meta={"seed": seed},
    )
    return SyntheticStudy(
        layers=layers,
        cell_size=cell_size,
        crs=crs,
        transform=transform,
        truth=truth,
        observed=observed,
        probability=probability,
        labels=labels,
        intercept=b0,
        seed=seed,
    )
