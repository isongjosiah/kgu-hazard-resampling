"""The common format every hazard loader returns.

Whatever the source (a survey CSV, landslide polygons, flood rasters), a loader
hands back one of two things, so nothing downstream needs to know where the
labels came from:

* :class:`VectorLabels` for points or polygons, with ``label`` 1 (hazard) or
  0 (checked and found free of the hazard).
* :class:`RasterEvents` for gridded labels such as satellite flood maps, with
  1, 0, or -1 (unknown: not observed, cloud, permanent water).

Every load also returns a :class:`LoadReport` saying what was read, kept and
dropped, and why. Nothing is dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd
    from affine import Affine
    from shapely.geometry.base import BaseGeometry

PRESENCE = 1
ABSENCE = 0
UNKNOWN = -1


class AbsenceKind(str, Enum):
    """What a 0 label means for a dataset. This decides which results it can support."""

    OBSERVED = "observed"
    """Places were visited or inspected and found free of the hazard (e.g. GE-LUCAS)."""

    WITHIN_FOOTPRINT = "within_footprint"
    """Everything inside a systematically mapped area was checked, so unmapped
    places inside that area are absences, but nowhere outside it."""

    PARTIAL = "partial"
    """Absences are observed but incomplete (e.g. floods missed under cloud)."""

    NONE = "none"
    """Presence only. Absences must be constructed, so risk numbers cannot be checked."""


@dataclass
class LoadReport:
    """What a loader read, kept and dropped. Printed on every load."""

    dataset: str
    n_read: int
    n_presence: int
    n_absence: int
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        kept = self.n_presence + self.n_absence
        lines = [
            f"{self.dataset}: kept {kept:,} of {self.n_read:,} "
            f"({self.n_presence:,} hazard, {self.n_absence:,} no hazard)"
        ]
        for reason, n in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  dropped {n:,}: {reason}")
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class VectorLabels:
    """Point or polygon labels in one projected CRS.

    ``features`` always has the columns ``sample_id`` (unique str), ``label``
    (0 or 1) and ``geometry``. Any extra columns a loader keeps (country codes,
    land cover, landslide type) come after them.
    """

    dataset: str
    features: gpd.GeoDataFrame
    absence_kind: AbsenceKind
    report: LoadReport
    footprint: BaseGeometry | None = None
    """The area that was systematically mapped, if the dataset has one."""

    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def crs(self):
        return self.features.crs

    def validate(self) -> None:
        f = self.features
        missing = [c for c in ("sample_id", "label", "geometry") if c not in f.columns]
        if missing:
            raise ValueError(f"{self.dataset}: missing columns {missing}")
        if f.crs is None:
            raise ValueError(f"{self.dataset}: features have no CRS")
        if f.crs.is_geographic:
            raise ValueError(
                f"{self.dataset}: features are in a geographic CRS ({f.crs.to_string()}). "
                "Distances and areas need a projected CRS."
            )
        bad = set(np.unique(f["label"].to_numpy())) - {PRESENCE, ABSENCE}
        if bad:
            raise ValueError(f"{self.dataset}: labels must be 0 or 1, found {sorted(bad)}")
        if f["sample_id"].duplicated().any():
            dup = f.loc[f["sample_id"].duplicated(), "sample_id"].iloc[0]
            raise ValueError(f"{self.dataset}: sample_id is not unique (e.g. {dup!r})")
        if f.geometry.isna().any() or f.geometry.is_empty.any():
            raise ValueError(f"{self.dataset}: empty geometries must be dropped by the loader")
        if self.absence_kind is AbsenceKind.NONE and (f["label"] == ABSENCE).any():
            raise ValueError(
                f"{self.dataset}: presence-only dataset contains 0 labels; "
                "constructed absences belong to the sampling step, not the loader"
            )


@dataclass
class RasterLabel:
    """One gridded label map, e.g. one flood event. Values are 1, 0 or -1 (unknown)."""

    event_id: str
    label: np.ndarray
    transform: Affine
    crs: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label.ndim != 2:
            raise ValueError(f"event {self.event_id}: label must be 2-D")
        bad = set(np.unique(self.label)) - {PRESENCE, ABSENCE, UNKNOWN}
        if bad:
            raise ValueError(f"event {self.event_id}: label values must be 1, 0 or -1: {bad}")

    @property
    def cell_size(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))


@dataclass
class RasterEvents:
    """All gridded label maps for one dataset."""

    dataset: str
    events: list[RasterLabel]
    absence_kind: AbsenceKind
    report: LoadReport
    meta: dict[str, Any] = field(default_factory=dict)


Labels = VectorLabels | RasterEvents
