"""The analysis grid: one CRS, one cell size, cells aligned to whole multiples of it."""

from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine


@dataclass(frozen=True)
class GridSpec:
    """A north-up grid whose cell edges fall on multiples of ``res`` (so grids line up)."""

    crs: str
    res: float
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def covering(
        cls, bounds: tuple[float, float, float, float], crs: str, res: float = 30.0
    ) -> GridSpec:
        """The smallest aligned grid that covers ``bounds`` (in ``crs``)."""
        left, bottom, right, top = bounds
        if not (right > left and top > bottom):
            raise ValueError(f"empty bounds {bounds}")
        return cls(
            crs=crs,
            res=float(res),
            left=math.floor(left / res) * res,
            bottom=math.floor(bottom / res) * res,
            right=math.ceil(right / res) * res,
            top=math.ceil(top / res) * res,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.left, self.bottom, self.right, self.top)

    @property
    def shape(self) -> tuple[int, int]:
        return (
            round((self.top - self.bottom) / self.res),
            round((self.right - self.left) / self.res),
        )

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.left, 0.0, -self.res, self.top)
