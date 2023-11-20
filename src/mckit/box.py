from __future__ import annotations

from typing import Self

import numpy as np

from numpy.typing import NDArray

# noinspection PyUnresolvedReferences,PyPackageRequirements
from mckit.geometry import EX, EY, EZ
from mckit.geometry import GLOBAL_BOX as _GLOBAL_BOX
from mckit.geometry import Box as _Box
from mckit.utils import make_hashable


class Box(_Box):
    def __init__(self, center, wx, wy, wz, ex=EX, ey=EY, ez=EZ):
        _Box.__init__(self, center, wx, wy, wz, ex=ex, ey=ey, ez=ez)

    @classmethod
    def from_geometry_box(cls, geometry_box: _Box) -> Self:
        return cls(
            geometry_box.center,
            *geometry_box.dimensions,
            geometry_box.ex,
            geometry_box.ey,
            geometry_box.ez,
        )

    @classmethod
    def from_corners(cls, min_vals: NDArray, max_vals: NDArray) -> Self:
        if not np.all(min_vals < max_vals):
            raise ValueError("Unsorted boundaries values")
        center = 0.5 * (min_vals + max_vals)
        widths = max_vals - min_vals
        return cls(center, *widths, ex=EX, ey=EY, ez=EZ)

    @classmethod
    def from_bounds(
        cls, minx: float, maxx: float, miny: float, maxy: float, minz: float, maxz: float
    ) -> Self:
        min_vals = np.array([minx, miny, minz])
        max_vals = np.array([maxx, maxy, maxz])
        return cls.from_corners(min_vals, max_vals)

    def __repr__(self):
        exm = "" if np.array_equal(self.ex, EX) else f"ex={self.ex}"
        eym = "" if np.array_equal(self.ey, EY) else f"ey={self.ey}"
        ezm = "" if np.array_equal(self.ez, EZ) else f"ez={self.ez}"
        msgs = [x for x in [exm, eym, ezm] if x]
        emsg = ", " + ", ".join(msgs) if msgs else ""
        wx, wy, wz = self.dimensions
        return f"Box({self.center}, {wx}, {wy}, {wz}{emsg})"

    def __eq__(self, other):
        return (
            np.array_equal(self.center, other.center)
            and np.array_equal(self.dimensions, other.dimensions)
            and np.array_equal(self.ex, other.ex)
            and np.array_equal(self.ey, other.ey)
            and np.array_equal(self.ez, other.ez)
        )

    def __hash__(self):
        return hash(make_hashable((self.center, self.dimensions, self.ex, self.ey, self.ez)))

    def __getstate__(self):
        center = self.center
        wx, wy, wz = self.dimensions
        ex = self.ex
        ey = self.ey
        ez = self.ez
        return center, wx, wy, wz, ex, ey, ez

    def __setstate__(self, state):
        center, wx, wy, wz, ex, ey, ez = state
        self.__init__(center, wx, wy, wz, ex, ey, ez)


GLOBAL_BOX = Box.from_geometry_box(_GLOBAL_BOX)
