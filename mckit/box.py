import numpy as np

# noinspection PyUnresolvedReferences,PyPackageRequirements
from .geometry import EX, EY, EZ
from .geometry import GLOBAL_BOX as _GLOBAL_BOX
from .geometry import Box as _Box


class Box(_Box):
    __doc__ = _Box.__doc__

    def __init__(self, center, wx, wy, wz, ex=EX, ey=EY, ez=EZ):
        _Box.__init__(self, center, wx, wy, wz, ex=ex, ey=ey, ez=ez)

    @classmethod
    def from_geometry_box(cls, geometry_box: _Box):
        return cls(
            geometry_box.center,
            *geometry_box.dimensions,
            geometry_box.ex,
            geometry_box.ey,
            geometry_box.ez,
        )

    def __eq__(self, other):
        return (
            np.array_equal(self.center, other.center)
            and np.array_equal(self.dimensions, other.dimensions)
            and np.array_equal(self.ex, other.ex)
            and np.array_equal(self.ey, other.ey)
            and np.array_equal(self.ez, other.ez)
        )

    def __hash__(self):
        return hash((self.center, self.dimensions, self.ex, self.ey, self.ez))

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
