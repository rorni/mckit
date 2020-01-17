from typing import Optional

# noinspection PyUnresolvedReferences,PyPackageRequirements
from mckit.surface import EX, Plane
from mckit.parser.common import Index, NumberedItemNotFound


class DummySurface(Plane):
    """To substitute surface when it's not found"""
    def __init__(self, name: int):
        super().__init__(EX, 0.0, name=name, comment="dummy")

    def __str__(self):
        return self.name()

    def __repr__(self):
        return f"DummySurface({self.name()})"


def raise_on_absent_surface_strategy(name: int) -> Optional[DummySurface]:
    raise SurfaceNotFound(name)


def dummy_on_absent_surface_strategy(name: int) -> Optional[DummySurface]:
    return DummySurface(name)


class SurfaceStrictIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(raise_on_absent_surface_strategy, kwargs)


class SurfaceDummyIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(dummy_on_absent_surface_strategy, kwargs)


class SurfaceNotFound(NumberedItemNotFound):
    kind = 'Surface'