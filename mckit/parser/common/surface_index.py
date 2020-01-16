from typing import Optional

from mckit import surface
from mckit.parser.common import Index, NumberedItemNotFound


class DummySurface(surface.Surface):
    """To substitute surface when it's not found"""
    def __init__(self, name: int):
        super().__init__(name=name, comment="dummy")

    def transform(self, tr):
        raise NotImplementedError(
            "Dummy surface cannot transform and this is intended.\n \
             Please, remove it from a model or replace with real surface."
        )


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