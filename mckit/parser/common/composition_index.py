from typing import Optional

from mckit import material
from mckit.parser.common import Index, NumberedItemNotFound


class DummyMaterial(material.Material):
    def __init__(self, name: int, *, density: Optional[float] = None, concentration: Optional[float] = None) -> None:
        assert (density is None) ^ (concentration is None), "Specify only one of the parameters"
        if density is None:
            super().__init__(composition=DummyComposition(name), concentration=concentration * 1.0e24)
        else:
            super().__init__(composition=DummyComposition(name), density=density)


class DummyComposition(material.Composition):
    """To substitute composition when it's not found"""
    def __init__(self, name: int):
        super().__init__(name=name, weight=[(1001, 1.0)], comment="dummy")


def raise_on_absent_composition_strategy(name: int) -> Optional[DummyComposition]:
    raise CompositionNotFound(name)


def dummy_on_absent_composition_strategy(name: int) -> Optional[DummyComposition]:
    return DummyComposition(name)


class CompositionStrictIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(raise_on_absent_composition_strategy, kwargs)


class CompositionDummyIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(dummy_on_absent_composition_strategy, kwargs)


class CompositionNotFound(NumberedItemNotFound):
    kind = 'Composition'
