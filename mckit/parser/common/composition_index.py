from typing import Iterable, Optional

from mckit.material import Composition, Material
from mckit.utils.Index import Index, NumberedItemNotFoundError


class DummyMaterial(Material):
    def __init__(
        self,
        name: int,
        *,
        density: Optional[float] = None,
        concentration: Optional[float] = None,
    ) -> None:
        assert (density is None) ^ (
            concentration is None
        ), "Specify only one of the parameters"
        if density is None:
            # noinspection PyTypeChecker
            super().__init__(
                composition=DummyComposition(name), concentration=concentration * 1.0e24
            )
        else:
            super().__init__(composition=DummyComposition(name), density=density)


# noinspection PyTypeChecker
class DummyComposition(Composition):
    """To substitute composition when it's not found"""

    def __init__(self, name: int):
        super().__init__(name=name, weight=[(1001, 1.0)], comment="dummy")


def raise_on_absent_composition_strategy(name: int) -> Optional[DummyComposition]:
    raise CompositionNotFoundError(name)


def dummy_on_absent_composition_strategy(name: int) -> Optional[DummyComposition]:
    return DummyComposition(name)


class CompositionStrictIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(raise_on_absent_composition_strategy, **kwargs)

    @classmethod
    def from_iterable(cls, items: Iterable[Composition]) -> "CompositionStrictIndex":
        index = cls()
        index.update((c.name(), c) for c in items)
        return index


class CompositionDummyIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(dummy_on_absent_composition_strategy, **kwargs)


class CompositionNotFoundError(NumberedItemNotFoundError):
    kind = "Composition"
