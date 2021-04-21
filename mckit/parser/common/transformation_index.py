from typing import Iterable, Optional

from mckit.transformation import Transformation
from mckit.utils.Index import Index, NumberedItemNotFoundError


class DummyTransformation(Transformation):
    """To substitute transformation when it's not found"""

    def __init__(self, name: int):
        super().__init__(name=name, comment="dummy")


def raise_on_absent_transformation_strategy(name: int) -> Optional[DummyTransformation]:
    raise TransformationNotFoundError(name)


def dummy_on_absent_transformation_strategy(name: int) -> Optional[DummyTransformation]:
    return DummyTransformation(name)


class TransformationStrictIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(raise_on_absent_transformation_strategy, **kwargs)

    @classmethod
    def from_iterable(
        cls, items: Iterable[Transformation]
    ) -> "TransformationStrictIndex":
        index = cls()
        index.update((c.name(), c) for c in items)
        return index


class TransformationDummyIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(dummy_on_absent_transformation_strategy, **kwargs)


class TransformationNotFoundError(NumberedItemNotFoundError):
    kind = "Transformation"
