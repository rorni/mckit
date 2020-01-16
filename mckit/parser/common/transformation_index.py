from typing import Optional

from mckit import surface as surf
from mckit.parser.common import Index, NumberedItemNotFound


class DummyTransformation(surf.Transformation):
    """To substitute transformation when it's not found"""
    def __init__(self, name: int):
        super().__init__(name=name, comment="dummy")


def raise_on_absent_transformation_strategy(name: int) -> Optional[DummyTransformation]:
    raise TransformationNotFound(name)


def dummy_on_absent_transformation_strategy(name: int) -> Optional[DummyTransformation]:
    return DummyTransformation(name)


class TransformationStrictIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(raise_on_absent_transformation_strategy, kwargs)


class TransformationDummyIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(dummy_on_absent_transformation_strategy, kwargs)


class TransformationNotFound(NumberedItemNotFound):
    kind = 'Transformation'
