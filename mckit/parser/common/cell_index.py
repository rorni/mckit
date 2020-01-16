from typing import Optional

import mckit.body as body
from mckit.parser.common import Index, NumberedItemNotFound


class DummyCell(body.Body):
    """To substitute cell when it's not found"""
    def __init__(self, name: int):
        super().__init__(name=name, comment="dummy")


def raise_on_absent_cell_strategy(name: int) -> Optional[DummyCell]:
    raise CellNotFound(name)


def dummy_on_absent_cell_strategy(name: int) -> Optional[DummyCell]:
    return DummyCell(name)


class CellStrictIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(raise_on_absent_cell_strategy, kwargs)


class CellDummyIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(dummy_on_absent_cell_strategy, kwargs)


class CellNotFound(NumberedItemNotFound):
    kind = 'Cell'
