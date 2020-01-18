class NumberedItemNotFoundError(KeyError):
    kind: str = ''

    def __init__(self, item: int, *args, **kwargs):
        super().__init__(args, kwargs)
        self._item = item

    def __str__(self):
        return f"{self.kind} #{self._item} is not found"



class CellNotFoundError(NumberedItemNotFoundError):
    kind = 'Cell'

