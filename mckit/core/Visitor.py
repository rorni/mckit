from contextlib import contextmanager
from typing import Any


class Acceptor:

    def accept(self, visitor: 'Visitor') -> Any:
        lookup = "visit_" + type(self).__qualname__.replace(".", "_").lower()
        return getattr(visitor, lookup)(self)


class Visitor:

    @contextmanager
    def visit(self, acceptor: Acceptor) -> Any:
        lookup = "accept_" + type(self).__qualname__.replace(".", "_").lower()
        return getattr(acceptor, lookup)(self)
