"""Utility methods to access a package data."""
from __future__ import annotations

from typing import TYPE_CHECKING

import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files  # pragma: no cover

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.abc import Traversable
    from importlib.resources import Package
    from pathlib import Path


def path_resolver(package: Package) -> Callable[[str], Path | Traversable]:
    """Create method to find data path.

    Args:
        package: the package below which the data is stored.

    Returns:
        callable which appends the argument to the package folder as Path.
    """
    return files(package).joinpath
