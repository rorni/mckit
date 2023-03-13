"""Utility methods to access a package data."""
from __future__ import annotations

from typing import Callable

import sys

from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.abc import Traversable
    from importlib.resources import Package, files
else:
    from importlib_resources import files


def path_resolver(package: Package) -> Callable[[str], Path | Traversable]:
    """Create method to find data path.

    Args:
        package: the package below which the data is stored.

    Returns:
        callable which appends the argument to the package folder adt returns as Path.
    """
    return files(package).joinpath
