"""Utility methods to access a package data."""
from __future__ import annotations

import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files


def path_resolver(package):
    """Create method to find data path.

    Args:
        package: the package below which the data is stored.

    Returns:
        callable which appends the argument to the package folder adt returns as Path.
    """

    return files(package).joinpath
