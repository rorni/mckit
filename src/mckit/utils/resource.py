"""Utility methods to access a package data."""
from __future__ import annotations

from typing import Callable

import importlib.resources as rc

from pathlib import Path


def path_resolver(package: rc.Package) -> Callable[[rc.Resource], Path]:
    """Create method to find data path.

    Args:
        package: the package below which the data is stored.

    Yields:
        Path to resource object, may be temporary, if the resource is from zip.
    """

    resolver = rc.files(package)

    return resolver.joinpath
