"""Utility methods to access a package data."""

from typing import Callable, cast

from pathlib import Path

import pkg_resources as pkg


def filename_resolver(package: str) -> Callable[[str], str]:
    """Create method to find data file name.

    Uses resource manager to handle all the cases of the deployment.

    Args:
        package: the package below which the data is stored.

    Returns:
        callable which appends the argument to the package folder.
    """
    resource_manager = pkg.ResourceManager()  # type: ignore[attr-defined]

    def func(resource: str) -> str:
        return cast(str, resource_manager.resource_filename(package, resource))

    func.__doc__ = f"Computes file names for resources located in {package}"

    return func


def path_resolver(package: str) -> Callable[[str], Path]:
    """Create method to find data path.

    Uses :func:`file_resolver`.

    Args:
        package: the package below which the data is stored.

    Returns:
        callable which appends the argument to the package folder adt returns as Path.
    """
    resolver = filename_resolver(package)

    def func(resource: str) -> Path:
        filename = resolver(resource)
        return Path(filename)

    func.__doc__ = f"Computes Path for resources located in {package}"

    return func
