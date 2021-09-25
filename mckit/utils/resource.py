import inspect

from pathlib import Path

import pkg_resources as pkg


def filename_resolver(package=None):
    if package is None:
        module = inspect.getmodule(inspect.stack()[1][0])
        package = module.__name__

    resource_manager = pkg.ResourceManager()

    def func(resource):
        return resource_manager.resource_filename(package, resource)

    func.__doc__ = f"Computes file names for resources located in {package}"

    return func


def path_resolver(package=None):

    resolver = filename_resolver(package)

    def func(resource):
        filename = resolver(resource)
        return Path(filename)

    if package is None:
        package = "caller package"

    # noinspection PyCompatibility
    func.__doc__ = f"Computes Path for resources located in {package}"

    return func
