#!/usr/bin/env python3

# https://stackoverflow.com/questions/60073711/how-to-build-c-extensions-via-poetry
import os
import os.path as path
import sys
import numpy as np

from setuptools import Extension, find_packages, setup
from distutils.errors import (
    CCompilerError, DistutilsExecError, DistutilsPlatformError
)
from distutils.command.build_ext import build_ext


def get_dirs(environment_variable):
    include_dirs = os.environ.get(environment_variable, "")

    if include_dirs:
        include_dirs = include_dirs.split(os.pathsep)
    else:
        include_dirs = []

    return include_dirs


def append_if_not_present(destination, value):
    if isinstance(value, list):
        destination.extend(value)
    elif value not in destination:
        destination.append(value)


include_dirs = get_dirs("INCLUDE_PATH")
append_if_not_present(include_dirs, np.get_include())

library_dirs = get_dirs("LIBRARY_PATH")

if sys.platform.startswith('linux'):
    geometry_dependencies = [
        'mkl_intel_lp64',
        'mkl_core',
        'mkl_sequential',
        'nlopt',
    ]
    conda_include_dir = path.join(sys.prefix, "include")
    append_if_not_present(include_dirs, conda_include_dir)
    append_if_not_present(library_dirs, path.join(sys.prefix, "lib"))
else:
    geometry_dependencies = [
        'mkl_intel_lp64_dll',
        'mkl_core_dll',
        'mkl_sequential_dll',
        'libnlopt-0',
    ]
    nlopt_inc = get_dirs("NLOPT")
    append_if_not_present(include_dirs, nlopt_inc)
    nlopt_lib = get_dirs("NLOPT")
    append_if_not_present(library_dirs, nlopt_lib)
    mkl_inc = sys.prefix + '\\Library\\include'
    append_if_not_present(include_dirs, mkl_inc)
    mkl_lib = sys.prefix + '\\Library\\lib'
    append_if_not_present(library_dirs, mkl_lib)

geometry_sources = [path.join("mckit", "src", src) for src in [
    "geometrymodule.c",
    "box.c",
    "surface.c",
    "shape.c",
    "rbtree.c",
]]

ext_modules = [
    Extension(
        "mckit.geometry",
        sources=geometry_sources,
        include_dirs=include_dirs,
        libraries=geometry_dependencies,
        library_dirs=library_dirs,

    )
]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}}
    )
