#!/usr/bin/env python3

# https://stackoverflow.com/questions/60073711/how-to-build-c-extensions-via-poetry
from typing import List, Union

import os
import os.path as path
import sys

from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

# TODO dvp: check if the issue with build requirements is resolved and poetry updated
# see https://github.com/python-poetry/poetry/pull/2794
# import numpy as np  # numpy is in build requirements, so, it should be available
# workaround
try:
    import numpy as np
except ImportError:
    import subprocess

    subprocess.check_call("pip install numpy".split())
    import numpy as np

from build_nlopt import build_nlopt
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# see https://habr.com/ru/post/210450/
from setuptools.dist import Distribution

build_nlopt()


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


def get_dirs(environment_variable):
    dirs = os.environ.get(environment_variable, "")

    if dirs:
        dirs = dirs.split(os.pathsep)
    else:
        dirs = []

    return dirs


def insert_directories(
    destination: List[str], value: Union[str, List[str]]
) -> List[str]:
    dirs = []
    if isinstance(value, list):
        dirs.extend(value)
    elif value not in destination:
        dirs.append(value)
    for old in destination:
        if old not in dirs:
            dirs.append(old)
    return dirs


include_dirs = get_dirs("INCLUDE_PATH")

include_dirs = insert_directories(include_dirs, np.get_include())

library_dirs = get_dirs("LIBRARY_PATH")
platform = sys.platform.lower()

if platform.startswith("linux"):
    geometry_dependencies = [
        "mkl_rt",  # dvp: use -lmkl_rt instead of -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
        "nlopt",
    ]
    python_include_dir = path.join(sys.prefix, "include")
    include_dirs = insert_directories(include_dirs, python_include_dir)
    library_dirs = insert_directories(library_dirs, path.join(sys.prefix, "lib"))
    extra_compile_args = ["-O3", "-w"]
elif "win" in platform and "darwin" not in sys.platform.lower():
    geometry_dependencies = [
        "mkl_intel_lp64_dll",
        "mkl_core_dll",
        "mkl_sequential_dll",
        "libnlopt-0",
    ]
    nlopt_inc = get_dirs("NLOPT")
    include_dirs = insert_directories(include_dirs, nlopt_inc)
    nlopt_lib = get_dirs("NLOPT")
    library_dirs = insert_directories(library_dirs, nlopt_lib)
    mkl_inc = sys.prefix + "\\Library\\include"
    include_dirs = insert_directories(include_dirs, mkl_inc)
    mkl_lib = sys.prefix + "\\Library\\lib"
    library_dirs = insert_directories(library_dirs, mkl_lib)
    extra_compile_args = ["/O2"]
else:
    raise EnvironmentError(f"Cannot recognize platform {platform}")

geometry_sources = [
    path.join("mckit", "src", src)
    for src in ["geometrymodule.c", "box.c", "surface.c", "shape.c", "rbtree.c"]
]


ext_modules = [
    Extension(
        "mckit.geometry",
        sources=geometry_sources,
        include_dirs=include_dirs,
        libraries=geometry_dependencies,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
    ),
]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed("File not found. Could not compile C extension.")

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed("Could not compile C extension.")


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": ExtBuilder},
            "package_data": {"mckit": ["data/isotopes.dat", "libnlopt-0.dll"]},
            "distclass": BinaryDistribution,
            "install_requires": [
                "numpy>=1.13",
                "mkl-devel",
                "mkl",
                "mkl-include",
                "nlopt",
            ],
        }
    )
