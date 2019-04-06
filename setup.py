import os
import os.path as path
import sys

import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


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

extensions = [
    Extension(
        "mckit.geometry",
        geometry_sources,
        include_dirs=include_dirs,
        libraries=geometry_dependencies,
        library_dirs=library_dirs
    )
]

setup(
    name='mckit',
    version='0.1.1',
    packages=find_packages(),
    package_data={'mckit': ['data/isotopes.dat', 'libnlopt-0.dll']},
    url='https://gitlab.iterrf.ru/Rodionov/mckit',
    license='',
    author='Roman Rodionov',
    author_email='r.rodionov@iterrf.ru',
    description='Tool for handling neutronic models and results',
    install_requires=['numpy', 'scipy', 'ply', 'mkl-include'],
    ext_modules=extensions,
    # data_files=[('.', ['libnlopt-0.dll'])]
)
