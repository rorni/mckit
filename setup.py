import os
import os.path as path
import sys

import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution
from setuptools.command.test import test as TestCommand

# See recomendations in https://docs.pytest.org/en/latest/goodpractices.html
# noinspection PyAttributeOutsideInit

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

_title__ = 'mckit'
__author__ = 'Roman Rodionov'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018-2019 Roman Rodinov'
__ver_major__ = 1
__ver_minor__ = 0
__ver_patch__ = 0
__version_info__ = (__ver_major__, __ver_minor__, __ver_patch__)
__ver_sub__ = ''
__version__ = "%d.%d.%d%s" % (__ver_major__, __ver_minor__, __ver_patch__, __ver_sub__)

# update version values available to code from version module
with open("mckit/version.py") as fid:
    code = compile(fid.read(), "version.py", mode="exec")
    exec(code)


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
        library_dirs=library_dirs,

    )
]

packages = find_packages(
    include=("mckit", "mckit.*",),
    exclude=(
        "adhoc",
        "build*",
        "data",
        "dist",
        "doc",
        "examples",
        "htmlcov",
        "notebook",
        "tutorial",
        "tests",
        "src",
    ),
)

setup(
    name=_title__,
    version=__version__,
    packages=packages,
    package_data={'mckit': ['data/isotopes.dat', 'libnlopt-0.dll']},
    url='https://gitlab.iterrf.ru/Rodionov/mckit',
    license=__license__,
    author=__author__,
    author_email='r.rodionov@iterrf.ru',
    description='Tool for handling neutronic models and results',
    install_requires=[
        'click>=6.7',
        'click-log>=0.3.2',
        'mkl-devel',
        'numpy',
        'ply',
        'scipy',
        'tomlkit',
        'datetime',
    ],
    ext_modules=extensions,
    tests_require=['pytest', 'pytest-cov>=2.3.1'],
    cmdclass={'test': PyTest},
    entry_points={
        'console_scripts': [
            'mckit = mckit.cli.runner:mckit',
        ]
    },
)
