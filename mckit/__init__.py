# flake8: noqa F401, E402

import os
import platform
import sys

from ctypes import cdll
from pathlib import Path


def get_shared_lib_name(name: str) -> str:
    sys_name = platform.system()
    if sys_name == "Linux":
        return f"lib{name}.so.1"
    if sys_name == "Darwin":
        return f"lib{name}.dylib"
    if sys_name == "Windows":
        return f"{name}.dll"


def find_file(_file: str, *directories: Path):
    """Find a file in directories"""
    for d in directories:
        path = d / _file
        if path.exists():
            return path
    raise EnvironmentError(f"Cannot find {_file} in directories {directories}")


if platform.system() == "Windows":
    dirs = [
        Path(__file__).parent,
        Path(sys.prefix, "Library", "bin"),
    ]
    if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
        for _dir in dirs:
            os.add_dll_directory(_dir)
    cdll.LoadLibrary(find_file("nlopt.dll", *dirs))  # to guarantee dll loading
else:
    if (
        os.environ.get("LD_LIBRARY_PATH") is None
    ):  # a user can use other location form mkl library.
        # preload library
        mkl_lib_path = Path(sys.prefix, "lib", get_shared_lib_name("mkl_rt"))
        assert (
            mkl_lib_path.exists()
        ), f"The MKL library should be either available at {mkl_lib_path}, or with LD_LIBRARY_PATH"
        mkl_Lib = cdll.LoadLibrary(str(mkl_lib_path))


from .body import Body, Shape
from .fmesh import FMesh
from .material import AVOGADRO, Composition, Element, Material
from .parser.mctal_parser import read_mctal
from .parser.meshtal_parser import read_meshtal
from .surface import Cone, Cylinder, GQuadratic, Plane, Sphere, Torus, create_surface
from .transformation import Transformation
from .universe import Universe
from .version import (
    __author__,
    __copyright__,
    __license__,
    __summary__,
    __title__,
    __version__,
)

__doc__ = __summary__
