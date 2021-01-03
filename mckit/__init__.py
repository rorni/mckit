# flake8: noqa F401

import os
import platform
import sys

from ctypes import cdll
from pathlib import Path

if platform.system() == "Windows":
    dll_path = os.path.join(sys.prefix, "Library", "bin")
    nlopt_lib_path = os.path.join(dll_path, "nlopt.dll")
    assert os.path.exists(
       nlopt_lib_path
    ), f"nlopt.dll should be in ${nlopt_lib_path} before importing mckit"
    if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
        os.add_dll_directory(dll_path)
    nlopt_lib = cdll.LoadLibrary(nlopt_lib_path)  # to guarantee dll loading 
else:
    if os.environ.get("LD_LIBRARY_PATH") is None:  # a user can use other location form mkl library.
        # preload library
        mkl_lib_path = Path(sys.prefix, "lib", "libmkl_rt.so.1")
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
