# flake8: noqa F401

import os
import sys
import platform

if platform.system() == "Windows":
    lib_path = os.path.join(sys.prefix, "Library", "bin")
    assert os.path.exists(
        os.path.join(lib_path, "nlopt.dll")
    ), f"nlopt.dll should be in ${lib_path} before importing mckit"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib_path)


from .universe import Universe
from .surface import create_surface, Plane, Sphere, Cylinder, Cone, Torus, GQuadratic
from .body import Shape, Body
from .transformation import Transformation
from .material import AVOGADRO, Element, Composition, Material
from .fmesh import FMesh
from .parser.meshtal_parser import read_meshtal
from .parser.mctal_parser import read_mctal
from .version import (
    __author__,
    __copyright__,
    __license__,
    __title__,
    __version__,
    __summary__,
)


__doc__ = __summary__
