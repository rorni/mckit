# flake8: noqa F401, E402

import os
import platform
import sys

from ctypes import cdll
from pathlib import Path


def get_shared_lib_name(name: str) -> str:
    sys_name = platform.system()
    if sys_name == "Linux":
        return f"lib{name}.so"
    if sys_name == "Darwin":
        return f"lib{name}.dylib"
    if sys_name == "Windows":
        return f"{name}.dll"


def find_file(_file: str, *directories: Path) -> Path:
    """Find a file in directories"""
    for d in directories:
        path = d / _file
        if path.exists():
            return path.absolute()
    raise EnvironmentError(f"Cannot find {_file} in directories {directories}")


if platform.system() == "Windows":
    dirs = [
        Path(__file__).parent,
        Path(sys.prefix, "Library", "bin"),
    ]
    if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
        for _dir in dirs:
            os.add_dll_directory(str(_dir))
    dll_path = str(find_file("nlopt.dll", *dirs))
    print("---***", dll_path)
    cdll.LoadLibrary(dll_path)  # to guarantee dll loading
elif platform.system() == "Linux":
    if (
        os.environ.get("LD_LIBRARY_PATH") is None
    ):  # a user can use other location for the MKL and nlopt libraries.

        def preload_library(_lib_path):
            suffixes = ".so.2 .so.1 .so.0 .so".split()
            for s in suffixes:
                p = _lib_path.with_suffix(s)
                if p.exists():
                    return cdll.LoadLibrary(str(p))
            return None

        libs = list(
            map(lambda x: Path(sys.prefix, "lib", f"{x}.so"), ["libmkl_rt", "libnlopt"])
        )

        for lib in libs:
            loaded_lib = preload_library(lib)
            assert (
                loaded_lib is not None
            ), f"The library should be either available at {lib}, or with LD_LIBRARY_PATH"


from .body import Body, Shape  # noqa
from .fmesh import FMesh  # noqa
from .material import AVOGADRO, Composition, Element, Material  # noqa
from .parser.mctal_parser import read_mctal  # noqa
from .parser.meshtal_parser import read_meshtal  # noqa
from .surface import (  # noqa
    Cone,
    Cylinder,
    GQuadratic,
    Plane,
    Sphere,
    Torus,
    create_surface,
)
from .transformation import Transformation  # noqa
from .universe import Universe  # noqa
from .version import (  # noqa
    __author__,
    __copyright__,
    __license__,
    __summary__,
    __title__,
    __version__,
)

__doc__ = __summary__
