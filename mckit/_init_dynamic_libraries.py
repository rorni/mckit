from typing import List

import os
import sys
import sysconfig

from ctypes import cdll
from logging import getLogger
from pathlib import Path

_LOG = getLogger(__name__)

HERE = Path(__file__).parent.absolute()
WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()
MACOS = sys.platform.startswith("darwin")


def library_base_name(_lib_name: str) -> str:
    return _lib_name if WIN else "lib" + _lib_name


SUFFIXES = (
    [".dll"]
    if WIN
    else ".2.dylib .1.dylib .0.dylib .dylib".split()
    if MACOS
    else ".so.2 .so.1 .so.0 .so".split()
)


LIBRARY_DIRECTORIES: List[Path] = []

if WIN:
    LIBRARY_DIRECTORIES.append(Path(sys.prefix, "Library", "lib"))
else:
    ld_library_path = os.environ.get(
        "DYLD_LIBRARY_PATH" if MACOS else "LD_LIBRARY_PATH"
    )
    if ld_library_path:
        LIBRARY_DIRECTORIES.extend(map(Path, ld_library_path.split(":")))
    LIBRARY_DIRECTORIES.append(Path(sys.prefix, "lib"))

LIBRARY_DIRECTORIES.append(HERE)


def preload_library(_lib_name: str) -> bool:
    for d in LIBRARY_DIRECTORIES:
        for s in SUFFIXES:
            p = Path(d, library_base_name(_lib_name)).with_suffix(s)
            if p.exists():
                cdll.LoadLibrary(str(p))
                _LOG.info("Found library: {}", p.absolute())
                return True
    return False


def init():
    if WIN:
        if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
            for _dir in LIBRARY_DIRECTORIES:
                os.add_dll_directory(str(_dir))
    for lib_name in ["mkl_rt", "nlopt"]:
        if not preload_library(lib_name):
            raise EnvironmentError(f"Cannot preload library {lib_name}")


init()

import mckit.geometry as geometry  # noqa

from mckit.body import Body, Shape  # noqa
from mckit.surface import (  # noqa
    Cone,
    Cylinder,
    GQuadratic,
    Plane,
    Sphere,
    Torus,
    create_surface,
)
