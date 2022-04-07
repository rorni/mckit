import os
import sys
import sysconfig

from ctypes import cdll
from logging import getLogger
from pathlib import Path

_LOG = getLogger(__name__)


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


LIBRARY_DIRECTORIES = (
    [Path(sys.prefix, "Library", "lib")] if WIN else [Path(sys.prefix, "lib")]
) + [Path(__file__).parent]


def preload_library(_lib_name: str) -> None:
    for d in LIBRARY_DIRECTORIES:
        for s in SUFFIXES:
            p = Path(d, library_base_name(_lib_name)).with_suffix(s)
            if p.exists():
                cdll.LoadLibrary(str(p))
                _LOG.info("Found library: {}", p.absolute())
                return


def init():
    if WIN:
        if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
            for _dir in LIBRARY_DIRECTORIES:
                os.add_dll_directory(str(_dir))
    else:
        ld_library_path_variable = "DYLD_LIBRARY_PATH" if MACOS else "LD_LIBRARY_PATH"
        if os.environ.get(ld_library_path_variable) is not None:
            _LOG.info(
                "Using library load path from environment variable {}",
                ld_library_path_variable,
            )
        for lib_name in ["mkl_rt", "nlopt"]:
            preload_library(lib_name)


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
