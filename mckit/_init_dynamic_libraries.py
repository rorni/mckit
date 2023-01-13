from typing import Generator, List

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


SUFFIX = ".dll" if WIN else ".dylib" if MACOS else ".so"

if WIN or MACOS:

    def combine_version_and_suffix(version: int, suffix: str) -> str:
        return f".{version}{suffix}"  # .2.dll or .2.dylib

else:  # Linux

    def combine_version_and_suffix(version: int, suffix: str) -> str:
        return f"{suffix}.{version}"  # .so.2


def iterate_suffixes_with_version(max_version: int = 2) -> Generator["str", None, None]:
    while max_version >= 0:
        yield combine_version_and_suffix(max_version, SUFFIX)
        max_version -= 1
    yield SUFFIX


SHARED_LIBRARY_DIRECTORIES: List[Path] = []

if WIN:
    SHARED_LIBRARY_DIRECTORIES.append(Path(sys.prefix, "Library", "bin"))
else:
    ld_library_path = os.environ.get("DYLD_LIBRARY_PATH" if MACOS else "LD_LIBRARY_PATH")
    if ld_library_path:
        SHARED_LIBRARY_DIRECTORIES.extend(map(Path, ld_library_path.split(":")))
    SHARED_LIBRARY_DIRECTORIES.append(Path(sys.prefix, "lib"))

SHARED_LIBRARY_DIRECTORIES.append(HERE)


def preload_library(lib_name: str, max_version: int = 2) -> None:
    for d in SHARED_LIBRARY_DIRECTORIES:
        for s in iterate_suffixes_with_version(max_version):
            p = Path(d, library_base_name(lib_name)).with_suffix(s)
            if p.exists():
                cdll.LoadLibrary(str(p))
                _LOG.info("Found library: {}", p.absolute())
                return
    raise EnvironmentError(f"Cannot preload library {lib_name}")


def init():
    if WIN:
        if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
            for _dir in SHARED_LIBRARY_DIRECTORIES:
                os.add_dll_directory(str(_dir))
    preload_library("mkl_rt", max_version=2)
    preload_library("nlopt", max_version=0)


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
