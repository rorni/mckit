from __future__ import annotations

import os
import sys
import sysconfig

from collections.abc import Generator
from ctypes import cdll
from logging import getLogger
from pathlib import Path

_LOG = getLogger(__name__)

HERE = Path(__file__).parent.absolute()
WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()
MACOS = sys.platform.startswith("darwin")


def _library_base_name(_lib_name: str) -> str:
    return _lib_name if WIN else "lib" + _lib_name


SUFFIX = ".dll" if WIN else ".dylib" if MACOS else ".so"

if WIN or MACOS:

    def _combine_version_and_suffix(version: int, suffix: str) -> str:
        return f".{version}{suffix}"  # .2.dll or .2.dylib

else:  # Linux

    def _combine_version_and_suffix(version: int, suffix: str) -> str:
        return f"{suffix}.{version}"  # .so.2


def _iterate_suffixes_with_version(max_version: int = 2) -> Generator[str, None, None]:
    while max_version >= 0:
        yield _combine_version_and_suffix(max_version, SUFFIX)
        max_version -= 1
    yield SUFFIX


SHARED_LIBRARY_DIRECTORIES: list[Path] = []

if WIN:
    SHARED_LIBRARY_DIRECTORIES.append(Path(sys.prefix, "Library", "bin"))
else:
    ld_library_path = os.environ.get("DYLD_LIBRARY_PATH" if MACOS else "LD_LIBRARY_PATH")
    if ld_library_path:
        SHARED_LIBRARY_DIRECTORIES.extend(map(Path, ld_library_path.split(":")))
    SHARED_LIBRARY_DIRECTORIES.append(Path(sys.prefix, "lib"))

SHARED_LIBRARY_DIRECTORIES.append(HERE)


def _preload_library(lib_name: str, max_version: int = 2) -> None:
    for d in SHARED_LIBRARY_DIRECTORIES:
        for s in _iterate_suffixes_with_version(max_version):
            p = Path(d, _library_base_name(lib_name)).with_suffix(s)
            if p.exists():
                cdll.LoadLibrary(str(p))
                _LOG.info("Found library: {}", p.absolute())
                return
    raise OSError(f"Cannot preload library {lib_name}")


def _init():
    if WIN and hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
        for _dir in SHARED_LIBRARY_DIRECTORIES:
            os.add_dll_directory(str(_dir))
    _preload_library("mkl_rt", max_version=2)
    _preload_library("nlopt", max_version=0)
    # geometry_so = next(Path(sysconfig.get_paths()["purelib"], "mckit").glob("geometry*"))
    # cdll.LoadLibrary(str(geometry_so))


_init()

# We have to import these libraries here, after preloading libraries.
from mckit import geometry
from mckit.body import Body, Shape
from mckit.surface import Cone, Cylinder, GQuadratic, Plane, Sphere, Torus, create_surface
