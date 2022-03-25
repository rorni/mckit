import os
import sys

from ctypes import cdll
from enum import IntEnum
from pathlib import Path


class Platform(IntEnum):
    linux = 0
    darwin = 1
    windows = 2

    @classmethod
    def define(cls) -> "Platform":
        platform = sys.platform
        if platform == "win32":
            platform = "windows"
        return cls[platform]


PLATFORM = Platform.define()

if PLATFORM is Platform.linux:
    SUFFIXES = ".so.2 .so.1 .so.0 .so".split()
    LD_LIBRARY_PATH = "LD_LIBRARY_PATH"
elif PLATFORM is Platform.darwin:
    SUFFIXES = ".2.dylib .1.dylib .0.dylib .dylib".split()
    LD_LIBRARY_PATH = "DYLD_LIBRARY_PATH"
else:
    SUFFIXES = []


def _preload_library(_lib_path, suffixes):
    print("***--- search for library: ", _lib_path)
    for s in suffixes:
        p = _lib_path.with_suffix(s)
        if p.exists():
            print("***--- found library: ", p.absolute())
            return cdll.LoadLibrary(str(p))
    return None


def init() -> None:
    if os.environ.get(LD_LIBRARY_PATH) is not None:
        return
    load_mkl()
    load_nlopt()


def load_nlopt():
    if PLATFORM is Platform.windows:
        dll_path = Path(__file__).parent / "nlopt.dll"
        cdll.LoadLibrary(str(dll_path))
    else:
        lib_path = Path(__file__).parent / "libnlopt"
        lib = _preload_library(lib_path, SUFFIXES)
        assert (
            lib is not None
        ), f"The library {lib_path} should be either available or accessible with LD_LIBRARY_PATH"


def load_mkl():
    if PLATFORM is Platform.windows:
        pass
    else:
        lib_path = Path(sys.prefix, "lib", "libmkl_rt")
        lib = _preload_library(lib_path, SUFFIXES)
        assert (
            lib is not None
        ), f"The library {lib_path} should be either available or accessible with LD_LIBRARY_PATH"


# _init()
