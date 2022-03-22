import os
import platform
import sys

from ctypes import cdll
from pathlib import Path

from mckit.utils.io import find_file_in_directories


def _preload_library(_lib_path, suffixes):
    print("***--- search for library: ", _lib_path)
    for s in suffixes:
        p = _lib_path.with_suffix(s)
        if p.exists():
            print("***--- found library: ", p.absolute())
            return cdll.LoadLibrary(str(p))
    return None


def init() -> None:
    system = platform.system()
    if system == "Windows":
        _init_for_windows()
    else:
        libs = list(
            map(
                lambda x: Path(sys.prefix, "lib", x),
                ["libmkl_rt", "libnlopt"],
            )
        )

        if sys.platform == "linux":
            # a user can use other location for the MKL and nlopt libraries.
            if os.environ.get("LD_LIBRARY_PATH") is not None:
                return
            suffixes = ".so.2 .so.1 .so.0 .so".split()
        elif sys.platform == "darwin":
            if os.environ.get("DYLD_LIBRARY_PATH") is not None:
                return
            suffixes = ".2.dylib .1.dylib .0.dylib .dylib".split()
        else:
            raise EnvironmentError(f"Unknown platform: {sys.platform}")

        for lib in libs:
            loaded_lib = _preload_library(lib, suffixes)
            assert (
                loaded_lib is not None
            ), f"The library {lib} should be either available at {Path(sys.prefix, 'lib')}, or with LD_LIBRARY_PATH"


def _init_for_windows():
    dirs = [
        Path(__file__).parent,
        Path(sys.prefix, "Library", "bin"),
    ]
    dll_path = str(find_file_in_directories("nlopt.dll", *dirs))
    if hasattr(os, "add_dll_directory"):  # Python 3.7 doesn't have this method
        for _dir in dirs:
            os.add_dll_directory(str(_dir))
    print("---***", dll_path)
    cdll.LoadLibrary(dll_path)  # to guarantee dll loading


# _init()
