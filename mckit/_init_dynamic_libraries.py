from typing import List, Optional

import os
import sys

from ctypes import cdll
from enum import IntEnum
from logging import getLogger
from pathlib import Path

_LOG = getLogger(__name__)


class Platform(IntEnum):
    windows = 0
    linux = 1
    darwin = 2

    @classmethod
    def define(cls) -> "Platform":
        platform_name = sys.platform
        if platform_name == "win32":
            platform_name = "windows"
        return cls[platform_name]

    def library_base_name(self, lib_name: str) -> str:
        if self is Platform.windows:
            return lib_name
        return "lib" + lib_name

    def suffixes(self) -> List[str]:
        if self is Platform.windows:
            return [".dll"]

        if PLATFORM is Platform.linux:
            return ".so.2 .so.1 .so.0 .so".split()

        if PLATFORM is Platform.darwin:
            return ".2.dylib .1.dylib .0.dylib .dylib".split()

    def library_directories(self) -> List[Path]:

        # if a library is built and installed to python environment
        if self is Platform.windows:
            lib_dirs = [Path(sys.prefix, "Library", "lib")]
        else:
            lib_dirs = [Path(sys.prefix, "lib")]

        # if a prebuilt library is installed with pip from mckit wheel
        lib_dirs.append(Path(__file__).parent)

        return lib_dirs

    def ld_library_path_variable(self) -> Optional[str]:

        if self is Platform.linux:
            return "LD_LIBRARY_PATH"

        if self is Platform.darwin:
            return "DYLD_LIBRARY_PATH"

        return None

    def preload_library(self, lib_name: str) -> None:
        for d in self.library_directories():
            for s in self.suffixes():
                p = Path(d, self.library_base_name(lib_name)).with_suffix(s)
                if p.exists():
                    cdll.LoadLibrary(str(p))
                    _LOG.info("Found library: {}", p.absolute())
                    return
        return None

    def init(self) -> None:

        if self is not Platform.windows:
            variable = self.ld_library_path_variable()
            if os.environ.get(variable) is not None:
                _LOG.info(
                    "Using library load path from environment variable {}", variable
                )
                return

        for lib_name in ["mkl_rt", "nlopt"]:
            self.preload_library(lib_name)


PLATFORM = Platform.define()


def init():
    _LOG.debug("Working on platform {}", PLATFORM.name)
    PLATFORM.init()
