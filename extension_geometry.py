import os
import sys

from functools import partial
from pathlib import Path

from extension_utils import SYSTEM_WINDOWS
from setuptools import Extension

# See MKL linking options for various versions of MKL and OS:
# https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
#
# Windows (oneMKL 2021, MS-C/C++,64bit, c-32bit, shared library, no cluster):
# mkl_intel_lp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb.lib


def _init() -> Extension:
    if SYSTEM_WINDOWS:
        extra_compile_args = ["/O2"]
        _libraries = [
            "mkl_rt",
            "nlopt",
        ]
    else:
        #
        # On Linux MKL requires full path to the library:
        # https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/pypi-package-for-mkl-2021-2-0-is-not-linkable-under-Linux-and/m-p/1275110?profile.language=ru
        #
        lib_dir = Path(sys.prefix, sys.platlibdir)
        if not lib_dir.is_dir():
            raise EnvironmentError(f"{lib_dir} is not a valid directory")
        # TODO dvp: implement logic for other possible suffixes
        mkl_rt_path = lib_dir / "libmkl_rt.so.2"
        if not mkl_rt_path.exists():
            raise EnvironmentError(f"{mkl_rt_path} is not a valid path to MKL library.")
        extra_compile_args = ["-O3", "-w", str(mkl_rt_path)]
        _libraries = [
            "nlopt",
        ]

    return Extension(
        "mckit.geometry",
        sources=list(map(str, Path("mckit", "src").glob("*.c"))),
        libraries=_libraries,
        extra_compile_args=extra_compile_args,
        language="c",
    )


geometry_extension = _init()
