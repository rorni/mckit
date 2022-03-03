from typing import List

from pathlib import Path

from extension_utils import SYSTEM_WINDOWS, get_library_dir
from setuptools import Extension

# See MKL linking options for various versions of MKL and OS:
# https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
#
# Windows (oneMKL 2021, MS-C/C++,64bit, c-32bit, shared library, no cluster):
# mkl_intel_lp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb.lib


def _make_full_names(lib_dir: Path, mkl_libs: List[str]) -> List[str]:
    """Add library directory and extension to an MKL library name.

    According to the recommendation
    https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/pypi-package-for-mkl-2021-2-0-is-not-linkable-under-Linux-and/m-p/1275110?profile.language=ru
    on Linux MKL requires full path to the library.
    """
    # TODO dvp: implement logic for other possible suffixes in future MKL versions
    lib_paths = list(map(lambda p: lib_dir / f"lib{p}.so.2", mkl_libs))
    for p in lib_paths:
        if not p.exists():
            raise EnvironmentError(f"{p} is not a valid path to an MKL library.")
    return list(map(str, lib_paths))


def _init() -> Extension:
    if SYSTEM_WINDOWS:
        # TODO dvp: check new Intel recommended options
        # mkl_intel_ilp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb12.lib
        # /DMKL_ILP64 -I"%MKLROOT%\include"
        extra_compile_args = ["/O2"]
        _libraries = [
            "mkl_rt",
            "nlopt",
        ]
        extra_link_args = []
    else:
        # Linker options (Linux 64bit, gcc, tbb):
        # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.rhcqqp
        # -L${MKLROOT}/lib/intel64 -Wl, --no-as-needed -lmkl_intel_ilp64 -lmkl_tbb_thread - lmkl_core -lpthread -lm -ldl
        # Compiler options:
        # -DMKL_ILP64 -m64 -I"${MKLROOT}/include"
        lib_dir = get_library_dir(check=True)
        # mkl_libs = [
        #     "mkl_intel_ilp64",
        #     "mkl_tbb_thread",
        #     "mkl_core",
        # ]  # - this is recommended
        mkl_libs = ["mkl_rt"]  # - this works
        extra_compile_args = [
            "-O3",
            "-Wall",
            "-m64",
        ]
        extra_link_args = ["-Wl,--no-as-needed"] + _make_full_names(lib_dir, mkl_libs)
        _libraries = ["nlopt", "pthread", "m", "dl"]

    define_macros = [("MKL_ILP64", None)]
    return Extension(
        "mckit.geometry",
        sources=list(map(str, Path("mckit", "src").glob("*.c"))),
        libraries=_libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
        define_macros=define_macros,
    )


geometry_extension = _init()
