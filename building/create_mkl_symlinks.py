"""Add symbolic links to MKL shared libraries.

INTEL PyPA package supporting team refused to provide links to the libraries with proper names.
https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/pypi-package-for-mkl-2021-2-0-is-not-linkable-under-Linux-and/m-p/1275110?profile.language=ru
This approach on Linux/Darwin MKL requires to provide full path to the library for linking.

From their comment:

    g++ mkl_sample.o /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so.1 -lpthread -lm -ldl

I did that in our build scripts and this worked. But than I've found that we need CMakeLists based
building framework. Goal: to enable work with both Python and C code in powerful IDEs
and eventually migrate to pybind11 or nanobind.

It appeared that CMake find_library function doesn't recognize the MKL libraries.

The script creates the symbolic links with "correct" names to make both linker and cmake happy about MKL.
Run this script from Cmake script before searching for MKL.
"""

import sys

from pathlib import Path

from building.extension_utils import MACOS, WIN

if WIN:
    raise OSError("No need to create symlinks to MKL libraries on Windows")

if MACOS:
    pattern = "libmkl_*.dylib"
    suffix = ".dylib"
else:
    pattern = "libmkl_*.so*"
    suffix = ".so"


def create_mkl_symlinks() -> None:
    """Create symbolic links to MKL libraries installed in the Python instance.

    See module documentation :ref:`create_mkl_symlinks`.
    """
    lib_dir = Path(sys.prefix) / "lib"
    for p in filter(Path.is_file, lib_dir.glob(pattern)):
        link = lib_dir / (p.name.split(".")[0] + suffix)
        if link.is_symlink() or link.exists():
            continue
        print(link, "->", p)
        link.symlink_to(p)


if __name__ == "__main__":
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)
    create_mkl_symlinks()


__all__ = ["create_mkl_symlinks"]
