import os

from functools import partial

from extension_utils import SYSTEM_WINDOWS, extra_compile_args
from setuptools import Extension

# See MKL linking options for various versions of MKL and OS:
# https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
#
# Windows (oneMKL 2021, MS-C/C++,64bit, c-32bit, shared library, no cluster):
# mkl_intel_lp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb.lib

_libraries = [
    "mkl_rt",
    "nlopt",
]

_sources_root = os.path.join("mckit", "src")
_sources = list(
    map(
        partial(os.path.join, _sources_root),
        ["geometrymodule.c", "box.c", "surface.c", "shape.c", "rbtree.c"],  # noqa
    )
)

geometry_extension = Extension(
    "mckit.geometry",
    sources=_sources,
    libraries=_libraries,
    extra_compile_args=extra_compile_args,
    language="c",
)
