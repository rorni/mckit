import os

from functools import partial

from extension_utils import extra_compile_args
from setuptools import Extension

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
