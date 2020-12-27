import os

from functools import partial

from extension_utils import extra_compile_args, include_dirs, library_dirs
from setuptools import Extension

geometry_dependencies = [
    "mkl_rt",
    "nlopt",
]

geometry_sources = ["geometrymodule.c", "box.c", "surface.c", "shape.c", "rbtree.c"]
geometry_sources_root = os.path.join("mckit", "src")
geometry_sources = list(
    map(partial(os.path.join, geometry_sources_root), geometry_sources)
)

geometry_extension = Extension(
    "mckit.geometry",
    sources=geometry_sources,
    include_dirs=include_dirs,
    libraries=geometry_dependencies,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    language="c",
)
