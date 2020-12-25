import os
import platform
import sys

import importlib.metadata as meta

from distutils.sysconfig import get_python_inc


import numpy as np

from extension_utils import SYSTEM_WINDOWS, get_dirs, insert_directories
from setuptools import Extension

mkl_distribution = meta.distribution("mkl")
site_packages = mkl_distribution.locate_file(".")

include_dirs = get_dirs("INCLUDE_PATH")
include_dirs = insert_directories(include_dirs, np.get_include())
include_dirs = insert_directories(include_dirs, get_python_inc())
library_dirs = get_dirs("LIBRARY_PATH")

geometry_dependencies = [
    "mkl_rt",
    "nlopt",
]

if SYSTEM_WINDOWS:
    include_dirs = insert_directories(
        include_dirs, os.path.join(sys.prefix, "Library/include")
    )
    library_dirs = insert_directories(
        library_dirs, os.path.join(sys.prefix, "libs")
    )  # for PythonXX.dll
    library_dirs = insert_directories(
        library_dirs, os.path.join(sys.prefix, "Library/lib")
    )
    extra_compile_args = ["/O2"]
else:
    if platform.system() != "Linux":
        print(
            f"--- WARNING: the build scenario is not tested on platform {platform.system()}.",
            "             Trying the scenario for Linux.",
            sep="\n",
        )
    include_dirs = insert_directories(include_dirs, os.path.join(sys.prefix, "include"))
    library_dirs = insert_directories(library_dirs, os.path.join(sys.prefix, "lib"))
    extra_compile_args = ["-O3", "-w"]


geometry_sources = [
    os.path.join("mckit", "src", src)
    for src in ["geometrymodule.c", "box.c", "surface.c", "shape.c", "rbtree.c"]
]

geometry_extension = Extension(
    "mckit.geometry",
    sources=geometry_sources,
    include_dirs=include_dirs,
    libraries=geometry_dependencies,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    language="c",
)
