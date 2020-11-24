#!/usr/bin/env python3

# https://stackoverflow.com/questions/60073711/how-to-build-c-extensions-via-poetry
import os
import os.path as path
import platform
import re
import subprocess
import sys
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from distutils.sysconfig import get_python_inc, get_config_var
from distutils.version import LooseVersion
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import tempfile

try:
    import numpy as np
except ImportError:
    np = None

# from setuptools_cpp import CMakeExtension, ExtensionBuilder, Pybind11Extension
# from setuptools_cpp import ExtensionBuilder


def get_cmake_version() -> LooseVersion:
    try:
        out = subprocess.check_output(["cmake", "--version"])
    except OSError:  # pragma: no cover
        raise RuntimeError("CMake must be installed to build nlopt")
    cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))  # type: ignore
    return cmake_version


def do_build_nlopt(nlopt_source_dir, debug=False) -> None:
    nlopt_path = Path(nlopt_source_dir)
    if not nlopt_path.exists():
        raise FileNotFoundError(nlopt_source_dir)
    if not nlopt_path.is_dir():
        raise ValueError(f"Path {nlopt_path} is not a directory")
    cfg = "Debug" if debug else "Release"
    cmake_args = [
        "--config",
        cfg,
        f"-DCMAKE_INSTALL_PREFIX={sys.prefix}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DPYTHON_INCLUDE_DIR={get_python_inc()}",
        f"-DPYTHON_LIBRARY={get_config_var('LIBDIR')}",
    ]
    if platform.system() == "Windows":  # pragma: no cover
        # cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
        if sys.maxsize > 2 ** 32:
            cmake_args += ["-A", "x64"]
    else:
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
    # cmake_args += ["-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_path("include"))]
    env = os.environ.copy()
    # env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), dist_version)
    with tempfile.TemporaryDirectory() as build_temp:
        # Path(build_temp).mkdir(parents=True, exist_ok=True)
        cmd = ["cmake"] + cmake_args + [nlopt_source_dir]
        print(f"Execute CMake configure: {cmd}")
        if 0 == subprocess.check_call(cmd, cwd=build_temp, env=env):
            cmd = ["cmake", "--build", "."]
            print(f"Execute CMake build: {cmd}")
            if 0 == subprocess.check_call(cmd, cwd=build_temp):
                cmd = ["cmake", "--install", "."]
                print(f"Execute CMake install: {cmd}")
                if 0 != subprocess.check_call(cmd, cwd=build_temp):
                    raise EnvironmentError("Failed to execute CMake build")
            else:
                raise EnvironmentError("Failed to execute CMake build")
        else:
            raise EnvironmentError("Failed to execute CMake configure")


def build_nlopt():
    cmake_version = get_cmake_version()
    if cmake_version < "3.19.0":
        raise RuntimeError(
            "CMake >= 3.19.0 is required: CMake should support `cmake --install .`"
        )
    nlopt_path = (Path.cwd() / "3rd-party/nlopt").absolute()
    do_build_nlopt(str(nlopt_path))


def get_dirs(environment_variable):
    dirs = os.environ.get(environment_variable, "")

    if dirs:
        dirs = dirs.split(os.pathsep)
    else:
        dirs = []

    return dirs


def prepend_if_not_present(destination, value):
    if isinstance(value, list):
        destination.extend(value)
    elif value not in destination:
        destination.append(value)


include_dirs = get_dirs("INCLUDE_PATH")

if np is not None:
    prepend_if_not_present(include_dirs, np.get_include())

library_dirs = get_dirs("LIBRARY_PATH")

if sys.platform.startswith("linux"):
    geometry_dependencies = ["mkl_intel_lp64", "mkl_core", "mkl_sequential", "nlopt"]
    python_include_dir = path.join(sys.prefix, "include")
    prepend_if_not_present(include_dirs, python_include_dir)
    prepend_if_not_present(library_dirs, path.join(sys.prefix, "lib"))
else:
    geometry_dependencies = [
        "mkl_intel_lp64_dll",
        "mkl_core_dll",
        "mkl_sequential_dll",
        "libnlopt-0",
    ]
    nlopt_inc = get_dirs("NLOPT")
    prepend_if_not_present(include_dirs, nlopt_inc)
    nlopt_lib = get_dirs("NLOPT")
    prepend_if_not_present(library_dirs, nlopt_lib)
    mkl_inc = sys.prefix + "\\Library\\include"
    prepend_if_not_present(include_dirs, mkl_inc)
    mkl_lib = sys.prefix + "\\Library\\lib"
    prepend_if_not_present(library_dirs, mkl_lib)

geometry_sources = [
    path.join("mckit", "src", src)
    for src in ["geometrymodule.c", "box.c", "surface.c", "shape.c", "rbtree.c"]
]

ext_modules = [
    Extension(
        "mckit.geometry",
        sources=geometry_sources,
        include_dirs=include_dirs,
        libraries=geometry_dependencies,
        library_dirs=library_dirs,
    ),
]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed("File not found. Could not compile C extension.")

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed("Could not compile C extension.")


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    build_nlopt()
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": ExtBuilder},
            "package_data": {"mckit": ["data/isotopes.dat", "libnlopt-0.dll"]},
        }
    )


if __name__ == "__main__":
    build_nlopt()
