#!/usr/bin/env python3

import os
import platform
import re
import subprocess
import sys

from distutils.sysconfig import get_config_var, get_python_inc
from distutils.version import LooseVersion
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def get_cmake_version() -> LooseVersion:
    try:
        out = subprocess.check_output(["cmake", "--version"])
    except OSError:  # pragma: no cover
        raise RuntimeError("CMake must be installed to build nlopt")
    cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))  # type: ignore
    return cmake_version


def do_build_nlopt(
    nlopt_source_dir: str, build_directory: str = None, debug=False
) -> None:
    nlopt_path = Path(nlopt_source_dir)
    if not nlopt_path.exists():
        raise FileNotFoundError(nlopt_source_dir)
    if not nlopt_path.is_dir():
        raise ValueError(f"Path {nlopt_path} is not a directory")
    cfg = "Debug" if debug else "Release"
    cmake_args = [
        f"-DCMAKE_INSTALL_PREFIX={sys.prefix}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DPYTHON_INCLUDE_DIR={get_python_inc()}",
        f"-DPYTHON_LIBRARY={get_config_var('LIBDIR')}",
    ]
    build_args = ["--config", cfg]
    if platform.system() == "Windows":  # pragma: no cover
        # cmake_args += [
		#	"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
		#]
        if sys.maxsize > 2 ** 32:
            cmake_args += ["-A", "x64"]
        build_args += ["--", "/m"]
    else:
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
    env = os.environ.copy()
    cmd = ["cmake"] + cmake_args + [nlopt_source_dir]
    print(f"Execute CMake configure: {cmd}")
    if 0 == subprocess.check_call(cmd, cwd=build_directory, env=env):
        cmd = ["cmake", "--build", "."] + build_args
        print(f"Execute CMake build: {cmd}")
        if 0 == subprocess.check_call(cmd, cwd=build_directory):
            cmd = ["cmake", "--install", "."]
            print(f"Execute CMake install: {cmd}")
            if 0 != subprocess.check_call(cmd, cwd=build_directory):
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
    argv = sys.argv[1:]
    if argv:
        build_directory = argv[0]
    else:
        build_directory = nlopt_path / "build"
    build_directory.mkdir(parents=True, exist_ok=True)
    do_build_nlopt(str(nlopt_path), str(build_directory))


# TODO dvp: add more functionality to this build script: debug, clean build and so on.

if __name__ == "__main__":
    build_nlopt()
