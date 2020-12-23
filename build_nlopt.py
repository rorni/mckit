#!/usr/bin/env python3

import os
import platform
import re
import shutil
import sys

from distutils.sysconfig import get_config_var, get_python_inc
from distutils.version import LooseVersion
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from pathlib import Path
from setuptools import Extension
from subprocess import check_call
from typing import Dict, List
from setuptools.command.build_ext import build_ext


try:
    import numpy as np
except ImportError:
    np = None

MIN_CMAKE_VERSION = "3.18.4"
SYSTEM_WINDOWS = platform.system() == "Windows"


def check_cmake_installed() -> None:
    try:
        check_call(["cmake", "--version"])
    except OSError:  # pragma: no cover
        raise EnvironmentError("CMake must be installed to build nlopt")
    # cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))  # type: ignore
    # return cmake_version


def get_nlopt_version(source_dir: Path) -> str:
    execute_command("git submodule update --init --recursive --depth=1".split(), cwd=Path.cwd())
    with open(source_dir / "CMakeLists.txt") as f:
        content = f.read()
        version = []
        for s in ("MAJOR", "MINOR", "BUGFIX"):
            m = re.search(f"NLOPT_{s}_VERSION *['\"](.+)['\"]", content)
            version.append(m.group(1))
        version = ".".join(version)
        return version


def create_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def execute_command(cmd: List[str], cwd: Path, env: Dict[str, str] = os.environ):
    print(cwd.as_posix(), ':', ' '.join(cmd))
    check_call(cmd, cwd=cwd, env=env)


class NLOptBuildExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])
        # Source dir should be at the root directory
        self.source_dir = Path(__file__).parent.absolute() / "3rd-party" / "nlopt"
        self.version = get_nlopt_version(self.source_dir)


class BuildFailed(Exception):
    pass


class NLOptBuild(build_ext):
    def run(self):

        if platform.system() not in ("Windows", "Linux", "Darwin"):
            raise RuntimeError(f"Unsupported os: {platform.system()}")

        for ext in self.extensions:
            if isinstance(ext, NLOptBuildExtension):
                self.build_extension(ext)

    @property
    def config(self):
        return "Debug" if self.debug else "Release"

    def build_extension(self, ext: Extension):

        if not isinstance(ext, NLOptBuildExtension):
            try:
                build_ext.build_extension(self, ext)
            except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
                raise BuildFailed("Could not compile C extension.")
            return

        check_cmake_installed()

        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        _ed = ext_dir.as_posix()
        # - make sure path ends with delimiter
        # - required for auto-detection of auxiliary "native" libs
        if not _ed.endswith(os.path.sep):
            _ed += os.path.sep

        build_dir = create_directory(Path(self.build_temp))

        install_prefix = sys.prefix
        if SYSTEM_WINDOWS:
            install_prefix = os.path.join(install_prefix, "Library")

        # CMake configure
        cmd = [
            "cmake",
            "-LAH",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            f"-DPYTHON_INCLUDE_DIR={get_python_inc()}",
            f"-DPYTHON_LIBRARY={get_config_var('LIBDIR')}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={_ed}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DNLOPT_GUILE=OFF",
            "-DNLOPT_MATLAB=OFF",
            "-DNLOPT_OCTAVE=OFF",
            ext.source_dir.as_posix()
        ]

        if platform.system() == "Windows":
            cmd.insert(2, f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{self.config.upper()}={_ed}")

        execute_command(
            cmd=cmd,
            cwd=build_dir,
            env={
                **os.environ.copy(),
                "CXXFLAGS": f'{os.environ.get("CXXFLAGS", "")} -DVERSION_INFO="{self.distribution.get_version()}"'
            })

        # CMake build
        execute_command([
            'cmake',
            '--build',
            '.',
            '--config',
            self.config,
            "--",
            "-m" if platform.system() == "Windows" else "-j2"
        ], cwd=build_dir)

        execute_command([
            'cmake',
            'install',
            '.',
        ], cwd=build_dir)

        package_dir = str((Path(__file__).parent / "mckit").absolute())
        if platform.system() == "Windows":
            nlopt_so = str(Path(build_dir) / "Release/nlopt.dll")
            shutil.copy2(nlopt_so, package_dir)
        else:
            for f in Path(build_dir).glob("libnlopt.so*"):
                shutil.copy2(str(f), package_dir)


#         # Copy over the important bits
#         nlopt_py = build_dir / "extern" / "nlopt" / "src" / "swig" / "nlopt.py"
#         if not nlopt_py.exists():
#             raise RuntimeError("swig python file was not generated")
#
#         shutil.copy2(nlopt_py, ext_dir / "nlopt.py")
#         with open(ext_dir / "__init__.py", 'w') as f:
#             f.write(f"""
# from .nlopt import *
# __version__ = '{ext.version}'
# """.strip() + "\n")


# def do_build_nlopt(
#     nlopt_source_dir: str, build_directory: str = None, debug=False
# ) -> None:
#     nlopt_path = Path(nlopt_source_dir)
#     if not nlopt_path.exists():
#         raise FileNotFoundError(nlopt_source_dir)
#     if not nlopt_path.is_dir():
#         raise ValueError(f"Path {nlopt_path} is not a directory")
#     cfg = "Debug" if debug else "Release"
#     install_prefix = sys.prefix
#     if SYSTEM_WINDOWS:
#         install_prefix = os.path.join(install_prefix, "Library")
#     cmake_args = [
#         f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
#         f"-DPYTHON_EXECUTABLE={sys.executable}",
#         f"-DPYTHON_INCLUDE_DIR={get_python_inc()}",
#         f"-DPYTHON_LIBRARY={get_config_var('LIBDIR')}",
#     ]
#     build_args = ["--config", cfg]
#     if SYSTEM_WINDOWS:  # pragma: no cover
#         if sys.maxsize > 2 ** 32:
#             cmake_args += ["-A", "x64"]
#         build_args += ["--", "/m"]
#     else:
#         cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
#     env = os.environ.copy()
#     cmd = ["cmake"] + cmake_args + [nlopt_source_dir]
#     print(f"Execute CMake configure: {cmd}")
#     if 0 == check_call(cmd, cwd=build_directory, env=env):
#         cmd = ["cmake", "--build", "."] + build_args
#         print(f"Execute CMake build: {cmd}")
#         if 0 == check_call(cmd, cwd=build_directory):
#             cmd = ["cmake", "--install", "."]
#             print(f"Execute CMake install: {cmd}")
#             if 0 != check_call(cmd, cwd=build_directory):
#                 raise EnvironmentError("Failed to execute CMake build")
#         else:
#             raise EnvironmentError("Failed to execute CMake build")
#     else:
#         raise EnvironmentError("Failed to execute CMake configure")


# def build_nlopt():
#     cmake_version = get_cmake_version()
#     if cmake_version < MIN_CMAKE_VERSION:
#         raise RuntimeError(
#             f"CMake >= {MIN_CMAKE_VERSION} is required: CMake should support `cmake --install .`"
#         )
#     if 0 == subprocess.check_call(
#         "git submodule update --init --recursive --depth=1".split()
#     ):
#         nlopt_path = (Path.cwd() / "3rd-party/nlopt").absolute()
#         build_directory = nlopt_path / "build"
#         build_directory.mkdir(parents=True, exist_ok=True)
#         do_build_nlopt(str(nlopt_path), str(build_directory))
#     else:
#         raise EnvironmentError("Failed to update Git submodules")


#
# if __name__ == "__main__":
#     build_nlopt()
