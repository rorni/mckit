import os
import platform
import shutil
import sys

from distutils.sysconfig import customize_compiler, get_config_var, get_python_inc
from pathlib import Path

from extension_nlopt import NLOptBuildExtension
from extension_utils import (
    SYSTEM_WINDOWS,
    create_directory,
    execute_command,
)
from setuptools import Extension
from setuptools.command.build_ext import build_ext

MIN_CMAKE_VERSION = "3.18.4"


class BuildFailed(Exception):
    pass


class MCKitBuilder(build_ext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compiler is None:
            from distutils.ccompiler import new_compiler

            self.compiler = new_compiler(
                compiler=self.compiler, dry_run=self.dry_run, force=self.force
            )
            customize_compiler(self.compiler)

    def run(self):

        if platform.system() not in ("Windows", "Linux", "Darwin"):
            raise EnvironmentError(f"Unsupported os: {platform.system()}")

        for extension in self.extensions:
            self.build_extension(extension)

    @property
    def config(self):
        return "Debug" if self.debug else "Release"

    def build_extension(self, extension: Extension) -> None:

        if not isinstance(extension, NLOptBuildExtension):
            build_ext.build_extension(self, extension)
            return

        ext_dir = Path(self.get_ext_fullpath(extension.name)).parent.absolute()
        _ed = ext_dir.as_posix()
        # - make sure path ends with delimiter
        # - required for auto-detection of auxiliary "native" libs
        if not _ed.endswith(os.path.sep):
            _ed += os.path.sep

        build_dir = create_directory(Path(self.build_temp))

        install_prefix = sys.prefix
        if SYSTEM_WINDOWS:
            install_prefix = os.path.join(install_prefix, "Library")

        print("--- CMake configure")
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
            extension.source_dir.as_posix(),
        ]

        if platform.system() == "Windows":
            cmd.insert(
                2, f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{self.config.upper()}={_ed}"
            )

        execute_command(
            cmd=cmd,
            cwd=build_dir,
            # env={
            #     **os.environ.copy(),
            #     "CXXFLAGS": f'{os.environ.get("CXXFLAGS", "")} -DVERSION_INFO="{self.distribution.get_version()}"',
            # },
        )

        print("--- CMake build")
        execute_command(
            [
                "cmake",
                "--build",
                ".",
                "--config",
                self.config,
                "--",
                "-m" if SYSTEM_WINDOWS else "-j2",
            ],
            cwd=build_dir,
        )

        print("--- CMake install")
        execute_command(
            [
                "cmake",
                "--install",
                ".",
            ],
            cwd=build_dir,
        )

        if SYSTEM_WINDOWS:
            nlopt_dll = str(Path(build_dir) / "Release/nlopt.dll")
            mckit_folder = Path(extension.source_dir).parent.parent / 'mckit'
            assert mckit_folder.is_dir()
            assert (mckit_folder / '__init__.py').exists()
            shutil.copy2(nlopt_dll, str(mckit_folder))
            print(f"--- nlopt.dll is updated in {mckit_folder}")