from typing import Dict, List

import os
import sys

from pathlib import Path
from subprocess import check_call

from extension_utils import SYSTEM_WINDOWS, create_directory  # , python_inc, site


def execute_command(
    cmd: List[str], cwd: Path = Path.cwd(), env: Dict[str, str] = os.environ
) -> None:
    print(f"--- {cwd.as_posix()}: {' '.join(cmd)}")
    check_call(cmd, cwd=cwd, env=env)


def build_nlopt(*, install_prefix: str = None, build_dir: Path = None) -> None:

    source_dir = Path(__file__).parent.absolute() / "3rd-party" / "nlopt"
    if not source_dir.exists():
        execute_command(
            ["git", "submodule", "update", "--init", "--recursive", "--depth=1"]
        )

    # TODO dvp: check if something is to be done to support build isolation, use build_ext.build_temp probably?
    if build_dir is None:
        build_dir = create_directory(source_dir / "build")

    if install_prefix is None:
        install_prefix = sys.prefix
        if SYSTEM_WINDOWS:
            install_prefix = os.path.join(install_prefix, "Library")

    print(f"--- Installing NLOpt to {install_prefix}")

    cmd = [
        "cmake",
        "-LAH",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        # f"-DPYTHON_INCLUDE_DIR={python_inc}",
        # f"-DPYTHON_LIBRARY={get_config_var('LIBDIR')}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DNLOPT_GUILE=OFF",
        "-DNLOPT_MATLAB=OFF",
        "-DNLOPT_OCTAVE=OFF",
        source_dir.as_posix(),
    ]

    # if SYSTEM_WINDOWS:
    #     cmd.insert(
    #         2, f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{self.config.upper()}={_ed}"
    #     )

    execute_command(
        cmd=cmd,
        cwd=build_dir,
    )

    execute_command(
        [
            "cmake",
            "--build",
            ".",
            "--config",
            "Release",
            "--",
            "-m" if SYSTEM_WINDOWS else "-j2",
        ],
        cwd=build_dir,
    )

    execute_command(
        [
            "cmake",
            "--install",
            ".",
        ],
        cwd=build_dir,
    )


if __name__ == "__main__":
    build_nlopt()
