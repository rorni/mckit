#!python
from __future__ import annotations

import os
import sys

from pathlib import Path
from subprocess import check_call

sys.path.append(str(Path(__file__).parent.parent))
print(sys.path)

from building.extension_utils import WIN, create_directory


def execute_command(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] = os.environ
) -> None:
    if cwd is None:
        cwd = Path.cwd()
    check_call(cmd, cwd=cwd, env=env)  # noqa: S603


def build_nlopt(
    *, install_prefix: str | None = None, build_dir: Path | None = None, clean=True
) -> Path:
    source_dir = Path(__file__).parent.parent.absolute() / "3rd-party" / "nlopt"
    if not source_dir.exists():
        execute_command(["git", "submodule", "update", "--init", "--recursive", "--depth=1"])

    # TODO dvp: check if something is to be done to support build isolation,
    #           use build_ext.build_temp probably?
    if build_dir is None:
        build_dir = create_directory(source_dir / "build", clean=clean)

    if install_prefix is None:
        install_prefix = sys.prefix
        if WIN:
            install_prefix = os.path.join(install_prefix, "Library")

    cmd = [
        "cmake",
        "-LAH",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DNLOPT_GUILE=OFF",
        "-DNLOPT_MATLAB=OFF",
        "-DNLOPT_OCTAVE=OFF",
        source_dir.as_posix(),
    ]

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
            "-m" if WIN else "-j2",
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

    return build_dir


if __name__ == "__main__":
    build_nlopt()
