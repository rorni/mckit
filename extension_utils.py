from typing import Dict, List, Union

import os
import platform
import shutil

from pathlib import Path
from subprocess import check_call

SYSTEM_WINDOWS = platform.system() == "Windows"


def check_cmake_installed() -> None:
    try:
        check_call(["cmake", "--version"])
    except OSError:  # pragma: no cover
        raise EnvironmentError("CMake must be installed to build nlopt")


def create_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def execute_command(
    cmd: List[str], cwd: Path, env: Dict[str, str] = os.environ
) -> None:
    print(cwd.as_posix(), ":", " ".join(cmd))
    check_call(cmd, cwd=cwd, env=env)


def get_dirs(environment_variable: str) -> List[str]:
    dirs = os.environ.get(environment_variable, "")

    if dirs:
        dirs = dirs.split(os.pathsep)
    else:
        dirs = []

    return dirs


def insert_directories(
    destination: List[str], value: Union[str, List[str]]
) -> List[str]:
    dirs = []
    if isinstance(value, list):
        dirs.extend(value)
    elif value not in destination:
        dirs.append(value)
    for old in destination:
        if old not in dirs:
            dirs.append(old)
    return dirs
