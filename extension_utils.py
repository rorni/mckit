from typing import List, Union

import platform
import shutil

from pathlib import Path

SYSTEM_WINDOWS = platform.system() == "Windows"

extra_compile_args = ["/O2"] if SYSTEM_WINDOWS else ["-O3", "-w"]


def create_directory(path: Path, clean: bool = True) -> Path:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def check_directories(*directories: str) -> None:
    for directory in directories:
        if not Path(directory).is_dir():
            raise EnvironmentError(f"The directory {directory} does not exist")


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
