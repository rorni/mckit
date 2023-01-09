from typing import List, Union

import shutil
import sys
import sysconfig

from pathlib import Path

WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()
MACOS = sys.platform.startswith("darwin")


def get_library_dir(check: bool = False) -> Path:
    root_prefix = Path(sys.prefix)

    if WIN:
        root_prefix = root_prefix / "Library"

    library_dir = root_prefix / "lib"

    if check:
        if not library_dir.is_dir():
            raise EnvironmentError(f"{library_dir} is not a valid directory")

    return library_dir


def create_directory(path: Path, clean: bool = True) -> Path:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def check_directories(*directories: str) -> None:
    for directory in directories:
        if not isinstance(directory, str):
            raise TypeError(f"The value {directory} is not a string")
        if not Path(directory).is_dir():
            raise EnvironmentError(f"The directory {directory} does not exist")


def insert_directories(destination: List[str], value: Union[str, List[str]]) -> List[str]:
    dirs = value
    if not isinstance(value, list):
        dirs = [dirs]
    for old in destination:
        if old not in dirs:
            dirs.append(old)
    return dirs
