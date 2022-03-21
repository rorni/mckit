from typing import Generator

import os

from pathlib import Path

MCNP_ENCODING = "Cp1251"
"""The encoding used in SuperMC when creating MCNP models code. Some symbols are not Unicode."""


def get_root_dir(environment_variable_name, default):
    return Path(os.getenv(environment_variable_name, default)).expanduser()


def make_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_dirs(*dirs: Path) -> Generator[Path, None, None]:
    yield from map(make_dir, dirs)


def check_if_path_exists(p: Path) -> Path:
    if p.exists():
        return p
    raise FileNotFoundError(f'Path "{p}" does not exist')


def check_if_all_paths_exist(*paths: Path) -> Generator[Path, None, None]:
    yield from map(check_if_path_exists, paths)


class FindFileInDirectoriesError(EnvironmentError):
    def __init__(self, _file, directories):
        super().__init__(f"Cannot find {_file} in directories {directories}")
        self.file = _file
        self.directories = directories


def find_file_in_directories(_file: str, *directories: Path) -> Path:
    """Find a file in directories

    Args:
        _file: a file to find
        directories: list of directories to search the file in

    Raises:
        FindFileInDirectoriesError: if the `_file` is not found in the specified `directories`
    """
    for d in directories:
        path = d / _file
        if path.exists():
            return path.absolute()
    raise FindFileInDirectoriesError(_file, directories)
