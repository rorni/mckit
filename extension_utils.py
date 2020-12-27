from typing import Dict, List

import distutils.log as log
import os
import platform
import shutil

from distutils.sysconfig import get_python_inc
from pathlib import Path
from subprocess import check_call

import numpy as np

SYSTEM_WINDOWS = platform.system() == "Windows"


def create_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


np_include = np.get_include()
drop_parts = 5 if SYSTEM_WINDOWS else 6
site = Path(*(Path(np_include).parts[:-drop_parts]))
python_inc = get_python_inc()
include_dirs = [python_inc, np_include]

if SYSTEM_WINDOWS:
    include_dirs.append(str(site / "Library/include"))
    library_dirs = [str(site / "libs"), str(site / "Library/lib")]
    extra_compile_args = ["/O2"]
else:
    if platform.system() != "Linux":
        print(
            f"--- WARNING: the build scenario is not tested on platform {platform.system()}.",
            "             Trying the scenario for Linux.",
            sep="\n",
        )
    include_dirs.append(str(site / "include"))
    library_dirs = [str(site / "lib")]
    extra_compile_args = ["-O3", "-w"]


def check_directories(*directories: str) -> None:
    for directory in directories:
        if not Path(directory).is_dir():
            raise EnvironmentError(f"The directory {directory} does not exist")


check_directories(*include_dirs)
check_directories(*library_dirs)
