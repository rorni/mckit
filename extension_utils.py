import os
import platform
import shutil
import sys
import distutils.log as log
from distutils.sysconfig import get_python_inc
from pathlib import Path

SYSTEM_WINDOWS = platform.system() == "Windows"


def create_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


# try:
#     import importlib_metadata as meta
# except ImportError:
#     import importlib.metadata as meta  # type: ignore
#
# setuptools_dist = meta.distribution('setuptools')
#
# # np_include = np.get_include()
# # drop_parts = 5 if SYSTEM_WINDOWS else 6
# drop_parts = 3
# # this should be path to isolated build
# site = Path(*setuptools_dist.parts[:-drop_parts])


def check_directories(*directories: str) -> None:
    for directory in directories:
        if not Path(directory).is_dir():
            raise EnvironmentError(f"The directory {directory} does not exist")


def compute_extension_properties(build_path: Path):
    """We compute this properties when build_extension is called."""
    log.info(f"--- computing directories for {build_path}")
    python_inc = get_python_inc()
    include_dirs = [python_inc]

    if SYSTEM_WINDOWS:
        include_dirs.append(str(build_path / "Library/include"))
        library_dirs = [
            os.path.join(sys.base_prefix, "libs"),
            str(build_path / "Library/lib"),
        ]
        extra_compile_args = ["/O2"]
    else:
        if platform.system() != "Linux":
            print(
                f"--- WARNING: the build scenario is not tested on platform {platform.system()}.",
                "             Trying the scenario for Linux.",
                sep="\n",
            )
        include_dirs.append(str(build_path / "include"))
        library_dirs = [str(build_path / "lib")]
        extra_compile_args = ["-O3", "-w"]

    check_directories(*include_dirs)
    check_directories(*library_dirs)

    return include_dirs, library_dirs, extra_compile_args
