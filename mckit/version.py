try:
    import importlib_metadata as meta
except ImportError:
    import importlib.metadata as meta

__title__ = "mckit"
__distribution__ = meta.distribution(__title__)
__meta_data__ = __distribution__.metadata
__author__ = __meta_data__["Author"]
__author_email__ = __meta_data__["Author-email"]
__license__ = __meta_data__["License"]
__summary__ = __meta_data__["Summary"]
__copyright__ = (
    "Copyright 2018-2020 Roman Rodionov"  # TODO dvp: move to meta (project.toml)
)

__version__ = meta.version("mckit")

try:
    __version__ = meta.version(__name__)
except meta.PackageNotFoundError:
    __version__ = "0.0.0"

__version_info__ = tuple(
    map(int, __version__.split(".")[:3])
)  # leaving only the 3 first digits from version
