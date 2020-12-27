try:
    import importlib_metadata as meta
except ImportError:
    import importlib.metadata as meta  # type: ignore

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
__version__ = __distribution__.version
#
# The version from metadata may have several formats:
#  - in release version (there are no prepatch suffixes) it will be plain 1.5.0 for example
#  - in prepatch version running separately 1.5.0-alpha1
#  - in prepatch versions test runs it will be 1.5.0a1
#
