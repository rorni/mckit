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
# VERSION_PATTERN = re.compile(
#     r"""
#     ^
#     v?
#     (?:
#         (?:(?P<epoch>[0-9]+)!)?                           # epoch
#         (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
#         (?P<pre>                                          # pre-release
#             [-_.]?
#             (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
#             [-_.]?
#             (?P<pre_n>[0-9]+)?
#         )?
#         (?P<post>                                         # post release
#             (?:-(?P<post_n1>[0-9]+))
#             |
#             (?:
#                 [-_.]?
#                 (?P<post_l>post|rev|r)
#                 [-_.]?
#                 (?P<post_n2>[0-9]+)?
#             )
#         )?
#         (?P<dev>                                          # dev release
#             [-_.]?
#             (?P<dev_l>dev)
#             [-_.]?
#             (?P<dev_n>[0-9]+)?
#         )?
#     )
#     (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?       # local version
#     $
# """,
#     re.IGNORECASE | re.VERBOSE,
# )
