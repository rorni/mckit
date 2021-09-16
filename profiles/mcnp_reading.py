"""
Code to profile on large mcnp files.
Not using pytest.
"""
from zipfile import ZipFile

from mckit import Universe
from mckit.constants import MCNP_ENCODING
from mckit.parser.mcnp_input_parser import read_mcnp_text
from mckit.parser.mcnp_input_sly_parser import ParseResult, from_text
from mckit.utils.resource import path_resolver

data_filename_resolver = path_resolver("benchmarks")
with ZipFile(data_filename_resolver("data/4M.zip")) as data_archive:
    CLITE_TEXT = data_archive.read("clite.i").decode(encoding=MCNP_ENCODING)


def test_old_mcnp_reading():
    universe: Universe = read_mcnp_text(CLITE_TEXT)
    assert len(universe) == 150


def test_sly_mcnp_reading():
    result: ParseResult = from_text(CLITE_TEXT)
    assert (
        result.title
        == "C-LITE VERSION 1 RELEASE 131031 ISSUED 31/10/2013 - Halloween edition"
    )
    assert len(result.universe) == 150


if __name__ == "__main__":
    test_old_mcnp_reading()
    test_sly_mcnp_reading()
