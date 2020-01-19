import pytest
from zipfile import ZipFile
from mckit.utils.resource import path_resolver
from mckit.constants import MCNP_ENCODING
from mckit import read_mcnp_text, Universe
from mckit.parser.mcnp_input_sly_parser import ParseResult, from_text

data_filename_resolver = path_resolver('benchmarks')


def load_source():
    inp = ZipFile(data_filename_resolver("data/data.zip"))
    source = inp.read("cmodel.i")
    return source.decode(encoding=MCNP_ENCODING)


SOURCE = load_source()


def f(delay=1.0):
    return load_source()


def test_old_mcnp_reading(benchmark):
    result: Universe = benchmark(read_mcnp_text, SOURCE)
    pass


def test_sly_mcnp_reading(benchmark):
    result: ParseResult = benchmark(from_text, SOURCE)
    pass






if __name__ == '__main__':
    pytest.main()
