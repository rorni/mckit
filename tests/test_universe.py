import pytest

from mckit.universe import Universe
from mckit.parser.mcnp_input_parser import read_mcnp


@pytest.fixture(scope='module', params=['tests/universe1.i'])
def universe(request):
    return read_mcnp(request.param)


class TestUniverse:
    def test_transform(self):
        raise NotImplementedError

    def test_apply_fill(self):
        raise NotImplementedError

    def test_transform(self):
        raise NotImplementedError

    def test_simplify(self):
        raise NotImplementedError

    def test_get_surfaces(self):
        raise NotImplementedError

    def test_get_materials(self):
        raise NotImplementedError

    def test_get_universes(self, universe):
        u = universe.get_universes()
        assert u == {1, 2}


