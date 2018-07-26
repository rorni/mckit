import unittest


@unittest.skip
class TestUniverse(unittest.TestCase):
    def test_universe_creation(self):
        raise NotImplementedError

    def test_model_reading(self):
        raise NotImplementedError

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

    def test_get_universes(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
