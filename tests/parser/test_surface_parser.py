import pytest
import mckit.parser.surface_parser as srp
from mckit.surface import (
    Cone, Cylinder, Plane, Sphere, GQuadratic, Torus, create_surface
)


def test_surface_parser():
    srp.parse()


if __name__ == '__main__':
    pytest.main()
