"""The mckit package code root."""
from __future__ import annotations

import mckit._init_dynamic_libraries as init_lib

from mckit._init_dynamic_libraries import (
    Body,
    Cone,
    Cylinder,
    GQuadratic,
    Plane,
    Shape,
    Sphere,
    Torus,
    create_surface,
)
from mckit.fmesh import FMesh
from mckit.material import AVOGADRO, Composition, Element, Material
from mckit.parser.mctal_parser import read_mctal
from mckit.parser.meshtal_parser import read_meshtal
from mckit.transformation import Transformation
from mckit.universe import Universe
from mckit.version import (
    __author__,
    __copyright__,
    __license__,
    __summary__,
    __title__,
    __version__,
)
