# -*- coding: utf-8 -*-

from .universe import Universe
from .surface import create_surface, Plane, Sphere, Cylinder, Cone, Torus, \
    GQuadratic
from .body import Shape, Body
from .transformation import Transformation
from .material import AVOGADRO, Element, Composition, Material
from .fmesh import FMesh
from .parser.mcnp_input_parser import read_mcnp
from .parser.meshtal_parser import read_meshtal
from .parser.mctal_parser import read_mctal

__all__ = [
    'Universe', 'create_surface', 'Plane', 'Sphere', 'Cylinder', 'Cone',
    'Torus', 'GQuadratic', 'Shape', 'Body', 'Transformation', 'AVOGADRO',
    'Element', 'Composition', 'Material', 'read_meshtal', 'read_mctal',
    'FMesh', 'read_mcnp'
]

__title__ = 'mckit'
__author__ = 'Roman Rodionov'
__license__ = 'MIT'
__copyright__ = 'Copyright 2018-2019 Roman Rodinov'
__ver_major__ = 1
__ver_minor__ = 0
__ver_patch__ = 0
__version_info__ = (__ver_major__, __ver_minor__, __ver_patch__)
__ver_sub__ = ''
__version__ = "%d.%d.%d%s" % (__ver_major__, __ver_minor__, __ver_patch__, __ver_sub__)
