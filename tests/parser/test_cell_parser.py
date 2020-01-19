from typing import Optional, List
import pytest
import mckit.parser.cell_parser as clp
from mckit.body import (
    Body, TGeometry
)
from mckit.parser.common import (
    CellStrictIndex, CellDummyIndex,
    DummySurface, SurfaceStrictIndex, SurfaceDummyIndex,
    DummyMaterial, DummyComposition, CompositionStrictIndex,
    Index
)
from mckit.material import Material
from mckit.transformation import Transformation


def create_cell(
        cell_no: int,
        geometry: TGeometry,
        material: Optional[Material] = None,
        transformation: Optional[Transformation] = None,
        **options
):
    if not options:
        options = dict()
    options['name'] = cell_no

    def convert_integers_to_surfaces(_geometry):
        for i, e in enumerate(_geometry):
            if isinstance(e, int):
                _geometry[i] = DummySurface(e)
            elif isinstance(e, list):
                convert_integers_to_surfaces(e)

    convert_integers_to_surfaces(geometry)

    if material:
        options['MAT'] = material
    if transformation:
        options['TRCL'] = transformation

    return Body(geometry, **options)


@pytest.mark.parametrize("text,expected_types,expected_values", [
    (
            "1 0",
            ["INTEGER", "ZERO"],
            [1, 0]
    ),
    (
            "1 1 -0.083",
            ["INTEGER", "INTEGER", "FLOAT"],
            [1, 1, -0.083]
    ),
])
def test_cell_lexer(text, expected_types, expected_values):
    lexer = clp.Lexer()
    tokens = list(lexer.tokenize(text))
    actual_types = list(f.type for f in tokens)
    actual_values = list(f.value for f in tokens)
    assert actual_types == expected_types
    assert actual_values == expected_values


@pytest.mark.parametrize("text,expected,surfaces", [
    ("1 0 1", create_cell(1, [1]), [1]),
    ("1 0 -1", create_cell(1, [1, 'C']), [1]),
    ("1 0 1 2", create_cell(1, [1, 2, 'I']), [1, 2]),
    ("1 0 1 -2", create_cell(1, [1, 2, 'C', 'I']), [1, 2]),
    ("1 0 1  2 -3", create_cell(1, [1, 2, 'I', 3, 'C', 'I']), [1, 2, 3]),
    ("1 0 (1  2) : 3", create_cell(1, [1, 2, 'I', 3, 'U']), [1, 2, 3]),
    ("1 0 (1  2) : (3 4)", create_cell(1, [1, 2, 'I', 3, 4, 'I', 'U']), [1, 2, 3, 4]),
])
def test_parser_geometry(text, expected, surfaces):
    surfaces_index = create_dummy_surface_index(surfaces)
    actual = clp.parse(text, surfaces=surfaces_index)
    assert actual == expected


@pytest.mark.parametrize("text,expected,surfaces,compositions", [
    (
            "1 1 -1.0 1",
            create_cell(1, [1], material=DummyMaterial(1, density=1.0)),
            [1],
            [1],
    ),
    (
            "1 1 .08 1",
            create_cell(1, [1], material=DummyMaterial(1, concentration=0.08)),
            [1],
            [1],
    ),
])
def test_parser_with_materials(text, expected, surfaces, compositions):
    surfaces_index = create_dummy_surface_index(surfaces)
    composition_index = create_dummy_composition_index(compositions)
    actual = clp.parse(text, surfaces=surfaces_index, compositions=composition_index)
    assert actual == expected


@pytest.mark.parametrize("text,expected,surfaces", [
    (
            "1 0 1 IMP:n=1.0",
            create_cell(1, [1], **{"IMPN": 1.0}),
            [1],
    ),
    (
            "1 0 1 vol 1.0",
            create_cell(1, [1], **{"VOL": 1.0}),
            [1],
    ),
])
def test_parser_with_attributes(text, expected, surfaces):
    surfaces_index = create_dummy_surface_index(surfaces)
    actual = clp.parse(text, surfaces=surfaces_index)
    assert actual == expected


@pytest.mark.parametrize("text,expected,surfaces,cells", [
    (
            "2 like 1 but imp:p=2.0",
            create_cell(2, [1], **{"IMPN": 1.0, "IMPP": 2.0}),
            [1],
            [create_cell(1, [1], **{"IMPN": 1.0})]
    ),
])
def test_parser_with_like_spec(text, expected, surfaces, cells):
    surfaces_index = create_dummy_surface_index(surfaces)
    cells_index = CellStrictIndex.from_iterable(cells)
    actual = clp.parse(text, cells=cells_index, surfaces=surfaces_index)
    assert actual == expected


@pytest.mark.parametrize("text,expected", [
    (
        """93    0   ((
            16 24   -87 92     -89 -90 -91 :
            16 24   -93 88     -89 -90 -91 :
            16 24    94 93 -95 -89 -90 -91 :
            16 24   95 -96     -89 -90 -91 (97:98:99:100:101) :
            16 24   96 -92     -89 -90 -91 (102:103:104:105:106) 
          )
          (
            -466:467:468:-469                                                   $ UPPER PORT OPENING THROUGH BIOSHIELD, CENTRAL
          )
          (
            -482:483:484:-485                                                   $ UPPER PORT OPENING THROUGH BIOSHIELD, Y+ [-20-DEGREE CLOCKWISE]
          )
          (
            -498:499:500:-501                                                   $ UPPER PORT OPENING THROUGH BIOSHIELD, Y- [+20-DEGREE CLOCKWISE]
          )
          (
           -514:515:516:-517                                                    $ EQ. PORT OPENING THROUGH BIOSHIELD, CENTRAL
          )
          (
           -530:531:532:-533                                                    $ EQ. PORT OPENING THROUGH BIOSHIELD, Y+ [-20-DEGREE CLOCKWISE]
          )
          (
           -546:547:548:-549                                                    $ EQ. PORT OPENING THROUGH BIOSHIELD, Y- [+20-DEGREE CLOCKWISE]
          )
          (
           -562:563:564:-565                                                    $ LOWER PORT OPENING THROUGH BIOSHIELD, CENTRAL
          )
          (                                                                     $ LOWER PORT PLUG (Y-)
            1451:-1452:1453:-1454 : -85 -74 : 79 74 :-434:91
          )
C                                                                               $ LOWER PORT PLUG (Y+)
           (90:-16:1456:-1455:1457:-1460: -1458 1459 )
           (1455:1461:-16:1462:
             -1463:-1466: 1464 1465 :-1467:-1468: 1469 1470 :75)
           (-1461:-16:-1466:-1471:1472:1473: 75 76 74 : -81 -74 ))
C
          (
            -565:+563:-16:-482                                                  $ LOWER PORT OPENING THROUGH BIOSHIELD, Y+ [-20-DEGREE CLOCKWISE]
          )
          IMP:N=1.000000  IMP:P=1.000000
          FILL=93
        """, 93
    )
])
def test_failures(text, expected):
    surfaces_index = SurfaceDummyIndex()
    cells_index = CellDummyIndex()
    actual = clp.parse(text, cells=cells_index, surfaces=surfaces_index)
    assert actual is not None
    assert actual.name() == expected



def create_dummy_surface_index(surfaces: List[int]) -> Index:
    return SurfaceStrictIndex.from_iterable(map(DummySurface, surfaces))


def create_dummy_composition_index(compositions: List[int]) -> Index:
    return CompositionStrictIndex.from_iterable(map(DummyComposition, compositions))


if __name__ == '__main__':
    pytest.main()
