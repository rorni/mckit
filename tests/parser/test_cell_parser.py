from typing import Optional, List, Dict
import pytest
import mckit.parser.cell_parser as clp
from mckit.body import (
    Body, TGeometry
)
from mckit.parser.common import (
    CellStrictIndex,
    DummySurface, SurfaceStrictIndex,
    DummyMaterial, DummyComposition, CompositionStrictIndex
)
from mckit.surface import Surface
from mckit.material import Composition, Material
from mckit.transformation import Transformation


# def test_dummy_material():
#     o = DummyMaterial(1, 1.0)
#     assert isinstance(o, Material)

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
    cells_index = CellStrictIndex()
    cells_index.update((c.name(), c) for c in cells)
    actual = clp.parse(text, cells=cells_index, surfaces=surfaces_index)
    assert actual == expected


def create_dummy_surface_index(surfaces: List[int]) -> Dict[int, Surface]:
    surfaces_index = SurfaceStrictIndex()
    surfaces_index.update((s, DummySurface(s)) for s in surfaces)
    return surfaces_index


def create_dummy_composition_index(compositions: List[int]) -> Dict[int, Composition]:
    composition_index = CompositionStrictIndex()
    composition_index.update((s, DummyComposition(s)) for s in compositions)
    return composition_index


if __name__ == '__main__':
    pytest.main()
