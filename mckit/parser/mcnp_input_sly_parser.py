# -*- coding: utf-8 -*-
"""
 Read and parse MCNP file text.
"""
from attr import attrs, attrib
from itertools import repeat
from pathlib import Path
from typing import Iterable, Union, TextIO, Optional, Generator, Callable, List

from .mcnp_section_parser import (
    parse_sections_text, distribute_cards, Card as TextCard, InputSections, Kind
)
from mckit.constants import MCNP_ENCODING
from mckit.card import Card
from mckit.universe import Universe, produce_universes
from mckit.parser.common import (
    Index, CellStrictIndex, SurfaceStrictIndex, CompositionStrictIndex, TransformationStrictIndex,
    CellNotFoundError,
)
from mckit.parser.transformation_parser import parse as parse_transformation, Transformation
from mckit.parser.material_parser import parse as parse_composition, Composition
from mckit.parser.surface_parser import parse as parse_surface, Surface
from mckit.parser.cell_parser import parse as parse_cell, Body


@attrs
class ParseResult:
    universe: Universe = attrib()
    cells: List[Body] = attrib()
    cells_index: CellStrictIndex = attrib()
    surfaces: List[Surface] = attrib()
    surfaces_index: SurfaceStrictIndex = attrib()
    compositions: List[Composition] = attrib()
    compositions_index: CompositionStrictIndex = attrib()
    transformations: List[Transformation] = attrib()
    transformations_index: TransformationStrictIndex = attrib()
    sections: InputSections = attrib()


def from_file(path: Union[str, Path]) -> ParseResult:
    with open(path, enconding=MCNP_ENCODING) as fid:
        return from_stream(fid)


def from_stream(stream: TextIO) -> ParseResult:
    text = stream.read()
    return from_text(text)


def from_text(text: str) -> ParseResult:
    sections: InputSections = parse_sections_text(text)
    text_materials, text_transformations, _, _, _ = distribute_cards(sections.data_cards)
    # type: List[TextCard], List[TextCard]
    transformations = parse_transformations(text_transformations)
    transformations_index = TransformationStrictIndex.from_iterable(transformations)
    compositions = parse_compositions(text_materials)
    compositions_index = CompositionStrictIndex.from_iterable(compositions)
    surfaces = parse_surfaces(sections.surface_cards, transformations_index)
    surfaces_index = SurfaceStrictIndex.from_iterable(surfaces)
    cells, cells_index = parse_cells(sections.cell_cards, surfaces_index, compositions_index, transformations_index)
    universe = produce_universes(cells)
    return ParseResult(
        universe=universe,
        cells=cells,
        cells_index=cells_index,
        surfaces=surfaces,
        surfaces_index=surfaces_index,
        compositions=compositions,
        compositions_index=compositions_index,
        transformations=transformations,
        transformations_index=transformations_index,
        sections=sections,
    )


def parse_section(
    text_cards: Iterable[TextCard],
    expected_kind: Kind,
    parser: Callable[[str], Card]
) -> Generator[Card]:
    
    def iterator() -> Generator[Card]:
        comment: Optional[str] = None
        for text_card in text_cards:
            if text_card.kind is Kind.comment:
                assert comment is None
                comment = text_card.text
            else:
                assert text_card.kind is expected_kind
                card = parser(text_card.text)
                if comment:
                    card.options['prepending_comment'] = comment
                    comment = None
                card.options['original'] = text_card.text
                yield card
    
    return iterator()


def parse_transformations(text_cards: Iterable[TextCard]) -> List[Card]:
    return list(
        i for i in parse_section(text_cards, Kind.TRANSFORMATION, parse_transformation)
    )


def parse_compositions(text_cards: Iterable[TextCard]) -> List[Card]:
    return list(
       i for i in parse_section(text_cards, Kind.MATERIAL, parse_composition)
    )


def parse_surfaces(text_cards: Iterable[TextCard], transformations: Index) -> SurfaceStrictIndex:

    def parser(text: str):
        return parse_surface(text, transformations=transformations)

    return list(
        i for i in parse_section(text_cards, Kind.SURFACE, parser)
    )


def extract_number(text_card: TextCard):
    return int(text_card.text.split(maxsplit=1)[0])


class MissedCellsError(RuntimeError):
    def __init__(self, missed_cells: List[int]):
        self.missed_cells = missed_cells
        msg = f"Not found cells: {missed_cells}"
        super().__init__(msg)

    @classmethod
    def from_text_cards(cls, missed_cells: Iterable[TextCard]):
        missed_cells_numbers: List[int] = list(i for i in map(extract_number, missed_cells))
        return cls(missed_cells_numbers)


def parse_cells(
        text_cards: List[TextCard],
        surfaces: Index,
        compositions: Index,
        transformations: Index,
) -> SurfaceStrictIndex:
    size = len(text_cards)
    cells_index = CellStrictIndex()
    cells_to_process = list(i for i in range(size))
    cells = list(i for i in repeat(None, size))

    def parser(text: str):
        return parse_cell(
            text,
            cells=cells_index,
            surfaces=surfaces,
            compositions=compositions,
            transformations=transformations,
        )

    while cells_to_process:
        cells_to_process_length = len(cells_to_process)
        comment: Optional[str] = None
        new_cells_to_process = []
        for i in cells_to_process:
            text_card = text_cards[i]
            if text_card.kind is Kind.comment:
                assert comment is None
                comment = text_card.text
                new_cells_to_process.append(i)
            else:
                assert text_card.kind is Kind.CELL
                try:
                    card: Body = parser(text_card.text)
                except CellNotFoundError:
                    new_cells_to_process.append(i)
                    continue
                if comment:
                    card.options['prepending_comment'] = comment
                    comment = None
                    assert new_cells_to_process[-1] == i - 1, "Comment should be in cards to process list"
                    new_cells_to_process = new_cells_to_process[:-1]
                card.options['original'] = text_card.text
                cells[i] = card
                cells_index[card.name()] = card
        if cells_to_process_length == len(new_cells_to_process):
            missed_cells_cards = list(text_cards[i] for i in new_cells_to_process)
            raise MissedCellsError(missed_cells_cards)
        cells_to_process = new_cells_to_process
    cells = list(c for c in cells if c is not None)
    return cells, cells_index

