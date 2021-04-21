# -*- coding: utf-8 -*-
"""
 Read and parse MCNP file text.
"""
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    NewType,
    Optional,
    TextIO,
    Tuple,
    Union,
)

from itertools import repeat
from pathlib import Path

from attr import attrib, attrs
from mckit.card import Card
from mckit.constants import MCNP_ENCODING
from mckit.parser.cell_parser import Body
from mckit.parser.cell_parser import parse as parse_cell
from mckit.parser.common import (
    CellNotFoundError,
    CellStrictIndex,
    CompositionStrictIndex,
    ParseError,
    SurfaceStrictIndex,
    TransformationStrictIndex,
)
from mckit.parser.material_parser import Composition
from mckit.parser.material_parser import parse as parse_composition
from mckit.parser.surface_parser import Surface
from mckit.parser.surface_parser import parse as parse_surface
from mckit.parser.transformation_parser import Transformation
from mckit.parser.transformation_parser import parse as parse_transformation
from mckit.universe import Universe, produce_universes
from mckit.utils.Index import Index

from .mcnp_section_parser import Card as TextCard
from .mcnp_section_parser import (
    InputSections,
    Kind,
    distribute_cards,
    parse_sections_text,
)

T = NewType("T", Any)
T1 = NewType("T1", Any)
T2 = NewType("T2", Any)
Pair = NewType("Pair", Tuple[T1, T2])
YieldGenerator = NewType("YieldGenerator", Generator[T, None, None])


@attrs
class ParseResult:
    universe: Universe = attrib()
    cells: List[Body] = attrib()
    cells_index: CellStrictIndex = attrib()
    surfaces: List[Surface] = attrib()
    surfaces_index: SurfaceStrictIndex = attrib()
    compositions: Optional[List[Composition]] = attrib()
    compositions_index: Optional[CompositionStrictIndex] = attrib()
    transformations: Optional[List[Transformation]] = attrib()
    transformations_index: Optional[TransformationStrictIndex] = attrib()
    sections: InputSections = attrib()

    @property
    def title(self):
        return self.sections.title


def from_file(path: Union[str, Path]) -> ParseResult:
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding=MCNP_ENCODING) as fid:
        return from_stream(fid)


def from_stream(stream: TextIO) -> ParseResult:
    text = stream.read()
    return from_text(text)


def from_text(text: str) -> ParseResult:
    sections: InputSections = parse_sections_text(text)
    if sections.data_cards:
        # fmt: off
        text_compositions, text_transformations, _, _, _ = distribute_cards(
            sections.data_cards
        )  # type: List[TextCard], List[TextCard], List[TextCard], List[TextCard], List[TextCard],
        # fmt: on
        transformations = parse_transformations(text_transformations)
        transformations_index = TransformationStrictIndex.from_iterable(transformations)
        compositions = parse_compositions(text_compositions)
        compositions_index = CompositionStrictIndex.from_iterable(compositions)
    else:
        transformations = None
        transformations_index = None
        compositions = None
        compositions_index = None

    surfaces = parse_surfaces(sections.surface_cards, transformations_index)
    surfaces_index = SurfaceStrictIndex.from_iterable(surfaces)
    cells, cells_index = parse_cells(
        sections.cell_cards, surfaces_index, compositions_index, transformations_index
    )
    universe = produce_universes(
        cells
    )  # TODO dvp: apply title from sections to the topmost universe
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


TSectionGenerator = NewType(
    "TSectionGenerator",
    Generator[Union[Transformation, Composition, Surface], None, None],
)


def join_comments(text_cards: Iterable[TextCard]):
    def _iter():
        comment: Optional[str] = None
        for card in text_cards:
            if card.is_comment:
                assert comment is None, f"Comment is already set {comment[:70]}"
                comment = card.text
            else:
                assert (
                    not card.is_comment
                ), f"Pair of comment is found, second one is: {comment[:70]}"
                yield card, comment
                comment = None

    return list(_iter())


def parse_section(
    text_cards: Iterable[TextCard], expected_kind: Kind, parser: Callable[[str], Card]
) -> TSectionGenerator:
    text_cards_with_comments = join_comments(text_cards)

    def iterator() -> TSectionGenerator:
        for text_card, comment in text_cards_with_comments:
            assert text_card.kind is expected_kind
            try:
                card = parser(text_card.text)
            except (ValueError, ParseError) as ex:
                raise ValueError(f"Failed to parse card '{text_card}'") from ex
            if comment:
                card.options["comment_above"] = comment
            # card.options['original'] = text_card.text
            yield card

    return iterator()


def parse_transformations(text_cards: Iterable[TextCard]) -> List[Transformation]:
    return list(parse_section(text_cards, Kind.TRANSFORMATION, parse_transformation))


def parse_compositions(text_cards: Iterable[TextCard]) -> List[Composition]:
    return list(parse_section(text_cards, Kind.MATERIAL, parse_composition))


def parse_surfaces(
    text_cards: Iterable[TextCard], transformations: Index
) -> List[Surface]:
    def parser(text: str):
        return parse_surface(text, transformations=transformations)

    return list(parse_section(text_cards, Kind.SURFACE, parser))


def extract_number(text_card: TextCard):
    return int(text_card.text.split(maxsplit=1)[0])


class MissedCellsError(RuntimeError):
    def __init__(self, missed_cells: List[int]):
        self.missed_cells = missed_cells
        msg = f"Not found cells: {missed_cells}"
        super().__init__(msg)

    @classmethod
    def from_text_cards(cls, missed_cells: Iterable[TextCard]):
        missed_cells_numbers: List[int] = list(map(extract_number, missed_cells))
        return cls(missed_cells_numbers)


def parse_cells(
    text_cards: List[TextCard],
    surfaces: Index,
    compositions: Index,
    transformations: Index,
) -> Tuple[List[Body], CellStrictIndex]:
    text_cards_with_comments = join_comments(text_cards)
    size = len(text_cards_with_comments)
    cells_index = CellStrictIndex()
    cells_to_process = list(range(size))
    cells = list(repeat(None, size))

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
        new_cells_to_process = []
        for i in cells_to_process:
            text_card, comment = text_cards_with_comments[i]
            assert text_card.kind is Kind.CELL
            try:
                card: Body = parser(text_card.text)
                assert card is not None, (
                    "Failed to process cell %s" % text_card.text[:70]
                )
            except CellNotFoundError:
                new_cells_to_process.append(i)
                continue
            if comment:
                card.options["comment_above"] = comment
            card.options["original"] = text_card.text
            cells[i] = card
            cells_index[card.name()] = card
        if cells_to_process_length == len(new_cells_to_process):
            missed_cells_cards = list(text_cards[i] for i in new_cells_to_process)
            raise MissedCellsError.from_text_cards(missed_cells_cards)
        cells_to_process = new_cells_to_process

    if any(map(lambda c: c is None, cells)):
        cells = list(filter(lambda c: c is not None, cells))

    return cells, cells_index
