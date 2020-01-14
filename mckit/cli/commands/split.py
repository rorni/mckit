# -*- coding: utf-8 -*-

"""
Разложение текста модели на секции.

Читает модель, извлекает и раскладывает в указанную директорию отедельно файлы для ячеек, поверхностей,
материалов, трансформаций, sdef и прочие карты. Файлы соответственно: cells.txt, surfaces.txt,
materials.txt, transformations.txt, sdef.txt, cards.txt
"""
import logging
from pathlib import Path
import typing as tp
from typing import List, Iterable

import mckit.parser.mcnp_section_parser as sp
from mckit.parser.mcnp_section_parser import Card
from .common import check_if_path_exists, MCNP_ENCODING

OUTER_LINE = "=" * 40
INNER_LINE = "-" * 40


def print_text(
        text: str,
        output_dir: Path,
        section_file_name: str,
        override: bool
) -> None:
    if text:
        out = output_dir / section_file_name
        check_if_path_exists(out, override)
        out.write_text(text, encoding=MCNP_ENCODING)


def print_cards(
        cards: Iterable[Card],
        output_dir: Path,
        section_file_name: str,
        override: bool
) -> None:
    if cards:
        out = output_dir / section_file_name
        check_if_path_exists(out, override)
        with out.open("w", encoding=MCNP_ENCODING) as fid:
            for card in cards:
                print(card.text, file=fid)


def distribute_cards(
        cards: Iterable[Card]
) -> tp.Tuple[List[Card], List[Card], List[Card], List[Card], List[Card]]:
    comment: tp.Optional[sp.Card] = None

    def append(_cards: List[Card], _card: Card) -> None:
        nonlocal comment
        if comment:
            _cards.append(comment)
            comment = None
        _cards.append(_card)

    materials, transformations, sdef, tallies, others = [], [], [], [], []
    # type: List[Card], List[Card], List[Card], List[Card], List[Card]

    for card in cards:
        if card.is_comment:
            assert comment is None
            comment = card
        elif card.is_material:
            append(materials, card)
        elif card.is_transformation:
            append(transformations, card)
        elif card.is_sdef:
            append(sdef, card)
        elif card.is_tally:
            append(tallies, card)
        else:
            append(others, card)

    if comment:
        others.append(comment)

    return materials, transformations, sdef, tallies, others


def split(
        output_dir: Path,
        mcnp_file_name: tp.Union[str, Path],
        override: bool,
        separators=False,
) -> None:
    logger = logging.getLogger(__name__)
    logger.debug("Splitting model from %s", mcnp_file_name)
    if isinstance(mcnp_file_name, str):
        mcnp_file_name = Path(mcnp_file_name)
    assert output_dir.is_dir()
    with open(mcnp_file_name, encoding=MCNP_ENCODING) as fid:
        sections: sp.InputSections = sp.parse_sections(fid)
    print_text(sections.title, output_dir, "title.txt", override)
    print_cards(sections.cell_cards, output_dir, "cells.txt", override)
    print_cards(sections.surface_cards, output_dir, "surfaces.txt", override)
    if sections.data_cards:
        materials, transformations, sdef, tallies, others = distribute_cards(sections.data_cards)
        print_cards(materials, output_dir, "materials.txt", override)
        print_cards(transformations, output_dir, "transformations.txt", override)
        print_cards(sdef, output_dir, "sdef.txt", override)
        print_cards(tallies, output_dir, "tallies.txt", override)
        print_cards(others, output_dir, "cards.txt", override)
    print_text(sections.remainder, output_dir, "remainder.txt", override)
    logger.debug("The parts of %s are saved to %s", mcnp_file_name, output_dir)
    if separators:
        write_separators(output_dir, mcnp_file_name.stem)


def write_separators(output: Path, model: str) -> None:
    for section in "cells surfaces materials transformations tallies".split():
        for start_end in "start end".split():
            if start_end == "start":
                first_line = OUTER_LINE
                second_line = INNER_LINE
            else:
                first_line = INNER_LINE
                second_line = OUTER_LINE
            text = (
                "c\n" +
                "c   " + first_line + "\n" +
                "c\n" +
                f"c   {start_end} of {model} {section}\n" +
                "c\n" +
                "c   " + second_line + "\n" +
                "c\n"
            )
            path: Path = output / f"{section}_{start_end}.txt"
            path.write_text(text, encoding=MCNP_ENCODING)
    path = output / "new_line.txt"
    path.write_text("\n")
