# -*- coding: utf-8 -*-

"""
Разложение модели на teкстовые секции.

Читает модель, извлекает и раскладывает в указанную директорию отедельно файлы для ячеек, поверхностей,
материалов, трансформаций, sdef и прочие карты. Файлы соответственно: cells.txt, surfaces.txt,
materials.txt, transformations.txt, sdef.txt, cards.txt
"""
from attr import attrs, attrib
from datetime import datetime
import logging
from pathlib import Path
import mckit.parser.mcnp_section_parser as sp
import click
import mckit as mk
import typing as tp
from .common import check_if_path_exists, save_mcnp, get_default_output_directory, MCNP_ENCODING


def print_text(text, output, base_name, override):
    if  text:
        out = output / base_name
        check_if_path_exists(out, override)
        with open(out, "w", encoding=MCNP_ENCODING) as fid:
            print(text, file=fid)


def print_cards(cards, output, base_name, override):
    if  cards:
        out = output / base_name
        check_if_path_exists(out, override)
        with open(out, "w", encoding=MCNP_ENCODING) as fid:
            for c in cards:
                print(c.text, file=fid)


@attrs
class _distributor(object):
    """
        Need this class because inner function append() somehow doesn't see comment variable
    """
    comment: sp.Card = None

    def append(self, cards: tp.List[sp.Card], card: sp.Card):
        if self.comment:
            cards.append(self.comment)
            self.comment = None
        cards.append(card)

    def __call__(self, cards: tp.Iterable[sp.Card]) -> tp.Tuple[tp.List,tp.List,tp.List]:
        self.comment = None
        materials, transformations, sdef, others = [], [], [], []

        for card in cards:
            if card.is_comment:
                assert self.comment is None
                self.comment = card
            elif card.is_material:
                self.append(materials, card)
            elif card.is_transformation:
                self.append(transformations, card)
            elif card.is_sdef:
                self.append(sdef, card)
            else:
                self.append(others, card)

        if self.comment:
            others.append(self.comment)

        self.comment = None

        return materials, transformations, sdef, others


def distribute_cards(cards: tp.Iterable[sp.Card]) -> tp.Tuple[tp.List,tp.List,tp.List]:
    return _distributor()(cards)


def split(output:Path, source, override:bool, separators=False):
    logger = logging.getLogger(__name__)
    logger.debug("Splitting model from %s", source)
    source = Path(source)
    assert output.is_dir()
    with open(source, encoding=MCNP_ENCODING) as fid:
        sections: sp.InputSections = sp.parse_sections(fid)
    print_text(sections.title, output, "title.txt", override)
    print_cards(sections.cell_cards, output, "cells.txt", override)
    print_cards(sections.surface_cards, output, "surfaces.txt", override)
    if sections.data_cards:
        materials, transformations, sdef, others = distribute_cards(sections.data_cards)
        print_cards(materials, output, "materials.txt", override)
        print_cards(transformations, output, "transformations.txt", override)
        print_cards(sdef, output, "sdef.txt", override)
        print_cards(others, output, "cards.txt", override)
    print_text(sections.remainder, output, "remainder.txt", override)
    logger.debug("The parts of %s are saved to %s", source, output)
    if separators:
        model = source.stem
        write_separators(output, model)


def write_separators(output: Path, model: str):
    outer_line = "=" * 40
    inner_line = "-" * 40
    for kind in "cells surfaces materials transformations".split():
        for start_end in "start end".split():
            if start_end == "start":
                first_line = outer_line
                second_line = inner_line
            else:
                first_line = inner_line
                second_line = outer_line
            text = (
                "c\n" +
                "c   " + first_line + "\n" +
                "c\n" +
                f"c   {start_end} of {model} {kind}\n" +
                "c\n" +
                "c   " + second_line + "\n" +
                "c\n"
            )
            path: Path = output / f"{kind}_{start_end}.txt"
            path.write_text(text, encoding=MCNP_ENCODING)
    path: Path = output / "new_line.txt"
    path.write_text("\n")





