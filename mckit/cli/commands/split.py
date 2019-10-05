# -*- coding: utf-8 -*-

"""
Разложение модели на teкстовые секции.

Читает модель, извлекает и раскладывает в указанную директорию отедельно файлы для ячеек, поверхностей,
материалов, трансформаций, sdef и прочие карты. Файлы соответственно: cells.txt, surfaces.txt,
materials.txt, transformations.txt, sdef.txt, cards.txt
"""
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


def distribute_cards(cards: tp.Iterable[sp.Card]) -> tp.Tuple[tp.List,tp.List,tp.List]:
    comment = None
    materials, transformations, sdef, others = [], [], [], []

    def append(_cards, _card):
        global comment
        if comment:
            _cards.append(comment)
            comment = None
        _cards.append(_card)

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
        else:
            append(others, card)

    if comment:
        others.append(comment)

    return materials, transformations, sdef, others


def split(output:Path, source, override:bool):
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





