# -*- coding: utf-8 -*-
"""
Apply transformation to a model.
"""
from pathlib import Path

from mckit.utils.logging import logger

from mckit.parser.mcnp_input_sly_parser import from_file, ParseResult
from .common import save_mcnp
from mckit.parser import transformation_parser as trans_parse
from mckit.transformation import Transformation
from mckit.universe import Universe


def transform(transformation, output, source, override):
    logger.info("Transforming model from {s}", s=source)
    if Path(output).exists() and not override:
        raise FileExistsError(
            f"File {output} already exists. Remove it or use --override option"
        )
    parse_result: ParseResult = from_file(source)
    src: Universe = parse_result.universe
    trans: Transformation = trans_parse.parse(transformation)
    dst = src.transform(trans)
    save_mcnp(dst, output, override)
