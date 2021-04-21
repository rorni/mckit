from typing import List

import sys

from contextlib import contextmanager
from pathlib import Path

import click
import mckit.version as meta

from click_loguru import ClickLoguru
from mckit.cli.commands import (
    do_check,
    do_compose,
    do_decompose,
    do_split,
    do_transform,
)
from mckit.cli.commands.common import get_default_output_directory
from mckit.utils import MCNP_ENCODING
from mckit.utils.logging import logger

NAME = meta.__title__
VERSION = meta.__version__
LOG_FILE_RETENTION = 3
NO_LEVEL_BELOW = 30


def stderr_log_format_func(msgdict):
    """Do level-sensitive formatting.

    Just a copy from click-loguru so far."""

    if msgdict["level"].no < NO_LEVEL_BELOW:
        return "<level>{message}</level>\n"
    return "<level>{level}</level>: <level>{message}</level>\n"


click_loguru = ClickLoguru(
    NAME,
    VERSION,
    stderr_format_func=stderr_log_format_func,
    retention=LOG_FILE_RETENTION,
    log_dir_parent=".logs",
    timer_log_level="info",
)

context = {}


@click_loguru.logging_options
@click.group(help=meta.__summary__)
@click_loguru.init_logger()
@click_loguru.stash_subcommand()
@click.option("--override/--no-override", default=False)
@click.version_option(VERSION, prog_name=NAME)
@logger.catch(reraise=True)
def mckit(
    verbose: bool, quiet: bool, logfile: bool, profile_mem: bool, override: bool
) -> None:
    if quiet:
        logger.level("WARNING")
    if verbose:
        logger.level("TRACE")
    logger.info("Running {}", NAME)
    logger.debug("Working dir {}", Path(".").absolute())
    #
    # TODO dvp: add customized logger configuring from a configuration toml-file.
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # obj = ctx.ensure_object(dict)
    # obj["DEBUG"] = debug
    context["OVERRIDE"] = override


@mckit.command()
@click.option(
    "--output", "-o", default=None, help="Output directory, default: <source>.universes"
)
@click.option(
    "--fill-descriptor",
    "-f",
    default="fill-descriptor.toml",
    help="TOML file for FILL descriptors",
)
@click.argument(
    "source", metavar="<source>", type=click.Path(exists=True), nargs=1, required=True
)
def decompose(output, fill_descriptor, source):
    """Separate an MCNP model to envelopes and filling universes"""
    logger.info(f"Processing {source}")
    return do_decompose(output, fill_descriptor, source, context["OVERRIDE"])


@mckit.command()
@click.option("--output", "-o", required=True, help="Output file")
@click.option("--fill-descriptor", default=None, help="TOML file for FILL descriptors")
@click.argument(
    "source", metavar="<source>", type=click.Path(exists=True), nargs=1, required=True
)
def compose(output, fill_descriptor, source):
    """Merge universes and envelopes into MCNP model using merge descriptor"""
    if fill_descriptor is None:
        fill_descriptor = Path(source).absolute().parent / "fill-descriptor.toml"
    else:
        fill_descriptor = Path(fill_descriptor)
    if not fill_descriptor.exists():
        raise click.UsageError(f'Cannot find fill descriptor file "{fill_descriptor}"')
    logger.info(
        'Composing "{output}", from envelopes "{source}" with fill descriptor "{fill_descriptor}"',
        output=output,
        source=source,
        fill_descriptor=fill_descriptor,
    )
    return do_compose(output, fill_descriptor, source, context["OVERRIDE"])


@mckit.command()
@click.option("--output", "-o", default=None, help="Output directory")
@click.option(
    "--separators/--no-separators",
    default=False,
    help="Write files with decorative comments to separate this model sections (cells, surfaces etc.) on concatenation",
)
@click.argument(
    "source", metavar="<source>", type=click.Path(exists=True), nargs=1, required=True
)
def split(output, source, separators):
    """Splits MCNP model to text portions (opposite to concat)"""
    if output is None:
        output = get_default_output_directory(source, ".split")
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger.info(
        'Splitting "{source}" to directory "{output}"', source=source, output=output
    )
    return do_split(output, source, context["OVERRIDE"], separators)


# noinspection PyCompatibility
@contextmanager
def resolve_output(output, exist_ok=False, encoding=MCNP_ENCODING):
    if output:
        output = Path(output)
        if exist_ok or not output.exists():
            fid = output.open("w", encoding=encoding)
        else:
            raise click.UsageError(
                f"Specify --override option to override existing file '{output}'"
            )
    else:
        fid = sys.stdout
    try:
        yield fid
    finally:
        if fid is not sys.stdout:
            fid.close()


# noinspection PyCompatibility
@mckit.command()
@click_loguru.init_logger()
@click.option(
    "--output",
    "-o",
    metavar="<output>",
    type=click.Path(exists=False),
    required=False,
    help="File to write the concatenated parts",
)
@click.option(
    "--parts-encoding",
    metavar="<output>",
    type=click.Path(exists=False),
    required=False,
    default=MCNP_ENCODING,
    help=f"Encoding to read parts (default:{MCNP_ENCODING})",
)
@click.option(
    "--output-encoding",
    metavar="<output>",
    type=click.Path(exists=False),
    required=False,
    default=MCNP_ENCODING,
    help=f"Encoding to write output (default:{MCNP_ENCODING})",
)
@click.argument(
    "parts", metavar="<part...>", type=click.Path(exists=True), nargs=-1, required=True
)
def concat(output, parts_encoding, output_encoding, parts):
    """Concat text files. (will filter texts according specification in future)"""
    override = context["OVERRIDE"]
    with resolve_output(output, exist_ok=override, encoding=output_encoding) as out_fid:
        for f in parts:
            f = Path(f)
            # TODO dvp: Add filtering of a part's text here. Implement as external scripts call.
            #           Should be configurable
            print(f.read_text(encoding=parts_encoding), file=out_fid, end="")


# noinspection PyCompatibility
@mckit.command()
@click_loguru.init_logger()
@click.argument(
    "sources", metavar="<source>", type=click.Path(exists=True), nargs=-1, required=True
)
def check(sources: List[click.Path]) -> None:
    """Read MCNP model(s) and show statistics and clashes."""
    for source in sources:
        do_check(source)


# noinspection PyCompatibility
@mckit.command()
@click_loguru.init_logger()
@click.option(
    "--transformation",
    "-t",
    type=click.INT,
    required=True,
    help="Transformation in MCNP format",
)
@click.option(
    "--output",
    "-o",
    type=click.STRING,
    required=True,
    help="Output file",
)
@click.option(
    "--transformations",
    "-i",
    default="transformations.txt",
    type=click.Path(exists=True),
    help="Output file",
)
@click.argument(
    "source", metavar="<source>", type=click.Path(exists=True), nargs=1, required=True
)
def transform(
    output: click.STRING,
    transformation: click.STRING,
    transformations: click.Path,
    source: click.Path,
) -> None:
    """Transform MCNP model(s) with one of specified transformatio."""
    do_transform(
        Path(output),
        transformation,
        Path(str(transformations)),
        Path(str(source)),
        context["OVERRIDE"],
    )
    logger.info("File {} is transformed to {}", source, output)


if __name__ == "__main__":
    mckit(obj={})
