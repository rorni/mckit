from __future__ import absolute_import, division, print_function, unicode_literals


import logging
import sys

import click
import click_log

from contextlib import contextmanager
from pathlib import Path

from mckit import __version__
from mckit.utils import MCNP_ENCODING
from mckit.cli.commands.common import get_default_output_directory
from mckit.cli.commands import do_decompose, do_compose, do_split

__LOG = logging.getLogger(__name__)
click_log.basic_config(__LOG)
__LOG.level = logging.INFO


if sys.version_info.major < 3:  # pragma: no cover
    click.echo("Only Python 3 is supported in concat_mcnp")
    sys.exit(1)


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--override/--no-override', default=False)
@click.pass_context
@click_log.simple_verbosity_option(__LOG, default="INFO")
@click.version_option(__version__)
def mckit(ctx, debug, override):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    obj = ctx.ensure_object(dict)
    obj['DEBUG'] = debug
    obj['OVERRIDE'] = override


@mckit.command()
@click.pass_context
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--fill-descriptor", "-f", default="fill-descriptor.toml", help="TOML file for FILL descriptors")
@click.argument(
    "source",
    metavar="<source>",
    type=click.Path(exists=True),
    nargs=1,
    required=True,
)
def decompose(ctx, output, fill_descriptor, source):
    __LOG.info(f"Processing {source}")
    return do_decompose(output, fill_descriptor, source, ctx.obj['OVERRIDE'])


@mckit.command()
@click.pass_context
@click.option("--output", "-o", required=True, help="Output file")
@click.option(
    "--fill-descriptor",
    default=None,
    help="TOML file for FILL descriptors",
)
@click.argument(
    "source",
    metavar="<source>",
    type=click.Path(exists=True),
    nargs=1,
    required=True,
)
def compose(ctx, output, fill_descriptor, source):
    if fill_descriptor is None:
        fill_descriptor = Path(source).absolute().parent / "fill-descriptor.toml"
    else:
        fill_descriptor = Path(fill_descriptor)
    if not fill_descriptor.exists():
        raise click.UsageError(f"Cannot find fill descriptor file \"{fill_descriptor}\"")
    __LOG.info(f"Composing \"{output}\", from envelopes \"{source}\" with fill descriptor \"{fill_descriptor}\"")
    return do_compose(output, fill_descriptor, source, ctx.obj['OVERRIDE'])


@mckit.command()
@click.pass_context
@click.option("--output", "-o", default=None, help="Output directory")
@click.argument(
    "source",
    metavar="<source>",
    type=click.Path(exists=True),
    nargs=1,
    required=True,
)
def split(ctx, output, source):
    if output is None:
        output = get_default_output_directory(source, ".split")
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    __LOG.info(f"Splitting \"{source}\" to directory \"{output}\"")
    return do_split(output, source, ctx.obj['OVERRIDE'])


# noinspection PyCompatibility
@contextmanager
def resolve_output(output, exist_ok=False, encoding=MCNP_ENCODING):
    if output:
        output = Path(output)
        if exist_ok or not output.exists():
            fid = output.open("w", encoding=encoding)
        else:
            raise click.UsageError(f"Specify --override option to override existing file '{output}'")
    else:
        fid = sys.stdout
    try:
        yield fid
    finally:
        if fid is not sys.stdout:
            fid.close()


# noinspection PyCompatibility
@mckit.command()
@click.pass_context
@click.option(
    "--output", "-o",
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
    "parts",
    metavar="<part...>",
    type=click.Path(exists=True),
    nargs=-1,
    required=True,
)
def concat(
        ctx,
        output,
        parts_encoding,
        output_encoding,
        parts
):
    override = ctx.obj['OVERRIDE']
    with resolve_output(output, exist_ok=override, encoding=output_encoding) as out_fid:
        for f in parts:
            f = Path(f)
            print(f.read_text(encoding=parts_encoding), file=out_fid, end="")


if __name__ == '__main__':
    mckit(obj={})
