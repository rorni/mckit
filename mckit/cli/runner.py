from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys

import click
import click_log

from mckit import __version__
from mckit.cli.commands import do_decompose

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
@click.option("--output", "-o", default="universes", help="Output directory")
@click.option("--fill-descriptor", "-o", default="fill-descriptor.toml", help="TOML file for FILL descriptors")
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


if __name__ == '__main__':
    mckit(obj={})
