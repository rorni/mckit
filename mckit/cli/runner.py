from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import sys
# noinspection PyCompatibility
from pathlib import Path
from contextlib import contextmanager

import click
import click_log

from mckit import __version__
from .commands import do_decompose

__LOG = logging.getLogger(__name__)
click_log.basic_config(__LOG)
__LOG.level = logging.INFO


if sys.version_info.major < 3:  # pragma: no cover
    click.echo("Only Python 3 is supported in concat_mcnp")
    sys.exit(1)

@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
@click_log.simple_verbosity_option(__LOG, default="INFO")
@click.version_option(__version__)
def mckit(ctx, debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug


@mckit.command()
@click.pass_context
@click.argument(
    "source",
    metavar="<source>",
    type=click.Path(exists=True),
    nargs=1,
    required=True,
)
def decompose(ctx, source):
    __LOG.info(f"Processing {source}")
    do_decompose(ctx, source)


if __name__ == '__main__':
    mckit(obj={})
