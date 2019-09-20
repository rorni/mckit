import typing as tp
import logging
from pathlib import Path
import click
from mckit import Universe
# This is the encoding swallowing non ascii (neither unicode) symbols happening in MCNP models code
MCNP_ENCODING = "cp1251"


def save(model: Universe, path: tp.Union[str, Path], override: bool):
    if isinstance(path, str):
        path = Path(path)
    if not override and path.exists():
        logger = logging.getLogger(__name__)
        errmsg = """\
        Cannot override existing file \"{}\".
        Please remove the file or specify --override option"""
        errmsg = errmsg.format(path)
        logger.error(errmsg)
        raise click.UsageError(errmsg)
    else:
        model.save(path, encoding=MCNP_ENCODING)