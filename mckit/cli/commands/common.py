import typing as tp
import logging
from pathlib import Path
import click
from mckit import Universe
# This is the encoding swallowing non ascii (neither unicode) symbols happening in MCNP models code
MCNP_ENCODING = "cp1251"


def check_if_path_exists(path: tp.Union[str, Path], override: bool):
    if isinstance(path, str):
        path = Path(path)
    if not override and path.exists():
        logger = logging.getLogger(__name__)
        errmsg = f"""\
        Cannot override existing file \"{path}\".
        Please remove the file or specify --override option"""
        logger.error(errmsg)
        raise click.UsageError(errmsg)


def save_mcnp(model: Universe, path: tp.Union[str, Path], override: bool):
    check_if_path_exists(path, override)
    model.save(path, encoding=MCNP_ENCODING)


def get_default_output_directory(source, suffix):
    return Path(Path(source).with_suffix(suffix).name)


