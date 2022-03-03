import pytest

from click.testing import CliRunner
from mckit.cli.logging import logger


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def disable_log():
    try:
        logger.disable("mckit.cli")
        yield
    finally:
        logger.enable("mckit.cli")
