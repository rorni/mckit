import pytest
from loguru import logger
from click.testing import CliRunner


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
