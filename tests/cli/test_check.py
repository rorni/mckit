from collections import namedtuple
import logging
import click_log
import pytest
from click.testing import CliRunner
from mckit.utils.resource import filename_resolver
from mckit.cli.runner import mckit,  __version__
from mckit.cli.commands.common import get_default_output_directory

# skip the pylint warning on fixture names
# pylint: disable=redefined-outer-name

# skip the pylint warning on long names: test names should be descriptive
# pylint: disable=invalid-name


test_logger = logging.getLogger(__name__)
click_log.basic_config(test_logger)
test_logger.level = logging.DEBUG


@pytest.fixture
def runner():
    return CliRunner()


data_filename_resolver = filename_resolver('tests')


def test_when_there_is_no_args(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(mckit, args=['check'], catch_exceptions=False)
        assert result.exit_code != 0, "Should fail when no arguments provided"
        assert 'Usage:' in result.output


def test_not_existing_mcnp_file(runner):
    result = runner.invoke(mckit, args=["check", "not-existing.imcnp"], catch_exceptions=False)
    assert result.exit_code > 0
    assert "Path \"not-existing.imcnp\" does not exist" in result.output


@pytest.mark.parametrize("source, expected", [
    (
        "cli/data/simple_cubes.mcnp",
        "cells;surfaces;compositions"
    ),
])
def test_good_path(runner, source, expected):
    source = data_filename_resolver(source)
    result = runner.invoke(mckit, args=['check', source], catch_exceptions=False)
    assert result.exit_code == 0, "Should success"
    for e in expected.split(';'):
        assert e in result.output


