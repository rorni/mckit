from mckit.cli.runner import mckit, __version__
import logging
import click_log
import pytest
from pathlib import Path
import tomlkit as tk
from click.testing import CliRunner
import mckit as mk
from mckit.utils.resource import filename_resolver

# skip the pylint warning on fixture names
# pylint: disable=redefined-outer-name

# skip the pylint warning on long names: test names should be descriptive
# pylint: disable=invalid-name


test_logger = logging.getLogger(__name__)
click_log.basic_config(test_logger)
test_logger.level = logging.INFO


@pytest.fixture
def runner():
    return CliRunner()


data_filename_resolver = filename_resolver('tests.cli')


def test_help_compose(runner):
    result = runner.invoke(mckit, args=["compose", "--help"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Usage: mckit compose" in result.output


def test_when_there_is_no_args(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(mckit, args=['compose'], catch_exceptions=False)
        assert result.exit_code != 0, "Should fail when no arguments provided"
        assert 'Usage:' in result.output


def test_not_existing_envelopes_file(runner):
    result = runner.invoke(mckit, args=["compose", "not-existing.imcnp"], catch_exceptions=False)
    assert result.exit_code > 0
    assert "Path \"not-existing.imcnp\" does not exist" in result.output


def test_when_output_is_not_specified(runner):
    source = data_filename_resolver("data/simple_cubes.mcnp")
    result = runner.invoke(mckit, args=["compose", str(source)], catch_exceptions=False)
    assert result.exit_code > 0
    assert "Missing option \"--output\"" in result.output


@pytest.mark.parametrize("source, output, expected", [
    ("data/universes/envelopes.i", "simple_cubes_restored.i", "data/simple_cubes.mcnp"),
])
def test_when_fill_descriptor_is_not_specified(runner, source, output, expected):
    source = data_filename_resolver(source)
    with runner.isolated_filesystem():
        result = runner.invoke(mckit, args=['compose', "--output", output, source], catch_exceptions=False)
        assert result.exit_code == 0, "Should success using fill_descriptor in the same directory as source file"
        assert Path(output).exists(), \
            f"Should create file {output} file"
        actual = mk.read_mcnp(output)
        expected = mk.read_mcnp(data_filename_resolver(expected))
        assert actual.has_equivalent_cells(expected), "Cells differ"
