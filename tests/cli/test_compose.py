from mckit.cli.runner import mckit,  __version__
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


def test_not_existing_envelops_file(runner):
    result = runner.invoke(mckit, args=["compose", "not-existing.imcnp"], catch_exceptions=False)
    assert result.exit_code > 0
    assert "Path \"not-existing.imcnp\" does not exist" in result.output


def test_when_output_is_not_specified(runner):
    source = data_filename_resolver("data/simple_cubes.mcnp")
    result = runner.invoke(mckit, args=["compose", str(source)], catch_exceptions=False)
    assert result.exit_code > 0
    assert "Missing option \"--output\"" in result.output


@pytest.mark.parametrize("source, output, expected", [
    ("data/universes/envelops.i", "simple_cubes_restored.i", "data/simple_cubes.mcnp"),
])
def test_when_fill_descriptor_is_not_specified(runner, source, output, expected):
    source = data_filename_resolver(source)
    with runner.isolated_filesystem():
        result = runner.invoke(mckit, args=['compose', "--output", output, source], catch_exceptions=False)
        assert result.exit_code == 0, "Should success using fill_descriptor in the same directory as source file"
        assert Path(output).exists(), \
            f"Should create file {output} file"
        actual = mk.read_mcnp(output)
        expected = mk.read_mcnp(expected)
        assert actual.has_equivalent_cells(expected), "Cells differ"


# @pytest.mark.parametrize("source,expected", [
#     ("cli/data/simple_cubes.mcnp", "envelops.i u1.i u2.i".split()),
# ])
# def test_when_only_source_is_specified(runner, source, expected):
#     source = data_filename_resolver(source)
#     with runner.isolated_filesystem():
#         result = runner.invoke(mckit, args=['decompose', source], catch_exceptions=False)
#         assert result.exit_code == 0, \
#             "Should success without specified output: " + result.output
#         output = Path("universes")
#         for f in expected:
#             assert (output / f).exists(), \
#                 f"Should store the file {f} in the default directory 'universes'"
#
#
# @pytest.mark.parametrize("source,output,expected", [
#     ("cli/data/simple_cubes.mcnp", "split-1", "envelops.i u1.i u2.i".split()),
# ])
# def test_when_output_is_specified(runner, source, output, expected):
#     source = data_filename_resolver(source)
#     with runner.isolated_filesystem():
#         output = Path(output)
#         assert not output.exists(), f"The {output} directory should not exist before mckit run"
#         result = runner.invoke(mckit, args=['decompose', "--output", str(output), source], catch_exceptions=False)
#         assert result.exit_code == 0
#         assert output.exists(), f"The {output} directory should exist after mckit run"
#         for f in expected:
#             assert (output / f).exists(), \
#                 f"Should store the file {f} in the default directory 'universes'"
#
#
# def test_when_output_file_exists_and_override_is_not_specified(runner):
#     source = data_filename_resolver("cli/data/simple_cubes.mcnp")
#     with runner.isolated_filesystem() as prefix:
#         output = Path(prefix) / "universes/envelops.i"
#         output.parent.mkdir(parents=True)
#         output.touch(exist_ok=False)
#         result = runner.invoke(
#             mckit,
#             args=["decompose", source],
#             catch_exceptions=False
#         )
#         assert result.exit_code != 0, \
#             "Should fail when output file exists and --override is not specified"
#
#
# def test_when_output_file_exists_and_override_is_specified(runner):
#     source = data_filename_resolver("cli/data/simple_cubes.mcnp")
#     with runner.isolated_filesystem() as prefix:
#         output = Path(prefix) / "universes/envelops.i"
#         output.parent.mkdir(parents=True)
#         output.touch(exist_ok=False)
#         result = runner.invoke(
#             mckit,
#             args=["--override", "decompose", source],
#             catch_exceptions=False
#         )
#         assert result.exit_code == 0, \
#             "Should success when output file exists and --override is specified"
#
#
#
# def test_fill_descriptor(runner):
#     source = data_filename_resolver("cli/data/simple_cubes.mcnp")
#     with runner.isolated_filesystem() as prefix:
#         output = Path(prefix) / "universes/fill-descriptor.toml"
#         result = runner.invoke(
#             mckit,
#             args=["decompose", source],
#             catch_exceptions=False
#         )
#         assert result.exit_code == 0, "Should success"
#         assert output.exists()
#         with output.open() as fid:
#             fill_descriptor = fid.read()
#             assert fill_descriptor.find('simple_cubes.mcnp')
#             fill_descriptor = tk.parse(fill_descriptor)
#             assert 'created' in fill_descriptor
#             assert '2' in fill_descriptor
#             assert 'universe' in fill_descriptor['2']
#             assert 1 == fill_descriptor['2']['universe']
#             assert 'u1.i' == fill_descriptor['2']['file']
#
#
#
# def test_fill_descriptor_when_fill_descriptor_file_is_specified(runner):
#     source = data_filename_resolver("cli/data/simple_cubes.mcnp")
#     with runner.isolated_filesystem() as prefix:
#         output = Path(prefix) / "universes/fill-descriptor-special.toml"
#         result = runner.invoke(
#             mckit,
#             args=["decompose", "--fill-descriptor", str(output), source],
#             catch_exceptions=False
#         )
#         assert result.exit_code == 0, "Should success"
#         assert output.exists()