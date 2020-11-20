from mckit.universe_analyser import UniverseAnalyser
from mckit.parser import from_file
from mckit.utils import path_resolver

cli_data = path_resolver("tests.cli")


def test_test_universe_analyser():
    universe_path = cli_data("data/shared_surface.mcnp")
    assert universe_path.exists()
    universe = from_file(universe_path).universe
    analyzer = UniverseAnalyser(universe)
    assert 0 == len(analyzer.cell_duplicates)
    assert 0 == len(analyzer.surface_duplicates)
