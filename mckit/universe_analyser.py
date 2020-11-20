from typing import cast, List

from mckit import Body, Composition, Transformation
from mckit.universe import Universe, collect_transformations
from mckit.surface import Surface
from mckit.utils.Index import IndexOfNamed, StatisticsCollector

from mckit.utils.named import Name


class UniverseAnalyser:
    def __init__(self, universe: Universe):
        self.universe = universe
        universes = self.universe.get_universes()
        self.universe_duplicates = StatisticsCollector(ignore_equal=True)
        self.universes_index = IndexOfNamed.from_iterable(
            universes,
            on_duplicate=self.universe_duplicates,
        )
        self.cell_duplicates = StatisticsCollector()
        self.cell_index = IndexOfNamed[Name, Body].from_iterable(
            cast(List[Body], sum(map(list, universes), start=[])),
            on_duplicate=self.cell_duplicates,
        )
        self.surface_duplicates = StatisticsCollector(ignore_equal=True)
        self.surface_index = IndexOfNamed[Name, Surface].from_iterable(
            universe.get_surfaces_list(inner=True),
            on_duplicate=self.surface_duplicates,
        )
        self.composition_duplicates = StatisticsCollector(ignore_equal=True)
        self.composition_index = IndexOfNamed[Name, Composition].from_iterable(
            universe.get_compositions(),
            on_duplicate=self.composition_duplicates,
        )
        self.transformation_duplicates = StatisticsCollector(ignore_equal=True)
        self.transformation_index = IndexOfNamed[Name, Transformation].from_iterable(
            collect_transformations(universe, recursive=True),
            on_duplicate=self.transformation_duplicates,
        )

    # cells_universe_map: Dict[Name, List[Name]]
    # surfaces_universe_map: Dict[Name, List[Name]]
    # compositions_universe_map: Dict[Name, List[Name]]
    # transformations_universe_map: Dict[Name, List[Name]]
