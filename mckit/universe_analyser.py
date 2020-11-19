from typing import cast, List

from mckit import Body, Composition, Transformation
from mckit.universe import Universe, collect_transformations
from mckit.surface import Surface
from mckit.utils.Index import IndexOfNamed, ignore_equal_objects_strategy

from mckit.utils.named import Name


class UniverseAnalyser:
    def __init__(self, universe: Universe):
        self.universe = universe
        universes = self.universe.get_universes()
        self.universes_index = IndexOfNamed.from_iterable(
            universes,
            on_duplicate=ignore_equal_objects_strategy,
        )
        self.cell_index = IndexOfNamed[Name, Body].from_iterable(
            cast(List[Body], map(list, *universes))
        )
        self.surface_index = IndexOfNamed[Name, Surface].from_iterable(
            universe.get_surfaces(inner=True),
            on_duplicate=ignore_equal_objects_strategy,
        )
        self.composition_index = IndexOfNamed[Name, Composition].from_iterable(
            universe.get_compositions(),
            on_duplicate=ignore_equal_objects_strategy,
        )
        self.transformation_index = IndexOfNamed[Name, Transformation].from_iterable(
            collect_transformations(universe, recursive=True),
            on_duplicate=ignore_equal_objects_strategy,
        )

    # cells_universe_map: Dict[Name, List[Name]]
    # surfaces_universe_map: Dict[Name, List[Name]]
    # compositions_universe_map: Dict[Name, List[Name]]
    # transformations_universe_map: Dict[Name, List[Name]]
