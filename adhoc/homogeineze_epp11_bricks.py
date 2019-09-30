import typing as tp
import numpy as np
import mckit as mk
from pathlib import Path

universes_to_process = [97]  # , 98, 99]


def get_path(uninverse_name: int) -> Path:
    return Path(f"u{uninverse_name:d}.i")


def load_universe(universe_name: int) -> mk.Universe:
    universe_path = get_path(universe_name)
    assert universe_path.exists()
    universe = mk.read_mcnp(universe_path)
    return universe

def process_universe(universe_name: int, min_volume=0.001) -> tp.NoReturn:
    universe = load_universe(universe_name)
    cell: mk.Body = universe[0]
    volume0 = cell.shape.volume(min_volume=min_volume)
    print(f"Volume of the root cell: {volume0}")
    universe.apply_fill(cell.name())
    universe.simplify(min_volume=0.1)
    total_volume = 0.0
    material_index = {}
    material_volume_map = {}
    for cell in universe:
        material = cell.material()
        volume = cell.shape.volume(min_volume=min_volume)
        total_volume += volume
        if material is not None:
            material_name = material.composition.name()  # TODO dvp: reverse dependency Composition-Material?
            if material_name not in material_index:
                material_index[material_name] = material
            material_volume = material_volume_map.get(material_name, 0.0) + volume
            material_volume_map[material_name] = material_volume
    print(f'Total volume: {total_volume}')
    material_volume_fractions = []
    for k, v in material_volume_map.items():
        material_volume_fractions.append((material_index[k], v/volume0))
    mixed_material = mk.Material.mixture(*material_volume_fractions, fraction_type='volume')
    composition = mixed_material.composition
    composition.rename(universe_name)
    print(
f"""c
c Material for filler #{universe_name}
c Density {mixed_material.density:.3g}, g/ccm
c"""
    )
    print(mixed_material.composition.mcnp_repr())


def main():
    for universe_name in universes_to_process:
        process_universe(universe_name)


if __name__ == "__main__":
    main()
