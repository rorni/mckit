"""
"""
import sys
import os

# import matplotlib.pyplot as plt
# import seaborn as sb
# import pandas as pd
# import numpy as np
# import scipy as sp
# import scipy.constants as sc
import typing as tp
from typing import Union, NoReturn, List, Callable
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from joblib import (
    Memory,
    # Parallel, delayed, wrap_non_picklable_objects, effective_n_jobs
)
from joblib.externals.loky import set_loky_pickler

import dotenv
from pathlib import Path
import numpy as np

sys.path.append("..")

import mckit as mk
from mckit import *
import mckit.geometry as mg


def assert_all_paths_exist(*paths):
    def apply(p: Path):
        assert p.exists(), "Path \"{}\" doesn't exist".format(p)
    map(apply, paths)


def make_dirs(*dirs):
    def apply(d: Path):
        d.mkdir(parents=True, exist_ok=True)
    map(apply, dirs)



def get_root_dir(environment_variable_name, default):
    return Path(os.getenv(environment_variable_name, default)).expanduser()


dotenv.load_dotenv(dotenv_path=".env", verbose=True)
HFSR_ROOT = get_root_dir("HFSR_ROOT", "~/dev/mcnp/hfsr")
CMODEL_ROOT = get_root_dir("CMODEL_ROOT", "~/dev/mcnp/c-model")

assert_all_paths_exist(HFSR_ROOT, CMODEL_ROOT)

# NJOBS = effective_n_jobs()
# print(f"NJOBS: {NJOBS}")
# set_loky_pickler()

def is_sorted(a: np.ndarray) -> bool:
    return np.all(np.diff(a) > 0)


def select_from(cell: mk.Body, to_select: np.ndarray) -> bool:
    name: int = cell.name()
    index: int = to_select.searchsorted(name)
    return index < to_select.size and to_select[index] == name


def decorate_select(
    select: tp.Union[List, np.ndarray, Callable[[mk.Body, np.ndarray], bool]]
) -> Callable[[mk.Body, np.ndarray], bool]:
    if select is not None:
        if isinstance(select, list):
            select = np.array(select, dtype=np.int32)
        if isinstance(select, np.ndarray):
            assert is_sorted(select)
            assert 0 < select[0]
            return lambda c: select_from(c, select)
    return select


class BoundingBoxAdder(object):

    def __init__(self, tolerance: float, add_boxes_to):
        self.tolerance = tolerance
        self.select = decorate_select(add_boxes_to)
        self.add_boxes_to = add_boxes_to

    def __call__(self, cell: mk.Body):
        if self.select is None or self.select(cell):
            cell.bounding_box = cell.shape.bounding_box(tol=self.tolerance)

    def __getstate__(self):
        return self.tolerance, self.add_boxes_to

    def __setstate__(self, state):
        tolerance, add_boxes_to = state
        self.__init__(tolerance, add_boxes_to)



def attach_bounding_boxes(model: mk.Universe, tolerance: float = 10.0, add_boxes_to=None) -> NoReturn:
    # pool: ThreadPool = ThreadPool()
    pool: Pool = Pool()
    bba = BoundingBoxAdder(tolerance, add_boxes_to)
    pool.map(bba, model)



# def attach_bounding_boxes(model: mk.Universe, tolerance: float = 1.0) -> tp.NoReturn:
#     boxes = Parallel(n_jobs=NJOBS, backend='multiprocessing')(
#         delayed(compute_bounding_box)(c.shape, tolerance) for c in model
#     )
#     for _i, cell in enumerate(model):
#         cell.bounding_box = boxes[_i]


mem = Memory(location=".cache", verbose=2)


@mem.cache
def load_model(path: str, tolerance: float = 10.0, add_boxes_to=None) -> mk.Universe:
    # The cp1251-encoding reads C-model with various kryakozyabrs
    model: mk.Universe = read_mcnp(path, encoding="Cp1251")
    attach_bounding_boxes(model, tolerance, add_boxes_to)
    return model


def subtract_model_from_model(
    minuend: mk.Universe,
    subtrahend: mk.Universe,
) -> mk.Universe:
    new_universe = minuend.copy()
    changed = False
    for _i, a_cell in enumerate(new_universe):
        new_cell = subtract_model_from_cell(a_cell, subtrahend, simplify=True)
        if new_cell is not a_cell:
            changed = True
            new_universe[_i] = new_cell
    if changed:
        return new_universe
    else:
        return minuend


def subtract_model_from_cell(
    cell: mk.Shape,
    model: mk.Universe,
    simplify: bool = True,
) -> mk.Shape:
    new_cell = cell
    cbb = cell.bounding_box
    for b_cell in model:
        if cbb.check_intersection(b_cell.bounding_box):
            comp = b_cell.shape.complement()
            new_cell = new_cell.intersection(comp)
    if simplify and new_cell is not cell:
        new_cell = new_cell.simplify(box=cbb, min_volume=0.1)
    return new_cell


# new_cells.extend(b_model)

antenna_envelop = load_model(str(HFSR_ROOT / "models/antenna/box.i"))
envelops = load_model(str(CMODEL_ROOT / "universes/envelopes.i"), tolerance=1.0, add_boxes_to=[11, 14, 75])
envelops_original = envelops.copy()

cells_to_fill = [11, 14, 75]
cells_to_fill_indexes = [c - 1 for c in cells_to_fill]
universes_dir = CMODEL_ROOT / "universes"
assert universes_dir.is_dir()
universes = {}

for i in cells_to_fill_indexes:
    envelop = envelops[i]
    new_envelop = subtract_model_from_cell(envelop, antenna_envelop)
    assert new_envelop is not envelop, \
        f"Envelope ${envelop.name()} should be changed on intersect with antenna envelope"
    envelops[i] = new_envelop
antenna_envelop.rename(start_cell=200000, start_surf=200000)
envelops.extend(antenna_envelop)




for i in cells_to_fill:
    universe_path = universes_dir / f"u{i}.i"
    universe = read_mcnp(universe_path, encoding="cp1251")
    universes[i] = universe