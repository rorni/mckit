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
from joblib import (
    Memory, Parallel, delayed, wrap_non_picklable_objects, effective_n_jobs
)
from joblib.externals.loky import set_loky_pickler

import dotenv
from pathlib import Path

sys.path.append("..")

import mckit as mk
from mckit import *
import mckit.geometry as mg


def assert_pathes_exist(*pathes):
    for p in pathes:
        assert p.exists(), "Path \"{}\" doesn't exist".format(p)


def makedirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_root_dir(environment_variable_name, default):
    return Path(os.getenv(environment_variable_name, default)).expanduser()


dotenv.load_dotenv(dotenv_path=".env", verbose=True)
HFSR_ROOT = get_root_dir("HFSR_ROOT", "~/dev/mcnp/hfsr")
CMODEL_ROOT = get_root_dir("CMODEL_ROOT", "~/dev/mcnp/c-model")

assert_pathes_exist(HFSR_ROOT, CMODEL_ROOT)

NJOBS = effective_n_jobs()
print(f"NJOBS: {NJOBS}")
set_loky_pickler()

# @wrap_non_picklable_objects
def compute_bounding_box(shape: mk.Shape, tolerance: float) -> mg.Box:
    return shape.bounding_box(tol=tolerance)


def attach_bounding_boxes(model: mk.Universe, tolerance: float = 1.0) -> tp.NoReturn:
    boxes = Parallel(n_jobs=NJOBS, backend='multiprocessing')(
        delayed(compute_bounding_box)(c.shape, tolerance) for c in model
    )
    for _i, cell in enumerate(model):
        cell.bounding_box = boxes[_i]


mem = Memory(location=".cache", verbose=2)


@mem.cache
def load_model(path: str, tolerance: float = 1.0) -> mk.Universe:
    # The cp1251-encoding reads C-model with various kryakozyabrs
    model = read_mcnp(path, encoding="Cp1251")
    attach_bounding_boxes(model, tolerance)
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
envelops = load_model(str(CMODEL_ROOT / "universes/envelopes.i"))
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