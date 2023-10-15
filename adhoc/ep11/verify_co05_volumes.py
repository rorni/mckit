#!python
"""Find ratio of a volume fo steel with Co0.5 fraction to other steel volume.

The model pc11_shields8_Co05.i is variant where the specific material was applied for
frame parts, which are in the integrator scope - fraction of Co was increased to 0.5%.
The material's id is 207302.
All other parts (diagnostics) use the same materials as in baseline model.

We want to show that the portion of the specific steel is much less than of other steel
and cannot affect SDDR value too much.
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np

import dotenv

from mckit.geometry import EX, EY, EZ, Box
from mckit.parser import from_file
from tqdm import tqdm

dotenv.load_dotenv()

WRK_DIR = Path(os.getenv("MCKIT_MODEL", "~/dev/mcnp/ep11/wrk/pc/Co05-volumes")).expanduser()
if not WRK_DIR.is_dir():
    raise FileNotFoundError(WRK_DIR)

MODEL_PATH = WRK_DIR / "pc11_shields8_Co05.i"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)


def from_bounds(
    minx: float, maxx: float, miny: float, maxy: float, minz: float, maxz: float
) -> Box:
    min_vals = np.array([minx, miny, minz])
    max_vals = np.array([maxx, maxy, maxz])
    if not np.all(min_vals < max_vals):
        raise ValueError("Unsorted boundaries values")
    center = 0.5 * (min_vals + max_vals)
    widths = max_vals - min_vals
    return Box(center, *widths, ex=EX, ey=EY, ez=EZ)


def create_global_box():
    """Define the box surrounding the model.

    This cell 196000 with zero importance.
    Surfaces:
    206001:-206002:206003:206005:-206004:-206000
    206000 PX 830
    206001 PX 3350
    206002 PY -1200
    206003 PY 1200
    206004 PZ -620
    206005 PZ 740
    """
    return from_bounds(830.0, 3350, -1200.0, 1200.0, -620, 740)


GLOBAL_BOX = create_global_box()


model_info = from_file(MODEL_PATH)
cells_index = model_info.cells_index
model = model_info.universe


def collect_cell_materials_map(model):
    pass


def main():
    """TODO..."""
    pass


if __name__ == "__main__":
    main()
