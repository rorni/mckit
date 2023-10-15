#!python
"""Find ratio of a volume fo steel with Co0.5 fraction to other steel volume.

The model pc11_shields8_Co05.i is variant where the specific material was applied for
frame parts, which are in the integrator scope - fraction of Co was increased to 0.5wt%.
All other parts (diagnostics) use the same materials as in baseline model.

We want to show that the portion of the specific steel is much less than of other steel
and cannot affect SDDR value too much.
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np

import dotenv

from mckit.geometry import Box
from mckit.parser import from_file
from tqdm import tqdm

dotenv.load_dotenv()

MODEL_PATH = Path(os.getenv("MCKIT_MODEL", "~/dev/mcnp/ep11/wrk/pc/Co05-volumes")).expanduser()
if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

WRK_DIR = MODEL_PATH.parent


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
    return Box.from_bounds()


def main():
    """TODO..."""
    pass


if __name__ == "__main__":
    main()
