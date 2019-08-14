import sys
import os

# import matplotlib.pyplot as plt
# import seaborn as sb
# import pandas as pd
# import numpy as np
# import scipy as sp
# import scipy.constants as sc
import typing as tp
from joblib import Memory
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


def attach_bounding_boxes(model: mk.Universe, tolerance: int = 10) -> tp.NoReturn:
    for c in model:
        c.bounding_box = c.shape.bounding_box(tol=tolerance)


# def collect_bounding_boxes(model: mk.Universe, tolerance: int = 10) -> tp.List[mg.Box]:
#     return list(scan_bounding_boxes(model, tolerance))

mem = Memory(location=".cache")


@mem.cache
def load_model(path: Path, tolerance: int = 10) -> mk.Universe:
    # The cp1251-encoding reads C-model with various kryakozyabrs
    model = read_mcnp(str(path), encoding="Cp1251")
    attach_bounding_boxes(model, tolerance)
    return model

antenna_envelop = load_model(HFSR_ROOT / "models/antenna/box.i")
envelops = load_model(CMODEL_ROOT / "universes/envelopes.i")