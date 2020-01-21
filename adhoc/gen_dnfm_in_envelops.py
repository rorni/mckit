import os
import dotenv
import logging
import numpy as np
from pathlib import Path
from mckit import *
from mckit.parser.mcnp_input_sly_parser import from_file
from mckit.utils import assert_all_paths_exist, get_root_dir
dotenv.load_dotenv(dotenv_path=".env", verbose=True)
DNFM_ROOT = get_root_dir("DNFM_ROOT", "~/dev/mcnp/dnfm")
# CMODEL_ROOT = get_root_dir("CMODEL_ROOT", "~/dev/mcnp/c-model")
MODEL_DIR = DNFM_ROOT / "models/c-model"
LOG = logging.getLogger(__name__)
logging.basicConfig(
    # format='%(asctime)s - %(levelname)-7s - %(name)-20s - %(message)s',
    format='%(asctime)s - %(levelname)-7s - %(message)s',
    level=logging.DEBUG,
)
LOG.info("DNFM_ROOT=%s", DNFM_ROOT)
# LOG.info("CMODEL_ROOT=%s", CMODEL_ROOT)
assert_all_paths_exist(DNFM_ROOT, MODEL_DIR)
NJOBS = os.cpu_count()
# print(f"NJOBS: {NJOBS}")
# set_loky_pickler()

dnfm_box = from_file(MODEL_DIR/'DNFM_box.i').universe
dnfm = from_file(MODEL_DIR/'DNFM_NEW_LOC.i').universe
void = from_file(MODEL_DIR/'void.i')
dnfm_box.rename(start_surf=3300, start_cell=1100)
new_cells = []
box = dnfm_box._cells[0].shape.complement()
for c in void:
    c.options.pop('FILL', None)
    if c.name() in {64, 85, 165}:
        new_cells.append(c.intersection(box).simplify(min_volume=0.1))
    else:
        new_cells.append(c)
new_cells.append(dnfm_box._cells[0])
new_univ = Universe(new_cells, name_rule='keep')
new_univ.save(MODEL_DIR/'new_env_v1.i')

LOG.info("Success!")

