from typing import List
import os
import dotenv
import logging
from pathlib import Path
from mckit.parser.mcnp_input_sly_parser import from_file, Card, Universe
from mckit.utils import assert_all_paths_exist, get_root_dir
dotenv.load_dotenv(dotenv_path=".env", verbose=True)
DNFM_ROOT: Path = get_root_dir("DNFM_ROOT", "~/dev/mcnp/dnfm")
CMODEL_ROOT: Path = get_root_dir("CMODEL_ROOT", "~/dev/mcnp/cmodel")
MODEL_DIR: Path = DNFM_ROOT / "models/c-model"
LOG: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    # format='%(asctime)s - %(levelname)-7s - %(name)-20s - %(message)s',
    format='%(asctime)s - %(levelname)-7s - %(message)s',
    level=logging.DEBUG,
)
LOG.info("DNFM_ROOT=%s", DNFM_ROOT)
LOG.info("CMODEL_ROOT=%s", CMODEL_ROOT)
assert_all_paths_exist(CMODEL_ROOT, DNFM_ROOT, MODEL_DIR)
NJOBS = os.cpu_count()
# print(f"NJOBS: {NJOBS}")
# set_loky_pickler()

dnfm_box: Universe = from_file(MODEL_DIR/'DNFM_box.i').universe
dnfm: Universe = from_file(MODEL_DIR/'DNFM_NEW_LOC.i').universe
envelopes: Universe = from_file(CMODEL_ROOT / 'cmodel.universes/envelopes.i').universe
dnfm_box.rename(start_surf=3300, start_cell=1100)
new_cells: List[Card] = []
box = dnfm_box[0].shape.complement()
for c in envelopes:
    c.options.pop('FILL', None)
    if c.name() in {64, 85, 165}:
        new_cells.append(c.intersection(box).simplify(min_volume=0.1))
    else:
        new_cells.append(c)
new_cells.append(dnfm_box[0])
new_univ = Universe(new_cells, name_rule='keep')
new_univ.save(MODEL_DIR/'new_env_v1.i')

LOG.info("Success!")

