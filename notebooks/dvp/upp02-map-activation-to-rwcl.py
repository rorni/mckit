#!/usr/bin/env python
# coding: utf-8

# # Extract results of activation computations for UPP02 to RWCL
#
# Dmitry Portnov, July 2021
#
# ## Goals and input data
#
# Extract data from "component_to_cell-up02-1.json", which correspondes to component_to_cell-1.json, attached to https://jira.iterrf.ru:8443/browse/UPP-93, and form a table of values corresponding to RWCL "Radwaste_Checklist_UPP02.xlsx" and "Radwaste_Checklist_UP02_ISS_PCSS.xlsx".
#
# All the files are in the same folder on 10.106.203.11: "d:\dvp\dev\mcnp\upp\wrk\up02".
#
# ## Scenario
#
# - Load data from json as map RWCL item -> list of cells in the MCNP model
# - Load activation data as map cell -> activation values for 12 day of cooliing
#
#

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


from __future__ import annotations
import typing as tp


# In[3]:


import os, sys


# In[5]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import json
import pathlib

from collections import defaultdict
from pathlib import Path
from pprint import pprint, pformat


# In[7]:


import numpy as np
import dotenv
from tqdm import tqdm


# In[8]:


import pandas as pd


# In[10]:


import r2s_rfda as r2s


# In[11]:


import mckit as mc
import r2s_rfda.fetch as r2sft
import r2s_rfda.utils as r2sut
import r2s_rfda.launcher as r2sln

# from r2s_rfda.fetch import load_result_config, load_data, load_result_config
# from r2s_rfda import utils
# from r2s_rfda.launcher import load_config


# In[10]:


# If exists local ".env" file, then load it to environment.
dotenv.load_dotenv();


# In[11]:


def all_exist(*paths: Path) -> bool:
    for p in paths: # type: Path
        if not p.exists():
            raise FileNotFoundError(p)
def all_are_files(*paths: Path) -> bool:
    return all(map(Path.is_file, paths))
def all_are_dirs(*paths: Path) -> bool:
    return all(map(Path.is_dir, paths))


# In[13]:


ROOT: Path = Path(os.getenv("UP02_ACT_ROOT", r'd:\dvp\dev\mcnp\upp\wrk\up02'))
r2swrk: Path = ROOT  # for UP02 computation were done in the root directory
component_to_cell_json = ROOT / "json-3" / "component_to_cell3.json"
inp_keys_json = ROOT / "json-3" / "inp-keys.json"
inp_iss_keys_json = ROOT / "json-3" / "inp-iss-keys.json"
rwclinput: Path = ROOT / "Radwaste_Checklist_UPP02.xlsx"
rwclinput_iss: Path = ROOT / "Radwaste_Checklist_UP02_ISS_PCSS.xlsx"

all_exist(ROOT, r2swrk, component_to_cell_json, inp_keys_json, inp_iss_keys_json, rwclinput, rwclinput_iss)
assert all_are_dirs(ROOT, r2swrk)
assert all_are_files(pjson, rwclinput, rwclinput_iss)


# ## Load "component => cells" mapping from JSON

# In[14]:


with component_to_cell_json.open() as fid:
    component_to_cell = json.load(fid)
pprint(sorted(component_to_cell.keys()))


# # Load "RWCL key => Component" mapping

# In[100]:


with inp_keys_json.open() as fid:
    inp_keys = json.load(fid)
# pprint(sorted(inp_keys.keys()))
pprint(inp_keys)


# In[20]:


with inp_iss_keys_json.open() as fid:
    inp_iss_keys = json.load(fid)
pprint(inp_iss_keys)


# Note: JSON doesn't support comments and trailing commas, the JSON texts are fixed

# ## Load Cell->activation data mapping from R2S working folder

# In[21]:


r2s_launch_conf = r2sln.load_config(r2swrk)
r2s_result_conf = r2sft.load_result_config(r2swrk)


# In[22]:


time = r2sut.convert_time_literal('12d')
time_labels = list(sorted(r2s_result_conf['gamma'].keys()))
time += time_labels[r2s_launch_conf['zero']]
closest_lab = r2sut.find_closest(time, time_labels)
# closest_lab


# In[23]:


r2s_result_conf.keys()


# In[24]:


activation_paths = r2s_result_conf["activity"]
activation_paths
act_data = r2sft.load_data(r2swrk / activation_paths[closest_lab])


# In[25]:


# Estimate correction factor for tqdm progressbar total value.
its=80625951000
its/act_data._data.nnz


# In[26]:


tqdm_correction = 6400


# In[27]:


def load_cell_activity(cell_excel_path: Path):
    if cell_excel_path.exists():
        cell_activity_df = pd.read_excel(cell_excel_path, sheet_name="activity", index_col=0)
    else:
        cell_activity = defaultdict(float)
        with tqdm(total=act_data._data.nnz * tqdm_correction) as pbar:
            for cnt, ((g, c, i, j, k), act) in enumerate(act_data.iter_nonzero()):
                cell_activity[c] += act
                #print(act_data.gbins[g], c, i, j, k, act)
                #i += 1
                #if i == 1000:
                #    break
                if cnt % 1000 == 0:
                    pbar.update(cnt)
        cell_excel_path.parent.mkdir(parents=True, exist_ok=True)
        index = np.array(sorted(cell_activity.keys()), dtype=np.int32)
        data  = np.fromiter((cell_activity[k] for k in index), dtype=float)
        cell_activity_df = pd.DataFrame(data, columns=["activity"], index=index)
        with pd.ExcelWriter(cell_excel_path, engine="openpyxl") as excel:
            cell_activity_df.to_excel(excel, sheet_name="activity")
    return cell_activity_df



# In[146]:


def load_cell_nuclide_activity(cell_excel_path: Path, act_data, nuclide: str):
    label = nuclide + " activity"
    loaded_from_cache = False
    if cell_excel_path.exists():
        try:
            cell_activity_df = pd.read_excel(cell_excel_path, sheet_name=label, index_col=0)
            loaded_from_cache = True
        except ValueError:
            loaded_from_cache = False

    if not loaded_from_cache:
        nuclide_idx = np.searchsorted(act_data.gbins, nuclide)
        assert act_data.gbins[nuclide_idx] == nuclide
        cell_activity = defaultdict(float)
        with tqdm(total=act_data._data.nnz * tqdm_correction) as pbar:
            for cnt, ((g, c, i, j, k), act) in enumerate(act_data.iter_nonzero()):
                if nuclide_idx == g:
                    cell_activity[c] += act
                if cnt % 1000 == 0:
                    pbar.update(cnt)
        cell_excel_path.parent.mkdir(parents=True, exist_ok=True)
        index = np.array(sorted(cell_activity.keys()), dtype=np.int32)
        data  = np.fromiter((cell_activity[k] for k in index), dtype=float)
        cell_activity_df = pd.DataFrame(data, columns=["activity"], index=index)
        with pd.ExcelWriter(cell_excel_path, engine="openpyxl", mode="a") as excel:
            cell_activity_df.to_excel(excel, sheet_name=label)
    return cell_activity_df



# In[147]:


cell_activity_df = load_cell_activity(Path.cwd().parent.parent / "wrk/up02/cell_data.xlsx")
cell_activity_df


# In[148]:


cell_h3_activity_df = load_cell_nuclide_activity(Path.cwd().parent.parent / "wrk/up02/cell_h3_activity.xlsx", act_data, "H3")
cell_h3_activity_df


# In[29]:


cell_activity_df.loc[320000]["activity"]


# # In vessel RWCL

# In[41]:


rwclinput_df = pd.read_excel(rwclinput, sheet_name="UPP02", header=0, usecols=[2,3], index_col=0).iloc[3:]  # Yes, UPP02 for UPP08
rwclinput_df


# In[44]:


mass_column = rwclinput_df.columns[0]


# In[45]:


in_vessel_components = rwclinput_df.copy()


# In[57]:


in_vessel_components["Activity, Bq"] = np.zeros(len(in_vessel_components.index), dtype=float)


# In[150]:


in_vessel_components["H3 Activity, Bq"] = np.zeros(len(in_vessel_components.index), dtype=float)


# In[120]:


def collect_cells_map(rwcl_key, inp_key_map, component_to_cell_map) -> tp.Optional[Dict[int, float]]:
    keys_in_cell_map = inp_key_map[rwcl_key]
    if keys_in_cell_map:
        cell_fraction_map = defaultdict(float)
        for k in keys_in_cell_map:
            cells_fraction_map_for_key = component_to_cell_map[k]
            if cells_fraction_map_for_key:
                for k, v in cells_fraction_map_for_key.items():
                    cell_fraction_map[int(k)] += v
        for k, v in cell_fraction_map.items():
            if 1.0 < v:
                print(f"Warning: ramp down fraction owverflow in cell {k}: {v:.3g}")
                cell_fraction_map[k] = 1.0
        return cell_fraction_map
    else:
        return None


# In[121]:


def compute_activity(cell_fraction_map: Dict[int, float], cell_activity: pd.DataFrame) -> float:

    def mapper(pair: tp.Tuple[int, float]) -> float:
        cell, fraction = pair
        assert isinstance(cell, int), f"Integer cell number is expected, {cell} is found"
        try:
            result = cell_activity.loc[cell].activity * fraction
        except KeyError:
            print(f"Warning: cannot find cell {cell} in cell activity data")
            result = 0.0
        return result

    return sum(map(mapper, cell_fraction_map.items()))


# In[122]:


for rwclid in in_vessel_components.index:
    cell_fraction_map = collect_cells_map(rwclid, inp_keys, component_to_cell)
    if cell_fraction_map is not None:
        activity = compute_activity(cell_fraction_map, cell_activity_df)
        in_vessel_components.loc[rwclid, ["Activity, Bq"]] = activity
    else:
        print(f"Warning: cannot find activity for RWCL component {rwclid}")


# In[159]:


cell_h3_activity_df.rename(columns={"H3 activity":"activity"}, inplace=True)


# In[160]:


for rwclid in in_vessel_components.index:
    cell_fraction_map = collect_cells_map(rwclid, inp_keys, component_to_cell)
    if cell_fraction_map is not None:
        activity = compute_activity(cell_fraction_map, cell_h3_activity_df)
        in_vessel_components.loc[rwclid, ["H3 Activity, Bq"]] = activity
    else:
        print(f"Warning: cannot find activity for RWCL component {rwclid}")


# In[161]:


in_vessel_components


# In[162]:


ivc1 = in_vessel_components.copy()


# In[168]:


ivc1["Mass, kg"] = np.fromiter(map(lambda x: float(x.split()[0]), ivc1[mass_column]), dtype=float)


# In[174]:


ivc1["Unit Activity, Bq/kg"] = ivc1["Activity, Bq"] / ivc1["Mass, kg"]


# In[175]:


ivc1["Unit H3 Activity, Bq/kg"] = ivc1["H3 Activity, Bq"] / ivc1["Mass, kg"]


# In[176]:


ivc1


# In[216]:


def create_msg(row):
#     print("Row:", row)
    activity = row["Activity, Bq"]
    unit_activity = row["Unit Activity, Bq/kg"]
    h3_activity = row["H3 Activity, Bq"]
    unit_h3_activity = row["Unit H3 Activity, Bq/kg"]
    if activity > 0.0:
        return f"Activity {activity:.2g} Bq\n({unit_activity:.2g}, Bq/kg),\nH3 Activity {h3_activity:.2g} Bq\n({unit_h3_activity:.2g}, Bq/kg)"
    else:
        return ""



# ## Interpolate missed values in in-vessel components (ivc1).
#
# "PP tubes" and "PP Generics..." unit values should me the same as for "PP structure", totals - proportional to masses.
#
# "Rear box shielding plates (8 pieces)" - as for "Rear frame shielding9"

# In[3]:


def update_missed_components(df, refkey, missed_keys):
    unit_activity = df.loc[refkey, ["Unit Activity, Bq/kg"]].values[0]
    unit_h3_activity = df.loc[refkey, ["Unit H3 Activity, Bq/kg"]].values[0]
    df.loc[missed_keys, ["Unit Activity, Bq/kg"]] =  unit_activity
    df.loc[missed_keys, ["Unit H3 Activity, Bq/kg"]] = unit_h3_activity
    masses = df.loc[missed_keys, ["Mass, kg"]].values
    df.loc[missed_keys, ["Activity, Bq"]] = unit_activity * masses
    df.loc[missed_keys, ["H3 Activity, Bq"]] = unit_h3_activity * masses
    return df.loc[missed_keys]


# In[344]:


def update_missed_pp_components():
    missed_keys = ["PP Tubes", "PP Generics (pads, skids, gripping - 8 pieces)", "Generic Screws (50 pieces)"]
    refkey = "PP structure"
    return update_missed_components(ivc1, refkey, missed_keys)

update_missed_pp_components()


# In[345]:


def update_missed_rb_components():
    missed_keys = ["Rear box shielding plates (8 pieces)", "Rear frame shielding9"]
    refkey = "Rear frame shielding8"
    return update_missed_components(ivc1, refkey, missed_keys)

update_missed_rb_components()


# In[346]:


# ivc1=ivc1.reindex(columns=["Mass, kg", "Activity, Bq", "Unit Activity, Bq/kg", "H3 Activity, Bq", "Unit H3 Activity, Bq/kg"])
ivc1["Radiological data"] = ivc1.apply(create_msg, axis=1)


# In[347]:


ivc1


# In[348]:


with pd.ExcelWriter(ROOT / "in-vessel.xlsx", engine="openpyxl", mode="w") as excel:
    ivc1.to_excel(excel, sheet_name="result")


# # ISS components

# In[273]:


rwclinput_iss_df = pd.read_excel(rwclinput_iss, sheet_name="UPP02", header=0, usecols=[2,3], index_col=0).iloc[3:]
rwclinput_iss_df.rename(columns={mass_column: "Mass, kg"}, inplace=True)
rwclinput_iss_df


# In[274]:


for i in rwclinput_iss_df.index:
    print(i, ":", inp_iss_keys[i])


# In[275]:


iss_components = rwclinput_iss_df.copy()


# In[276]:


iss_components["Activity, Bq"] = np.zeros(len(iss_components.index))
iss_components["Unit Activity, Bq/kg"] = np.zeros(len(iss_components.index))
iss_components["H3 Activity, Bq"] = np.zeros(len(iss_components.index))
iss_components["Unit H3 Activity, Bq/kg"] = np.zeros(len(iss_components.index))


# In[277]:


for rwclid in iss_components.index:
    cell_fraction_map = collect_cells_map(rwclid, inp_iss_keys, component_to_cell)
    if cell_fraction_map is not None:
        activity = compute_activity(cell_fraction_map, cell_activity_df)
        iss_components.loc[rwclid, ["Activity, Bq"]] = activity
    else:
        print(f"Warning: cannot find activity for RWCL component {rwclid}")


# In[278]:


for rwclid in iss_components.index:
    cell_fraction_map = collect_cells_map(rwclid, inp_iss_keys, component_to_cell)
    if cell_fraction_map is not None:
        activity = compute_activity(cell_fraction_map, cell_h3_activity_df)
        iss_components.loc[rwclid, ["H3 Activity, Bq"]] = activity
    else:
        print(f"Warning: cannot find activity for RWCL component {rwclid}")


# In[279]:


iss_components


# ## Set "Unit" values

# In[280]:


iss_components["Unit Activity, Bq/kg"] = iss_components["Activity, Bq"] / iss_components["Mass, kg"]
iss_components["Unit H3 Activity, Bq/kg"] = iss_components["H3 Activity, Bq"] / iss_components["Mass, kg"]
iss_components


# ## Correct "Bioshield #" pieces
#
# All 27 components "Bioshield \<number\>" are presented  as a single component in our model, so the totals should be distributed proportionaly to masses
# Unit values should the same for all these components.

# In[281]:


bioshield_components = list(f"Bioshield {i}" for i in range(1, 28))


# In[282]:


iss_components.loc[bioshield_components]


# In[283]:


bioshield_mass = iss_components.loc[bioshield_components]["Mass, kg"].sum()
bioshield_mass


# In[284]:


bioshield_unit_activity = iss_components.loc[bioshield_components[0], ["Activity, Bq"]].values[0] / bioshield_mass
bioshield_unit_activity


# In[285]:


bioshield_unit_h3_activity = iss_components.loc[bioshield_components[0], ["H3 Activity, Bq"]].values[0] / bioshield_mass
bioshield_unit_h3_activity


# In[287]:


iss_components_bak = iss_components.copy()


# In[288]:


iss_components.loc[bioshield_components, ["Unit Activity, Bq/kg"]] = bioshield_unit_activity
iss_components.loc[bioshield_components, ["Unit H3 Activity, Bq/kg"]] = bioshield_unit_h3_activity
iss_components


# In[299]:


# set total activity proportional to mass
masses = iss_components.loc[bioshield_components, ["Mass, kg"]].values
iss_components.loc[bioshield_components, ["Activity, Bq"]] =  iss_components.loc[bioshield_components, ["Unit Activity, Bq/kg"]].values * masses
iss_components.loc[bioshield_components, ["H3 Activity, Bq"]] =  iss_components.loc[bioshield_components, ["Unit H3 Activity, Bq/kg"]].values * masses
iss_components


# ## Interpolate missed data for ISS
#
# *Note*: port cell data are not computed in this model

# In[349]:


def update_missed_iss_components():
    missed_keys = ["IS Locking System", "IS Cable tubes", "IS Add Protection", "IS Protection Frame"]
    refkey = "IS Bogies"
    update_missed_components(iss_components, refkey, missed_keys)
    missed_keys = ["IS Walkways"]
    refkey = "IS Chasis"
    update_missed_components(iss_components, refkey, missed_keys)

update_missed_iss_components()


# In[350]:


iss_components["Radiological data"] = iss_components.apply(create_msg, axis=1)
iss_components


# In[351]:


with pd.ExcelWriter(ROOT / "iss.xlsx", engine="openpyxl", mode="w") as excel:
    iss_components.to_excel(excel, sheet_name="result")


# In[352]:


ROOT


# In[ ]:
