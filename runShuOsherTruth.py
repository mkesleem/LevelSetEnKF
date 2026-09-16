# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT


import numpy as np
from pathlib import Path
import subprocess
import utils_pv_1d as utils_pv
import utils_main
import os

ENSEMBLE_NORMAL_PARAMS = {
    "x0": {"mean": 0.05, "std": 0.0},
    "rhoL": {"mean": 3.857143, "std": 0.0},
    "uL": {"mean": 2.629369, "std": 0.0},
    "pL": {"mean": 10.3333, "std": 0.0},
    "rhoR": {"mean": 1.0, "std": 0.0},
    "uR": {"mean": 0.0, "std": 0.0},
    "pR": {"mean": 1.0, "std": 0.0},
}

# folder paths
BASE = Path(__file__).resolve().parent
dir_truth = BASE/"shuOsher.truth"
dir_truth_IC = dir_truth / "IC"

# Files to keep (just filenames, not paths)
keep_files = {"CMakeLists.txt", "UserDefinedState.cpp"}

# Iterate over all items in the folder
for filename in os.listdir(dir_truth_IC):
    file_path = os.path.join(dir_truth_IC, filename)
    
    # Skip the files you want to keep
    if filename in keep_files:
        continue
    
    # Remove files or folders
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # delete file or symlink
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)  # delete folder recursively
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")

# generate IC
subprocess.run(["./Allrun", "input0.st"], cwd=dir_truth, check=True)

# read IC
vtr_f = dir_truth / "results" / f"solution_0000.vtr"
sol_i = utils_pv.load_solution(vtr_f, dir_truth/"input.st")

# get data
x = sol_i['x']
rho = sol_i['rho']
u = sol_i['u']
p = sol_i['p']

# change data
truth_params = {
    name: np.array([v["mean"]])
    for name, v in ENSEMBLE_NORMAL_PARAMS.items()
}
L = x < truth_params["x0"]
R = x >= truth_params["x0"]
rho[L] = truth_params["rhoL"]
rho[R] = truth_params["rhoR"] + 0.2 * np.sin(
    10 * np.pi * (x[R] - truth_params["x0"])
)
u[L] = truth_params["uL"]
u[R] = truth_params["uR"]
p[L] = truth_params["pL"]
p[R] = truth_params["pR"]
out_vtr = dir_truth / "IC" / f"solution_0000_analysis.vtr"

utils_pv.write_vtk_analysis(
    template_vtr=vtr_f,
    input_st_path=dir_truth / "input.st",
    rho_full=rho,
    u_full=u,
    p_full=p,
    out_vtr_path=out_vtr
)
utils_main.update_userdefinedstate(out_vtr, dir_truth)
utils_main.build_case(dir_truth)

# time-march
subprocess.run(["./Allrun", "input.st"], cwd=dir_truth, check=True)

for filename in os.listdir(dir_truth_IC):
    file_path = os.path.join(dir_truth_IC, filename)
    
    # Skip the files you want to keep
    if filename in keep_files:
        continue
    
    # Remove files or folders
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # delete file or symlink
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)  # delete folder recursively
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")