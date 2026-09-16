# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


from pathlib import Path
import h5py
import utils_time_march_1d
import levelSet1d as levelSet
import numpy as np
from make_initial_ensemble_1d import make_initial_ensemble
from utils_main import run_allclean

BASE = Path(__file__).resolve().parent

# --- Initialization ---
CASE_NAME = "toro"
run_allclean(BASE,CASE_NAME)

Ne = 1
TEMPLATE_NAME = CASE_NAME+".template"
ENSEMBLE_NORMAL_PARAMS = {
    "x0": {"mean": 0.5, "std": 0.1},
    "rhoL": {"mean": 5.99924, "std": 0.2},
    "pL": {"mean": 460.894, "std": 0.1*460.894},
    "rhoR": {"mean": 5.99242, "std": 0.1},
    "pR": {"mean": 46.0950, "std": 0.1*46.0950},
}
# time march information
seed = 0
MAX_TIME = 0.007

# run
make_initial_ensemble(
    BASE,
    BASE/TEMPLATE_NAME,
    CASE_NAME,
    n=Ne,
    seed=seed,
    ensemble_normal_params=ENSEMBLE_NORMAL_PARAMS,
)
utils_time_march_1d.ensemble_init(BASE, CASE_NAME, Ne)
utils_time_march_1d.ensemble_run(BASE, CASE_NAME, Ne, MAX_TIME)

# read
prim, x = utils_time_march_1d.read_solution(BASE,CASE_NAME,0,Ne)

# specify number of discontinuities
NdVec = np.asarray([3,2,2])

# fit level set
LF, LS, LW = levelSet.train(x,prim[:,:,0],NdVec,lam1=1e2,lamB=1e2)

# save level-set fit
out_fn = BASE / f"data_{CASE_NAME}_LS.h5"
with h5py.File(out_fn, "w") as f:
    f.create_dataset("x", data=x)
    f.create_dataset("t", data=MAX_TIME)
    f.create_dataset("LF", data=LF)
    f.create_dataset("prim", data=prim[:,:,0])
    f.create_dataset("LS", data=LS)
    f.create_dataset("LW", data=LW)
    f.create_dataset("NdVec", data=NdVec)

print(f"Level-set data written to: {out_fn}")
