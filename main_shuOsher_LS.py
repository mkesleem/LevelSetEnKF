# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


from pathlib import Path
import h5py
import utils_time_march_1d
import utils_time_march_shuOsher
import levelSet1d as levelSet
import numpy as np
from make_initial_ensemble_shuOsher import make_initial_ensemble
from utils_main import run_allclean

BASE = Path(__file__).resolve().parent

# --- Initialization ---
CASE_NAME = "shuOsher"
run_allclean(BASE,CASE_NAME)

Ne = 1
TEMPLATE_NAME = CASE_NAME+".template"
ENSEMBLE_NORMAL_PARAMS = {
    "x0": {"mean": 0.1, "std": 0.04},
    "rhoL": {"mean": 3.857143, "std": 0.4},
    "uL": {"mean": 2.629369, "std": 0.2},
    "pL": {"mean": 10.3333, "std": 0.1*10.3333},
    "rhoR": {"mean": 1.0, "std": 0.1},
    "uR": {"mean": 0.0, "std": 0.0},
    "pR": {"mean": 1.0, "std": 0.1*1.0},
}
# time march information
seed = 0
MAX_TIME = 0.12

# run
samples = make_initial_ensemble(
    BASE,
    BASE/TEMPLATE_NAME,
    CASE_NAME,
    n=Ne,
    seed=seed,
    ensemble_normal_params=ENSEMBLE_NORMAL_PARAMS,
)
assert np.all(samples["x0"] > 0)
utils_time_march_1d.ensemble_init(BASE, CASE_NAME, Ne)
utils_time_march_shuOsher.write_ensemble_state_init(
    BASE, CASE_NAME, 0, Ne, samples
)
utils_time_march_1d.ensemble_run(BASE, CASE_NAME, Ne, MAX_TIME)

# read
prim, x = utils_time_march_1d.read_solution(BASE,CASE_NAME,0,Ne)

# specify number of discontinuities
NdVec = np.asarray([1,1,1])

# fit level set
LF, LS, LW = levelSet.train(x,prim[:,:,0],NdVec,lam1=1e-1,lamB=1e2)

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
