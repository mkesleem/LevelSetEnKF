# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


import numpy as np
from pathlib import Path
import subprocess
import sys
import levelSet1d as levelSet
import h5py
import utils_time_march_1d
import utils_da
from make_initial_ensemble_1d import make_initial_ensemble
from utils_main import run_allclean

BASE = Path(__file__).resolve().parent

# --- Initialization ---
CASE_NAME = "sod"
run_allclean(BASE,CASE_NAME)

Ne = 50
NobsX = 4
OBS_X = np.linspace(0.0,1.0,NobsX+2)[1:-1]
REL_NOISE, ABS_NOISE = 0.1, 0.05
seed = 0
rng = np.random.default_rng(seed)
NobsX = len(OBS_X)
TEMPLATE_NAME = CASE_NAME+".template"
ENSEMBLE_NORMAL_PARAMS = {
    "x0": {"mean": 0.5, "std": 0.1},
    "rhoL": {"mean": 1.0, "std": 0.05},
    "pL": {"mean": 1.0, "std": 0.05},
    "rhoR": {"mean": 0.125, "std": 0.006},
    "pR": {"mean": 0.1, "std": 0.005},
}
fname_master = BASE / f"{CASE_NAME}_master.h5"

# time march information
MAX_TIME = 0.2
dt = 0.002
t = np.arange(0, MAX_TIME + dt, dt)
NTIME = len(t)
enkfBool = np.zeros(NTIME,dtype=bool)
enkfBool[30::5] = True
enkfBool[-1] = False
t_enkf = t[enkfBool]
enkfCount = np.arange(NTIME)[enkfBool]
if enkfCount.size > 0:
    dt_enkf = np.concatenate(([t_enkf[0]], t_enkf[1:] - t_enkf[:-1]))
    dt_final = t[-1] - t_enkf[-1]
else:
    dt_enkf = 0
    dt_final = t[-1] - t[0]
NENKF = len(t_enkf)
sMat = np.zeros((NENKF,NobsX))
yMat = np.zeros((NENKF,NobsX))
workers = np.min([Ne,16])

# get number of discontinuities
NdVec = np.asarray([2,1,1])
NdVec0 = np.asarray([1,0,1])
Nd = np.max(NdVec)
alignVecLF = np.zeros(Nd+1)
alignVecLF[0] = 1

# regularization
lam1 = 1e2
lamB = 1e2

# initialize
make_initial_ensemble(
    BASE,
    BASE/TEMPLATE_NAME,
    CASE_NAME,
    n=Ne,
    seed=seed,
    ensemble_normal_params=ENSEMBLE_NORMAL_PARAMS,
)
utils_time_march_1d.ensemble_init(BASE, CASE_NAME, Ne, workers)

# read
prim_initial, x_initial = utils_time_march_1d.read_solution(BASE,CASE_NAME,0,Ne)
prim_initial_mean = levelSet.ensemble_mean(
    prim_initial, x_initial, NdVec0, lam1=lam1, lamB=lamB, workers=workers
)
mean_t_idx = np.concatenate(([0], enkfCount, [NTIME - 1]))

# write initial .h5 file
with h5py.File(fname_master, "w") as f:
    f.create_dataset("sMat", data=sMat)
    f.create_dataset("yMat", data=yMat)
    f.create_dataset("xObs", data=OBS_X)
    f.create_dataset("t_idx", data=0)
    f.create_dataset("Ne", data=Ne)
    f.create_dataset("enkfBool", data=enkfBool)
    f.create_dataset("t", data=t)
    f.create_dataset("mean_t_idx", data=mean_t_idx)

    dset = f.create_dataset(
        "prim_a_mean",
        shape=(NENKF + 2,) + prim_initial_mean.shape,
        dtype=prim_initial_mean.dtype,
    )
    dset[0] = prim_initial_mean
    dset = f.create_dataset(
        "prim_f_mean",
        shape=(NENKF + 2,) + prim_initial_mean.shape,
        dtype=prim_initial_mean.dtype,
    )
    dset[0] = prim_initial_mean

# --- DA Loop ---
for t_idx in range(NENKF):
    print(f"\n===== DA STEP {t_idx} =====")

    # forecast
    utils_time_march_1d.ensemble_run(
        BASE,
        CASE_NAME,
        Ne,
        dt_enkf[t_idx],
        workers,
        t_idx
    )

    # get observations
    yMat[t_idx],sMat[t_idx] = utils_da.get_observations_1d(
        BASE,
        CASE_NAME,
        t_idx,
        enkfCount,
        OBS_X,
        REL_NOISE,
        ABS_NOISE,
        rng
    )

    # read data
    prim, x = utils_time_march_1d.read_solution(BASE,CASE_NAME,t_idx,Ne)

    # EnKF
    prim_a,prim_f_LS,prim_a_mean,prim_f_mean = levelSet.EnKF(
        prim,
        x,
        yMat[t_idx],
        sMat[t_idx],
        NdVec,
        OBS_X,
        rng,
        alignVecLF,
        lam1=lam1,
        lamB=lamB,
        sharp_bool=True,
        workers=workers
    )

    # write analysis state solution (m2c)
    utils_time_march_1d.write_solution(BASE,CASE_NAME,t_idx,prim_a,Ne)

    # write to .h5 file
    with h5py.File(fname_master, "r+") as f:
        if "prim_f_LS" not in f:
            f.create_dataset(
                "prim_f_LS",
                shape=(NENKF,) + prim_f_LS.shape,
                dtype=prim_f_LS.dtype,
            )

        f["prim_f_LS"][t_idx] = prim_f_LS
        f["prim_a_mean"][t_idx + 1] = prim_a_mean
        f["prim_f_mean"][t_idx + 1] = prim_f_mean
        f["sMat"][t_idx] = sMat[t_idx]
        f["yMat"][t_idx] = yMat[t_idx]
        f["t_idx"][()] = t_idx + 1

# final time-march
print(f"\n===== DA COMPLETE =====")
utils_time_march_1d.ensemble_run(BASE, CASE_NAME, Ne, dt_final, workers, NENKF)

# read final solution
prim_final, x_final = utils_time_march_1d.read_solution(BASE,CASE_NAME,NENKF,Ne)
prim_final_mean = levelSet.ensemble_mean(
    prim_final, x_final, NdVec, lam1=lam1, lamB=lamB,
    workers=workers
)

with h5py.File(fname_master, "r+") as f:
    f["prim_a_mean"][-1] = prim_final_mean
    f["prim_f_mean"][-1] = prim_final_mean
    f["t_idx"][()] = NENKF + 1

print(f"\n===== Write data =====")
out_fn = BASE / f"data_{CASE_NAME}.h5"
subprocess.run(
    [
        sys.executable,
        str(BASE / "writeToH5_1D.py"),
        CASE_NAME,
        str(out_fn),
        "--base-dir",
        str(BASE),
    ],
    check=True,
)

print(f"SUCCESS: Full DA Cycle Complete. Data written to: {out_fn}")
