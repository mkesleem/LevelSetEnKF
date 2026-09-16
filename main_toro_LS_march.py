# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


import h5py
import numpy as np
from pathlib import Path
import levelSet1d as levelSet
import utils_time_march_1d
from make_initial_ensemble_1d import make_initial_ensemble
from utils_main import run_allclean

BASE = Path(__file__).resolve().parent

# --- Initialization ---
CASE_NAME = "toro"
run_allclean(BASE,CASE_NAME)
TRACK_CASES = {
    "prim": f"{CASE_NAME}_prim",
    "primLS": f"{CASE_NAME}_primLS",
    "primLS_sharp": f"{CASE_NAME}_primLS_sharp",
}

Ne = 1
seed = 0
TEMPLATE_NAME = CASE_NAME+".template"
ENSEMBLE_NORMAL_PARAMS = {
    "x0": {"mean": 0.5, "std": 0.1},
    "rhoL": {"mean": 5.99924, "std": 0.2},
    "pL": {"mean": 460.894, "std": 0.1*460.894},
    "rhoR": {"mean": 5.99242, "std": 0.1},
    "pR": {"mean": 46.0950, "std": 0.1*46.0950},
}
# time march information
MAX_TIME = 0.0245
dt = 0.0245/70
t = np.arange(0, MAX_TIME + dt, dt)
NTIME = len(t)
enkfBool = np.zeros(NTIME,dtype=bool)
enkfBool[20::10] = True
enkfBool[-1] = False
t_enkf = t[enkfBool]
enkfCount = np.arange(NTIME)[enkfBool]
if len(np.where(enkfBool)[0]) > 0:
    dt_enkf = np.concatenate(([t_enkf[0]], t_enkf[1:] - t_enkf[:-1]))
    dt_final = t[-1] - t_enkf[-1]
else:
    dt_enkf = 0
    dt_final = t[-1]-t[0]

NENKF = len(t_enkf)
solutions_t0 = {}

# get number of discontinuities
NdVec = np.asarray([3,2,2])

for track_case in TRACK_CASES.values():
    make_initial_ensemble(
        BASE,
        BASE / TEMPLATE_NAME,
        track_case,
        n=Ne,
        seed=seed,
        ensemble_normal_params=ENSEMBLE_NORMAL_PARAMS,
    )
    utils_time_march_1d.ensemble_init(BASE,track_case,Ne,workers=1)

# --- DA Loop ---
for t_idx in range(NENKF):
    print(f"\n===== DA STEP {t_idx} =====")

    for track, track_case in TRACK_CASES.items():
        utils_time_march_1d.ensemble_run(BASE,track_case,Ne,dt_enkf[t_idx],workers=1,t_idx=t_idx)
        solution, x = utils_time_march_1d.read_solution(
            BASE,track_case,t_idx,Ne
        )

        if track != "prim":
            LF, LS, LW = levelSet.train(
                x,solution[:,:,0],NdVec,lam1=1e2,lamB=1e2
            )
            if track == "primLS_sharp":
                LW = np.full_like(LW, 1e-12)
            solution[:,:,0] = levelSet.eval(LF,LS,LW,NdVec)

        if t_idx == 0:
            solutions_t0[track] = solution.copy()

        # Use this track's state as the initial condition for its next segment.
        utils_time_march_1d.write_solution(
            BASE,track_case,t_idx,solution,Ne
        )

# final time-march
print(f"\n===== DA COMPLETE =====")
solutions_final = {}
for track, track_case in TRACK_CASES.items():
    utils_time_march_1d.ensemble_run(BASE,track_case,Ne,dt_final,workers=1,t_idx=NENKF)
    solutions_final[track], x = utils_time_march_1d.read_solution(
        BASE,track_case,NENKF,Ne
    )

if solutions_t0.keys() != TRACK_CASES.keys():
    raise RuntimeError("Not all three trajectories produced a solution at t_idx == 0")

print(f"\n===== Write data =====")
out_fn = BASE / f"data_{CASE_NAME}_LS_march.h5"
with h5py.File(out_fn, "w") as f:
    f.create_dataset("x", data=x)
    f.create_dataset("t", data=t)
    for track in TRACK_CASES:
        f.create_dataset(f"{track}_t0", data=solutions_t0[track])
        f.create_dataset(f"{track}_final", data=solutions_final[track])

print(f"Marching level-set data written to: {out_fn}")
