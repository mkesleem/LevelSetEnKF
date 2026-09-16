# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


import numpy as np
from pathlib import Path
import subprocess
import levelSet2d as levelSet
import h5py
import utils_time_march_2d
import utils_time_march
from make_initial_ensemble_2d import make_initial_ensemble
from utils_main import (
    run_allclean, cleanSolution0, update_input_for_forecast
)

def main():
    # --- Initialization ---
    BASE = Path(__file__).resolve().parent
    CASE_NAME = "bw2d"
    run_allclean(BASE,CASE_NAME)

    seed = 0
    TEMPLATE_NAME = CASE_NAME+".template"
    ENSEMBLE_NORMAL_PARAMS = {
        "x0": {"mean": 1.0, "std": 0.08},
        "y0": {"mean": 1.0, "std": 0.08},
        "rho1": {"mean": 1.0, "std": 0.05},
        "rho2": {"mean": 1.0, "std": 0.05},
        "p1": {"mean": 1000.0, "std": 1000 * 0.1},
        "p2": {"mean": 0.01, "std": 0.1 * 0.01},
        "ra": {"mean": 0.45, "std": 0.05},
    }

    # time march information
    MAX_TIME = 0.0025

    make_initial_ensemble(
        BASE,
        BASE/TEMPLATE_NAME,
        CASE_NAME,
        n=1,
        seed=seed,
        ensemble_normal_params=ENSEMBLE_NORMAL_PARAMS
    )
    dir_i = BASE / (CASE_NAME + f"_{0:03d}")
    subprocess.run(["./Allrun", "input0.st"], cwd=dir_i, check=True)
    cleanSolution0(case_dir=dir_i)
    utils_time_march_2d.write_ensemble_state(BASE,CASE_NAME,0,1)

    utils_time_march.m2c_update_max_time(BASE,CASE_NAME,MAX_TIME,1)
    update_input_for_forecast(BASE/(CASE_NAME + f"_{0:03d}"), 0)
    subprocess.run(
        ["./Allrun", "input.st"],
        cwd=BASE/(CASE_NAME+f"_{0:03d}"),
        check=True
    )

    # get size
    NdVec = np.asarray([2,1,1,1])

    # read data
    prim_full, x_full, y_full = utils_time_march_2d.read_solution(
        BASE,CASE_NAME,0,1
    )
    prim = prim_full[:,1:-1,1:-1,:]
    x = x_full[1:-1,1:-1]
    y = y_full[1:-1,1:-1]

    # set number of workers
    workers = 16

    # construct level set representation
    LF, LS, LW = levelSet.train(
        prim[:,:,:,0],
        x,
        y,
        NdVec,
        lam1=1e2,
        lamB=1e2,
        workers=workers
    )

    # evaluate
    primLS = prim.copy()
    primLS[:,:,:,0] = levelSet.eval(LF,LS,LW,NdVec)

    # save level-set fit
    out_fn = BASE / f"data_{CASE_NAME}_LS.h5"
    with h5py.File(out_fn, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("t", data=MAX_TIME)
        f.create_dataset("prim", data=prim[:,:,:,0])
        f.create_dataset("primLS", data=primLS[:,:,:,0])
        f.create_dataset("LF", data=LF)
        f.create_dataset("LS", data=LS)
        f.create_dataset("LW", data=LW)
        f.create_dataset("NdVec", data=NdVec)

    print(f"Level-set data written to: {out_fn}")


if __name__ == "__main__":
    main()
