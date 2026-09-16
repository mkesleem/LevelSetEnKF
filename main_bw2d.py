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
import utils_da as utils_da
from make_initial_ensemble_2d import make_initial_ensemble
from utils_main import (
    run_allclean, cleanSolution0, update_input_for_forecast
)

def main():

    # Initialize
    BASE = Path(__file__).resolve().parent
    CASE_NAME = "bw2d"
    CASE_NAME_TRUTH = "bw2d"
    run_allclean(BASE,CASE_NAME)

    # get observation coordinates
    NobsD = 4
    OBS_Xc = np.sort(np.linspace(0.0,2.0,NobsD+2)[1:-1])
    OBS_X, OBS_Y = np.meshgrid(OBS_Xc,OBS_Xc)
    NobsXY = len(OBS_X.ravel())

    Ne = 30
    workers = np.min([16,Ne])
    REL_NOISE, ABS_NOISE = 0.1, 0.05
    seed = 0
    rng = np.random.default_rng(seed)
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
    fname_master = BASE / f"{CASE_NAME}_master.h5"

    # time march information
    MAX_TIME = 0.01
    dt = MAX_TIME / 100
    t = np.arange(0, MAX_TIME + dt, dt)
    NTIME = len(t)
    enkfBool = np.zeros(NTIME,dtype=bool)
    enkfBool[25::10] = True
    enkfBool[-1] = False
    t_enkf = t[enkfBool]
    enkfCount = np.arange(NTIME)[enkfBool]
    dt_enkf = np.concatenate(([t_enkf[0]], t_enkf[1:] - t_enkf[:-1]))
    dt_final = t[-1] - t_enkf[-1]
    NENKF = len(t_enkf)
    sMat = np.zeros((NENKF,NobsXY))
    yMat = np.zeros((NENKF,NobsXY))

    # t_idx records the number of completed forecast segments.
    with h5py.File(fname_master, "w") as f:
        f.create_dataset("sMat", data=sMat)
        f.create_dataset("yMat", data=yMat)
        f.create_dataset("xObs", data=OBS_X)
        f.create_dataset("yObs", data=OBS_Y)
        f.create_dataset("t_idx", data=0)
        f.create_dataset("Ne", data=Ne)
        f.create_dataset("enkfBool", data=enkfBool)
        f.create_dataset("t", data=t)

    make_initial_ensemble(
        BASE,
        BASE/TEMPLATE_NAME,
        CASE_NAME,
        n=Ne,
        seed=seed,
        ensemble_normal_params=ENSEMBLE_NORMAL_PARAMS
    )
    for i in range(Ne):
        dir_i = BASE / (CASE_NAME + f"_{i:03d}")
        subprocess.run(["./Allrun", "input0.st"], cwd=dir_i, check=True)
        cleanSolution0(case_dir=dir_i)

    utils_time_march_2d.write_ensemble_state(BASE,CASE_NAME,0,Ne)

    # --- DA Loop ---
    for t_idx in range(NENKF):
        print(f"\n===== DA STEP {t_idx} =====")

        utils_time_march.m2c_update_max_time(
            BASE, CASE_NAME, dt_enkf[t_idx], Ne
        )
        for i in range(Ne):
            update_input_for_forecast(BASE/(CASE_NAME + f"_{i:03d}"), t_idx)
            subprocess.run(
                ["./Allrun", "input.st"],
                cwd=BASE/(CASE_NAME+f"_{i:03d}"),
                check=True
            )

        # get size
        NdVec = np.asarray([2,1,1,1])

        # get observations
        yMat[t_idx],sMat[t_idx] = utils_da.get_observations_2d(
            BASE,
            CASE_NAME_TRUTH,
            t_idx,
            enkfCount,
            OBS_X,
            OBS_Y,
            REL_NOISE,
            ABS_NOISE,
            rng
        )

        # read data
        prim_full, x_full, y_full = utils_time_march_2d.read_solution(
            BASE,CASE_NAME,t_idx,Ne)
        prim = prim_full[:,1:-1,1:-1,:]
        x = x_full[1:-1,1:-1]
        y = y_full[1:-1,1:-1]

        # determine which features to align
        Nd = NdVec.max()
        alignVec = np.zeros(Nd+1, dtype=bool)
        alignVec[0] = 1

        # apply EnKF
        prim_a,primLS = levelSet.EnKF(
            prim,
            x,
            y,
            yMat[t_idx],
            sMat[t_idx],
            NdVec,
            OBS_X,
            OBS_Y,
            rng,
            lam1=1e2,
            lamB=1e2,
            alignVecLF=alignVec,
            workers=workers
        )

        # write solution
        prim_a_full = np.zeros_like(prim_full)
        primLS_full = np.zeros_like(prim_full)
        for vdx in range(4):
            for edx in range(Ne):
                prim_a_full[vdx,:,:,edx] = np.pad(
                    prim_a[vdx,:,:,edx],
                    pad_width=1,
                    mode='constant',
                    constant_values=prim_a[vdx,0,0,edx]
                )
                primLS_full[vdx,:,:,edx] = np.pad(
                    primLS[vdx,:,:,edx],
                    pad_width=1,
                    mode='constant',
                    constant_values=primLS[vdx,0,0,edx]
                )
        utils_time_march_2d.write_solution(
            BASE,
            CASE_NAME,
            t_idx,
            primLS_full,
            Ne,
            analysisBool=False
        )
        utils_time_march_2d.write_solution(
            BASE,
            CASE_NAME,
            t_idx,
            prim_a_full,
            Ne,
            analysisBool=True
        )

        # Save observations and the number of completed forecast segments.
        with h5py.File(fname_master, "r+") as f:
            f["sMat"][t_idx] = sMat[t_idx]
            f["yMat"][t_idx] = yMat[t_idx]
            f["t_idx"][()] = t_idx + 1


    # final time-march
    utils_time_march.m2c_update_max_time(BASE, CASE_NAME, dt_final, Ne)
    for i in range(Ne):
        update_input_for_forecast(BASE/(CASE_NAME + f"_{i:03d}"), NENKF)
        subprocess.run(
            ["./Allrun", "input.st"],
            cwd=BASE/(CASE_NAME+f"_{i:03d}"),
            check=True
        )

    # Include the final post-DA forecast segment in the completed count.
    with h5py.File(fname_master, "r+") as f:
        f["t_idx"][()] = NENKF + 1

    print("SUCCESS: Full DA Cycle Complete.")


if __name__ == "__main__":
    main()
