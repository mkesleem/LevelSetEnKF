#!/usr/bin/env python3
# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


import utils_main
import utils_pv_1d as utils_pv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ANALYSIS_VTR_NAME = "solution_analysis.vtr"


def _write_ensemble_state_init_member(
        base_dir: utils_main.Path,
        case_name: str,
        t_idx: int,
        i: int,
        samples: dict,
        quiet: bool = True,
    ) -> None:
    """
    Prepare m2c restart files for ensemble member i.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution filename
    - i: zero-based ensemble-member index
    - samples: dictionary of samples parameters for member i
    - quiet: suppress update and build output if true

    Output:
    - None

    Loads solution_XXXX_0001.vtr, falling back to solution_XXXX_0000.vtr,
    writes IC/solution_analysis.vtr, updates UserDefinedState.cpp, and rebuilds
    the member's initial-condition code.
    """
    dir_i = base_dir / (case_name + f"_{i:03d}")
    vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0001.vtr"
    if not vtr_f.exists():
        vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0000.vtr"
    sol_i = utils_pv.load_solution(vtr_f, dir_i/"input.st")
    
    out_vtr = dir_i / "IC" / ANALYSIS_VTR_NAME
    x = sol_i['x']
    rho = sol_i['rho']
    u = sol_i['u']
    p = sol_i['p']

    # change data
    L = x < samples["x0"][i]
    R = x >= samples["x0"][i]
    rho[L] = samples["rhoL"][i]
    rho[R] = samples["rhoR"][i] + 0.2 * np.sin(
        10 * np.pi * (x[R] - samples["x0"][i])
    )
    u[L] = samples["uL"][i]
    u[R] = samples["uR"][i]
    p[L] = samples["pL"][i]
    p[R] = samples["pR"][i]

    utils_pv.write_vtk_analysis(
        template_vtr=vtr_f,
        input_st_path=dir_i / "input.st",
        rho_full=rho,
        u_full=u,
        p_full=p,
        out_vtr_path=out_vtr
    )
    utils_main.update_userdefinedstate(out_vtr, dir_i,quiet=quiet)
    utils_main.build_case(dir_i,quiet=quiet)


def write_ensemble_state_init(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    Ne: int,
    samples: dict,
    workers: int | None = None,
) -> None:
    """
    Prepare m2c restart files for an ensemble.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution filenames
    - Ne: ensemble size
    - samples: Ne samples of problem parameters
    - workers: number of worker threads, or None for the executor default

    Output:
    - None

    """

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for member in range(Ne):
            futures.append(executor.submit(
                _write_ensemble_state_init_member,
                base_dir,
                case_name,
                t_idx,
                member,
                samples,
            ))
        for future in futures:
            future.result()
