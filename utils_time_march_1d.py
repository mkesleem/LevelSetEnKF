# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.

"""Read, write, initialize, and advance one-dimensional ensembles."""

import utils_main
import utils_time_march
import utils_pv_1d as utils_pv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ANALYSIS_VTR_NAME = "solution_analysis.vtr"


def _write_ensemble_state_member(
    base_dir: utils_main.Path,
    case_name: str,
    t_idx: int,
    i: int,
    quiet: bool = True,
) -> None:
    """
    Prepare m2c restart files for ensemble member i.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution filename
    - i: zero-based ensemble-member index
    - quiet: suppress update and build output if true

    Output:
    - None

    Loads solution_XXXX_0001.vtr, falling back to solution_XXXX_0000.vtr,
    writes IC/solution_analysis.vtr, updates UserDefinedState.cpp, and rebuilds
    the member's initial-condition code.
    """
    dir_i = base_dir / (case_name + f"_{i:03d}")
    vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0001.vtr"
    if not vtr_f.exists(): vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0000.vtr"
    sol_i = utils_pv.load_solution(vtr_f, dir_i/"input.st")
    
    out_vtr = dir_i / "IC" / ANALYSIS_VTR_NAME
    rho_an = sol_i['rho']
    u_an = sol_i['u']
    p_an = sol_i['p']

    utils_pv.write_vtk_analysis(
        template_vtr=vtr_f,
        input_st_path=dir_i / "input.st",
        rho_full=rho_an,
        u_full=u_an,
        p_full=p_an,
        out_vtr_path=out_vtr
    )
    utils_main.update_userdefinedstate(out_vtr, dir_i, quiet=quiet)
    utils_main.build_case(dir_i, quiet=quiet)


def write_ensemble_state(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    Ne: int,
    executor=None,
    quiet: bool = True,
) -> None:
    """
    Prepare m2c restart files for an ensemble.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution filenames
    - Ne: ensemble size
    - executor: active executor used to process members in parallel
    - quiet: suppress update and build output if true

    Output:
    - None

    """
    
    futures = []
    for ensemble in range(Ne):
        futures.append(executor.submit(
            _write_ensemble_state_member,
            base_dir,
            case_name,
            t_idx,
            ensemble,
            quiet=quiet,
        ))
    for future in futures:
        future.result()
    

def write_solution(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    primLS: np.ndarray,
    Ne: int,
) -> None:
    """
    Write analyzed fields for every ensemble member.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used to select each template solution
    - primLS: 3 by Nx by Ne array ordered as density, velocity, pressure
    - Ne: ensemble size

    Output:
    - None

    Writes each member to IC/solution_analysis.vtr and updates the boundary
    states in input.st using the first and last values of each profile.
    """
    for i in range(Ne):

        # load solution
        dir_i = base_dir / (case_name + f"_{i:03d}")
        dir_i_vtr = dir_i / "results" 
        vtr_base = f"solution_{t_idx:04d}"
        vtr_f = utils_time_march.get_last_solution(dir_i_vtr, vtr_base)

        # write output solution
        out_vtr = dir_i / "IC" / ANALYSIS_VTR_NAME
        rho_an = primLS[0,:,i]
        u_an = primLS[1,:,i]
        p_an = primLS[2,:,i]
        utils_pv.write_vtk_analysis(
            template_vtr=vtr_f,
            input_st_path=dir_i / "input.st",
            rho_full=rho_an,
            u_full=u_an,
            p_full=p_an,
            out_vtr_path=out_vtr,
        )


def read_solution(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    Ne: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the latest solution for every ensemble member at one cycle.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution filename prefix
    - Ne: ensemble size

    Output:
    - prim: 3 by Nx by Ne array ordered as density, x-velocity, pressure
    - x: Nx coordinates including the two boundary points

    For each member, the highest numbered solution_XXXX_####.vtr is loaded.
    """

    dir_0 = base_dir / (case_name + f"_{0:03d}")
    vtr_f = dir_0 / "results" / f"solution_{t_idx:04d}_0001.vtr"
    if not vtr_f.exists(): vtr_f = dir_0 / "results" / f"solution_{t_idx:04d}_0000.vtr"
        
    sol_0 = utils_pv.load_solution(vtr_f, dir_0/"input.st")
    x = sol_0['x']
    Nx = len(x)
    Nv = 3
    prim = np.zeros((Nv,Nx,Ne))

    for i in range(Ne):
        dir_i = base_dir / (case_name + f"_{i:03d}")
        dir_i_vtr = dir_i / "results" 
        vtr_base = f"solution_{t_idx:04d}"
        vtr_f = utils_time_march.get_last_solution(dir_i_vtr, vtr_base)
        if not vtr_f.exists(): vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0000.vtr"
        
        # get solution
        sol_i = utils_pv.load_solution(vtr_f, dir_i/"input.st")
        prim[0,:,i] = sol_i['rho']
        prim[1,:,i] = sol_i['u']
        prim[2,:,i] = sol_i['p']
        
    return prim, x


def ensemble_init(
    base_dir: utils_main.Path,
    case_name: str,
    Ne: int,
    workers: int | None = None,
    quiet: bool = True,
) -> None:
    """
    Generate restartable initial conditions for an ensemble in parallel.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - Ne: ensemble size
    - workers: number of worker threads, or None for the executor default
    - quiet: suppress per-member solver, update, and build output if true

    Output:
    - None

    Runs each member from input0.st, normalizes its initial result files,
    writes IC/solution_analysis.vtr, updates UserDefinedState.cpp, and builds
    the initial-condition code.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        print(f"Generate {Ne} initial conditions")
        futures = []
        for member in range(Ne):
            futures.append(executor.submit(
                utils_main.run_initial_member,
                base_dir,
                case_name,
                member,
                quiet=quiet,
            ))
        for future in futures:
            future.result()

        print(f"Write {Ne} initial conditions to file")
        write_ensemble_state(
            base_dir,
            case_name,
            0,
            Ne,
            executor=executor,
            quiet=quiet,
        )


def ensemble_run(
    base_dir: utils_main.Path,
    case_name: str,
    Ne: int,
    max_time: float,
    workers: int | None = None,
    t_idx: int = 0,
    quiet: bool = True,
) -> None:
    """
    Advance every ensemble member through one forecast segment in parallel.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - Ne: ensemble size
    - max_time: segment duration written to each input.st file
    - workers: number of worker threads, or None for the executor default
    - t_idx: DA-cycle index used for output and archived input names
    - quiet: suppress per-member solver output if true

    Output:
    - None

    Updates the solver duration for all members, then runs each member from
    input.st using utils_main.run_forecast_member().
    """
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        print(f"Time march {Ne} trajectories")
        utils_time_march.m2c_update_max_time(
            base_dir,
            case_name,
            max_time,
            Ne,
        )
        futures = []
        for member in range(Ne):
            futures.append(executor.submit(
                utils_main.run_forecast_member,
                base_dir,
                case_name,
                member,
                t_idx, 
                quiet=quiet,
            ))
        for future in futures:
            future.result()
