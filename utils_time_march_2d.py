# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


import utils_main
import utils_pv_2d as utilities_pv
import utils_time_march
import numpy as np

def write_ensemble_state(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    Ne: int
) -> None:
    """
    Prepare m2c restart files for a two-dimensional problem, comprising
    Ne ensemble members.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution and restart filenames
    - Ne: ensemble size

    Output:
    - None

    Loads solution_XXXX_0001.vtr, falling back to solution_XXXX_0000.vtr,
    writes IC/solution_XXXX_IC.vtr, updates UserDefinedState.cpp, and rebuilds
    each member's initial-condition code.
    """
    for i in range(Ne):
        dir_i = base_dir / (case_name + f"_{i:03d}")
        vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0001.vtr"
        if not vtr_f.exists(): vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0000.vtr"
        sol_i = utilities_pv.load_solution(vtr_f, dir_i/"input.st")
        
        out_vtr = dir_i / "IC" / f"solution_{t_idx:04d}_IC.vtr"
        rho = sol_i['rho']
        u = sol_i['u']
        v = sol_i['v']
        p = sol_i['p']

        utilities_pv.write_vtk_analysis(
            template_vtr=vtr_f,
            input_st_path=dir_i / "input.st",
            x_full=sol_i["x"],
            y_full=sol_i["y"],
            rho_full=rho,
            u_full=u,
            v_full=v,
            p_full=p,
            out_vtr_path=out_vtr
        )
        utils_main.update_userdefinedstate(out_vtr, dir_i)
        utils_main.build_case(dir_i)


def write_solution(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    prim: np.ndarray,
    Ne: int,
    analysisBool: bool = True,
) -> None:
    """
    Write two-dimensional fields for every ensemble member.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used to select the template and name the output
    - prim: 4 by Ny by Nx by Ne array ordered as density, x-velocity,
      y-velocity, and pressure
    - Ne: ensemble size
    - analysisBool: write an analysis restart and rebuild each case if true;
      otherwise write a forecast snapshot

    Output:
    - None

    For each member, the highest numbered solution_XXXX_####.vtr is used as
    the template. The interior fields are written to the output VTR, and the
    Farfield block in input.st is updated from the supplied boundary values.
    """
    for i in range(Ne):

        # load solution
        dir_i = base_dir / (case_name + f"_{i:03d}")
        dir_i_vtr = dir_i / "results" 
        vtr_base = f"solution_{t_idx:04d}"
        vtr_f = utils_time_march.get_last_solution(dir_i_vtr, vtr_base)
        sol_i = utilities_pv.load_solution(vtr_f, dir_i/"input.st")

        # write output solution
        if analysisBool:
            out_vtr = dir_i / "IC" / f"solution_{t_idx:04d}_analysis.vtr"
        else:
            out_vtr = dir_i / "IC" / f"solution_{t_idx:04d}_forecast.vtr"
        rho = prim[0,:,:,i]
        u = prim[1,:,:,i]
        v = prim[2,:,:,i]
        p = prim[3,:,:,i]
        utilities_pv.write_vtk_analysis(
            template_vtr=vtr_f,
            input_st_path=dir_i / "input.st",
            x_full=sol_i["x"],
            y_full=sol_i["y"],
            rho_full=rho,
            u_full=u,
            v_full=v,
            p_full=p,
            out_vtr_path=out_vtr
        )
        if analysisBool:
            utils_main.update_userdefinedstate(out_vtr, dir_i)
            utils_main.build_case(dir_i)


def read_solution(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    Ne: int
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Read the latest two-dimensional solution for every ensemble member.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - t_idx: cycle index used in the solution filename prefix
    - Ne: ensemble size

    Output:
    - prim: 4 by Ny by Nx by Ne array ordered as density, x-velocity,
      y-velocity, and pressure
    - x: Ny by Nx array of x-coordinates including the boundary layer
    - y: Ny by Nx array of y-coordinates including the boundary layer

    For each member, the highest numbered solution_XXXX_####.vtr is loaded.
    """

    dir_0 = base_dir / (case_name + f"_{0:03d}")
    vtr_f = dir_0 / "results" / f"solution_{t_idx:04d}_0001.vtr"
    if not vtr_f.exists(): vtr_f = dir_0 / "results" / f"solution_{t_idx:04d}_0000.vtr"
        
    sol_0 = utilities_pv.load_solution(vtr_f, dir_0/"input.st")
    x = sol_0['x']
    y = sol_0['y']
    Ny,Nx = x.shape
    assert(x.shape==y.shape)
    Nv = 4
    prim = np.zeros((Nv,Ny,Nx,Ne))

    for i in range(Ne):
        dir_i = base_dir / (case_name + f"_{i:03d}")
        dir_i_vtr = dir_i / "results" 
        vtr_base = f"solution_{t_idx:04d}"
        vtr_f = utils_time_march.get_last_solution(dir_i_vtr, vtr_base)
        if not vtr_f.exists(): vtr_f = dir_i / "results" / f"solution_{t_idx:04d}_0000.vtr"
        
        # get solution
        sol_i = utilities_pv.load_solution(vtr_f, dir_i/"input.st")
        prim[0,:,:,i] = sol_i['rho']
        prim[1,:,:,i] = sol_i['u']
        prim[2,:,:,i] = sol_i['v']
        prim[3,:,:,i] = sol_i['p']
        
    return prim, x, y
