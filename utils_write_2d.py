# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


from pathlib import Path
import h5py
import numpy as np
import pyvista as pv
import utils_time_march

def read_vtr_fields(
    vtr_path: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Read 2D vtr file

    Input:
    - vtr_path: path to vtr file

    Output:
    - rho: Ny by Nx array of density
    - u: Ny by Nx array of x-velocity
    - v: Ny by Nx array of y-velocity
    - p: Ny by Nx array of pressure
    - x: Ny by Nx array of x-coordaintes
    - y: Ny by Nx array of y-coordaintes
    """
    sol = pv.read(str(vtr_path))
    nx = sol.dimensions[0]
    ny = sol.dimensions[1]
    x = sol.points[:, 0].reshape(ny, nx)
    y = sol.points[:, 1].reshape(ny, nx)
    rho = np.asarray(sol.point_data["density"]).reshape(ny, nx)
    u = np.asarray(sol.point_data["velocity"][:, 0]).reshape(ny, nx)
    v = np.asarray(sol.point_data["velocity"][:, 1]).reshape(ny, nx)
    p = np.asarray(sol.point_data["pressure"]).reshape(ny, nx)
    return rho, u, v, p, x, y


def write_h5_from_vtr(
    vtr_path: Path,
    h5_path: Path,
    *,
    t_value: float,
    cycle: int | None,
    member: int,
    kind: str
) -> None:
    """
    Write h5 file from VTR file

    Input:
    - vtr_path: path to vtr file
    - h5_path: path to h5 file
    - t_value: time
    - cycle: DA cycle
    - member: index of ensemble member
    - kind: unknown

    Output:
    - None
    """
    rho, u, v, p, x, y = read_vtr_fields(vtr_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("density", data=rho)
        f.create_dataset("u", data=u)
        f.create_dataset("v", data=v)
        f.create_dataset("pressure", data=p)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("t", data=t_value)
        f.attrs["Nx"] = x.shape[1]
        f.attrs["Ny"] = x.shape[0]
        f.attrs["t"] = t_value
        f.attrs["cycle"] = -1 if cycle is None else cycle
        f.attrs["ensemble"] = member
        f.attrs["kind"] = kind
        f.attrs["source_vtr"] = str(vtr_path)


def write_truth_h5(
    truth_dir: Path,
    dir_out: Path,
    cycles: list[int],
    t_index_by_cycle: dict[int, int],
    t: np.ndarray,
) -> None:
    """
    Write truth to h5 file

    Input:
    - truth_dir: directory containing truth solutions
    - dir_out: output directory
    - cycles: number of DA cycles
    - t_index_by_cycle: time index by DA cycle
    - t: Nt array of times

    Output:
    - None
    """
    results_dir = truth_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Truth results directory not found: {results_dir}")

    for cycle in cycles:
        t_index = t_index_by_cycle[cycle]
        truth_vtr = results_dir / f"solution_{t_index:04d}.vtr"
        if not truth_vtr.exists():
            raise FileNotFoundError(f"Truth VTR not found for cycle {cycle}: {truth_vtr}")

        write_h5_from_vtr(
            truth_vtr,
            dir_out / "Truth" / f"solution_{cycle:04d}.h5",
            t_value=float(t[min(t_index, len(t) - 1)]),
            cycle=cycle,
            member=-1,
            kind="Truth",
        )


def detect_ensemble_dirs(
    base_dir: Path,
    case_name: str,
) -> list[Path]:
    """
    Detect directory for ensemble members

    Input:
    - base_dir: base directory
    - case_name: case name

    Output:
    - dirs: list of directories
    """
    dirs = sorted(base_dir.glob(f"{case_name}_[0-9][0-9][0-9]"))
    if not dirs:
        raise FileNotFoundError(f"No ensemble directories found under {base_dir} matching {case_name}_###")
    return dirs


def detect_cycles(
    dir_0: Path
) -> list[int]:
    """
    Detect DA cycles in dir_0

    Input:
    - dir_0: directory to search in

    Output:
    - list: list of DA cycles
    """
    cycles = []
    for f in sorted((dir_0 / "results").glob("solution_*_0000.vtr")):
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[0] == "solution":
            cycles.append(int(parts[1]))

    if not cycles:
        raise FileNotFoundError(f"No solution_####_0000.vtr files found in {dir_0 / 'results'}")
    return cycles


def result_count_for_cycle(
    results_dir: Path,
    cycle: int
) -> int:
    """
    Count number of solutions per cycle

    Input:
    - results_dir: directory containing results
    - cycle: index of cycle number

    Output:
    - count: number of DA cycles
    """
    
    vtr_base = f"solution_{cycle:04d}"
    vtr_f = utils_time_march.get_last_solution(results_dir, vtr_base)
    if vtr_f is None:
        raise FileNotFoundError(f"No VTR files found for {vtr_base} in {results_dir}")
    return int(vtr_f.stem.split("_")[-1])