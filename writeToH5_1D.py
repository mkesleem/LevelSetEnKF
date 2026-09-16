# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT


import numpy as np
import h5py
import utils_time_march
import utils_pv_1d as utils_pv
from pathlib import Path
import argparse
import warnings

parser = argparse.ArgumentParser(
    description="Convert 1D VTR ensemble data to HDF5."
)

parser.add_argument(
    "CASE_NAME",
    type=str,
    help="Case name (e.g., bw, sod, toro, etc.)"
)

parser.add_argument(
    "out_fn",
    type=str,
    help="Output HDF5 filename"
)

parser.add_argument(
    "--base-dir",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="Directory containing CASE_NAME_###, CASE_NAME.truth, and CASE_NAME_master.h5"
)

args = parser.parse_args()

CASE_NAME = args.CASE_NAME
out_fn = args.out_fn
BASE = args.base_dir.resolve()
_missing_snapshot_warning_issued = False

def segment_input(case_dir: Path, segment_idx: int) -> Path:
    """Return the input snapshot associated with one forecast segment."""
    global _missing_snapshot_warning_issued
    snapshot = case_dir / f"input_{segment_idx:04d}.st"
    if snapshot.is_file():
        return snapshot
    if not _missing_snapshot_warning_issued:
        warnings.warn(
            f"Missing boundary-condition snapshot {snapshot}; falling back "
            "to the current input.st. Historical boundaries may be stale.",
            stacklevel=2,
        )
        _missing_snapshot_warning_issued = True
    return case_dir / "input.st"

# read master file
in_fn = BASE / f"{CASE_NAME}_master.h5"
with h5py.File(in_fn,"r") as f:
    sMat = f['sMat'][:]
    yMat = f['yMat'][:]
    xObs = f['xObs'][:]
    enkfBool = f['enkfBool'][:]
    Ne = int(f["Ne"][()])
    N_segments = int(f['t_idx'][()])
    t = f['t'][:]
    prim_f_LS = f['prim_f_LS'][:]
    prim_a_mean = f['prim_a_mean'][:]
    prim_f_mean = f['prim_f_mean'][:]
    mean_t_idx = f['mean_t_idx'][:]

# t_idx is the number of completed forecast segments.  Before any DA step
# finishes, segment 0 may still contain the initial state.
N_segments = max(1, N_segments)

# determine number of time steps
dir_0 = BASE / (CASE_NAME + f"_{0:03d}")
dir_0_vtr = dir_0 / "results" 
t_count_vec = np.zeros(N_segments,dtype=int)
for sdx in range(N_segments):
    vtr_base = f"solution_{sdx:04d}"
    vtr_f = utils_time_march.get_last_solution(dir_0_vtr, vtr_base)
    if vtr_f is None:
        raise FileNotFoundError(f"No VTR files found for {vtr_base} in {dir_0_vtr}")
    t_count_vec[sdx] = int(vtr_f.stem.split("_")[-1])

Nt = int(sum(t_count_vec))+1
if Nt > len(t):
    raise ValueError(f"Found {Nt} output times, but master file only contains {len(t)} times")
t = t[:Nt]

# ensembles
fn = dir_0_vtr / f"solution_{0:04d}_0000.vtr"
sol = utils_pv.load_solution(fn, segment_input(dir_0, 0))
x = sol["x"]
Nx = len(x)
rho = np.zeros((Nx,Nt,Ne))
u = np.zeros((Nx,Nt,Ne))
p = np.zeros((Nx,Nt,Ne))

for edx in range(Ne):
    dir_i = BASE / (CASE_NAME + f"_{edx:03d}")
    dir_i_vtr = dir_i / "results"
    for sdx in range(N_segments):
        for tdx in range(t_count_vec[sdx]+1):
            fn = dir_i_vtr / f"solution_{sdx:04d}_{tdx:04d}.vtr"
            sol = utils_pv.load_solution(fn, segment_input(dir_i, sdx))
            x_i = sol["x"]
            assert len(x_i)==Nx, "Error in length of x"
            if not np.allclose(x_i, x):
                raise ValueError(f"x grid mismatch in {fn}")
            if sdx > 0:
                this_t_index = int(sum(t_count_vec[:sdx])) + tdx
            else:
                this_t_index = tdx
            rho[:,this_t_index,edx] = sol["rho"]
            u[:,this_t_index,edx]   = sol["u"]
            p[:,this_t_index,edx]   = sol["p"]


# truth
truth_dir = BASE / f"{CASE_NAME}.truth"
fn = truth_dir / "results" / f"solution_{0:04d}.vtr"
sol = utils_pv.load_solution(fn, truth_dir / "input.st")
x_t = sol["x"]
Nx_t = len(x_t)
rho_t = np.zeros((Nx_t,Nt))
u_t = np.zeros((Nx_t,Nt))
p_t = np.zeros((Nx_t,Nt))
for tdx in range(Nt):
    fn = truth_dir / "results" / f"solution_{tdx:04d}.vtr"
    sol = utils_pv.load_solution(fn, truth_dir / "input.st")
    x_this = sol["x"]
    assert len(x_t)==Nx_t, "Error in length of x (truth)"
    if not np.allclose(x_this, x_t):
        raise ValueError(f"truth x grid mismatch in {fn}")
    rho_t[:,tdx] = sol["rho"]
    u_t[:,tdx]   = sol["u"]
    p_t[:,tdx]   = sol["p"]

with h5py.File(out_fn, "w") as f:
    f.create_dataset("x",   data=x)
    f.create_dataset("rho", data=rho)
    f.create_dataset("u",   data=u)
    f.create_dataset("p",   data=p)
    f.create_dataset("x_t",   data=x_t)
    f.create_dataset("rho_t", data=rho_t)
    f.create_dataset("u_t",   data=u_t)
    f.create_dataset("p_t",   data=p_t)
    f.create_dataset("yMat",   data=yMat)
    f.create_dataset("sMat",   data=sMat)
    f.create_dataset("xObs",   data=xObs)
    f.create_dataset("enkfBool", data=enkfBool)
    f.create_dataset("e",   data=enkfBool)
    f.create_dataset("t",   data=t)
    f.create_dataset("t_count_vec", data=t_count_vec)
    f.create_dataset("prim_f_LS", data=prim_f_LS)
    f.create_dataset("prim_a_mean", data=prim_a_mean)
    f.create_dataset("prim_f_mean", data=prim_f_mean)
    f.create_dataset("mean_t_idx", data=mean_t_idx)

print("Data written to: "+out_fn)
