#!/usr/bin/env python3
# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.
"""
Create perturbed two-dimensional ensemble cases from a template directory.

The module copies the template to case_name_000, case_name_001, and so on,
updates the circular initial state and far-field boundary conditions, and
records the sampled parameters. It does not execute the flow solver.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path
from utils_ensemble import (
    NUM,
    PAT_DENSITY,
    PAT_PRESS,
    WS,
    parse_ne_token,
    sample_ensemble,
    substitute_in_block,
    substitute_one,
)

# ---------------------- Regex helpers -------------------------------
PAT_POINT_X      = re.compile(rf"(Center_x{WS}={WS}){NUM}({WS};)", re.M)
PAT_POINT_Y      = re.compile(rf"(Center_y{WS}={WS}){NUM}({WS};)", re.M)
PAT_POINT_RA     = re.compile(rf"(Radius{WS}={WS}){NUM}({WS};)", re.M)
PAT_FARFIELD_BLOCK  = re.compile(r"under\s+Farfield\s*\{(?P<body>.*?)}", re.S)
PAT_INIT_BLOCK   = re.compile(r"under\s+InitialState\s*\{(?P<body>.*?)}", re.S)

# ---- existing: input0.st edits (diaphragm + IC + Inlet/Inlet2) ----
def apply_perturbations_to_text(
    text: str,
    x0: float,
    y0: float,
    ra: float,
    rho_in: float,
    p_in: float,
    rho_out: float,
    p_out: float,
) -> str:
    """
    Update the circular initial state and far field in input0.st.

    Input:
    - text: original input0.st text
    - x0: x-coordinate of the circle center
    - y0: y-coordinate of the circle center
    - ra: circle radius
    - rho_in: InitialState density inside the circle
    - p_in: InitialState pressure inside the circle
    - rho_out: Farfield density outside the circle
    - p_out: Farfield pressure outside the circle

    Output:
    - text: updated input0.st text
    """

    # 1) Define inner / outer region
    text, count = substitute_one(PAT_POINT_X, text, x0)
    if count != 1:
        raise ValueError("Center_x line not found or ambiguous.")
    text, count = substitute_one(PAT_POINT_Y, text, y0)
    if count != 1:
        raise ValueError("Center_y line not found or ambiguous.")
    text, count = substitute_one(PAT_POINT_RA, text, ra)
    if count != 1:
        raise ValueError("Radius line not found or ambiguous.")
    # 2) Farfield Density & Pressure
    text = substitute_in_block(
        PAT_FARFIELD_BLOCK,  PAT_DENSITY, text, rho_out, "Farfield"
    )
    text = substitute_in_block(
        PAT_FARFIELD_BLOCK,  PAT_PRESS,   text, p_out,   "Farfield"
    )
    # 3) InitialState
    text = substitute_in_block(
        PAT_INIT_BLOCK,   PAT_DENSITY, text, rho_in, "InitialState"
    )
    text = substitute_in_block(
        PAT_INIT_BLOCK,   PAT_PRESS,   text, p_in,   "InitialState"
    )
    return text

# ---- NEW: input.st edits (BoundaryConditions/Farfield only) --
def apply_bc_to_input(
    text: str,
    rho_out: float,
    p_out: float,
) -> str:
    """
    Update the Farfield boundary state in input.st.

    Input:
    - text: original input.st text
    - rho_out: Farfield density
    - p_out: Farfield pressure

    Output:
    - text: updated input.st text

    Velocity and material-ID fields are left unchanged.
    """
    text = substitute_in_block(
        PAT_FARFIELD_BLOCK,  PAT_DENSITY, text, rho_out, "Farfield"
    )
    text = substitute_in_block(
        PAT_FARFIELD_BLOCK,  PAT_PRESS,   text, p_out,   "Farfield"
    )
    return text

# ---------------------- Orchestration (no solver) -------------------
def make_initial_ensemble(
    base_dir: Path,
    template_dirname: str | Path,
    case_name: str,
    n: int,
    seed: int,
    ensemble_normal_params: dict,
) -> None:
    """
    Create perturbed two-dimensional ensemble case directories.

    Input:
    - base_dir: directory in which to create the ensemble cases
    - template_dirname: template directory name or path
    - case_name: prefix for the generated case directories
    - n: number of ensemble members
    - seed: random-number-generator seed
    - ensemble_normal_params: mapping of names to mean and standard deviation

    Output:
    - None: writes case_name_### directories and ensemble_samples.csv
    """

    base_dir = base_dir.resolve()
    template_dir = (base_dir / template_dirname).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    src_input0 = template_dir / "input0.st"
    src_input  = template_dir / "input.st"
    if not src_input0.is_file():
        raise FileNotFoundError(f"Missing input0.st in template: {src_input0}")
    if not src_input.is_file():
        raise FileNotFoundError(f"Missing input.st in template: {src_input}")

    # draw initial samples
    samples = sample_ensemble(
        n=n,
        ensemble_normal_params=ensemble_normal_params,
        seed=seed,
    )

    csv_path = base_dir / "ensemble_samples.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["run_id", "x0","y0","ra","rho1","rho2","p1","p2"])

        for i in range(n):
            x0 = samples["x0"][i]
            y0 = samples["y0"][i]
            ra = samples["ra"][i]
            r1 = samples["rho1"][i]
            r2 = samples["rho2"][i]
            p1 = samples["p1"][i]
            p2 = samples["p2"][i]
            run_name = case_name + f"_{i:03d}"
            run_dir = base_dir / run_name

            print(f"[{run_name}] copying template, updating input0.st and input.st(BC)")
            shutil.copytree(template_dir, run_dir, dirs_exist_ok=False)

            # input0.st
            text0 = (run_dir / "input0.st").read_text()
            new_text0 = apply_perturbations_to_text(
                text0,
                x0=float(x0),y0=float(y0),ra=float(ra),
                rho_in=float(r1), p_in=float(p1),
                rho_out=float(r2), p_out=float(p2),
            )
            (run_dir / "input0.st").write_text(new_text0)

            # input.st (BoundaryConditions only)
            text_in = (run_dir / "input.st").read_text()
            new_text_in = apply_bc_to_input(
                text_in,
                rho_out=float(r2), p_out=float(p2),
            )
            (run_dir / "input.st").write_text(new_text_in)

            writer.writerow([run_name, x0, y0, ra, r1, r2, p1, p2])

    print(f"\nCreated {n} initial runs: {case_name}_000 .. {case_name}_{n-1:03d}")
    print(f"Sample log: {csv_path}")

# ---------------------- CLI ----------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone ensemble generation."""
    p = argparse.ArgumentParser(description="Generate initial ensemble (no Allrun) from a template folder.")
    p.add_argument("Ne", type=str, help="Ensemble size; accepts '40' or 'Ne-40'")
    p.add_argument("--template", default="sod.template", help="Template folder name (default: sod.template)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    n = parse_ne_token(args.Ne)
    base = Path(__file__).resolve().parent
    make_initial_ensemble(base, args.template, n=n, seed=args.seed)
