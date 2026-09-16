#!/usr/bin/env python3
# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.
"""
Create perturbed Shu-Osher ensemble cases from a template directory.

The module copies the template to case_name_000, case_name_001, and so on,
updates the diaphragm and left/right flow states in input0.st and input.st,
and records the sampled parameters. It does not execute the flow solver.
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
PAT_POINT_X      = re.compile(rf"(Point_x{WS}={WS}){NUM}({WS};)", re.M)
PAT_INLET_BLOCK  = re.compile(r"under\s+Inlet\s*\{(?P<body>.*?)}", re.S)
PAT_INLET2_BLOCK = re.compile(r"under\s+Inlet2\s*\{(?P<body>.*?)}", re.S)
PAT_INIT_BLOCK   = re.compile(r"under\s+InitialState\s*\{(?P<body>.*?)}", re.S)

PAT_VELOCITY = re.compile(rf"(\bVelocityX{WS}={WS}){NUM}({WS};)")

# ---- existing: input0.st edits (diaphragm + IC + Inlet/Inlet2) ----
def apply_perturbations_to_text(
    text: str,
    x_diaphragm: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
) -> str:
    """
    Apply diaphragm, initial-state, and boundary perturbations to input0.st.

    Input:
    - text: original input0.st text
    - x_diaphragm: diaphragm location
    - rho_L: Inlet density
    - u_L: Inlet x-velocity
    - p_L: Inlet pressure
    - rho_R: Inlet2 and InitialState density
    - u_R: Inlet2 and InitialState x-velocity
    - p_R: Inlet2 and InitialState pressure

    Output:
    - text: updated input0.st text
    """
    # 1) Define inner / outer region
    text, count = substitute_one(PAT_POINT_X, text, x_diaphragm)
    if count != 1:
        raise ValueError("Center_x line not found or ambiguous.")
    # 2) Inlet BCs
    text = substitute_in_block(
        PAT_INLET_BLOCK,
        PAT_DENSITY,
        text,
        rho_L,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET_BLOCK,
        PAT_VELOCITY,
        text,
        u_L,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET_BLOCK,
        PAT_PRESS,
        text,
        p_L,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET2_BLOCK,
        PAT_DENSITY,
        text,
        rho_R,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET2_BLOCK,
        PAT_VELOCITY,
        text,
        u_R,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET2_BLOCK,
        PAT_PRESS,
        text,
        p_R,
        "Inlet"
    )
    # 3) InitialState
    text = substitute_in_block(
        PAT_INIT_BLOCK,
        PAT_DENSITY,
        text,
        rho_R,
        "InitialState"
    )
    text = substitute_in_block(
        PAT_INIT_BLOCK,
        PAT_VELOCITY,
        text,
        u_R,
        "InitialState"
    )
    text = substitute_in_block(
        PAT_INIT_BLOCK,
        PAT_PRESS,
        text,
        p_R,
        "InitialState"
    )
    return text

# ---- NEW: input.st edits (BoundaryConditions/Farfield only) --
def apply_bc_to_input(
    text: str,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
) -> str:
    """
    Update the Inlet and Inlet2 boundary states in input.st.

    Input:
    - text: original input.st text
    - rho_L: Inlet density
    - u_L: Inlet x-velocity
    - p_L: Inlet pressure
    - rho_R: Inlet2 density
    - u_R: Inlet2 x-velocity
    - p_R: Inlet2 pressure

    Output:
    - text: updated input.st text
    """
    text = substitute_in_block(
        PAT_INLET_BLOCK,
        PAT_DENSITY,
        text,
        rho_L,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET_BLOCK,
        PAT_VELOCITY,
        text,
        u_L,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET_BLOCK,
        PAT_PRESS,
        text,
        p_L,
        "Inlet"
    )
    text = substitute_in_block(
        PAT_INLET2_BLOCK,
        PAT_DENSITY,
        text,
        rho_R,
        "Inlet2"
    )
    text = substitute_in_block(
        PAT_INLET2_BLOCK,
        PAT_VELOCITY,
        text,
        u_R,
        "Inlet2"
    )
    text = substitute_in_block(
        PAT_INLET2_BLOCK,
        PAT_PRESS,
        text,
        p_R,
        "Inlet2"
    )
    return text

def make_initial_ensemble(
    base_dir: Path,
    template_dirname: str | Path,
    case_name: str,
    n: int,
    seed: int,
    ensemble_normal_params: dict,
    quiet: bool = True,
) -> dict:
    """
    Create perturbed Shu-Osher ensemble case directories.

    Input:
    - base_dir: directory in which to create the ensemble cases
    - template_dirname: template directory name or path
    - case_name: prefix for the generated case directories
    - n: number of ensemble members
    - seed: random-number-generator seed
    - ensemble_normal_params: mapping of names to mean and standard deviation
    - quiet: suppress per-member progress messages if true

    Output:
    - samples: mapping of parameter names to sampled ensemble values

    Also writes case_name_### directories and ensemble_samples.csv.
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

    samples = sample_ensemble(
        n=n,
        ensemble_normal_params=ensemble_normal_params,
        seed=seed,
    )

    csv_path = base_dir / "ensemble_samples.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["run_id", "x0","rhoL","uL","pL","rhoR","uR","pR"])

        for i in range(n):
            x0 = samples["x0"][i]
            rL = samples["rhoL"][i]
            uL = samples["uL"][i]
            pL = samples["pL"][i]
            rR = samples["rhoR"][i]
            uR = samples["uR"][i]
            pR = samples["pR"][i]
                      
            run_name = case_name + f"_{i:03d}"
            run_dir = base_dir / run_name

            if not quiet:
                print(f"[{run_name}] copying template, updating input0.st and input.st(BC)")
            shutil.copytree(template_dir, run_dir, dirs_exist_ok=False)

            # input0.st
            text0 = (run_dir / "input0.st").read_text()
            new_text0 = apply_perturbations_to_text(
                text0,
                x_diaphragm=float(x0),
                rho_L=float(rL),
                u_L=float(uL),
                p_L=float(pL),
                rho_R=float(rR),
                u_R=float(uR),
                p_R=float(pR),
            )
            (run_dir / "input0.st").write_text(new_text0)

            # input.st (BoundaryConditions only)
            text_in = (run_dir / "input.st").read_text()
            new_text_in = apply_bc_to_input(
                text_in,
                rho_L=float(rL), u_L=float(uL), p_L=float(pL),
                rho_R=float(rR), u_R=float(uR), p_R=float(pR),
            )
            (run_dir / "input.st").write_text(new_text_in)

            writer.writerow([run_name, x0, rL, uL, pL, rR, uR, pR])

    print(f"\nCreated {n} initial runs: {case_name}_000 .. {case_name}_{n-1:03d}")
    print(f"Sample log: {csv_path}")
    return samples

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
