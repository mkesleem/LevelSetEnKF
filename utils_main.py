# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]

# ------------------------------------------------------------
# Small shell helpers
# ------------------------------------------------------------
def _run(
    cmd: Iterable[str],
    *,
    cwd: PathLike | None = None,
    quiet: bool = True,
) -> None:
    """
    Run a command and raise an error if it fails.

    Input:
    - cmd: command and arguments
    - cwd: working directory, or None for the current directory
    - quiet: suppress the command and subprocess output if true

    Output:
    - None
    """

    if cwd is not None:
        cwd = Path(cwd)
    if not quiet:
        print(f"[run] ({cwd or Path.cwd()})$ {' '.join(cmd)}")
    subprocess.run(
        list(cmd), cwd=cwd, check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )


def run_allclean(
    base_dir: PathLike,
    case_name: str,
) -> None:
    """
    Run the repository's Allclean script for one case.

    Input:
    - base_dir: directory containing Allclean
    - case_name: case prefix to clean

    Output:
    - None

    The script is made executable if necessary. If it does not exist, this
    function prints a message and returns without raising an error.
    """
    base = Path(base_dir)
    script = base / "Allclean"
    if not script.exists():
        print("[allclean] Not found; skipping.")
        return
    if not os.access(script, os.X_OK):
        os.chmod(script, 0o755)
    _run(["./Allclean", case_name], cwd=base)


def case_dir(
    base_dir: PathLike,
    case_name: str,
    i: int,
) -> Path:
    """
    Construct the directory path for one ensemble member.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - i: zero-based ensemble-member index

    Output:
    - path: base_dir/case_name_###
    """
    return Path(base_dir) / (case_name + f"_{i:03d}")


# ------------------------------------------------------------
# Results/IC housekeeping
# ------------------------------------------------------------
def cleanSolution0(
    case_dir: PathLike = ".",
    results_subdir: str = "results",
    quiet: bool = True,
) -> None:
    """
    Reset an initial-condition results directory.

    Input:
    - case_dir: ensemble-member case directory
    - results_subdir: results-directory name within the case
    - quiet: suppress progress and warning messages if true

    Output:
    - None

    All files except solution_0000_0000.vtr and line_0000_0000.txt are
    removed. The retained VTR file is then copied to solution_0000_0001.vtr.
    A missing results directory or source VTR does not raise an error.
    """
    rdir = Path(case_dir, results_subdir)
    if not rdir.is_dir():
        if not quiet:
            print(f"[cleanSolution0] Directory not found: {rdir}")
        return

    keep = {"solution_0000_0000.vtr", "line_0000_0000.txt"}
    for f in rdir.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()
            if not quiet:
                print(f"[cleanSolution0] Removed {f.name}")

    src_vtr = rdir / "solution_0000_0000.vtr"
    dst_vtr = rdir / "solution_0000_0001.vtr"
    if src_vtr.exists():
        shutil.copy(src_vtr, dst_vtr)
        if not quiet:
            print(f"[cleanSolution0] Copied {src_vtr.name} -> {dst_vtr.name}")
    else:
        if not quiet:
            print("[cleanSolution0] WARNING: solution_0000_0000.vtr not found; nothing copied.")


def run_initial_member(
    base_dir: Path,
    case_name: str,
    i: int,
    quiet: bool = True,
) -> None:
    """
    Generate the initial output for one ensemble member.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - i: zero-based ensemble-member index
    - quiet: suppress solver and cleanup output if true

    Output:
    - None

    Runs ./Allrun input0.st in the member directory and then calls
    cleanSolution0().
    """
    dir_i = case_dir(base_dir, case_name, i)
    subprocess.run(
        ["./Allrun", "input0.st"], cwd=dir_i, check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    cleanSolution0(case_dir=dir_i,quiet=quiet)


# ------------------------------------------------------------
# Update C++ filename reference
# ------------------------------------------------------------
def update_userdefinedstate(
    out_path: str,
    case_dir: PathLike = ".",
    cpp_relpath: str = "IC/UserDefinedState.cpp",
    quiet: bool = True,
) -> None:
    """
    Update the VTR filename referenced by UserDefinedState.cpp.

    Input:
    - out_path: VTR path; only its basename is used
    - case_dir: ensemble-member case directory
    - cpp_relpath: path to UserDefinedState.cpp relative to case_dir
    - quiet: suppress the update message if true

    Output:
    - None

    Replaces the first 'std::string filename' declaration with an IC-relative
    path while preserving its indentation.
    """
    cpp_file = Path(case_dir, cpp_relpath)
    if not cpp_file.is_file():
        raise FileNotFoundError(f"UserDefinedState.cpp not found: {cpp_file}")

    rel_str = f"IC/{Path(out_path).name}"
    lines = cpp_file.read_text().splitlines(keepends=True)

    new_lines, replaced = [], False
    for line in lines:
        if "std::string filename" in line and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}std::string filename = "{rel_str}";\n')
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise RuntimeError(
            "Target line 'std::string filename = ...' not found."
        )

    cpp_file.write_text("".join(new_lines), encoding="utf-8")
    if not quiet:
        print(f"[update_userdefinedstate] {cpp_file} ← {rel_str}")


# ------------------------------------------------------------
# Build (cmake + make) in IC/
# ------------------------------------------------------------
def build_case(
    case_dir: PathLike = ".",
    ic_subdir: str = "IC",
    *,
    force_cmake: bool = False,
    quiet: bool = True,
) -> None:
    """
    Configure and build an ensemble member's initial-condition code.

    Input:
    - case_dir: ensemble-member case directory
    - ic_subdir: build-directory name within the case
    - force_cmake: run CMake even when CMakeCache.txt exists
    - quiet: suppress commands and build output if true

    Output:
    - None

    Runs 'cmake .' when configuration is required, followed by 'make'.
    """
    build_path = Path(case_dir, ic_subdir)
    if not build_path.is_dir():
        raise FileNotFoundError(f"Build directory not found: {build_path}")

    cache = build_path / "CMakeCache.txt"
    if force_cmake or not cache.exists():
        _run(["cmake", "."], cwd=build_path, quiet=quiet)
    else:
        if not quiet:
            print("[build_case] CMakeCache.txt found; skipping cmake (use force_cmake=True to refresh).")

    _run(["make"], cwd=build_path, quiet=quiet)
    if not quiet:
        print(f"[build_case] OK: {build_path}")


# ------------------------------------------------------------
# Update input.st file for solution and line name
# ------------------------------------------------------------
def update_input_for_forecast(
    case_dir: PathLike,
    t_idx: int,
    filename: str = "input.st",
    quiet: bool = True,
) -> None:
    """
    Update output names in a forecast input file for one DA cycle.

    Input:
    - case_dir: ensemble-member case directory
    - t_idx: cycle index used as a four-digit output suffix
    - filename: input-file name within case_dir
    - quiet: suppress the update message if true

    Output:
    - None

    Sets 'Solution' to solution_XXXX and 'FileName' to line_XXXX, where XXXX
    is the zero-padded cycle index.
    """
    case_dir = Path(case_dir)
    infile = case_dir / filename
    if not infile.is_file():
        raise FileNotFoundError(f"[update_input_for_forecast] {infile} not found.")

    tag = f"{t_idx:04d}"

    lines = infile.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("Solution"):
            new_lines.append(f'Solution = "solution_{tag}";')
        elif line.strip().startswith("FileName"):
            new_lines.append(f'FileName = "line_{tag}";')
        else:
            new_lines.append(line)

    infile.write_text("\n".join(new_lines) + "\n")
    if not quiet:
        print(f"[update_input_for_forecast] Updated {infile.name} for t_idx={t_idx}")


def run_forecast_member(
    base_dir: Path,
    case_name: str,
    i: int,
    t_idx: int,
    quiet: bool = True,
) -> None:
    """
    Run one forecast segment for one ensemble member.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - i: zero-based ensemble-member index
    - t_idx: DA-cycle index
    - quiet: suppress solver output if true

    Output:
    - None

    Updates input.st, archives it as input_XXXX.st, and runs
    ./Allrun input.st in the member directory.
    """
    dir_i = case_dir(base_dir, case_name, i)
    update_input_for_forecast(dir_i, t_idx)
    shutil.copyfile(
        dir_i / "input.st",
        dir_i / f"input_{t_idx:04d}.st",
    )
    subprocess.run(
        ["./Allrun", "input.st"], cwd=dir_i, check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
