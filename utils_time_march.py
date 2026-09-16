# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


import utils_main
import re
from pathlib import Path

def m2c_update_max_time(
    base_dir: utils_main.PathLike,
    case_name: str,
    new_max_time: float,
    Ne: int,
) -> None:
    """
    Update MaxTime in every ensemble member's input.st file.

    Input:
    - base_dir: parent directory of the ensemble cases
    - case_name: case prefix
    - new_max_time: value assigned to each MaxTime field
    - Ne: ensemble size

    Output:
    - None

    Updates case_name_000/input.st through case_name_{Ne-1:03d}/input.st.
    Raises ValueError if an input file does not contain a MaxTime field.
    """
    
    for i in range(Ne):
        dir_i = base_dir / (case_name + f"_{i:03d}")
        file_name = dir_i / "input.st"
        text = Path(file_name).read_text()

        # Replace the MaxTime line (handles spaces and formatting)
        pattern = r"(MaxTime\s*=\s*)([0-9.eE+-]+)(\s*;)"
        replacement = rf"\g<1>{new_max_time}\g<3>"

        new_text, count = re.subn(pattern, replacement, text)

        if count == 0:
            raise ValueError("MaxTime not found in file.")

        Path(file_name).write_text(new_text)


def get_last_solution(
    folder: utils_main.PathLike,
    vtr_base: str,
) -> Path | None:
    """
    Find the VTR solution with the largest numeric suffix.

    Input:
    - folder: directory containing the solution files
    - vtr_base: filename prefix before the final numeric suffix

    Output:
    - solution_path: path matching vtr_base_*.vtr with the largest suffix,
      or None if no files match

    A nonnumeric suffix is assigned an index of -1.
    """
    files = list(folder.glob(vtr_base+"_*.vtr"))
    
    if not files:
        return None
    
    def extract_idx(f: Path) -> int:
        try:
            return int(f.stem.split("_")[-1])
        except ValueError:
            return -1  # ignore malformed files
    
    return max(files, key=extract_idx)
