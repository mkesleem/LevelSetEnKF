# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT


from pathlib import Path
import h5py
import numpy as np
import utils_write_2d
import utils_time_march
import shutil
import xml.etree.ElementTree as ET


def pvd_time(pvd_file: Path, vtr_file: Path) -> float:
    if not pvd_file.is_file():
        raise FileNotFoundError(pvd_file)
    for entry in ET.parse(pvd_file).findall(".//DataSet"):
        if entry.get("file") == vtr_file.name:
            return float(entry.get("timestep"))
    raise ValueError(f"{vtr_file.name} is not listed in {pvd_file}")


# set params
BASE_DIR = Path("/home/michael/lsos")
CASE_NAME = "bw2d"
TRUTH_DIR = BASE_DIR / f"{CASE_NAME}.truth"

# destination folder
dir_out = BASE_DIR / "Plot/Plot2D/BW2D"

# get ensemble samples
ensemble_sample_file = BASE_DIR / "ensemble_samples.csv"
if not ensemble_sample_file.exists():
    raise FileNotFoundError(f"Ensemble sample file not found: {ensemble_sample_file}")
else:
    ensemble_sample_file_new = dir_out / "ensemble_samples.csv"
    ensemble_sample_file_new.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ensemble_sample_file, ensemble_sample_file_new)

# get time from the DA master file
master_file = BASE_DIR / f"{CASE_NAME}_master.h5"
if not master_file.exists():
    raise FileNotFoundError(f"Master H5 file not found: {master_file}")
with h5py.File(master_file, "r") as f:
    if "t" not in f:
        raise KeyError(f"Dataset 't' not found in master H5 file: {master_file}")
    t = np.asarray(f["t"])

# get ensembles
ensemble_dirs = utils_write_2d.detect_ensemble_dirs(BASE_DIR, CASE_NAME)

# get DA cycles
dir_0 = ensemble_dirs[0]
cycles = utils_write_2d.detect_cycles(dir_0)

# the sum of the completed time-march counts from earlier cycles.
t_index_by_cycle = {}
cumulative_count = 0
for cycle in cycles:
    t_index_by_cycle[cycle] = cumulative_count
    cumulative_count += utils_write_2d.result_count_for_cycle(dir_0 / "results", cycle)

final_cycle = cycles[-1] + 1
final_time_index = len(t) - 1
last_cycle = cycles[-1]
final_time = float(t[final_time_index])

# get solution prior to first DA cycle
first_cycle = cycles[0]
pre_da_step = utils_write_2d.result_count_for_cycle(dir_0 / "results", first_cycle)
pre_da_t_index = t_index_by_cycle[first_cycle] + pre_da_step
if not 0 <= pre_da_t_index < len(t):
    raise IndexError(f"Invalid pre-DA time index: {pre_da_t_index}")
pre_da_t_value = float(t[pre_da_t_index])

for dir_i in ensemble_dirs:
    edx = int(dir_i.name.rsplit("_", 1)[-1])
    dir_out_i = dir_out / dir_i.name
    results_dir = dir_i / "results"

    pre_da_vtr = results_dir / f"solution_{first_cycle:04d}_{pre_da_step:04d}.vtr"
    if not pre_da_vtr.exists():
        raise FileNotFoundError(
            f"Pre-DA ensemble VTR not found for ensemble {edx}: {pre_da_vtr}"
        )
    utils_write_2d.write_h5_from_vtr(
        pre_da_vtr,
        dir_out_i / f"pre_da_solution_{first_cycle:04d}_{pre_da_step:04d}.h5",
        t_value=pre_da_t_value,
        cycle=first_cycle,
        member=edx,
        kind="PreDA",
    )

    for cycle in cycles:
        t_index = t_index_by_cycle[cycle]
        if not 0 <= t_index < len(t):
            raise IndexError(f"Invalid time index {t_index} for cycle {cycle}")
        t_value = float(t[t_index])

        cycle_start_vtr = results_dir / f"solution_{cycle:04d}_0000.vtr"
        if cycle_start_vtr.exists():
            utils_write_2d.write_h5_from_vtr(
                cycle_start_vtr,
                dir_out_i / f"solution_{cycle:04d}_0000.h5",
                t_value=t_value,
                cycle=cycle,
                member=edx,
                kind="CycleStart",
            )
        else:
            raise FileNotFoundError(
                f"Missing cycle-start VTR for ensemble {edx}, "
                f"cycle {cycle}: {cycle_start_vtr}"
            )

    final_vtr = utils_time_march.get_last_solution(
        results_dir, f"solution_{last_cycle:04d}"
    )
    if final_vtr is None:
        raise FileNotFoundError(f"Missing final ensemble VTR: {results_dir}")
    cycle_start_time = float(t[t_index_by_cycle[last_cycle]])
    source_time = cycle_start_time + pvd_time(
        results_dir / f"solution_{last_cycle:04d}.pvd", final_vtr
    )
    if not np.isclose(source_time, final_time, rtol=0, atol=1e-9):
        raise ValueError(
            f"Final ensemble VTR {final_vtr} is at t={source_time}, "
            f"expected t={final_time}"
        )
    utils_write_2d.write_h5_from_vtr(
        final_vtr,
        dir_out_i / f"solution_{final_cycle:04d}_0000.h5",
        t_value=final_time,
        cycle=final_cycle,
        member=edx,
        kind="FinalTime",
    )

utils_write_2d.write_truth_h5(TRUTH_DIR, dir_out, cycles, t_index_by_cycle, t)

final_truth_vtr = TRUTH_DIR / "results" / f"solution_{final_time_index:04d}.vtr"
if not final_truth_vtr.exists():
    raise FileNotFoundError(f"Missing final truth VTR: {final_truth_vtr}")
truth_time = pvd_time(TRUTH_DIR / "results" / "solution.pvd", final_truth_vtr)
if not np.isclose(truth_time, final_time, rtol=0, atol=1e-9):
    raise ValueError(
        f"Final truth VTR {final_truth_vtr} is at t={truth_time}, "
        f"expected t={final_time}"
    )
utils_write_2d.write_h5_from_vtr(
    final_truth_vtr,
    dir_out / "Truth" / f"solution_{final_cycle:04d}.h5",
    t_value=final_time,
    cycle=final_cycle,
    member=-1,
    kind="FinalTime",
)

pre_da_truth_vtr = TRUTH_DIR / "results" / f"solution_{pre_da_t_index:04d}.vtr"
if not pre_da_truth_vtr.exists():
    raise FileNotFoundError(f"Pre-DA truth VTR not found: {pre_da_truth_vtr}")
utils_write_2d.write_h5_from_vtr(
    pre_da_truth_vtr,
    dir_out / "Truth" / f"pre_da_solution_{pre_da_t_index:04d}.h5",
    t_value=pre_da_t_value,
    cycle=first_cycle,
    member=-1,
    kind="PreDA",
)

print(f"Wrote H5 files for {len(ensemble_dirs)} ensemble member(s) plus truth, cycles {cycles}, to {dir_out}")
