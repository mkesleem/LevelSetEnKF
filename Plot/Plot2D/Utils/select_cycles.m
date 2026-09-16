% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function cycle_vec = select_cycles(base_dir, nplots)
% Select distinct truth cycles near evenly spaced targets for plotting.
%
% Input:
% - base_dir: directory containing the Truth directory
% - nplots: required number of cycles, including cycle zero and the final cycle
%
% Output:
% - cycle_vec: 1-by-nplots vector containing cycle zero (initial condition),
%   distinct intermediate DA cycles, and the final-time forecast
%
% Truth files must follow the naming convention solution_XXXX.h5.
%

truth_dir = fullfile(base_dir, "Truth");
truth_files = dir(fullfile(truth_dir, "solution_*.h5"));
truth_names = sort({truth_files.name});
if isempty(truth_names)
    error("No truth H5 files found in %s", truth_dir);
end

all_cycles = zeros(1, numel(truth_names));
for k = 1:numel(truth_names)
    tok = regexp(truth_names{k}, "solution_(\d+)\.h5", "tokens", "once");
    if isempty(tok)
        error("Unexpected truth filename: %s", truth_names{k});
    end
    all_cycles(k) = str2double(tok{1});
end

all_cycles = unique(all_cycles);
if nplots < 2 || nplots ~= floor(nplots)
    error("nplots must be an integer of at least 2.");
end
if ~ismember(0, all_cycles)
    error("No cycle-zero truth H5 file found in %s", truth_dir);
end
final_cycle = all_cycles(end);
final_file = fullfile(truth_dir, sprintf("solution_%04d.h5", final_cycle));
if ~strcmp(h5readatt(final_file, "/", "kind"), "FinalTime")
    error("Final-time truth H5 file is missing; rerun writeToH5_2D.py.");
end
da_available = all_cycles(all_cycles > 0 & all_cycles < final_cycle - 1);
if numel(da_available) < nplots - 2
    error("Expected at least %d intermediate DA cycles, found %d.", ...
        nplots - 2, numel(da_available));
end
da_targets = linspace(0, final_cycle, nplots);
da_selected = zeros(1, nplots - 2);
for k = 1:numel(da_selected)
    [~, idx] = min(abs(da_available - da_targets(k + 1)));
    da_selected(k) = da_available(idx);
    da_available(idx) = [];
end
cycle_vec = sort([0, da_selected, final_cycle]);


end
