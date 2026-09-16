% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function [cVec, prim_rmse, prim_spread] = compute_error_2d(base_dir, case_name)
% Compute two-dimensional ensemble RMSE and spread for every saved cycle.
%
% Input:
% - base_dir: directory containing Truth and ensemble-member directories
% - case_name: ensemble directory prefix; members match case_name_*
%
% Output:
% - cVec: 1-by-Nc vector of cycle indices, beginning with the pre-DA state
% - prim_rmse: 4-by-Nc array of spatial RMSE values for density,
%   x-velocity, y-velocity, and pressure
% - prim_spread: 4-by-Nc array of spatially averaged ensemble spreads for
%   the same variables
%
% The pre-DA state is read from the single pre_da_solution_*.h5 file in
% each directory. Later states are matched to Truth/solution_*.h5 using
% their cycle attributes.

ens_dirs = dir(fullfile(base_dir, case_name + "_*"));
ens_dirs = ens_dirs([ens_dirs.isdir]);
Ne = numel(ens_dirs);
if Ne == 0
    error("No ensemble folders found in %s", base_dir);
end

truth_dir = fullfile(base_dir, "Truth");
pre_da_truth_files = dir(fullfile(truth_dir, "pre_da_solution_*.h5"));
if numel(pre_da_truth_files) ~= 1
    error("Expected one pre-DA truth H5 file in %s; found %d", ...
        truth_dir, numel(pre_da_truth_files));
end

truth_files = dir(fullfile(truth_dir, "solution_*.h5"));
truth_names = sort({truth_files.name});
truth_names = truth_names(2:end);
Npost = numel(truth_names);
if Npost == 0
    error("No post-IC truth H5 files found in %s", truth_dir);
end
truth_names = [{pre_da_truth_files.name}, truth_names];
Nc = numel(truth_names);

vars = ["density", "u", "v", "pressure"];
Nv = numel(vars);
prim_rmse = zeros(Nv, Nc);
prim_spread = zeros(Nv, Nc);

for cdx = 1:Nc
    truth_file = fullfile(truth_dir, truth_names{cdx});
    cycle = h5readatt(truth_file, "/", "cycle");
    if cycle < 0
        cycle = cdx - 1;
    end

    for vdx = 1:Nv
        q_truth = h5read(truth_file, "/" + vars(vdx));
        [Ny, Nx] = size(q_truth);
        q_ens = zeros(Ny, Nx, Ne);

        for edx = 1:Ne
            ens_dir = fullfile(base_dir, ens_dirs(edx).name);
            if cdx == 1
                pre_da_ens_files = dir(fullfile(ens_dir, ...
                    "pre_da_solution_*.h5"));
                if numel(pre_da_ens_files) ~= 1
                    error("Expected one pre-DA ensemble H5 file in %s; found %d", ...
                        ens_dir, numel(pre_da_ens_files));
                end
                ens_file = fullfile(ens_dir, pre_da_ens_files.name);
            else
                ens_file = fullfile(ens_dir, ...
                    sprintf("solution_%04d_0000.h5", cycle));
            end
            if ~isfile(ens_file)
                error("Missing ensemble H5 file: %s", ens_file);
            end
            q_ens(:,:,edx) = h5read(ens_file, "/" + vars(vdx));
        end

        q_mean = mean(q_ens, 3);
        prim_rmse(vdx, cdx) = sqrt(mean((q_mean - q_truth).^2, "all"));
        prim_spread(vdx, cdx) = sqrt(mean(var(q_ens, 0, 3), "all"));
    end
end

cVec = [0, 1:Npost];

end
