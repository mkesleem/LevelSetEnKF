% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function [x, y, t_vec, truth_q, mean_q, far_q, far_idx, q_lim] = ...
    load_field_data_2d(base_dir, ens_dirs, cycle_vec, dataset)
% Load truth and ensemble statistics for one field at selected cycles.
%
% Input:
% - base_dir: directory containing Truth and ensemble-member directories
% - ens_dirs: ensemble directory-structure array returned by
%   find_ensemble_dirs
% - cycle_vec: vector of cycles to load
% - dataset: HDF5 dataset path to load
%
% Output:
% - x: Ny-by-Nx array of x-coordinates
% - y: Ny-by-Nx array of y-coordinates
% - t_vec: 1-by-Ncycles vector of solution times
% - truth_q: 1-by-Ncycles cell array of Ny-by-Nx truth fields
% - mean_q: 1-by-Ncycles cell array of Ny-by-Nx ensemble-mean fields
% - far_q: 1-by-Ncycles cell array containing the member farthest from truth
% - far_idx: 1-by-Ncycles vector of one-based farthest-member indices
% - q_lim: [minimum, maximum] shared color limits across all returned fields
%

truth_dir = fullfile(base_dir, "Truth");
Ncycles = numel(cycle_vec);
Ne = numel(ens_dirs);

truth_q = cell(1, Ncycles);
mean_q = cell(1, Ncycles);
far_q = cell(1, Ncycles);
far_idx = zeros(1, Ncycles);
t_vec = zeros(1, Ncycles);
q_min = Inf;
q_max = -Inf;

for cdx = 1:Ncycles
    cycle = cycle_vec(cdx);
    truth_file = fullfile(truth_dir, sprintf("solution_%04d.h5", cycle));
    if ~isfile(truth_file)
        error("Missing truth file: %s", truth_file);
    end

    truth_q{cdx} = h5read(truth_file, dataset);
    x_this = h5read(truth_file, "/x");
    y_this = h5read(truth_file, "/y");
    t_vec(cdx) = h5readatt(truth_file, "/", "t");
    if cdx == 1
        x = x_this;
        y = y_this;
    elseif ~isequal(x_this, x) || ~isequal(y_this, y)
        error("Truth grid changed in %s", truth_file);
    end

    [Ny, Nx] = size(truth_q{cdx});
    ens_q = zeros(Ny, Nx, Ne);
    rmse_member = zeros(Ne, 1);

    for edx = 1:Ne
        ens_file = fullfile(base_dir, ens_dirs(edx).name, ...
            sprintf("solution_%04d_0000.h5", cycle));
        if ~isfile(ens_file)
            error("Missing ensemble file: %s", ens_file);
        end

        ens_q(:,:,edx) = h5read(ens_file, dataset);
        rmse_member(edx) = sqrt(mean( ...
            (ens_q(:,:,edx) - truth_q{cdx}).^2, "all"));
    end

    mean_q{cdx} = mean(ens_q, 3);
    [~, far_idx(cdx)] = max(rmse_member);
    far_q{cdx} = ens_q(:,:,far_idx(cdx));

    q_min = min([q_min, min(truth_q{cdx}, [], "all"), ...
        min(mean_q{cdx}, [], "all"), min(far_q{cdx}, [], "all")]);
    q_max = max([q_max, max(truth_q{cdx}, [], "all"), ...
        max(mean_q{cdx}, [], "all"), max(far_q{cdx}, [], "all")]);
end

q_lim = [q_min, q_max];


end

