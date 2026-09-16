% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function fig = plot_bw2d_DA_field(base_dir, case_name, dataset, var_label)
% Plot truth, ensemble mean, and farthest-member fields across five cycles.
%
% Input:
% - base_dir: directory containing Truth and ensemble-member directories
% - case_name: ensemble directory prefix; members match case_name_*
% - dataset: HDF5 dataset path to plot, such as "/density" or "/pressure"
% - var_label: text used for the colorbar label (the variable to plot)
%
% Output:
% - fig: handle to the generated figure
%
% The farthest member at each cycle is the member with the largest spatial
% RMSE relative to the truth. All tiles use a common color scale.
%

FS_label = 14;
FS_title = 14;
Nplots = 5;

cycle_vec = select_cycles(base_dir, Nplots);
ens_dirs = find_ensemble_dirs(base_dir, case_name);
[x, y, t_vec, truth_q, mean_q, far_q, far_idx, q_lim] = ...
    load_field_data_2d(base_dir, ens_dirs, cycle_vec, dataset);

fig = figure(1); clf;
tiledlayout(3, Nplots, "TileSpacing", "compact", "Padding", "compact");
row_names = ["Truth", "Mean", "Farthest"];

for cdx = 1:Nplots
    plot_field_tile(x, y, truth_q{cdx}, q_lim);
    title_text = sprintf("$t=%.4f$", t_vec(cdx));
    title(title_text, ...
        "Interpreter", "latex", "FontSize", FS_title);
    if cdx == 1
        ylabel(row_names(1), "Interpreter", "latex", "FontSize", FS_label);
    end
end

for cdx = 1:Nplots
    plot_field_tile(x, y, mean_q{cdx}, q_lim);
    if cdx == 1
        ylabel(row_names(2), "Interpreter", "latex", "FontSize", FS_label);
    end
end

for cdx = 1:Nplots
    plot_field_tile(x, y, far_q{cdx}, q_lim);
    title(sprintf("Ens. %03d", far_idx(cdx) - 1), ...
        "Interpreter", "latex", "FontSize", FS_title);
    if cdx == 1
        ylabel(row_names(3), "Interpreter", "latex", "FontSize", FS_label);
    end
end

cb = colorbar;
cb.Layout.Tile = "east";
cb.Label.String = var_label;
cb.Label.Interpreter = "latex";


end
