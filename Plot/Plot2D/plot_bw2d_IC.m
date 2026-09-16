% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;

BASE_DIR = "/home/michael/lsos/Plot/Plot2D/BW2D";
CSV_FILE = fullfile(BASE_DIR, "ensemble_samples.csv");
NobsD = 4;

FS_label = 18;
FS_title = 18;
FS_tick = 14;


%% Files

if ~isfile(CSV_FILE)
    error("Missing ensemble sample file: %s", CSV_FILE);
end

samples = readtable(CSV_FILE, "VariableNamingRule", "preserve");
required_columns = ["x0", "y0", "ra"];
if ~all(ismember(required_columns, string(samples.Properties.VariableNames)))
    error("The CSV file must contain the columns: %s", ...
        strjoin(required_columns, ", "));
end


%% truth data

x_truth = 1.05;
y_truth = 1.05;
r_truth = 0.45;


%% Plot Pressure Circles

fig1 = figure(1); clf;
ax = axes(fig1);
hold(ax, "on"); box(ax, "on");

ens_color = [0.00 0.45 0.74];
theta = linspace(0, 2*pi, 361);

for edx = 1:height(samples)
    x_circle = samples.x0(edx) + samples.ra(edx) * cos(theta);
    y_circle = samples.y0(edx) + samples.ra(edx) * sin(theta);
    plot(ax, x_circle, y_circle, ...
        "Color", ens_color, ...
        "LineWidth", 1.0, ...
        "HandleVisibility", "off");
end

h_truth = plot(ax, x_truth + r_truth*cos(theta), y_truth + r_truth*sin(theta), ...
    "Color", "k", ...
    "LineWidth", 2.5, ...
    "DisplayName", "Truth");

h_ens = plot(ax, nan, nan, "-", ...
    "Color", [0.00 0.45 0.74], ...
    "LineWidth", 1.0, ...
    "DisplayName", "Ensembles");

OBS_Xc = sort(linspace(0.0, 2.0, NobsD + 2));
OBS_Xc = OBS_Xc(2:end-1);
[OBS_X, OBS_Y] = meshgrid(OBS_Xc, OBS_Xc);
h_obs = plot(ax, OBS_X(:), OBS_Y(:), "rx", ...
    "MarkerSize", 8, ...
    "LineWidth", 1.5, ...
    "DisplayName", "Pressure Sensors");

axis(ax, "square");
xlim(ax, [0 2]);
ylim(ax, [0 2]);
xlabel(ax, "$x$", "Interpreter", "latex", "FontSize", FS_label);
ylabel(ax, "$y$", "Interpreter", "latex", "FontSize", FS_label);
set(ax, "FontSize", FS_tick, "LineWidth", 1.0);
legend(ax, [h_ens, h_truth, h_obs], "Location", "best", "Interpreter", "latex");

width = 6;
height = 5.5;
set(fig1, "Units", "inches", "Position", [0.1 0.1 width height]);

