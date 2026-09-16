% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;
addpath("Utils")

BASE_DIR = "/home/michael/lsos/Plot/Plot2D/BW2D";
CASE_NAME = "bw2d";
DATASET = "/pressure";
VAR_LABEL = "$p$";

fig1 = plot_bw2d_DA_field(BASE_DIR, CASE_NAME, DATASET, VAR_LABEL);

width = 13;
height = 7;
set(fig1, "Units", "inches", "Position", [0.1 0.1 width height]);