% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;
addpath("Utils/")


%%

% get filename
filename = '../../data_toro.h5';
%filename = '../../data_shuOsher.h5';
%filename = '../../data_sod.h5';
assert(isfile(filename));

% read data
data = read_h5(filename);

[rmse,spread] = compute_err(data);
N_da = size(rmse,2);
da_cycle = (1:N_da)-1;


%% Plot

set(groot, ...
    'defaultTextInterpreter','latex', ...
    'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex');

titles = {'$\rho$','$u$','$p$'};

cRMSE   = [0.00 0.45 0.74];
cSpread = [0.85 0.33 0.10];

%% RMSE figure
fig1 = figure(1); clf;
tl1 = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

for vdx = 1:3
    ax = nexttile;
    hold(ax,'on'); box(ax,'on');% grid(ax,'on');

    h1 = plot(ax, da_cycle, rmse(vdx,:), 'o-', 'Color', cRMSE, 'LineWidth', 1.8);
    h2 = plot(ax, da_cycle, spread(vdx,:), 'o-', 'Color', cSpread, 'LineWidth', 1.8);

    thisYLim = [0,1.1*max([max(rmse(vdx,:)),max(spread(vdx,:))])];
    ylabel(ax,['\textbf{' titles{vdx} '}'],'FontSize',10);
    ylim(ax,thisYLim);
    xlim(ax,[1,N_da]-1);
    set(ax,'FontSize',10,'LineWidth',1.0);

    xlabel(ax,'$\mathrm{DA\ Cycle}$','FontSize',10);

    if vdx == 3
        lgd1 = legend([h1(1) h2(1)],["$\mathrm{RMSE}$","$\mathrm{Spread}$"],...
            'location','northeast','FontSize',10);
    end
end

width = 8; ar = 0.25;
height = width * ar;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height])
