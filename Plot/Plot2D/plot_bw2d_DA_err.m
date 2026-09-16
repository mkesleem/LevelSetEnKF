% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;

BASE_DIR = "/home/michael/lsos/Plot/Plot2D/BW2D";
CASE_NAME = "bw2d";
addpath('Utils');


%% Read Data

[cVec0, rmseMat0, spreadMat0] = compute_error_2d(BASE_DIR, CASE_NAME);

% final entry corresponds to final time (no DA; discard)
cVec = cVec0(1:end-1);
rmseMat = rmseMat0(:,1:end-1);
spreadMat = spreadMat0(:,1:end-1);


%% Plot

titles = {'$\rho$','$u$','$v$','$p$'};

ymax  = 1.1 * max([max(rmseMat,[],2), max(spreadMat,[],2)], [], 2);
ylims = [zeros(4,1), ymax];

cRMSE   = [0.00 0.45 0.74];
cSpread = [0.85 0.33 0.10];

fig1 = figure(1); clf;
tl1 = tiledlayout(1,4,'TileSpacing','compact','Padding','compact');

for vdx = 1:4
    ax = nexttile;
    hold(ax,'on'); box(ax,'on');

    h1 = plot(ax,cVec,rmseMat(vdx,:)  ,'o-','Color',cRMSE   ,'LineWidth',1.5,'MarkerSize',5);
    h2 = plot(ax,cVec,spreadMat(vdx,:),'o-','Color', cSpread,'LineWidth',1.5,'MarkerSize',5);

    ylabel(ax, ['\textbf{' titles{vdx} '}'],'Interpreter','latex');
    ylim(ax, ylims(vdx,:));
    xlim(ax, [min(cVec), max(cVec)])
    xticks(ax, cVec(1:2:end));
    set(ax, 'FontSize', 8, 'LineWidth', 1.0);

    xlabel(ax,'$\mathrm{DA\ Cycle}$','Interpreter','latex');
    if vdx == 4
        lgd1 = legend([h1 h2],["$\mathrm{RMSE}$","$\mathrm{Spread}$"],...
            'Interpreter','latex','Location','northeast');
    end
end

width = 8;
ar = 0.15;
height = width * ar;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height]);

