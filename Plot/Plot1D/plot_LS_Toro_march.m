% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;
addpath("Utils/")

if exist('brewermap','file')
    Colors = brewermap(8,'Dark2');
else
    Colors = lines(8);
end


%%

filename = '../../data_toro_LS_march.h5';

x = h5read(filename, '/x');
x = x(:);
Nx = numel(x);
Nt = 3;

prim_t0         = transpose(squeeze(h5read(filename,'/prim_t0'        )));
primLS_t0       = transpose(squeeze(h5read(filename,'/primLS_t0'      )));
primLS_sharp_t0 = transpose(squeeze(h5read(filename,'/primLS_sharp_t0')));

prim_final         = transpose(squeeze(h5read(filename,'/prim_final'        )));
primLS_final       = transpose(squeeze(h5read(filename,'/primLS_final'      )));
primLS_sharp_final = transpose(squeeze(h5read(filename,'/primLS_sharp_final')));

variableNames = {'$\rho$', '$u$', '$p$'};
legendNames = {'Finite Volume', 'Smooth Level Set', 'Sharp Level Set'};
columnTitles = {'$t_{LS,1} = 0.007$', ...
    '$T = 0.0245$'};

fig1 = plotComparison(x, ...
    {prim_t0, prim_final}, ...
    {primLS_t0, primLS_final}, ...
    {primLS_sharp_t0, primLS_sharp_final}, ...
    variableNames, legendNames, columnTitles, Colors);

width = 6.5; ar = 1.35;
height = width * ar;
set(fig1, 'Units', 'inches', 'Position', [0.1 0.1 width height])

fig2 = plotDensityComparison(x, ...
    {prim_t0, prim_final}, ...
    {primLS_t0, primLS_final}, ...
    {primLS_sharp_t0, primLS_sharp_final}, ...
    legendNames, columnTitles, Colors);

width = 6.5; ar = 0.5;
height = width * ar;
set(fig2, 'Units', 'inches', 'Position', [0.1 0.1 width height])

function fig1 = plotComparison(x,prim,primLS,primLSSharp,varNames,legNames,colTitles,Colors)
    fig1 = figure(1);
    layout = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    FS = 12;
    FS_leg = 12;

    fvC      = Colors(1,:);
    lsC      = Colors(2,:);
    lsSharpC = Colors(3,:);

    primMinVec = zeros(3,1);
    primMaxVec = zeros(3,1);
    for vdx = 1:3
        for tdx = 1:2
            thisPrimMin = min([min(primLSSharp{1}(vdx,:)),min(primLS{1}(vdx,:)),min(prim{1}(vdx,:))]);
            thisPrimMax = max([max(primLSSharp{1}(vdx,:)),max(primLS{1}(vdx,:)),max(prim{1}(vdx,:))]);
            primMinVec(vdx) = min([primMinVec(vdx),thisPrimMin]);
            primMaxVec(vdx) = max([primMaxVec(vdx),thisPrimMax]);
        end
    end
    
    ylimMat = zeros(3,2);
    ylimMat(:,1) = 1.15*primMinVec;
    ylimMat(:,2) = 1.15*primMaxVec;

    for vdx = 1:3
        for tdx = 1:2
            ax = nexttile(layout, 2*(vdx-1)+tdx);
            lh(3) = plot(ax,x,primLSSharp{tdx}(vdx,:),'-','Color',lsSharpC,'LineWidth',2.0);
            hold(ax,'on');
            lh(2) = plot(ax,x,primLS{tdx}(vdx,:),'-','Color',lsC,'LineWidth',2.0);
            lh(1) = plot(ax,x,prim{tdx}(vdx,:),'--','Color',fvC,'LineWidth', 2.0);

            % axis limits
            xlim(ax, [0.45, 0.85]);
            ylim(ax, ylimMat(vdx,:));

            if tdx == 1
                ylabel(ax,varNames{vdx},'Interpreter','latex','FontSize',FS);
            else
                ax.YTickLabel = [];
            end
            if vdx < 3
                ax.XTickLabel = [];
            else
                xlabel(ax,'$x$', 'Interpreter','latex','FontSize',FS);
            end
            if vdx == 1
                title(ax,colTitles{tdx},'Interpreter','latex','FontSize',FS);
            end
        end
    end

    lgd = legend(lh,legNames,'Orientation','horizontal','Interpreter','latex','FontSize',FS_leg,'Box','off');
    lgd.Layout.Tile = 'south';
end

function fig2 = plotDensityComparison(x,prim,primLS,primLSSharp,legNames,colTitles,Colors)
    fig2 = figure(2);
    layout = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    FS = 12;
    FS_leg = 12;

    fvC      = Colors(1,:);
    lsC      = Colors(2,:);
    lsSharpC = Colors(3,:);

    densityMin = inf;
    densityMax = -inf;
    for tdx = 1:2
        densityMin = min(densityMin, min([prim{tdx}(1,:), ...
            primLS{tdx}(1,:), primLSSharp{tdx}(1,:)]));
        densityMax = max(densityMax, max([prim{tdx}(1,:), ...
            primLS{tdx}(1,:), primLSSharp{tdx}(1,:)]));
    end
    densityRange = densityMax - densityMin;
    densityYLim = [densityMin - 0.05*densityRange, ...
        densityMax + 0.05*densityRange];

    for tdx = 1:2
        ax = nexttile(layout,tdx);
        lh(3) = plot(ax,x,primLSSharp{tdx}(1,:),'-', ...
            'Color',lsSharpC,'LineWidth',2.0);
        hold(ax,'on');
        lh(2) = plot(ax,x,primLS{tdx}(1,:),'-', ...
            'Color',lsC,'LineWidth',2.0);
        lh(1) = plot(ax,x,prim{tdx}(1,:),'--', ...
            'Color',fvC,'LineWidth',2.0);

        xlim(ax,[0.45,0.85]);
        ylim(ax,densityYLim);
        xlabel(ax,'$x$','Interpreter','latex','FontSize',FS);
        title(ax,colTitles{tdx},'Interpreter','latex','FontSize',FS);

        if tdx == 1
            ylabel(ax,'$\rho$','Interpreter','latex','FontSize',FS);
        else
            ax.YTickLabel = [];
        end
    end

    lgd = legend(lh,legNames,'Orientation','horizontal', ...
        'Interpreter','latex','FontSize',FS_leg,'Box','off');
    lgd.Layout.Tile = 'south';
end
