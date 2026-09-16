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

% get filename
filename = '../../data_sod_LS.h5';
assert(isfile(filename))
data = read_LS_h5(filename);
NdVec = data.NdVec;
LS = data.LS;
LF = data.LF;
x  = data.x;
prim = data.prim;

FS = 12;

width = 6.5;
ar = 1.20;
height = width * ar;

c1      = Colors(1,:);
c2      = Colors(2,:);
c3      = Colors(3,:);
cShock  = Colors(4,:);
cContact = Colors(5,:);
cGray   = [0.25 0.25 0.25];

lineColors = [c1; c2; c3];

x1 = 0.35;
x2 = 0.70;

fig1 = figure(1); clf;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height], ...
    'Color', 'w');

tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

% Density

vdx = 1;
thisNd = NdVec(vdx);
[~,idxMin] = min(abs(LS(:,1:thisNd,vdx)));
xs = x(idxMin);

ax = nexttile;
hold(ax,'on');

hLF = gobjects(thisNd+1,1);
for j = 1:thisNd+1
    hLF(j) = plot(x, LF(:,j,vdx), ...
        'Color', lineColors(j,:), ...
        'LineWidth', 1.7);
end

h0 = plot(x, prim(vdx,:), '--', ...
    'Color', cGray, 'LineWidth', 1.4);

hxs1 = xline(xs(1), ':', 'Color', cContact, 'LineWidth', 1.7);
hxs2 = xline(xs(2), ':', 'Color', cShock, 'LineWidth', 1.7);

ylabel("$\rho$","Interpreter","latex")
xlim([x1,x2]);
ylim([0,1.1])

legend([hLF;hxs1;hxs2;h0], ...
    ["$\rho_1$","$\rho_2$", "$\rho_3$", ...
     "$x_{s,\rho,1}$","$x_{s,\rho,2}$","$\rho$"], ...
    'Interpreter','latex','Location','eastoutside','Box','off','FontSize',FS)

set(ax,'TickLabelInterpreter','latex','FontSize',FS, ...
    'LineWidth',0.9,'Box','off','Layer','top', ...
    'XMinorTick','on','YMinorTick','on')
ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.08;
ax.XTickLabel = [];

% Velocity

vdx = 2;
thisNd = NdVec(vdx);
[~,idxMin] = min(abs(LS(:,1:thisNd,vdx)));
xs = x(idxMin);

ax = nexttile;
hold(ax,'on');

hLF = gobjects(thisNd+1,1);
for j = 1:thisNd+1
    hLF(j) = plot(x, LF(:,j,vdx), ...
        'Color', lineColors(j,:), ...
        'LineWidth', 1.7);
end

h0 = plot(x, prim(vdx,:), '--', ...
    'Color', cGray, 'LineWidth', 1.4);

hxs1 = xline(xs(1), ':', 'Color', cShock, 'LineWidth', 1.7);

ylabel("$u$","Interpreter","latex")
xlim([x1,x2]);
ylim([-0.1,1])

legend([hLF;hxs1;h0], ...
    ["$u_1$", "$u_2$","$x_{s,u,1}$","$u$"], ...
    'Interpreter','latex','Location','eastoutside','Box','off','FontSize',FS)

set(ax,'TickLabelInterpreter','latex','FontSize',FS, ...
    'LineWidth',0.9,'Box','off','Layer','top', ...
    'XMinorTick','on','YMinorTick','on')
ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.08;
ax.XTickLabel = [];

% Pressure

vdx = 3;
thisNd = NdVec(vdx);
[~,idxMin] = min(abs(LS(:,1:thisNd,vdx)));
xs = x(idxMin);

ax = nexttile;
hold(ax,'on');

hLF = gobjects(thisNd+1,1);
for j = 1:thisNd+1
    hLF(j) = plot(x, LF(:,j,vdx), ...
        'Color', lineColors(j,:), ...
        'LineWidth', 1.7);
end

h0 = plot(x, prim(vdx,:), '--', ...
    'Color', cGray, 'LineWidth', 1.4);

hxs1 = xline(xs(1), ':', 'Color', cShock, 'LineWidth', 1.7);

xlabel("$x$","Interpreter","latex")
ylabel("$p$","Interpreter","latex")
xlim([x1,x2]);
ylim([0,1.1])

legend([hLF;hxs1;h0], ...
    ["$p_1$", "$p_2$","$x_{s,p,1}$","$p$"], ...
    'Interpreter','latex','Location','eastoutside','Box','off','FontSize',FS)

set(ax,'TickLabelInterpreter','latex','FontSize',FS, ...
    'LineWidth',0.9,'Box','off','Layer','top', ...
    'XMinorTick','on','YMinorTick','on')
ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.08;

%% Density-only figure

vdx = 1;
thisNd = NdVec(vdx);
[~,idxMin] = min(abs(LS(:,1:thisNd,vdx)));
xs = x(idxMin);

width = 4.5;
ar    = 0.6;

fig2 = figure(2); clf;
set(fig2, 'Units', 'inches', ...
    'Position', [0.1 0.1 width ar*width], ...
    'Color', 'w');

ax2 = axes(fig2);
hold(ax2,'on');

hLF2 = gobjects(thisNd+1,1);
for j = 1:thisNd+1
    hLF2(j) = plot(ax2, x, LF(:,j,vdx), ...
        'Color', lineColors(j,:), ...
        'LineWidth', 1.7);
end

h02 = plot(ax2, x, prim(vdx,:), '--', ...
    'Color', cGray, 'LineWidth', 1.4);

hxs12 = xline(ax2, xs(1), ':', ...
    'Color', cContact, 'LineWidth', 1.7);
hxs22 = xline(ax2, xs(2), ':', ...
    'Color', cShock, 'LineWidth', 1.7);

xlabel(ax2, "$x$", "Interpreter", "latex")
ylabel(ax2, "$\rho$", "Interpreter", "latex")
xlim(ax2, [x1,x2]);
ylim(ax2, [0,1.1]);

legend(ax2, [hLF2;hxs12;hxs22;h02], ...
    ["$\rho_1$","$\rho_2$","$\rho_3$", ...
     "$x_{s,1}$","$x_{s,2}$","$\rho$"], ...
    'Interpreter','latex', ...
    'Location','eastoutside', ...
    'Orientation','vertical', ...
    'Box','off', ...
    'FontSize',FS)

set(ax2,'TickLabelInterpreter','latex','FontSize',FS, ...
    'LineWidth',0.9,'Box','off','Layer','top', ...
    'XMinorTick','on','YMinorTick','on')
ax2.GridAlpha = 0.15;
ax2.MinorGridAlpha = 0.08;
