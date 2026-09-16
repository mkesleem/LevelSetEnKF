% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;

fn = "../../data_bw2d_LS.h5";

prim   = h5read(fn, "/prim");
x      = h5read(fn, "/x");
y      = h5read(fn, "/y");
LF     = h5read(fn, "/LF");
LS     = h5read(fn, "/LS");
LW     = h5read(fn, "/LW");
primLS = h5read(fn, "/primLS");
NdVec  = h5read(fn, "/NdVec");

set(groot, ...
    'defaultTextInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...
    'defaultLegendInterpreter', 'latex');

cmap = turbo(256);

n = 256;
r = [linspace(0,1,n/2), ones(1,n/2)];
g = [linspace(0,1,n/2), linspace(1,0,n/2)];
b = [ones(1,n/2), linspace(1,0,n/2)];
rbmap = [r(:), g(:), b(:)];

LS1 = LS(:,:,1,1);
LS1_abs_max = max(abs(LS1), [], 'all');


%% Density LF

rhoMin = min(prim(:,:,1),[],'all');
rhoMax = max(prim(:,:,1),[],'all');

fig1 = figure(1); clf;
set(gcf,'Color','w');

tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

for ndx = 1:3
    nexttile(ndx);
    pcolor(x,y,LF(:,:,ndx,1));
    shading interp
    axis equal tight
    clim([rhoMin,rhoMax]);
    colormap(gca,cmap);
    colorbar;
    title(sprintf('$\\rho_%d$',ndx),'FontSize',13);
    xlabel('$x$');
    ylabel('$y$');
    set(gca,'FontSize',11,'LineWidth',1);
end

nexttile(4);
contourf(x,y,LS1,40,'LineStyle','none');
hold on;
contour(x,y,LS1,[0 0],'k','LineWidth',2);
hold off;

axis equal tight
colormap(gca,rbmap);
clim(gca,[-LS1_abs_max LS1_abs_max]);
colorbar;
title(sprintf('$\\varphi_{\\rho,1}$'),'FontSize',13);
xlabel('$x$');
ylabel('$y$');
set(gca,'FontSize',11,'LineWidth',1);

width = 6.5;
ar = 0.85;
height = width * ar;
set(gcf,'Units','inches','Position',[0.1 0.1 width height])


%% Velocity and pressure LF

varNames = ["u","v","p"];

fig2 = figure(2); clf;
set(gcf,'Color','w','Position',[100 100 950 1050]);

for vdx = 2:4
    vMin = min(prim(:,:,vdx),[],'all');
    vMax = max(prim(:,:,vdx),[],'all');
    for ndx = 1:2
        nexttile;
        pcolor(x,y,LF(:,:,ndx,vdx));
        shading interp
        axis equal tight
        clim([vMin, vMax]);
        colormap(gca,cmap);
        colorbar;
        title(sprintf('$%s\\;LF_%d$',varNames(vdx-1),ndx),'FontSize',13);
        xlabel('$x$');
        ylabel('$y$');
        set(gca,'FontSize',11,'LineWidth',1);
    end
end

width = 6.5;
ar = 1.3;
height = width * ar;
set(gcf,'Units','inches','Position',[0.1 0.1 width height])


%% Reconstructed primitive variables

varNames = ["\rho","u","v","p"];

fig3 = figure(3); clf;
set(gcf,'Color','w','Position',[100 100 1000 820]);
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

for vdx = 1:4
    vMin = min(prim(:,:,vdx),[],'all');
    vMax = max(prim(:,:,vdx),[],'all');

    nexttile;
    pcolor(x,y,primLS(:,:,vdx));
    shading interp
    axis equal tight
    clim([vMin, vMax]);
    colormap(gca,cmap);
    colorbar;
    title(sprintf('$%s$',varNames(vdx)),'FontSize',13);
    xlabel('$x$');
    ylabel('$y$');
    set(gca,'FontSize',11,'LineWidth',1);
end

width = 6.5;
ar = 0.85;
height = width * ar;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height])


%% Level set contours

figure(4); clf;
set(gcf,'Color','w','Position',[100 100 700 620]);

contourf(x,y,LS1,40,'LineStyle','none');
hold on;
contour(x,y,LS1,[0 0], 'k','LineWidth',2);
hold off;

axis equal tight
colormap(gca,rbmap);
clim(gca,[-LS1_abs_max LS1_abs_max]);
colorbar;
xlabel('$x$');
ylabel('$y$');
set(gca,'FontSize',12,'LineWidth',1);

width = 6.5;
ar = 0.75;
height = width * ar;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height])
