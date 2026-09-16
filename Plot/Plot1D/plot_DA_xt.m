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
x = data.x;
x_t = data.x_t;
t = data.t;
xObs = data.xObs;
yMat = data.yMat;
e = data.e;
Nx = data.Nx;
Nx_t = data.Nx_t;
Nt = data.Nt;
Nv = 3;
prim = data.prim;
prim_truth = data.prim_truth;
Ne = data.Ne;


%%

prim0e = zeros(Nv,Nx,Ne);
prim0t = zeros(Nv,Nx);
for vdx = 1:Nv
    prim0e(vdx,:,:) = reshape(prim{vdx}(:,1,:), [1,Nx,Ne]);
    prim0t(vdx,:) = reshape(interp1(x_t, prim_truth{vdx}(:,1), x, 'linear', 'extrap'), [1,Nx]);
end
err = zeros(Ne,1);
for edx = 1:Ne
    d = prim0e(:,:,edx) - prim0t;
    err(edx) = norm(d(:),2);
end
[~,edxFar] = max(err);

prim_a_plot_far = zeros(Nv,Nx,Nt);
prim_a_plot_mean = zeros(Nv,Nx,Nt);
prim_t_plot = zeros(Nv,Nx_t,Nt);
for tdx = 1:Nt
    for vdx = 1:Nv
        prim_a_plot_far(vdx,:,tdx) = reshape(prim{vdx}(:,tdx,edxFar), [1,Nx]);
        prim_a_plot_mean(vdx,:,tdx) = reshape(mean(squeeze(prim{vdx}(:,tdx,:)),2), [1,Nx]);
        prim_t_plot(vdx,:,tdx) = reshape(prim_truth{vdx}(:,tdx), [1,Nx_t]);
    end
end

tObs = t(e);

%%

fig1 = plot_xt_all(x, x_t, t, prim_a_plot_far, prim_a_plot_mean, prim_t_plot, xObs, tObs);
width = 6.5; ar = 0.6;
height = width * ar;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height])


%%

function fig = plot_xt_all(x, xt, t, prim_far, prim_mean, prim_truth, xObs, tObs)

vMin = zeros(3,1);
vMax = zeros(3,1);
for vdx = 1:3
    [vMin(vdx),vMax(vdx)] = computeMinMax(prim_far,prim_mean,prim_truth,vdx);
end

vars = ["Density, $\rho$", "Velocity, $u$", "Pressure, $p$"];
rows = ["Reference Truth", "Ensemble Mean", "Farthest Member"];

fig = figure; clf;
tiledlayout(3,3,'TileSpacing','compact','Padding','compact');

for rdx = 1:3
    switch rdx
        case 1
            thisX = xt;
            thisPrim = prim_truth;
        case 2
            thisX = x;
            thisPrim = prim_mean;
        case 3
            thisX = x;
            thisPrim = prim_far;
    end

    [X,T] = meshgrid(thisX,t);

    for vdx = 1:3
        ax = nexttile;
        pcolor(ax, X, T, squeeze(thisPrim(vdx,:,:))');
        shading(ax,'interp');
        if rdx == 1
            colorbar(ax);
        end
        clim(ax, [vMin(vdx),vMax(vdx)])

        if rdx == 3
            xlabel(ax,"$x$",'Interpreter','latex');
        else
            xticklabels(ax,[]);
        end
        if vdx == 1
            ylabel(ax, rows(rdx) + newline + "$t$", ...
                'Interpreter','latex', ...
                'Rotation',90);
        else
            yticklabels(ax,[]);
        end

        if rdx == 1
            title(ax, vars(vdx),'Interpreter','latex');
        end

        if vdx == 3
            xline(xObs,'k--','LineWidth',1.2);
            hold(ax,'on');
            %scatter(ax, repmat(reshape(xObs,[1,numel(xObs)]),[numel(tObs),1]), tObs, 10, 'k', 'filled');
        end

        set(ax,'TickLabelInterpreter','latex','FontSize',8);
    end
end

end


function [varMin,varMax] = computeMinMax(prim_far,prim_mean,prim_truth,vdx)

qFar = squeeze(prim_far(vdx,:,:));
qMean = squeeze(prim_mean(vdx,:,:));
qTruth = squeeze(prim_truth(vdx,:,:));
varMaxFar   = max(qFar(:));
varMaxMean  = max(qMean(:));
varMaxTruth = max(qTruth(:));
varMinFar   = min(qFar(:));
varMinMean  = min(qMean(:));
varMinTruth = min(qTruth(:));
varMax = max([varMaxFar,varMaxMean,varMaxTruth]);
varMin = min([varMinFar,varMinMean,varMinTruth]);

end
