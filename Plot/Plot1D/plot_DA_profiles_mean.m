% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT
clear; close all;
addpath("Utils/")


%%

if exist('brewermap','file')
    Colors = brewermap(8,'Dark2');
else
    Colors = lines(8);
end

% get filename
filename = '../../data_toro.h5';
%filename = '../../data_shuOsher.h5';
%filename = '../../data_sod.h5';

% read data
data = read_h5(filename);
x = data.x;
x_t = data.x_t;
t = data.t;
xObs = data.xObs;
yMat = data.yMat;
sMat = data.sMat;
e = data.e;
rho = data.prim{1};
u = data.prim{2};
p = data.prim{3};
rho_t = data.prim_truth{1};
u_t = data.prim_truth{2};
p_t = data.prim_truth{3};
Ne = data.Ne;

% Mean profiles are stored for the initial time, every DA time, and the
% final time.  Convert the zero-based Python time indices to MATLAB indices.
mean_t_idx = double(h5read(filename,'/mean_t_idx')) + 1;
prim_a_mean = orient_mean(h5read(filename,'/prim_a_mean'),numel(x),numel(mean_t_idx));

% get plotting
Nx = numel(x);
Nx_plot = Nx;
x_plot = linspace(min(x),max(x),Nx_plot);
Nt = numel(t); assert(numel(e)==Nt);

tdx_DA = find(e);
Nda_plot = min(3, numel(tdx_DA));
tdx_da_selected = zeros(Nda_plot, 1);
tdx_da_available = tdx_DA(:);
tdx_da_targets = round(linspace(1, Nt, Nda_plot + 2));
for k = 1:Nda_plot
    [~, idx] = min(abs(tdx_da_available - tdx_da_targets(k + 1)));
    tdx_da_selected(k) = tdx_da_available(idx);
    tdx_da_available(idx) = [];
end
tdx_plot = unique([1; tdx_da_selected; Nt]);
t_plot = t(tdx_plot);
Nt_plot = numel(t_plot);

rho_ens = cell(Nt_plot,1);
u_ens = cell(Nt_plot,1);
p_ens = cell(Nt_plot,1);
rho_ref = cell(Nt_plot,1);
u_ref = cell(Nt_plot,1);
p_ref = cell(Nt_plot,1);
obs_plot = cell(Nt_plot,1);
prim_a_mean_plot = cell(Nt_plot,3);

for k = 1:Nt_plot
    tdx = tdx_plot(k);

    % ensembles
    rho_ens{k} = reshape(interp1(x, squeeze(rho(:,tdx,:)), x_plot, 'linear')',[Ne,Nx_plot]);
    u_ens{k}   = reshape(interp1(x, squeeze(u(:,tdx,:)),   x_plot, 'linear')',[Ne,Nx_plot]);
    p_ens{k}   = reshape(interp1(x, squeeze(p(:,tdx,:)),   x_plot, 'linear')',[Ne,Nx_plot]);

    % reference truth
    rho_ref{k} = interp1(x_t, rho_t(:,tdx), x_plot, 'linear', 'extrap')';
    u_ref{k}   = interp1(x_t, u_t(:,tdx),   x_plot, 'linear', 'extrap')';
    p_ref{k}   = interp1(x_t, p_t(:,tdx),   x_plot, 'linear', 'extrap')';

    % get data to plot
    if e(tdx)
        thisTdx = sum(e(1:tdx));
        obs_plot_struct.y = yMat(:,thisTdx);
        obs_plot_struct.s = sMat(:,thisTdx);
    else
        obs_plot_struct.y = [];
        obs_plot_struct.s = [];
    end
    obs_plot{k} = obs_plot_struct;

    % means (available at initial, DA, and final times)
    mean_idx = find(mean_t_idx == tdx,1);
    if ~isempty(mean_idx)
        for vdx = 1:3
            prim_a_mean_plot{k,vdx} = interp1( ...
                x,prim_a_mean(vdx,:,mean_idx),x_plot,'linear');
        end
    end
end

% get extreme values
rho_max = max([max(rho,[],'all'),max(rho_t,[],'all')]);
u_max   = max([max(u  ,[],'all'),max(u_t  ,[],'all')]);
p_max   = max([max(p  ,[],'all'),max(p_t  ,[],'all'),1.01*max(yMat,[],'all')]);
rho_min = max([min(rho,[],'all'),min(rho_t,[],'all')]);
u_min   = max([min(u  ,[],'all'),min(u_t  ,[],'all')]);
p_min   = max([min(p  ,[],'all'),min(p_t  ,[],'all'),min(yMat,[],'all')]);
if rho_min >= -1e-10
    rho_min = 0;
else
    error("Negative density");
end
if p_min >= -1e-10
    p_min = 0;
else
    error("Negative pressure");
end
if u_min >= -1e-10
    u_min = -0.1*u_max;
else
    u_min = 1.1*u_min;
end

% get limits
rho_lim = [rho_min,1.1*rho_max];
u_lim   = [u_min,1.1*u_max  ];
p_lim   = [p_min,1.1*p_max  ];

fig1 = figure(1); clf;
for k = 1:Nt_plot

    % -------- Density --------
    subplot(Nt_plot,3,3*(k-1)+1); hold on;
    for n = 1:Ne
        ha = plot(x_plot,rho_ens{k}(n,:),'Color',Colors(1,:),'LineWidth',0.8);
    end
    ht = plot(x_plot,rho_ref{k},'k','LineWidth',2.5);
    if ~isempty(prim_a_mean_plot{k,1})
        ham = plot(x_plot,prim_a_mean_plot{k,1},'--', ...
            'Color',Colors(2,:),'LineWidth',2.0);
    end

    ylim(rho_lim); xlim([min(x_plot) max(x_plot)]);
    if k==1, title('Density, $\rho$','Interpreter','latex','FontSize',10); end
    ylabel(sprintf("$t=%.4f$",t_plot(k)),'Interpreter','latex','FontSize',10);
    if k==Nt_plot, xlabel('$x$','Interpreter','latex'); end
    grid on; box on;
    set(gca,'TickLabelInterpreter','latex','FontSize',10);

    if k==1
        if exist('ham','var')
            legend([ha(1),ht(1),ham], ...
                ["Analysis ensemble","Reference","Analysis mean"], ...
                'Interpreter','latex')
        else
            legend([ha(1),ht(1)],["Analysis ensemble","Reference"], ...
                'Interpreter','latex')
        end
    end

    % -------- Velocity --------
    subplot(Nt_plot,3,3*(k-1)+2); hold on;
    for n = 1:Ne
        plot(x_plot,u_ens{k}(n,:),'Color',Colors(1,:),'LineWidth',0.8);
    end
    plot(x_plot,u_ref{k},'k','LineWidth',2.5);
    if ~isempty(prim_a_mean_plot{k,2})
        plot(x_plot,prim_a_mean_plot{k,2},'--', ...
            'Color',Colors(2,:),'LineWidth',2.0);
    end

    ylim(u_lim); xlim([min(x_plot) max(x_plot)]);
    if k==1, title('Velocity, $u$','Interpreter','latex','FontSize',10); end
    if k==Nt_plot, xlabel('$x$','Interpreter','latex'); end
    grid on; box on;
    set(gca,'TickLabelInterpreter','latex','FontSize',10);

    % -------- Pressure --------
    subplot(Nt_plot,3,3*(k-1)+3); hold on;
    for n = 1:Ne
        plot(x_plot,p_ens{k}(n,:),'Color',Colors(1,:),'LineWidth',0.8);
    end
    plot(x_plot,p_ref{k},'k','LineWidth',2.5);
    if ~isempty(prim_a_mean_plot{k,3})
        plot(x_plot,prim_a_mean_plot{k,3},'--', ...
            'Color',Colors(2,:),'LineWidth',2.0);
    end

    % Observations
    thisObs = obs_plot{k};
    if ~isempty(thisObs.y)
        ho = errorbar(xObs,thisObs.y,thisObs.s,'.','MarkerSize',10,'Color',Colors(6,:));
    elseif k == 1
        ho = errorbar(xObs,NaN(size(xObs)),NaN(size(xObs)),'.','MarkerSize',10,'Color',Colors(6,:));
    end
    if k == 1 && exist('ho','var')
        legend(ho,"Observations",'Interpreter','latex')
    end

    ylim(p_lim); xlim([min(x_plot) max(x_plot)]);
    if k==1, title('Pressure, $p$','Interpreter','latex','FontSize',10); end
    if k==Nt_plot, xlabel('$x$','Interpreter','latex'); end
    grid on; box on;
    set(gca,'TickLabelInterpreter','latex','FontSize',10);

end

width = 8; ar = 1.35;
height = width * ar;
set(gcf, 'Units', 'inches', 'Position', [0.1 0.1 width height])


function q = orient_mean(q,Nx,Nmean)
%ORIENT_MEAN Return a saved mean field as [variable,x,mean time].
sz = size(q);
if numel(sz) ~= 3
    error('Expected prim mean data to be three-dimensional.');
end

idx_var = find(sz == 3,1);
idx_x = find(sz == Nx,1);
idx_mean = find(sz == Nmean,1);
if isempty(idx_var) || isempty(idx_x) || isempty(idx_mean) || ...
        numel(unique([idx_var,idx_x,idx_mean])) ~= 3
    error('Cannot orient prim mean data of size [%s].',num2str(sz));
end
q = permute(q,[idx_var,idx_x,idx_mean]);
end
