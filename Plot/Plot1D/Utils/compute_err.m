% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function [prim_rmse,prim_spread] = compute_err(data)
% Compute ensemble RMSE and spread at DA cycles.
%
% Input:
% - data: structure returned by read_h5 with fields:
%   - x: Nx vector of ensemble-grid coordinates
%   - x_t: Nx_t vector of truth-grid coordinates
%   - Nx: number of ensemble-grid points
%   - prim: array of Nx-by-Nt-by-Ne ensemble fields
%   - prim_truth: array of Nx_t-by-Nt truth fields
%   - e: Nt logical vector marking assimilation cycles
%
% Output:
% - prim_rmse: 3-by-Nc array of spatial RMSE values for density, velocity,
%   and pressure at the initial state and Nc-1 assimilation cycles
% - prim_spread: 3-by-Nc array of spatially averaged ensemble spreads at
%   the same cycles
%

x = data.x;
x_t = data.x_t;
e = data.e;
Nx = data.Nx;
prim = data.prim;
prim_t = data.prim_truth;

cdx_DA = find(e);
cdx_DA = [cdx_DA(1)-1;cdx_DA(:)];
prim_rmse = zeros(3,length(cdx_DA));
prim_spread = zeros(3,length(cdx_DA));
for cdx = 1:length(cdx_DA)

    for vdx = 1:3

        % get variables
        A = squeeze(prim{vdx}(:,cdx_DA(cdx),:));
        thisTruth = interp1(x_t, prim_t{vdx}(:,cdx_DA(cdx)), x, 'linear', 'extrap');

        % compute RMSE
        thisMean = reshape(mean(A,2),[Nx,1]);
        thisTruth = reshape(thisTruth,[Nx,1]);
        prim_rmse(vdx,cdx) = sqrt(mean((thisMean - thisTruth).^2));

        % compute spread
        prim_spread(vdx,cdx) = sqrt(mean(var(A,0,2)));
    end

end

end
