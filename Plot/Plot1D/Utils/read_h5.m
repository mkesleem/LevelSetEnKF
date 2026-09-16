% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function data = read_h5(filename)
% Read and orient one-dimensional ensemble, truth, and observation data.
%
% Input:
% - filename: path or filename of the HDF5 file
%
% Output:
% - data: structure with fields:
%   - filename: source HDF5 filename
%   - x: Nx-by-1 ensemble-grid coordinates
%   - Nx: number of ensemble-grid points
%   - x_t: Nx_t-by-1 truth-grid coordinates
%   - Nx_t: number of truth-grid points
%   - t: Nt-by-1 time values
%   - Nt: number of time values
%   - xObs: Nobs-by-1 observation coordinates
%   - prim: three-cell array of Nx-by-Nt-by-Ne ensemble fields ordered as
%     density, velocity, and pressure
%   - prim_truth: three-cell array of Nx_t-by-Nt truth fields in the same order
%   - Ne: ensemble size
%   - yMat: Nobs-by-Nt observation values
%   - sMat: Nobs-by-Nt observation-error standard deviations
%   - e: Nt-by-1 logical array marking data-assimilation cycles
%

% Read data
rho    = h5read(filename, '/rho');
u      = h5read(filename, '/u');
p      = h5read(filename, '/p');
rho_t  = h5read(filename, '/rho_t');
u_t    = h5read(filename, '/u_t');
p_t    = h5read(filename, '/p_t');
t      = h5read(filename, '/t');
x      = h5read(filename, '/x');
x_t    = h5read(filename, '/x_t');
xObs   = h5read(filename, '/xObs');
yMat   = h5read(filename, '/yMat');
sMat   = h5read(filename, '/sMat');
e_cell = h5read(filename, '/enkfBool');

% convert e_cell to logical
e = strcmpi(e_cell(:), 'true');

% get sizes
Nx    = numel(x);
Nx_t  = numel(x_t);
Nt    = numel(t);
NobsX = numel(xObs);

% reshape arrays
rho   = read_h5_reshape_ensemble(rho,  Nx  ,Nt);
u     = read_h5_reshape_ensemble(u  ,  Nx  ,Nt);
p     = read_h5_reshape_ensemble(p  ,  Nx  ,Nt);
rho_t = read_h5_reshape_truth(   rho_t,Nx_t,Nt);
u_t   = read_h5_reshape_truth(   u_t  ,Nx_t,Nt);
p_t   = read_h5_reshape_truth(   p_t  ,Nx_t,Nt);
yMat  = read_h5_reshape_obs(yMat,NobsX);
sMat  = read_h5_reshape_obs(sMat,NobsX);

% assemble solution
prim       = {rho  ,u  ,p  };
prim_truth = {rho_t,u_t,p_t};
Ne         = size(rho,3);

% create data structure
data.filename   = filename;
data.x          = reshape(x,[Nx,1]);
data.Nx         = Nx;
data.x_t        = reshape(x_t,[Nx_t,1]);
data.Nx_t       = Nx_t;
data.t          = reshape(t,[Nt,1]);
data.Nt         = Nt;
data.xObs       = reshape(xObs,[NobsX,1]);
data.prim       = prim;
data.prim_truth = prim_truth;
data.Ne         = Ne;
data.yMat       = yMat;
data.sMat       = sMat;
data.e          = e;

end
