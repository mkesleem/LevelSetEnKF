% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function data = read_LS_h5(filename)
% Read and orient a one-dimensional level-set solution from an HDF5 file.
%
% Input:
% - filename: path or filename of the HDF5 file
%
% Output:
% - data: structure with fields:
%   - x: Nx vector of spatial coordinates
%   - t: Nt vector of time values
%   - prim: Nv-by-Nx array of primitive variables
%   - LF: Nx-by-(Nd+1)-by-Nv array of level-set extensions
%   - LS: Nx-by-Nd-by-Nv array of level-set functions
%   - LW: event-width array with variables along the first dimension
%   - NdVec: number of events for each variable
%

% read data
x  = h5read(filename, '/x');
t  = h5read(filename, '/t');
LF = h5read(filename, '/LF');
LS = h5read(filename, '/LS');
LW = h5read(filename, '/LW');
prim = h5read(filename,'/prim');
NdVec = double(h5read(filename, '/NdVec'));

% reshape data
Nx = numel(x);
Nv = 3;
assert(size(prim,1)==Nx);
assert(size(prim,2)==Nv);
LF_ldx_x = size(LF)==Nx;
LS_ldx_x = size(LS)==Nx;
assert(all(LF_ldx_x==LS_ldx_x))
ldx_x = LF_ldx_x;
assert(sum(ldx_x)==1);
LF_ldx_v = size(LF)==Nv;
LS_ldx_v = size(LS)==Nv;
ldx_v = (LF_ldx_v == 1) & (LS_ldx_v == 1);
assert(sum(ldx_v)==1);
ldx_n = ~(ldx_v+ldx_x);
assert(sum(ldx_n+ldx_v+ldx_x)==3);
assert(all((ldx_n+ldx_v+ldx_x)==ones(1,3)));
idx_x = find(ldx_x);
idx_v = find(ldx_v);
idx_n = find(ldx_n);
[~,isort] = sort([idx_x,idx_n,idx_v]);
LF_sort = permute(LF,isort);
LS_sort = permute(LS,isort);
LW_sort = permute(LW,[2,1]);
prim_sort = permute(prim,[2,1]);
data = struct('x',x,'t',t,'prim',prim_sort,...
    'LF',LF_sort,'LS',LS_sort,'LW',LW_sort,'NdVec',NdVec);
end
