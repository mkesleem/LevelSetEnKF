% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function q = read_h5_reshape_ensemble(q,Nx,Nt)
% Orient a three-dimensional ensemble field by space, time, and member.
%
% Input:
% - q: three-dimensional field containing dimensions Nx, Nt, and Ne in an
%   unknown order
% - Nx: number of spatial grid points
% - Nt: number of time values
%
% Output:
% - q: Nx-by-Nt-by-Ne ensemble field
%
% The three dimension lengths must be unique so that their roles can be
% inferred unambiguously.
%

% verify number of dimensions
sz = size(q);
if numel(sz) ~= 3
    error('Expected ensemble array to be 3D');
end
if sum(sum(sz(:)==sz(:)')) ~= 3
    error("Size of each dimension is not unique");
end

% get indices corresponding to x and t
idxNx = find(sz == Nx, 1);
idxNt = find(sz == Nt, 1);
if isempty(idxNx) || isempty(idxNt) || idxNx == idxNt
    error('Cannot orient ensemble array');
end

% permute
idxRest = setdiff(1:3, [idxNx idxNt]);
q = permute(q, [idxNx idxNt idxRest]);

end
