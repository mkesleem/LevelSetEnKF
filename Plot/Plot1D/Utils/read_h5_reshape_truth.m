% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function q = read_h5_reshape_truth(q,Nx,Nt)
% Orient a two-dimensional truth field by space and time.
%
% Input:
% - q: Nx-by-Nt or Nt-by-Nx truth field
% - Nx: number of truth-grid points
% - Nt: number of time values
%
% Output:
% - q: Nx-by-Nt truth field
%

% verify size
sz = size(q);
if numel(sz) ~= 2
    error("Truth data has wrong number of dimensions");
end

% reshape
if sz(1) == Nx && sz(2) == Nt
    return;
elseif sz(2) == Nx && sz(1) == Nt
    q = transpose(q);
else
    error('Cannot orient truth array');
end

end
