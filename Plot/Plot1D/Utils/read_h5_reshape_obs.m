% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function q = read_h5_reshape_obs(q,Nobs)
% Orient a two-dimensional observation array by observation and time.
%
% Input:
% - q: Nobs-by-Nt or Nt-by-Nobs observation array
% - Nobs: number of observation locations
%
% Output:
% - q: Nobs-by-Nt observation array
%

sz = size(q);
if numel(sz) ~= 2
    error("Input array has wrong number of dimensions");
end
if sz(1) == Nobs
    return;
elseif sz(2) == Nobs
    q = transpose(q);
else
    error('Cannot orient observation array');
end
end
