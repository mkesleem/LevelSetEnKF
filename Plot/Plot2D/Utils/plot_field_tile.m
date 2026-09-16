% Copyright (c) 2026 Michael Sleeman
% SPDX-License-Identifier: MIT


function plot_field_tile(x, y, q, q_lim)
% Plot one two-dimensional field in the next tile.
%
% Input:
% - x: Ny-by-Nx array of x-coordinates
% - y: Ny-by-Nx array of y-coordinates
% - q: Ny-by-Nx field values
% - q_lim: [minimum, maximum] color limits
%
% Output:
% - None
%
% The tile uses interpolated shading, a square [0, 2]-by-[0, 2] domain,
% and the turbo colormap.
%

ax = nexttile;
pcolor(ax, x, y, q);
shading(ax, "interp");
axis(ax, "square");
xlim(ax, [0 2]);
ylim(ax, [0 2]);
clim(ax, q_lim);
colormap(ax, turbo);
ax.XTick = [0 1 2];
ax.YTick = [0 1 2];


end

