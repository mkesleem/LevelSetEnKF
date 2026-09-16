# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
import align2d


def ls_center(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Fit a circle to the zero level set of phi using least-squares

    Input:
    - phi: Ny by Nx level set function
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates

    Output:
    - cc: fitted circle center [cx, cy]
    - r: fitted radius, or NaN if a circle cannot be fitted
    """

    # get grid spacing
    dx = abs(x[0, 1] - x[0, 0])
    dy = abs(y[1, 0] - y[0, 0])

    # get points within a band of three grid points about the zero level set
    band = 3.0 * max(dx, dy)
    mask = np.isfinite(phi) & (np.abs(phi) <= band)

    # if no zero level set, return point where abs(phi) in minimized
    if not np.any(mask):
        idx = np.nanargmin(np.abs(phi))
        cc = np.asarray([x.ravel()[idx],y.ravel()[idx]])
        return cc,np.nan

    # get band
    xb = x[mask].ravel()
    yb = y[mask].ravel()

    # if more than three points, compute center
    if xb.size >= 3:
        A = np.column_stack([xb, yb, np.ones_like(xb)])
        rhs = -(xb**2 + yb**2)
        d, e, f = np.linalg.lstsq(A, rhs, rcond=None)[0]
        cx = -d/2
        cy = -e/2
        r  = np.sqrt((d**2+e**2)/4-f)
        cc = np.asarray([cx,cy])
        return cc,r
    else:
        idx = np.nanargmin(np.abs(phi))
        cc = np.asarray([x.ravel()[idx],y.ravel()[idx]])
        return cc,np.nan


def warp_radial_2d(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cc: np.ndarray,
    cf: np.ndarray,
) -> np.ndarray:
    """
    Recenter q from cf to cc (no radial scaling).

    Input:
    - q: Ny by Nx level set function to warp
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - cc: center of reference circle
    - cf: center of forecast circle

    Output:
    - q_align: level set function centered at cc
    """
    q_tmp = q[None,:,:]
    q_align = align2d.warp_radial_2d(q_tmp,1.0,0.0,x,y,cc,cf)
    return q_align[0]


def inverse_warp_radial_2d(
    q_align: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cc: np.ndarray,
    cf: np.ndarray,
) -> np.ndarray:
    """
    Undo the recentering from cf to cc (no radial scaling).

    Input:
    - q_align: Ny by Nx recentered level set function
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - cc: center of reference circle
    - cf: center of forecast circle

    Output:
    - q: level set function restored to center cf
    """
    q_align_tmp= q_align[None,:,:]
    q = align2d.inverse_warp_radial_2d(q_align_tmp,1.0,0.0,x,y,cc,cf)
    return q[0]
