# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import levelSet2d


def align_affine_profile_2d(
    qc: np.ndarray,
    qf: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cc: np.ndarray,
    cf: np.ndarray,
    rc: float,
    rf: float,
) -> tuple[float, float, np.ndarray]:
    """
    Radially align qf to qc using density gradient.

    Input:
    - qc: reference profile
    - qf: forecast profile to align
    - x: x-coordinates
    - y: y-coordinates
    - cc: reference center [cx, cy]
    - cf: forecast center [cx, cy]
    - rc: reference radius
    - rf: forecast radius

    Output:
    - a: optimized radial scaling factor
    - b: optimized radial translation
    - qf_align: forecast profile aligned with the reference profile
    """

    # get shape
    Nv, Ny, Nx = qc.shape
    assert(x.shape[0]==Ny)
    assert(x.shape[1]==Nx)
    assert(y.shape[0]==Ny)
    assert(y.shape[1]==Nx)
    assert(qf.shape[0]==Nv)
    assert(qf.shape[1]==Ny)
    assert(qf.shape[2]==Nx)

    # compute shock sensor
    s1c0,_ = levelSet2d.computeShockSensor(qc[0],x,y)
    s1f0,_ = levelSet2d.computeShockSensor(qf[0],x,y)
    s1c = np.sqrt(s1c0[:,:,0,0]**2+s1c0[:,:,1,0]**2)[None,:,:]
    s1f = np.sqrt(s1f0[:,:,0,0]**2+s1f0[:,:,1,0]**2)[None,:,:]

    s1c_max = np.max(np.abs(s1c))
    s1f_max = np.max(np.abs(s1f))

    s_tol = 1e-14
    if (not np.isfinite(s1c_max)
        or not np.isfinite(s1f_max)
        or s1c_max <= s_tol
        or s1f_max <= s_tol
    ):
        raise ValueError(
            "Cannot align profiles because at least one shock sensor is zero: "
            f"reference max={s1c_max:.3e}, forecast max={s1f_max:.3e}"
        )
    s1c = s1c / s1c_max
    s1f = s1f / s1f_max

    # get initial guess
    theta0 = np.array([1.0,rc-rf])
    lb = np.array([0.10,-2])
    ub = np.array([10.0, 2])

    sol = least_squares(
        residual_radial_2d,
        theta0,
        bounds=(lb, ub),
        args=(s1c,s1f,x,y,cc,cf),
        max_nfev=50,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    a, b = sol.x
    qf_align = warp_radial_2d(qf,a,b,x,y,cc,cf)

    return a,b,qf_align


def residual_radial_2d(
    theta: np.ndarray,
    qc_sensor: np.ndarray,
    qf_sensor: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cc: np.ndarray,
    cf: np.ndarray,
) -> np.ndarray:
    """
    Compute the radial-alignment residual between two shock sensors.

    Input:
    - theta: radial alignment parameters [a, b]
    - qc_sensor: reference shock-sensor field
    - qf_sensor: forecast shock-sensor field to align
    - x: x-coordinates
    - y: y-coordinates
    - cc: reference center [cx, cy]
    - cf: forecast center [cx, cy]

    Output:
    - residual: qf_align - qc_sensor
    """
    a,b = theta
    qf_align = warp_radial_2d(qf_sensor,a,b,x,y,cc,cf)
    return (qf_align - qc_sensor).ravel()


def warp_radial_2d(
    q: np.ndarray,
    a: float,
    b: float,
    x: np.ndarray,
    y: np.ndarray,
    cc: np.ndarray,
    cf: np.ndarray,
) -> np.ndarray:
    """
    1) Recenter q from center cf to center cc.
    2) Radially warp q according to [a,b].

    Input:
    - q: profile to warp
    - a: radial scaling factor
    - b: radial translation
    - x: x-coordinates
    - y: y-coordinates
    - cc: reference center [cx, cy]
    - cf: forecast center [cx, cy]

    Output:
    - q_align: radially aligned profile
    """
    dx = x - cc[0]
    dy = y - cc[1]
    r = np.hypot(dx, dy)
    r_safe = np.maximum(r, 1e-14)
    r_old = (r - b) / a
    x_old = cf[0] + r_old * dx / r_safe
    y_old = cf[1] + r_old * dy / r_safe
    q_align = np.zeros_like(q)
    pts = np.column_stack([y_old.ravel(),x_old.ravel()])
    Nv = q.shape[0]
    Ny = q.shape[1]
    Nx = q.shape[2]
    for v in range(Nv):
        interp = RegularGridInterpolator(
            (y[:,0],x[0,:]),
            q[v],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        q_align[v] = interp(pts).reshape(Ny, Nx)
    return q_align


def inverse_warp_radial_2d(
    q_align: np.ndarray,
    a: float,
    b: float,
    x: np.ndarray,
    y: np.ndarray,
    cc: np.ndarray,
    cf: np.ndarray,
) -> np.ndarray:
    """
    1) Invert radial warp according to parameters [a,b].
    2) Recenter the new profile at cf instead of cc.

    Input:
    - q_align: radially aligned profile
    - a: radial scaling factor
    - b: radial translation
    - x: x-coordinates
    - y: y-coordinates
    - cc: reference center [cx, cy]
    - cf: forecast center [cx, cy]

    Output:
    - q: profile with warp inverted
    """
    dx = x - cf[0]
    dy = y - cf[1]
    r = np.hypot(dx, dy)
    r_safe = np.maximum(r, 1e-14)
    r_new = a * r + b
    x_new = cc[0] + r_new * dx / r_safe
    y_new = cc[1] + r_new * dy / r_safe
    q = np.zeros_like(q_align)
    Nv = q_align.shape[0]
    Ny = q_align.shape[1]
    Nx = q_align.shape[2]
    pts = np.column_stack([y_new.ravel(),x_new.ravel()])
    for v in range(Nv):
        interp = RegularGridInterpolator(
            (y[:,0],x[0,:]),
            q_align[v],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        q[v] = interp(pts).reshape(Ny, Nx)
    return q
