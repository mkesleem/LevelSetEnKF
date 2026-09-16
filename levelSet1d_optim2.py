# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.optimize import least_squares


def model(
    theta: np.ndarray,
    x: np.ndarray,
    w_fixed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Evaluate a two-event, three-state level set reconstruction.

    hL = (tanh((x - xsL)/wL) + 1)/2
    hR = (tanh((x - xsR)/wR) + 1)/2
    u = (1 - hL)(1 - hR)uL + hL(1 - hR)uM + hL*hR*uR

    Input:
    - theta: parameters [uL, uM, uR, xsL, xsR, wL, wR], or without
      [wL, wR] if the widths are fixed
    - x: N spatial coordinates
    - w_fixed: two prescribed event widths, or None to obtain them from theta

    Output:
    - u: reconstruction
    """
    N = x.size
    uL = theta[:N]
    uM = theta[N:2*N]
    uR = theta[2*N:3*N]
    xsL = theta[3*N+0]
    xsR = theta[3*N+1]
    if w_fixed is None:
        wL  = theta[3*N+2]
        wR  = theta[3*N+3]
    else:
        wL = w_fixed[0]
        wR = w_fixed[1]

    heL = (np.tanh((x - xsL) / wL) + 1.0) / 2.0
    heR = (np.tanh((x - xsR) / wR) + 1.0) / 2.0
    tanh_member = np.column_stack(
        [(1.0 - heL) * (1.0 - heR),heL * (1.0 - heR),heL * heR,]
    )
    return np.sum(tanh_member * np.column_stack([uL, uM, uR]), axis=1)


def residual(
    theta: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    lam1: float,
    lamB: float,
    w_fixed: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assemble residuals for the regularized two-event level set fit.

    Input:
    - theta: parameters [uL, uM, uR, xsL, xsR, wL, wR], or without
      [wL, wR] if the widths are fixed
    - x: N spatial coordinates
    - u: N target profile values
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - w_fixed: two prescribed event widths, or None to obtain them from theta

    Output:
    - residual: concatenated fit, smoothness, and boundary residuals
    """
    N = x.size
    uL = theta[:N]
    uM = theta[N:2*N]
    uR = theta[2*N:3*N]

    # model mismatch
    r_data = model(theta, x, w_fixed) - u

    # smoothness
    r_smooth_L1 = lam1 * np.diff(uL, 1)
    r_smooth_M1 = lam1 * np.diff(uM, 1)
    r_smooth_R1 = lam1 * np.diff(uR, 1)

    r_bound_L = lamB * ( uL[ 0] - u[ 0] )
    r_bound_R = lamB * ( uR[-1] - u[-1] )
    
    return np.concatenate(
        [r_data,r_smooth_L1,r_smooth_M1,r_smooth_R1,[r_bound_L],[r_bound_R]]
    )


def fit_tanh(
    x: np.ndarray,
    u: np.ndarray,
    uL0: np.ndarray,
    uM0: np.ndarray,
    uR0: np.ndarray,
    xsL0: float,
    xsR0: float,
    wL0: float,
    wR0: float,
    lam1: float = 1e2,
    lamB: float = 1e2,
    LW_fixed: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    """
    Fit a regularized two-event level set representation to u.

    Input:
    - x: N spatial coordinates
    - u: N target profile values
    - uL0: initial left-state extension
    - uM0: initial middle-state extension
    - uR0: initial right-state extension
    - xsL0: initial left-event location
    - xsR0: initial right-event location
    - wL0: initial left-event width
    - wR0: initial right-event width
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - LW_fixed: prescribe wL=wL0 and wR=wR0 if true

    Output:
    - uL: optimized N-value left-state extension
    - uM: optimized N-value middle-state extension
    - uR: optimized N-value right-state extension
    - xsL: optimized left-event location
    - xsR: optimized right-event location
    - wL: optimized or prescribed left-event width
    - wR: optimized or prescribed right-event width
    """
    dx = np.mean(x[1:]-x[:-1])
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("x must be finite and strictly increasing")
    N = x.size

    x_ref = 0.5 * (x[0] + x[-1])
    u_ref = np.mean(u)
    u_floor = 1e-12 * max(1.0, np.max(np.abs(u)))
    u_scale = max(np.ptp(u), u_floor)
    x_opt = (x - x_ref) / dx
    u_opt = (u - u_ref) / u_scale
    uL0_opt = (uL0 - u_ref) / u_scale
    uM0_opt = (uM0 - u_ref) / u_scale
    uR0_opt = (uR0 - u_ref) / u_scale
    xsL0_opt = (xsL0 - x_ref) / dx
    xsR0_opt = (xsR0 - x_ref) / dx
    wL0_opt = wL0 / dx
    wR0_opt = wR0 / dx

    solver_options = dict(
        method="trf",
        loss="linear",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-10,
        gtol=1e-8,
        max_nfev=2000,
    )

    # initial uL and uR
    if LW_fixed:
        theta0 = np.concatenate(
            [uL0_opt,uM0_opt,uR0_opt,[xsL0_opt],[xsR0_opt]]
        )
        lower = np.concatenate(
            [np.full(3*N, -np.inf),[x_opt.min()],[x_opt.min()]]
        )
        upper = np.concatenate(
            [np.full(3*N, np.inf),[x_opt.max()],[x_opt.max()]]
        )
        result0 = least_squares(
            residual,
            theta0,
            args=(x_opt,u_opt,lam1,lamB,np.asarray([wL0_opt,wR0_opt])),
            bounds=(lower, upper),
            **solver_options,
        )
    else:
        theta0 = np.concatenate(
            [
                uL0_opt,
                uM0_opt,
                uR0_opt,
                [xsL0_opt],
                [xsR0_opt],
                [wL0_opt],
                [wR0_opt],
            ]
        )
        lower = np.concatenate(
            [np.full(3*N, -np.inf),
                [x_opt.min()],
                [x_opt.min()],
                [0.1],
                [0.5],
            ]
        )
        upper = np.concatenate(
            [np.full(3*N, np.inf),
                [x_opt.max()],
                [x_opt.max()],
                [100*np.ptp(x_opt)],
                [100*np.ptp(x_opt)],
            ]
        )
        result0 = least_squares(
            residual,
            theta0,
            args=(x_opt,u_opt,lam1,lamB,None),
            bounds=(lower, upper),
            **solver_options,
        )

    if not result0.success or result0.optimality > 1e-5:
        raise RuntimeError(
            "Coupled level-set fit did not converge: "
            f"status={result0.status}, cost={result0.cost:.3e}, "
            f"optimality={result0.optimality:.3e}, message={result0.message}"
        )

    theta = result0.x

    uL = u_ref + u_scale * theta[:N]
    uM = u_ref + u_scale * theta[N:2*N]
    uR = u_ref + u_scale * theta[2*N:3*N]
    xsL = x_ref + dx * theta[3*N+0]
    xsR = x_ref + dx * theta[3*N+1]
    if LW_fixed:
        wL = wL0
        wR = wR0
    else:
        wL  = dx * theta[3*N+2]
        wR  = dx * theta[3*N+3]

    return uL, uM, uR, xsL, xsR, wL, wR
