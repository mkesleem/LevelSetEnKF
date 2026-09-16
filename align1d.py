# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
import levelSet1d
from scipy.optimize import least_squares
from scipy.interpolate import interp1d


def computeOptimalWarp(
    qc: np.ndarray,
    qf: np.ndarray,
    x: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Align qf to qc in x using density gradient.

    Input:
    - qc: reference shock-sensor profile
    - qf: forecast shock-sensor profile to align
    - x: array of spatial coordinates

    Output:
    - a: scaling factor
    - b: translation
    - qf_align: aligned forecast profile
    """

    # compute shock sensor
    s1c = np.abs(levelSet1d.computeShockSensor(qc[0, :], x)[0])
    s1f = np.abs(levelSet1d.computeShockSensor(qf[0, :], x)[0])

    # take max gradient as initial guess
    xc0 = x[np.argmax(np.abs(s1c))]
    xf0 = x[np.argmax(np.abs(s1f))]

    # set bounds
    dx_max = np.abs(x[-1]-x[0])
    bounds = ([1e-3, -dx_max], [1e3, dx_max])

    # initial guess
    theta0 = np.asarray([1.0, xc0 - xf0])

    result = least_squares(
        residual,
        theta0,
        args=(s1c, s1f, x),
        bounds=bounds,
        max_nfev=100,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    a, b = result.x
    qf_align = warp(qf, a, b, x)
    return a, b, qf_align


def residual(
    theta: np.ndarray,
    qc: np.ndarray,
    qf: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Compute alignment error (residual) between aligned and reference profiles.

    Input:
    - theta: affine scaling and translation parameters ([a,b] = theta)
    - qc: reference profile
    - qf: forecast profile to align
    - x: array of spatial coordinates

    Output:
    - residual: difference between aligned and reference profiles
    """
    a, b = theta
    qf_align = warp(qf, a, b, x)
    return (qf_align[0, :] - qc).reshape(x.size)


def warp(
    q: np.ndarray,
    a: float,
    b: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    Warp q according to q_align(x) = q((x - b) / a).

    Input:
    - q: profile to warp
    - a: scaling factor
    - b: translation
    - x: spatial coordinates

    Output:
    - q_align: q warped according to the warping parameters
    """
    Nx = len(x)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    elif q.shape[0] == Nx:
        q = q.T

    q_align = np.zeros_like(q)
    x_old = (x - b) / a

    for vdx in range(q.shape[0]):
        interp = interp1d(
            x,
            q[vdx, :],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        q_align[vdx, :] = interp(x_old)

    return q_align


def warp_invert(
    q_align: np.ndarray,
    a: float,
    b: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    Invert the warp according to q(x) = q_align(a*x + b).

    Input:
    - q_align: warped profile to invert
    - a: scaling factor
    - b: translation
    - x: array of spatial coordinates

    Output:
    - q: recovered unwarped profile
    """
    Nx = len(x)
    if q_align.ndim == 1:
        q_align = q_align[np.newaxis, :]
    elif q_align.shape[0] == Nx:
        q_align = q_align.T

    q = np.zeros_like(q_align)
    x_new = a * x + b

    for vdx in range(q_align.shape[0]):
        interp = interp1d(
            x,
            q_align[vdx, :],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        q[vdx, :] = interp(x_new)

    return q
