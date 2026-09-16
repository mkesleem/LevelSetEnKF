# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT


import numpy as np
import utils_main
import utils_pv_1d
import utils_pv_2d
from scipy.interpolate import RegularGridInterpolator


def analysis_stochastic(
    X: np.ndarray,
    Y: np.ndarray,
    d: np.ndarray,
    R: np.ndarray,
    block_size: int | None = None,
) -> np.ndarray:
    """
    Perform a stochastic ensemble Kalman filter analysis update.

    Input:
    - X: Nstate by Ne forecast-state ensemble
    - Y: Nobs by Ne predicted-observation ensemble
    - d: Nobs by Ne perturbed-observation ensemble
    - R: Nobs by Nobs observation-error covariance matrix
    - block_size: number of state rows to update at once, or None

    Output:
    - X_a: Nstate by Ne analysis-state ensemble

    If Ne is one, X is returned unchanged. Blocking changes memory use but not
    the mathematical update.
    """
    Ne = X.shape[1]
    if Ne == 1:
        return X
    else:

        # compute perturbations
        Ne  = X.shape[1]
        Xp  = ( X - np.mean(X, axis=1, keepdims=True) ) / np.sqrt(Ne - 1)
        Yp  = ( Y - np.mean(Y, axis=1, keepdims=True) ) / np.sqrt(Ne - 1)

        # compute state covariance
        Cyy = ( Yp @ Yp.T )

        # solve linear system of equations
        RHS = d - Y
        LHS = Cyy + R
        b   = Yp.T @ (np.linalg.solve(LHS,RHS))
        
        # update state
        if block_size is None:
            return X + Xp @ b
        else:
            n = X.shape[0]
            Xa = np.zeros_like(X)
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                Xa[start:end] = (X[start:end]+Xp[start:end] @ b)

        return Xa
    
    
def get_observations_1d(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    enkfCount: np.ndarray,
    OBS_X: np.ndarray,
    REL_NOISE: float,
    ABS_NOISE: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy pressure observations from a 1D truth solution.

    Input:
    - base_dir: directory containing the truth case
    - case_name: truth-case name without the .truth suffix
    - t_idx: index of the current DA cycle
    - enkfCount: truth-snapshot indices for the DA cycles
    - OBS_X: Nobs sensor coordinates
    - REL_NOISE: relative observation-error level
    - ABS_NOISE: minimum absolute observation-error level
    - rng: NumPy random-number generator

    Output:
    - d: Nobs pressure observations with independent Gaussian noise
    - sigma_obs: Nobs observation-error standard deviations
    """
    truth_vtr = (
        base_dir
        / (case_name + ".truth/results")
        / f"solution_{enkfCount[t_idx]:04d}.vtr"
    )
    truth_sol = utils_pv_1d.load_solution(
        truth_vtr,
        base_dir/(case_name+".truth/input.st")
    )
    y_gt = np.interp(OBS_X, truth_sol["x"], truth_sol["p"])
    sigma_obs = np.maximum(REL_NOISE * np.abs(y_gt), ABS_NOISE)
    return y_gt + rng.normal(0, sigma_obs), sigma_obs


def get_observations_2d(
    base_dir: utils_main.PathLike,
    case_name: str,
    t_idx: int,
    enkfCount: np.ndarray,
    OBS_X: np.ndarray,
    OBS_Y: np.ndarray,
    REL_NOISE: float,
    ABS_NOISE: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy pressure observations from a 2D truth solution.

    Input:
    - base_dir: directory containing the truth case
    - case_name: truth-case name without the .truth suffix
    - t_idx: index of the current DA cycle
    - enkfCount: truth-snapshot indices for the DA cycles
    - OBS_X: sensor x-coordinates
    - OBS_Y: sensor y-coordinates
    - REL_NOISE: relative observation-error level
    - ABS_NOISE: minimum absolute observation-error level
    - rng: NumPy random-number generator

    Output:
    - d: Nobs pressure observations with independent Gaussian noise
    - sigma_obs: Nobs observation-error standard deviations
    """
    assert(t_idx < len(enkfCount))
    truth_vtr = (
        base_dir
        / (case_name + ".truth/results")
        / f"solution_{enkfCount[t_idx]:04d}.vtr"
    )
    truth_sol = utils_pv_2d.load_solution(
        truth_vtr, base_dir/(case_name+".truth/input.st")
    )
    p = truth_sol["p"]
    x = truth_sol["x"]
    y = truth_sol["y"]
    interp = RegularGridInterpolator(
        (y[:,0],x[0,:]),p,method="linear",bounds_error=False,fill_value=None
    )
    pts = np.column_stack((OBS_Y.ravel(),OBS_X.ravel()))
    y_gt = interp(pts)
    sigma_obs = np.maximum(REL_NOISE * np.abs(y_gt), ABS_NOISE)
    return y_gt + rng.normal(0, sigma_obs), sigma_obs


def forecast_inflation(
    X: np.ndarray,
    infl: float = 1.0,
) -> np.ndarray:
    """
    Apply multiplicative inflation to forecast ensemble anomalies.

    X_inflated = mean(X) + infl * (X - mean(X))

    Input:
    - X: Nstate by Ne forecast-state ensemble
    - infl: multiplicative anomaly-inflation factor

    Output:
    - X_inflated: Nstate by Ne inflated forecast-state ensemble

    If Ne is one, X is returned unchanged.
    """
    Ne = X.shape[1]
    if Ne == 1:
        return X
    else:
        Ne  = X.shape[1]
        Xm = np.mean(X, axis=1, keepdims=True)
        return Xm + infl * ( X - Xm )
