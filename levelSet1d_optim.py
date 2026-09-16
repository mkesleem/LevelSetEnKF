# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.optimize import least_squares
import levelSet1d_constant
import levelSet1d_optim2


def model(
    theta: np.ndarray,
    x: np.ndarray,
    w_fixed: float | None = None,
) -> np.ndarray:
    """
    Evaluate level set reconstruction.

    u(x) = (uL(x) + uR(x))/2 + (uR(x) - uL(x))/2 * tanh((x - xs)/w)

    Input:
    - theta: parameters [uL, uR, xs, w], or [uL, uR, xs] if w is fixed
    - x: N spatial coordinates
    - w_fixed: prescribed event width, or None to obtain w from theta

    Output:
    - u: reconstruction
    """
    N = x.size
    uL = theta[:N]
    uR = theta[N:2*N]
    if w_fixed is None:
        xs = theta[-2]
        w  = theta[-1]
    else:
        xs = theta[-1]
        w = w_fixed
    t = np.tanh((x - xs) / w)
    return 0.5 * (uL + uR) + 0.5 * (uR - uL) * t


def residual(
    theta: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    lam1: float,
    lamB: float,
    w_fixed: float | None = None,
) -> np.ndarray:
    """
    Assemble residuals for the regularized level set fit.

    Input:
    - theta: parameters [uL, uR, xs, w], or [uL, uR, xs] if w is fixed
    - x: N spatial coordinates
    - u: N target profile values
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - w_fixed: prescribed event width, or None to obtain w from theta

    Output:
    - residual: concatenated fit, smoothness, and boundary residuals
    """
    # fit error
    N = x.size
    uL = theta[:N]
    uR = theta[N:2*N]
    r_fit = model(theta, x, w_fixed) - u

    # smoothness + boundary value penalty
    r_smooth_L1 = lam1 * np.diff(uL, 1)
    r_smooth_R1 = lam1 * np.diff(uR, 1)
    r_bound_L = lamB * ( uL[ 0] - u[ 0] )
    r_bound_R = lamB * ( uR[-1] - u[-1] )
    
    return np.concatenate(
        [r_fit,r_smooth_L1,r_smooth_R1,[r_bound_L],[r_bound_R]]
    )


def fit_tanh(
    x: np.ndarray,
    u: np.ndarray,
    uL0: np.ndarray | None,
    uR0: np.ndarray | None,
    xs0: float | None,
    w0: float | None,
    lam1: float,
    lamB: float,
    LW_fixed: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fit level set representation to u.

    Input:
    - x: N spatial coordinates
    - u: N target profile values
    - uL0: initial left-state extension, or None (constant estimate if None)
    - uR0: initial right-state extension, or None (constant estimate if None)
    - xs0: initial event location, or None (estimate it from u if None)
    - w0: initial event width, or None (use w0=4*dx if None)
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - LW_fixed: prescribe w=w0 if LF_fixed is true

    Output:
    - uL: optimized N-value left-state extension
    - uR: optimized N-value right-state extension
    - xs: optimized event location
    - w: optimized or prescribed event width
    """
    Nx = len(x)
    dx = np.mean(x[1:]-x[:-1])
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("x must be finite and strictly increasing")

    N = x.size

    if xs0 is None:
        xs0 = x[np.argmax(np.abs(np.gradient(u)))]

    if w0 is None:
        w0 = 4*dx

    if uL0 is None:
        uL0 = u[0]*np.ones(Nx)
    else:
        assert(uL0.size==x.size)

    if uR0 is None:
        uR0 = u[-1]*np.ones(Nx)
    else:
        assert(uR0.size==x.size)

    # rescale variables for better performance of the optimization routine
    x_ref = 0.5 * (x[0] + x[-1])
    u_ref = np.mean(u)
    u_floor = 1e-12 * max(1.0, np.max(np.abs(u)))
    u_scale = max(np.ptp(u), u_floor)
    x_opt = (x - x_ref) / dx
    u_opt = (u - u_ref) / u_scale
    uL0_opt = (uL0 - u_ref) / u_scale
    uR0_opt = (uR0 - u_ref) / u_scale
    xs0_opt = (xs0 - x_ref) / dx
    w0_opt = w0 / dx

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
        theta0 = np.concatenate([uL0_opt,uR0_opt,[xs0_opt]])
        lower = np.concatenate([np.full(2*N, -np.inf),[x_opt.min()]])
        upper = np.concatenate([np.full(2*N, np.inf),[x_opt.max()]])
        result0 = least_squares(
            residual,
            theta0,
            args=(x_opt,u_opt,lam1,lamB,w0_opt),
            bounds=(lower, upper),
            **solver_options,
        )
    else:
        theta0 = np.concatenate(
            [uL0_opt,uR0_opt,[xs0_opt],[w0_opt]]
        )
        lower = np.concatenate(
            [np.full(2*N, -np.inf),[x_opt.min()],[0.1]]
        )
        upper = np.concatenate(
            [np.full(2*N, np.inf),[x_opt.max()],[100*np.ptp(x_opt)]]
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
            "Level-set fit did not converge: "
            f"status={result0.status}, cost={result0.cost:.3e}, "
            f"optimality={result0.optimality:.3e}, message={result0.message}"
        )

    theta = result0.x

    uL = u_ref + u_scale * theta[:N]
    uR = u_ref + u_scale * theta[N:2*N]
    if LW_fixed:
        xs = x_ref + dx * theta[-1]
        w = w0
    else:
        xs = x_ref + dx * theta[-2]
        w = dx * theta[-1]

    return uL, uR, xs, w


def train_inner(
    x: np.ndarray,
    u: np.ndarray,
    s1_pm: np.ndarray,
    idxMin: np.ndarray,
    Nd: int,
    lam1: float,
    lamB: float,
    LW_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a level-set representation containing Nd events to u.

    Input:
    - x: Nx spatial coordinates
    - u: Nx target profile values
    - s1_pm: Nx signed shock-sensor values
    - idxMin: Nd by 2 array of initial event-index intervals
    - Nd: number of events
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - LW_in: Nd prescribed event widths, or None to optimize the widths

    Output:
    - LF: (Nd + 1) by Nx state-extension array
    - LS: Nd by Nx level-set-function array
    - LW: Nd event widths
    - xs: Nd event-center coordinates
    - idxMin: updated Nd by 2 event-index intervals
    """

    # get size
    Nx = len(x)
    assert(len(u)==Nx)
    assert(len(s1_pm)==Nx)

    # compute event centre
    xs = levelSet1d_constant.computeEventCentre(x,s1_pm,idxMin)

    # compute level set function extensions
    LF = levelSet1d_constant.computeLF(u,idxMin)
    
    # compute level set function and width
    LS0 = x[np.newaxis,:] - xs[:,np.newaxis]
    LS = np.zeros_like(LS0)
    LW = np.zeros(Nd)
    for ndx in range(Nd):

        # handle specified width
        if LW_in is None:
            thisLW0 = None
            LW_fixed = False
        else:
            thisLW0 = LW_in[ndx]
            LW_fixed = True


        # local optimization
        iterMax = 100
        iter = 0
        while iter < iterMax:
            iter += 1
            idx_start = idxMin[ndx,0]
            idx_end = idxMin[ndx,1]
            thisIdx = np.arange(idx_start, idx_end + 1)
            thisX = x[thisIdx]
            thisU = u[thisIdx]
            thisUL0 = LF[ndx  ,thisIdx]
            thisUR0 = LF[ndx+1,thisIdx]
            uL_opt, uR_opt, xs_opt, w_opt = fit_tanh(
                thisX,
                thisU,
                thisUL0,
                thisUR0,
                xs[ndx],
                thisLW0,
                lam1,
                lamB,
                LW_fixed
            )

            # determine whether monotonically increasing or decreasing
            thisJdx = np.argmax(np.abs(s1_pm[thisIdx]))
            if thisJdx+1 < len(thisU):
                thisU2 = thisU[thisJdx+1]
            else:
                thisU2 = thisU[thisJdx]
            if thisJdx > 0:
                thisU1 = thisU[thisJdx-1]
            else:
                thisU1 = thisU[thisJdx]
            if thisU1 < thisU2:
                monoUpBool = True
            else:
                monoUpBool = False

            # verify convergence
            conv1 = abs(np.tanh((thisX[0]-xs_opt)/w_opt)+1) < 1e-3
            conv2 = abs(np.tanh((thisX[-1]-xs_opt)/w_opt)-1) < 1e-3
            if conv1 and conv2:
                break
            else:
                changeBool = False
                if not conv1:

                    if idxMin[ndx,0] > 0:
                        bool1 = (
                            monoUpBool and
                            u[idxMin[ndx,0]-1] < u[idxMin[ndx,0]]
                        )
                        bool2 = (
                            not monoUpBool 
                            and u[idxMin[ndx,0]-1] > u[idxMin[ndx,0]]
                        )
                    else:
                        bool1 = False
                        bool2 = False

                    if bool1 or bool2:
                        idxMin[ndx,0] += -1
                        changeBool = True
                if not conv2:
                    
                    if idxMin[ndx,1]+1 < Nx:
                        bool1 = (
                            monoUpBool
                            and u[idxMin[ndx,1]] < u[idxMin[ndx,1]+1]
                        )
                        bool2 = (
                            not monoUpBool
                            and u[idxMin[ndx,1]] > u[idxMin[ndx,1]+1]
                        )
                    else:
                        bool1 = False
                        bool2 = False
                    
                    if bool1 or bool2:
                        idxMin[ndx,1] +=  1
                        changeBool = True
            
                if not changeBool:
                    break

        # look at slope
        if monoUpBool:
            idxL = np.where(uL_opt[1:] < uL_opt[:-1])[0]
            if idxL.size > 0:
                idxL = idxL[0]
            else:
                idxL = thisIdx.size - 1
            idxR = np.where(uR_opt[1:] < uR_opt[:-1])[0]
            if idxR.size > 0:
                idxR = np.min([idxR[-1] + 1,thisIdx.size - 1])
            else:
                idxR = 0
        else:
            idxL = np.where(uL_opt[1:] > uL_opt[:-1])[0]
            if idxL.size > 0:
                idxL = idxL[0]
            else:
                idxL = thisIdx.size - 1
            idxR = np.where(uR_opt[1:] > uR_opt[:-1])[0]
            if idxR.size > 0:
                idxR = np.min([idxR[-1] + 1,thisIdx.size - 1])
            else:
                idxR = 0
        

        # smooth
        uL_smooth = uL_opt.copy()
        uL_smooth[idxL:] = uL_smooth[idxL]
        uR_smooth = uR_opt.copy()
        uR_smooth[:idxR+1] = uR_smooth[idxR]

        # update xs
        xs[ndx] = xs_opt

        if ndx > 0 and idxMin[ndx-1,1] == idxMin[ndx,0]:

            # get functions and values
            thisUL0 = LF[ndx-1]
            thisUM0 = np.zeros(Nx)
            thisUM0[thisIdx] = uL_smooth
            thisUM0[:thisIdx[0]] = uL_smooth[0]
            thisUM0[thisIdx[-1]:] = uL_smooth[-1]
            thisUR0 = np.zeros(Nx)
            thisUR0[thisIdx] = uR_smooth
            thisUR0[:thisIdx[0]] = uR_smooth[0]
            thisUR0[thisIdx[-1]:] = u[thisIdx[-1]:]
            if LW_in is None:
                wL = LW[ndx-1]
                wR = w_opt
            else:
                wL = LW_in[ndx-1]
                wR = LW_in[ndx]
            xsL = xs[ndx-1]
            xsR = xs[ndx]

            # restrict domain
            idx_start = idxMin[ndx-1,0]
            idx_end = idxMin[ndx,1]
            thisIdx = np.arange(idx_start, idx_end + 1)
            thisX = x[thisIdx]
            thisU = u[thisIdx]
            thisUL = thisUL0[thisIdx]
            thisUM = thisUM0[thisIdx]
            thisUR = thisUR0[thisIdx]

            # optimize
            (
                uL_opt,
                uM_opt,
                uR_opt,
                xsL_opt,
                xsR_opt,
                wL_opt,
                wR_opt,
            ) = levelSet1d_optim2.fit_tanh(
                thisX,
                thisU,
                thisUL,
                thisUM,
                thisUR,
                xsL,
                xsR,
                wL,
                wR,
                lam1=lam1,
                lamB=lamB,
                LW_fixed=LW_fixed,
            )

            LF[ndx-1,thisIdx] = uL_opt
            LF[ndx  ,thisIdx] = uM_opt
            LF[ndx  ,:thisIdx[0]] = uM_opt[0]
            LF[ndx  ,thisIdx[-1]:] = uM_opt[-1]
            LF[ndx+1,thisIdx] = uR_opt
            LS[ndx-1] = x - xsL_opt
            LS[ndx  ] = x - xsR_opt
            LW[ndx-1] = wL_opt
            LW[ndx  ] = wR_opt

        else:
            # update
            LF[ndx  ,thisIdx] = uL_smooth
            LF[ndx+1,thisIdx] = uR_smooth

            # extend
            LF[ndx  ,thisIdx[-1]:] = uL_smooth[-1]
            LF[ndx+1,:thisIdx[0]] = uR_smooth[0]
            
            LW[ndx] = w_opt
            LS[ndx] = x - xs_opt

    return LF, LS, LW, xs, idxMin
