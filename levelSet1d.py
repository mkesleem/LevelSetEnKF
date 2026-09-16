# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import utils_da
import levelSet1d_optim
import align1d


def ensemble_mean(
    prim: np.ndarray,
    x: np.ndarray,
    NdVec: np.ndarray,
    lam1: float = 1e-2,
    lamB: float = 1e-2,
    workers: int | None = None,
) -> np.ndarray:
    """
    Compute an ensemble mean in level set space.

    Input:
    - prim: Nv by Nx profile, or Nv by Nx by Ne ensemble
    - x: Nx spatial coordinates
    - NdVec: Nv event counts, one for each variable
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - workers: number of parallel fitting workers, or None for the default

    Output:
    - prim_mean: Nv by Nx mean profile reconstructed in level set space
    """
    if prim.ndim == 2:
        prim = prim[:, :, np.newaxis]

    if prim.ndim != 3:
        raise ValueError("prim must have shape (Nv, Nx) or (Nv, Nx, Ne)")

    Nv, Nx, Ne = prim.shape
    Nd = int(np.max(NdVec))

    if Nx != len(x):
        raise ValueError("The spatial dimension of prim must match x")

    LF = np.zeros((Nv, Nd + 1, Nx, Ne))
    LS = np.zeros((Nv, Nd, Nx, Ne))
    LW = np.zeros((Nv, Nd, Ne))

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
    ) as executor:
        futures = []

        for i in range(Ne):
            futures.append(
                executor.submit(
                    train,
                    x,
                    prim[:, :, i],
                    NdVec,
                    lam1,
                    lamB,
                )
            )

        for i, future in enumerate(futures):
            LF[:, :, :, i], LS[:, :, :, i], LW[:, :, i] = (
                future.result()
            )

    LF_mean = np.mean(LF, axis=3)
    LS_mean = np.mean(LS, axis=3)
    LW_mean = np.full((Nv, Nd), 1e-12)

    return eval(LF_mean, LS_mean, LW_mean, NdVec)


def computeShockSensor(
    rho: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute first- and second-derivative shock sensors.

    Input:
    - rho: Nx profile values
    - x: Nx uniformly spaced spatial coordinates

    Output:
    - s1: Nx first-derivative values
    - s2: Nx second-derivative values
    """
    # get sizes
    Nx = len(x)
    s1 = np.zeros(Nx)
    s2 = np.zeros(Nx)

    dx = x[1] - x[0]

    # first derivative (gradient)
    s1[1:-1] = (rho[2:] - rho[:-2]) / (2 * dx)
    s1[0] = (rho[1] - rho[0]) / dx
    s1[-1] = (rho[-1] - rho[-2]) / dx

    # second derivative
    s2[1:-1] = (rho[2:] - 2 * rho[1:-1] + rho[:-2]) / (dx ** 2)

    return s1, s2


def chooseIdxLS(
    s1: np.ndarray,
    s2: np.ndarray,
    Nd: int,
) -> np.ndarray:
    """
    Estimate the indices of Nd jump events.

    Input:
    - s1: Nx first-derivative shock sensor
    - s2: Nx second-derivative shock sensor; modified during selection
    - Nd: number of events

    Output:
    - idxLS: Nd sorted event-index estimates
    """
    idxLS = np.zeros(Nd,dtype=int)
    for ndx in range(Nd):
        idxLS[ndx], idx1, idx2 = chooseIdxLS_inner(s1,s2)
        s2[idx1:idx2+1] = 0
    
    return np.sort(idxLS)


def chooseIdxLS_inner(
    s1: np.ndarray,
    s2: np.ndarray,
) -> tuple[int, int, int]:
    """
    Estimate one event index and its surrounding monotone interval.
    idx1 < idxLS < idx2

    Input:
    - s1: Nx first-derivative shock sensor
    - s2: Nx second-derivative shock sensor

    Output:
    - idxLS: estimated event index
    - idx1: left interval index
    - idx2: right interval index
    """
    idx = np.argmax(np.abs(s2))
    idx1,idx2 = findConcaveIdx(idx,s1)
    idxLS = np.argmax(np.abs(s1[idx1:idx2+1]))+idx1
    idx1,idx2 = findConcaveIdx(idxLS,s1)
    return idxLS, idx1, idx2


def findConcaveIdx(
    idx: int,
    s: np.ndarray,
) -> tuple[int, int]:
    """
    Find the monotone interval of s surrounding idx.
    idx1 < idx < idx2

    Input:
    - idx: interior index about which to search
    - s: Nx sensor values

    Output:
    - idx1: left endpoint of the interval
    - idx2: right endpoint of the interval
    """

    # left value
    if s[idx-1] > s[idx]:
        left_candidates = np.where(s[:idx] <= s[1:idx+1])[0]
    else:
        left_candidates = np.where(s[:idx] >= s[1:idx+1])[0]

    # right value
    if s[idx+1] < s[idx]:
        right_candidates = np.where(s[idx+1:] >= s[idx:-1])[0]
    else:
        right_candidates = np.where(s[idx+1:] <= s[idx:-1])[0]

    # extract values
    jdx1 = left_candidates[-1] + 1 if left_candidates.size else 0
    jdx2 = right_candidates[0] + idx if right_candidates.size else len(s) - 1

    # check monotonicity
    d1 = s[jdx1+1:idx]-s[jdx1:idx-1]
    d2 = s[idx+1:jdx2+1]-s[idx:jdx2]
    is_monotone_1 = np.all(d1 >= 0) or np.all(d1 <= 0)
    is_monotone_2 = np.all(d2 >= 0) or np.all(d2 <= 0)
    is_monotone = is_monotone_1 and is_monotone_2
    if not is_monotone:
        raise ValueError(
            f"s[{jdx1}:{jdx2}] is not monotone."
        )

    return jdx1, jdx2


def chooseIdxMin(
    s1: np.ndarray,
    idxLS: np.ndarray,
) -> np.ndarray:
    """
    Find event intervals bounded by minima or sign changes in s1.

    Input:
    - s1: Nx first-derivative shock sensor
    - idxLS: Nd sorted event-index estimates

    Output:
    - idxMin: 2*Nd array of left and right interval indices
    """
    # get sizes
    Nx = len(s1)
    Nd = len(idxLS)

    # find local minima
    idxMinAll = findLocalMinima(np.abs(s1))

    # intialize
    idxMin = np.zeros((Nd,2),dtype=int)

    # inlet
    thisIdx1 = np.arange(0,idxLS[0]+1)
    if Nd > 1:
        thisIdx2 = np.arange(idxLS[0],idxLS[1]+1)
    else:
        thisIdx2 = np.arange(idxLS[0],Nx)
    idx1 = chooseIdxMin1(s1,idxMinAll,thisIdx1,idxLS[0])
    idx2 = chooseIdxMin2(s1,idxMinAll,thisIdx2,idxLS[0])
    assert idx1 < idxLS[0] and idx2 > idxLS[0]
    idxMin[0,:] = [idx1,idx2]

    # outlet
    if Nd > 1:
        thisIdx1 = np.arange(idxLS[-2],idxLS[-1]+1)
        thisIdx2 = np.arange(idxLS[-1],Nx)
        idx1 = chooseIdxMin1(s1,idxMinAll,thisIdx1,idxLS[-1])
        idx2 = chooseIdxMin2(s1,idxMinAll,thisIdx2,idxLS[-1])
        assert idx1 < idxLS[-1] and idx2 > idxLS[-1]
        idxMin[-1,:] = [idx1,idx2]

    # intermediate values
    for i in range(1,Nd-1):
        thisIdx1 = np.arange(idxLS[i-1],idxLS[i]+1)
        thisIdx2 = np.arange(idxLS[i],idxLS[i+1]+1)
        idx1 = chooseIdxMin1(s1,idxMinAll,thisIdx1,idxLS[i])
        idx2 = chooseIdxMin2(s1,idxMinAll,thisIdx2,idxLS[i])
        assert idx1 < idxLS[i] and idx2 > idxLS[i]
        idxMin[i,:] = [idx1,idx2]

    # verify that indices are sorted properly
    idxMinFlat = idxMin.flatten()
    is_sorted = np.all(idxMinFlat[:-1]<=idxMinFlat[1:])
    if not is_sorted:
        raise ValueError("idxMin is not sorted.")

    return idxMinFlat


def chooseIdxMin1(
    s1: np.ndarray,
    idxMinAll: np.ndarray,
    thisIdx: np.ndarray,
    idxLS: int,
) -> int:
    """
    Find the left boundary of an event interval; idx1 < idx

    Input:
    - s1: Nx first-derivative shock sensor
    - idxMinAll: Nx Boolean mask of candidate local minima
    - thisIdx: candidate indices to the left of the event
    - idxLS: event index

    Output:
    - idx1: selected left interval index
    """
    
    # takke absolute value
    s1_abs = np.abs(s1)

    # check whether s1 changes sign
    s1_sign = np.sign(s1)
    mask = s1_sign[thisIdx] != s1_sign[idxLS]
    idx_candidates = thisIdx[mask]

    if idx_candidates.size > 0:
        idx1_sign = idx_candidates[-1] + 1
    else:
        idx1_sign = None

    if idx1_sign is None: # if no sign change
        mask = idxMinAll[thisIdx]
        idx_candidates = thisIdx[mask]
        
        if idx_candidates.size > 0:
            idx1_min = idx_candidates[-1]
        else:
            idx1_min = None

        if idx1_min is None:
            base_val = s1[thisIdx[0]]
            mask = np.abs(s1[thisIdx] - base_val) < 1e-10
            idx1 = thisIdx[mask][-1]
        else:
            if (
                s1_abs[idx1_min] < min(s1_abs[idx1_min-1],s1_abs[idx1_min+1])
                or s1_abs[idx1_min] < 1e-10
            ):
                idx1 = idx1_min
            else:
                raise ValueError("Local minimum not computed correctly")

    
    else: # otherwise, sign change

        assert s1_sign[idx1_sign-1] != s1_sign[idxLS]
        assert np.all(s1_sign[idx1_sign:idxLS + 1] == s1_sign[idxLS])

        # check if a local minima is closer
        search_range = np.arange(idx1_sign, thisIdx[-1] + 1)
        mask = idxMinAll[search_range]
        idx_candidates = search_range[mask]

        if idx_candidates.size > 0:
            idx1_min = idx_candidates[-1]
            if (
                s1_abs[idx1_min] < min(s1_abs[idx1_min-1],s1_abs[idx1_min+1])
                or s1_abs[idx1_min] < 1e-10
            ):
                idx1 = idx1_min
            else:
                raise ValueError("Local minimum not computed correctly")

        else:
            idx1 = idx1_sign

    # adjust so that idx1 < idxLS
    if idx1 == idxLS:
        idx1 = idxLS - 1

    return idx1


def chooseIdxMin2(
    s1: np.ndarray,
    idxMinAll: np.ndarray,
    thisIdx: np.ndarray,
    idxLS: int,
) -> int:
    """
    Find the right boundary of an event interval; idx < idx2

    Input:
    - s1: Nx first-derivative shock sensor
    - idxMinAll: Nx Boolean mask of candidate local minima
    - thisIdx: candidate indices to the right of the event
    - idxLS: event index

    Output:
    - idx2: selected right interval index
    """

    # compute absolute value
    s1_abs = np.abs(s1)

    # get change in sign
    s1_sign = np.sign(s1)
    mask = s1_sign[thisIdx] != s1_sign[idxLS]
    idx_candidates = thisIdx[mask]

    if idx_candidates.size > 0:
        idx2_sign = idx_candidates[0] - 1
    else:
        idx2_sign = None

    if idx2_sign is None: # no sign change
        mask = idxMinAll[thisIdx]
        idx_candidates = thisIdx[mask]

        if idx_candidates.size > 0:
            idx2_min = idx_candidates[0]
        else:
            idx2_min = None

        if idx2_min is None:
            tail = np.arange(idxLS, s1.size)
            mask = np.abs(s1_abs[-1] - s1_abs[tail]) < 1e-10
            idx2 = tail[mask][0]
        else:
            if (
                s1_abs[idx2_min] < min(s1_abs[idx2_min-1],s1_abs[idx2_min+1])
                or s1_abs[idx2_min] < 1e-10
            ):
                idx2 = idx2_min
            else:
                raise ValueError("Local minimum not computed correctly")

    else: # sign change

        # verify
        assert s1_sign[idx2_sign + 1] != s1_sign[idxLS]
        assert np.all(s1_sign[idxLS:idx2_sign + 1] == s1_sign[idxLS])

        # check if there is a closer local minima
        search_range = np.arange(idxLS, idx2_sign + 1)
        mask = idxMinAll[search_range]
        idx_candidates = search_range[mask]

        if idx_candidates.size > 0:
            idx2_min = idx_candidates[0]

            if (
                s1_abs[idx2_min]
                < min(s1_abs[idx2_min - 1],s1_abs[idx2_min + 1])
                or s1_abs[idx2_min] < 1e-10
            ):
                idx2 = idx2_min
            else:
                raise ValueError("Local minimum not computed correctly")

        else:
            idx2 = idx2_sign

    # adjust so that idx2 > idxLS
    if idx2 == idxLS:
        idx2 = idxLS + 1

    return idx2


def findLocalMinima(
    s: np.ndarray,
) -> np.ndarray:
    """
    Identify strict interior local minima of a one-dimensional array.

    Input:
    - s: N values

    Output:
    - idxMinAll: N Boolean mask marking strict local minima
    """
    idxMinAll = np.zeros(len(s), dtype=bool)
    idxMinAll[1:-1] = ((s[1:-1] < s[:-2]) & (s[1:-1] < s[2:]))
    return idxMinAll


def eval(
    LF: np.ndarray,
    LS: np.ndarray,
    LW: np.ndarray,
    NdVec: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct physical profiles from their level set representation.

    Input:
    - LF: Nv by (Nd + 1) by Nx state extensions
    - LS: Nv by Nd by Nx level set functions
    - LW: Nv by Nd event widths
    - NdVec: Nv event counts, one for each variable

    Output:
    - q: Nv by Nx reconstructed profiles
    """

    # verify sizes
    Nd = np.max(NdVec)
    Nv = LF.shape[0]
    Nx = LF.shape[2]
    assert(LF.shape[1]==Nd+1)
    assert(LS.shape[0]==Nv)
    assert(LS.shape[1]==Nd)
    assert(LS.shape[2]==Nx)
    
    # evaluate
    q = np.zeros((Nv, Nx))
    for vdx in range(Nv):
        thisNd = NdVec[vdx]
        thisLW = LW[vdx,:thisNd]
        thisLF = LF[vdx,:thisNd+1]
        thisLS = LS[vdx,:thisNd]

        # smoothed Heaviside
        He = (np.tanh(thisLS / thisLW[:,np.newaxis]) + 1) / 2

        # build tanh_member (Nx, thisNd+1)
        tanh_member = np.zeros((thisNd+1,Nx))
        for fdx in range(thisNd+1):
            mask = np.ones(Nx)
            for jdx in range(thisNd):
                if jdx >= fdx:
                    mask *= (1 - He[jdx])
                else:
                    mask *= He[jdx]
            tanh_member[fdx] = mask

        # evaluate sum over fdx
        q[vdx] = np.sum(tanh_member * thisLF, axis=0)

    return q


def train(
    x: np.ndarray,
    prim: np.ndarray,
    NdVec: np.ndarray,
    lam1: float,
    lamB: float,
    LW0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a level set representation to multiple physical variables.

    Input:
    - x: Nx spatial coordinates
    - prim: Nv by Nx physical-variable profiles
    - NdVec: Nv event counts, one for each variable
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - LW0: Nv by Nd prescribed event widths, or None to optimize them

    Output:
    - LF: Nv by (Nd + 1) by Nx state extensions
    - LS: Nv by Nd by Nx level set functions
    - LW: Nv by Nd event widths
    """

    # get size
    assert(prim.ndim==2)
    Nv = prim.shape[0]
    Nx = len(x)
    assert prim.shape[1] == Nx, "Variables have wrong shape"
    assert len(NdVec) == Nv, "Number of discontinuities not specified correctly"
    Nd = int(np.max(NdVec))

    # compute shock sensor
    s1 = np.zeros((Nv,Nx))
    s2 = np.zeros((Nv,Nx))
    for vdx in range(Nv):
        s1[vdx], s2[vdx] = computeShockSensor(prim[vdx],x)

    # get indices
    idxLS     = np.zeros((Nv,Nd),dtype='int')
    idxMin    = np.zeros((Nv,2*Nd),dtype='int')
    for vdx in range(Nv):
        thisNd = NdVec[vdx]
        if thisNd == 0:
            continue
        idxLS[vdx,:thisNd]    = chooseIdxLS(s1[vdx],s2[vdx],thisNd)
        idxMin[vdx,:2*thisNd] = chooseIdxMin(s1[vdx],idxLS[vdx,:thisNd])

    # compute level set representation
    LF = np.zeros((Nv,Nd+1,Nx))
    LS = np.zeros((Nv,Nd,Nx))
    LW = np.zeros((Nv,Nd))
    xs = np.zeros((Nv,Nd))
    for vdx in range(Nv):
        thisNd = NdVec[vdx]
        thisIdxMin = idxMin[vdx,:2*thisNd].reshape(thisNd,2)
        thisU = prim[vdx]
        thisS1 = s1[vdx]
        if LW0 is None:
            thisLW0 = None
        else:
            thisLW0 = LW0[vdx]
        (
            LF[vdx,:thisNd+1],
            LS[vdx,:thisNd],
            LW[vdx,:thisNd],
            xs[vdx,:thisNd],
            _,
        ) = levelSet1d_optim.train_inner(
            x,
            thisU,
            thisS1,
            thisIdxMin,
            thisNd,
            lam1,
            lamB,
            thisLW0
        )
        
    return LF, LS, LW


def EnKF(
    prim: np.ndarray,
    x: np.ndarray,
    y_obs: np.ndarray,
    sigma_obs: np.ndarray,
    NdVec: np.ndarray,
    OBS_X: np.ndarray,
    rng: np.random.Generator,
    alignVecLF: np.ndarray | None = None,
    lam1: float = 1e-2,
    lamB: float = 1e-2,
    sharp_bool: bool = False,
    infl: float = 1.0,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply an EnKF update in level set space.

    1) Compute level set representations
    2) Registration of rarefaction waves
    3) EnKF in level set space
    4) Invert registration
    5) Transform back to physical space

    Input:
    - prim: Nv by Nx profile, or Nv by Nx by Ne forecast ensemble
    - x: Nx spatial coordinates
    - y_obs: Nobs observation values
    - sigma_obs: Nobs observation-error standard deviations
    - NdVec: Nv event counts, one for each variable
    - OBS_X: Nobs sensor x-coordinates
    - rng: NumPy random-number generator
    - alignVecLF: Boolean mask selecting state extensions for registration
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - sharp_bool: whether to use the sharpest median width for each variable
    - infl: inflation factor
    - workers: number of parallel fitting workers, or None for the default

    Output:
    - prim_a: Nv by Nx by Ne analysis ensemble
    - prim_f_LS: Nv by Nx by Ne reconstructed forecast ensemble
    - prim_a_mean: Nv by Nx analysis mean reconstructed in level set space
    - prim_f_mean: Nv by Nx forecast mean reconstructed in level set space
    """
    
    Nd = int(np.max(NdVec))
    Nx = len(x)
    Nv = prim.shape[0]
    if prim.ndim == 2:
        prim = prim[:,:,np.newaxis]
    Ne = prim.shape[2]
    assert(prim.shape[1]==Nx)

    if alignVecLF is None:
        alignVecLF = np.zeros(Nd+1)
    else:
        assert(len(alignVecLF)==Nd+1)
    
    n_obs = OBS_X.size
    if y_obs.size != n_obs or sigma_obs.size != n_obs:
        raise ValueError(
            "OBS_X, y_obs, and sigma_obs must have equal sizes; "
            f"got {n_obs}, {y_obs.size}, and {sigma_obs.size}"
        )

    # initialize
    LF_f   = np.zeros((Nv,Nd+1,Nx,Ne))
    LS_f   = np.zeros((Nv,Nd  ,Nx,Ne))
    LW_f_0 = np.zeros((Nv,Nd,Ne))

    # 1. Level Set Fitting
    print("Fit level representations")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
    ) as executor:

        # prepare calls (parallel, LW is unconstrained)
        futures = []
        for i in range(Ne):
            futures.append(
                executor.submit(
                    train,
                    x,
                    prim[:, :, i],
                    NdVec,
                    lam1,
                    lamB,
                )
            )

        # level sit fit (parallel, LW is unconstrained)
        for i, future in enumerate(futures):
            LF_f[:, :, :, i], LS_f[:, :, :, i], LW_f_0[:, :, i] = (
                future.result()
            )

        # compute LW to be used for fitting (median)
        LW_median_0 = np.nanmedian(LW_f_0,axis=2)
        if sharp_bool:
            LW_median_0 = LW_median_0.copy()
            LW_median_0[LW_median_0 < 1e-12] = np.nan
            LW_min = np.nanmin(LW_median_0, axis=1)
            LW_median = np.broadcast_to(
                LW_min[:, np.newaxis],
                LW_median_0.shape,
            ).copy()
            LW_median[np.isnan(LW_median_0)] = np.nan
        else:
            LW_median = LW_median_0.copy()

        # prepare calls (parallel, LW is LW_median)
        futures = []
        for i in range(Ne):
            futures.append(
                executor.submit(
                    train,
                    x,
                    prim[:, :, i],
                    NdVec,
                    lam1,
                    lamB,
                    LW0=LW_median,
                )
            )

        # level sit fit (parallel, LW is LW_median)
        for i, future in enumerate(futures):
            LF_f[:, :, :, i], LS_f[:, :, :, i], _ = (
                future.result()
            )
    
    # 2. Get Observations from Level Set
    prim_f_LS = np.zeros_like(prim)
    y_preds = np.zeros((len(OBS_X),Ne))
    for i in range(Ne):
        prim_f_LS[:,:,i] = eval(LF_f[:,:,:,i],LS_f[:,:,:,i],LW_median,NdVec)
        y_preds[:,i] = np.interp(OBS_X,x,prim_f_LS[2,:,i])
    
    # 3. Registration
    a_LF_f = np.full((Nd+1,Ne), np.nan)
    b_LF_f = np.full((Nd+1,Ne), np.nan)
    LF_f_align = LF_f.copy()
    for ndx in range(Nd+1):
        if alignVecLF[ndx]:
            for i in range(Ne):
                (
                    a_LF_f[ndx,i],
                    b_LF_f[ndx,i],
                    LF_f_align[:,ndx,:,i],
                ) = align1d.computeOptimalWarp(
                    LF_f[:,ndx,:,0],
                    LF_f[:,ndx,:,i],
                    x,
                )

    # compute forecast mean
    a_LF_f_mean = np.mean(a_LF_f,axis=1)
    b_LF_f_mean = np.mean(b_LF_f,axis=1)
    LF_align_f_mean = np.mean(LF_f_align,axis=3)
    LS_f_mean = np.mean(LS_f,axis=3)
    LF_f_mean = LF_align_f_mean.copy()
    for ndx in range(Nd+1):
        if alignVecLF[ndx]:
            a = a_LF_f_mean[ndx]
            b = b_LF_f_mean[ndx]
            LF_f_mean[:,ndx,:] = align1d.warp_invert(
                LF_align_f_mean[:,ndx,:],
                a,
                b,
                x,
            )

    # 4. Construct state vector
    X_LF = LF_f_align.reshape((Nv*(Nd+1)*Nx,Ne))
    X_LS = LS_f.reshape((Nv*(Nd+0)*Nx,Ne))
    state_f_all = np.vstack([
        X_LF,
        X_LS,
        a_LF_f,
        b_LF_f
    ])
    state_mask = ~np.isnan(state_f_all[:,0])
    assert(np.all(np.isnan(state_f_all[~state_mask,1:])))
    state_f = state_f_all[state_mask,:]
    assert(np.any(np.isnan(state_f)) == 0)

    # Split row counts
    n_LF = Nv*(Nd+1)*Nx
    n_LS = Nv*(Nd+0)*Nx
    n_a  = Nd+1
    n_b  = Nd+1
    i0   = 0
    i1   = i0 + n_LF
    i2   = i1 + n_LS
    i3   = i2 + n_a
    i4   = i3 + n_b

    state_f = utils_da.forecast_inflation(state_f,infl)

    # fitting complete
    print("Level set fit complete")

    # 5. EnKF Update
    obs_ens = y_obs[:, None] + rng.normal(
        0,
        sigma_obs[:, None],
        (len(OBS_X), Ne),
    )
    state_a = utils_da.analysis_stochastic(
        state_f,
        y_preds,
        obs_ens,
        np.diag(sigma_obs**2),
    )

    # 6. Extract updated state
    state_a_all = np.full((state_mask.size,Ne),np.nan)
    state_a_all[state_mask] = state_a
    X_LF_a = state_a_all[i0:i1]
    X_LS_a = state_a_all[i1:i2]
    a_LF_a = state_a_all[i2:i3]
    b_LF_a = state_a_all[i3:i4]
    LF_align_a = X_LF_a.reshape((Nv,Nd+1,Nx,Ne))
    LS_a       = X_LS_a.reshape((Nv,Nd+0,Nx,Ne))

    # compute mean
    a_LF_a_mean = np.mean(a_LF_a,axis=1)
    b_LF_a_mean = np.mean(b_LF_a,axis=1)
    LF_align_a_mean = np.mean(LF_align_a,axis=3)
    LS_a_mean = np.mean(LS_a,axis=3)
    LS_a       = X_LS_a.reshape((Nv,Nd+0,Nx,Ne))

    # 7. Invert the registration map
    LF_a = LF_align_a.copy()
    LF_a_mean = LF_align_a_mean.copy()
    for ndx in range(Nd+1):
        if alignVecLF[ndx]:
            a = a_LF_a_mean[ndx]
            b = b_LF_a_mean[ndx]
            if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
                raise RuntimeError(
                    f"Invalid analyzed warp: a={a}, b={b}, "
                    f"ndx={ndx}, ensemble={i}"
                )
            LF_a_mean[:,ndx,:] = align1d.warp_invert(
                LF_align_a_mean[:,ndx,:],
                a,
                b,
                x,
            )
            for i in range(Ne):
                a = a_LF_a[ndx, i]
                b = b_LF_a[ndx, i]
                if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
                    raise RuntimeError(
                        f"Invalid analyzed warp: a={a}, b={b}, "
                        f"ndx={ndx}, ensemble={i}"
                    )
                LF_a[:,ndx,:,i] = align1d.warp_invert(
                    LF_align_a[:,ndx,:,i],
                    a,
                    b,
                    x,
                )

    # 8. Evaluate the level set function
    prim_f_mean = eval(LF_f_mean,LS_f_mean,LW_median,NdVec)
    prim_a_mean = eval(LF_a_mean,LS_a_mean,LW_median,NdVec)
    prim_a = np.zeros((Nv,Nx,Ne))
    for i in range(Ne):
        prim_a[:,:,i] = eval(LF_a[:,:,:,i],LS_a[:,:,:,i],LW_median,NdVec)   

    return prim_a,prim_f_LS,prim_a_mean,prim_f_mean
