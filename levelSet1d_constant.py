# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np


def computeEventCentre(
    x: np.ndarray,
    s1: np.ndarray,
    idxMin: np.ndarray,
) -> np.ndarray:
    """
    Estimate the center of each jump discontinuity.

    Input:
    - x: Nx spatial coordinates
    - s1: Nx shock-sensor values
    - idxMin: Nd by 2 array of index intervals containing the events

    Output:
    - xs: Nd estimated event-center coordinates
    """
    Nd = idxMin.shape[0]
    xs = np.zeros(Nd,)
    for ndx in range(Nd):
        xs[ndx] = computeEventCentre_inner(x,s1,idxMin[ndx])
    return xs


def computeEventCentre_inner(
    x: np.ndarray,
    s1: np.ndarray,
    idxMin: np.ndarray,
) -> float:
    """
    Estimate one event center from the maximum absolute shock sensor.

    Input:
    - x: Nx spatial coordinates
    - s1: Nx shock-sensor values
    - idxMin: two indices bounding the event interval

    Output:
    - xs: estimated event-center coordinate
    """
    i_start = idxMin[0]
    i_end   = idxMin[1] + 1
    thisS = np.abs(s1[i_start:i_end].flatten())
    idxMax = np.argmax(np.abs(thisS))
    xs = x[idxMax+i_start]
    return xs


def computeLF(
    u: np.ndarray,
    idxMin: np.ndarray,
) -> np.ndarray:
    """
    Construct constant extensions of the profile between events.

    Each extension retains q in one smooth region and extends its boundary
    values as constants outside that region.

    Input:
    - u: Nx profile to extend
    - idxMin: Nd by 2 array of index intervals containing the events

    Output:
    - LF: (Nd + 1) by Nx array of extended profiles
    """
    Nx = len(u)
    Nd = idxMin.shape[0]

    if Nd == 0:
        return np.asarray(u)[np.newaxis, :].copy()

    LF = np.zeros((Nd+1,Nx))

    for ndx in range(Nd + 1):
        thisLF = np.zeros(Nx)

        if ndx == 0:
            i = idxMin[0, 0]
            thisLF[:i+1] = u[:i+1]
            thisLF[i+1:] = u[i]
            
        elif ndx == Nd:
            i = idxMin[ndx-1, 1]
            thisLF[i:] = u[i:]
            thisLF[:i] = u[i]
            
        else:
            i_prev = idxMin[ndx-1, 1]
            i_next = idxMin[ndx, 0]            
            thisLF[i_prev:i_next+1] = u[i_prev:i_next+1]
            thisLF[i_next+1:] = u[i_next]
            thisLF[:i_prev] = u[i_prev]

        LF[ndx] = thisLF

    return LF
