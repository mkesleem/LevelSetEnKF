# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree
from matplotlib.path import Path
from scipy.signal import argrelextrema
import levelSet1d, levelSet1d_optim
import utils_da
from scipy.ndimage import gaussian_filter1d
import align2d, align2d_LS
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext


def computeShockSensor(
    u: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first- and second-order spatial derivatives of a field.

    Input:
    - u: Ny by Nx field, optionally with Ne ensemble members
    - xg: Ny by Nx array of x-coordinates
    - yg: Ny by Nx array of y-coordinates

    Output:
    - s1: Ny by Nx by 2 by Ne array of derivatives du/dx and du/dy
    - s2: Ny by Nx by 3 by Ne array of derivatives d2u/dx2, d2u/dxdy,
      and d2u/dy2
    """
    
    dx = xg[0, 1] - xg[0, 0]
    dy = yg[1, 0] - yg[0, 0]
    assert(dx > 1e-10 and dy > 1e-10)

    if u.ndim == 2:
        u = u[:,:,np.newaxis]
    Ny, Nx, Ne = u.shape

    # First derivatives
    dfdx = np.zeros((Ny,Nx,Ne))
    dfdy = np.zeros((Ny,Nx,Ne))

    # x-derivative
    dfdx[:,1:-1,:] = (u[:,2:,:]-u[:,:-2,:]) / (2*dx)
    dfdx[:,0,:]    = (u[:,1,:]-u[:,0,:]) / dx
    dfdx[:,-1,:]   = (u[:,-1,:]-u[:,-2,:]) / dx

    # y-derivative
    dfdy[1:-1,:,:] = (u[2:,:,:] - u[:-2,:,:]) / (2*dy)
    dfdy[0,:,:]    = (u[1,:,:] - u[0,:,:]) / dy
    dfdy[-1,:,:]   = (u[-1,:,:] - u[-2,:,:]) / dy

    # Assemble gradient
    s1 = np.zeros((Ny,Nx,2,Ne))
    s1[:,:,0,:] = dfdx
    s1[:,:,1,:] = dfdy

    # Second derivatives
    d2fdx2  = np.zeros((Ny,Nx,Ne))
    d2fdy2  = np.zeros((Ny,Nx,Ne))
    d2fdxdy = np.zeros((Ny,Nx,Ne))

    # second derivative in x
    d2fdx2[:,1:-1,:] = (u[:,2:,:] - 2*u[:,1:-1,:] + u[:,:-2,:]) / dx**2
    d2fdx2[:,0,:] = (u[:, 2, :] - 2*u[:,1,:] + u[:,0,:]) / dx**2
    d2fdx2[:,-1,:] = (u[:, -1, :] - 2*u[:,-2,:] + u[:,-3,:]) / dx**2

    # second derivative in y
    d2fdy2[1:-1,:,:] = (u[2:,:,:] - 2*u[1:-1,:,:] + u[:-2,:,:]) / dy**2
    d2fdy2[0,:,:] = (u[2,:,:] - 2*u[1,:,:] + u[0,:,:]) / dy**2
    d2fdy2[-1,:,:] = (u[-1,:,:] - 2*u[-2,:,:] + u[-3,:,:]) / dy**2

    # cross derivative
    d2fdxdy[1:-1,1:-1,:] = (
        u[2:,2:,:] - u[2:,:-2,:] - u[:-2,2:,:] + u[:-2,:-2,:]
    ) / (4*dx*dy)

    # Assemble Hessian
    s2 = np.zeros((Ny,Nx,3,Ne))
    s2[:,:,0,:] = d2fdx2
    s2[:,:,1,:] = d2fdxdy
    s2[:,:,2,:] = d2fdy2

    return s1, s2


def train(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    NdVec: np.ndarray,
    lam1: float,
    lamB: float,
    LW_in: np.ndarray | None = None,
    workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit level set representations of the primitive variables.

    Input:
    - q: 4 by Ny by Nx array of primitive variables
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - NdVec: length-4 array containing the number of events per variable
    - lam1: regularization parameter to promote smoothness
    - lamB: regularization parameter to enforce boundary conditions
    - LW_in: optional 4 by Nd array of prescribed event widths
    - workers: number of parallel fitting workers, or None for the default
    - executor: optional executor to reuse for parallel fitting

    Output:
    - LF: 4 by (Nd + 1) by Ny by Nx array of level set extensions
    - LS: 4 by Nd by Ny by Nx array of level set functions
    - LW_out: 4 by Nd array of fitted event widths

    Nd = np.max(NdVec)
    """    

    # get grid size
    Ny, Nx = x.shape
    assert(y.shape[0]==Ny)
    assert(y.shape[1]==Nx)

    # verify size of q
    Nv = 4
    assert(q.shape[0] == Nv)
    assert(q.shape[1] == Ny)
    assert(q.shape[2] == Nx)

    # get number of interfaces
    Nd = int(np.max(NdVec))
    assert(len(NdVec) == Nv)

    if LW_in is not None:
        assert LW_in.shape == (Nv, Nd)
        LW_in_rho = LW_in[0,:NdVec[0]]
        LW_in_uv = LW_in[1,:NdVec[1]]
        LW_in_p = LW_in[3,:NdVec[3]]
        assert(np.abs(np.nanmax(LW_in[1,:]-LW_in[2,:]))<1e-12)
    else:
        LW_in_rho = None
        LW_in_uv = None
        LW_in_p = None

    # get variables for tracking interface
    q_track = np.zeros((Nv - 1, Ny, Nx))
    q_track[0] = q[0]
    q_track[1] = np.hypot(q[1], q[2])
    q_track[2] = q[3]

    # compute shock sensor
    s1 = np.zeros((Nv-1,Ny,Nx,2))
    for vdx in range(Nv-1): 
        thisS1,_ = computeShockSensor(q_track[vdx],x,y)
        s1[vdx] = thisS1[:,:,:,0]

    # Reuse a supplied worker pool, or create one for this training call.
    executor_context = (
        ProcessPoolExecutor(max_workers=workers)
        if executor is None
        else nullcontext(executor)
    )
    with executor_context as active_executor:
        line_data_rho = computeInterfaceLines(
            q[:1],
            x,
            y,
            s1[0],
            Nd=NdVec[0],
            lam1=lam1,
            lamB=lamB,
            LW_in=LW_in_rho,
            workers=workers,
            executor=active_executor
        )
        line_data_uv = computeInterfaceLines(
            q[1:3],
            x,
            y,
            s1[1],
            Nd=NdVec[1],
            lam1=lam1,
            lamB=lamB,
            LW_in=LW_in_uv,
            workers=workers,
            executor=active_executor
        )
        line_data_p = computeInterfaceLines(
            q[3:],
            x,
            y,
            s1[2],
            Nd=NdVec[3],
            lam1=lam1,
            lamB=lamB,
            LW_in=LW_in_p,
            workers=workers,
            executor=active_executor
        )

    # interpolate
    LF = np.full((Nv,Nd+1,Ny,Nx),np.nan)
    LS = np.full((Nv,Nd  ,Ny,Nx),np.nan)
    LW_out = np.full((Nv,Nd),np.nan)

    # density
    thisLF_rho,thisLS_rho,thisLW_rho = interpLineToGrid(line_data_rho,q[:1],x,y)
    thisNd = NdVec[0]
    LF[0,:thisNd+1,:,:] = thisLF_rho
    LS[0,:thisNd   :,:] = thisLS_rho
    LW_out[0,:thisNd]   = thisLW_rho

    # velocity
    thisLF_uv,thisLS_uv,thisLW_uv = interpLineToGrid(line_data_uv,q[1:3],x,y)
    thisNd = NdVec[1]
    LF[1:3,:thisNd+1,:,:] = thisLF_uv
    LS[1,  :thisNd  ,:,:] = thisLS_uv
    LS[2,  :thisNd  ,:,:] = thisLS_uv
    LW_out[1,:thisNd]     = thisLW_uv
    LW_out[2,:thisNd]     = thisLW_uv

    # pressure
    thisLF_p,thisLS_p,thisLW_p = interpLineToGrid(line_data_p,q[3:],x,y)
    thisNd = NdVec[3]
    LF[3,:thisNd+1,:,:] = thisLF_p
    LS[3,:thisNd   :,:] = thisLS_p
    LW_out[3,:thisNd]   = thisLW_p

    return LF,LS,LW_out


def computeInterfaceLines(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s1: np.ndarray,
    lam1: float,
    lamB: float,
    Nd: int = 1,
    LW_in: np.ndarray | None = None,
    workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
):
    """Fit one-dimensional level sets along normals to 2D interfaces.

    Input:
    - q: Nv by Ny by Nx array of fields to fit
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - s1: Ny by Nx by 2 array containing the tracking-field gradient
    - lam1: regularization parameter to promote smoothness
    - lamB: regularization parameter to enforce boundary conditions
    - Nd: number of interfaces to fit
    - LW_in: optional length-Nd array of prescribed interface widths
    - workers: number of parallel fitting workers, or None for the default
    - executor: optional executor to reuse for parallel fitting

    Output:
    - line_data_all: fitted normal-line data dictionaries, ordered by contour area
    """

    Ny,Nx = x.shape
    Nv = q.shape[0]
    assert(s1.shape == (Ny,Nx,2))
    assert(x.shape == (Ny,Nx))
    assert(y.shape == (Ny,Nx))
    assert(q.shape == (Nv,Ny,Nx))
    if LW_in is not None:
        assert(len(LW_in) == Nd)
        assert(LW_in.ndim==1)

    # compute norm of gradient for this variable
    s1_norm = np.sqrt(s1[:,:,0]**2 + s1[:,:,1]**2)
    s1_norm_mask = s1_norm.copy()
    s1_mask = s1.copy()

    line_data_all = []
    mask = np.zeros(x.shape, dtype=bool)
    area = np.zeros(Nd)
    LW_median = np.zeros(Nd)
    for ndx in range(Nd):

        # update gradient
        s1_norm_mask[mask] = 0

        # get width
        if LW_in is None:
            thisLW_in = None
        else:
            thisLW_in = np.asarray([LW_in[ndx]])

        # Prepare the contour geometry and interpolated normal-line data once.
        prepared_lines = prepare_interface_lines(
            q,
            x,
            y,
            s1_mask,
            s1_norm_mask,
        )

        # First fit: use supplied width (unconstrained if width not supplied).
        line_data_in = fit_interface_lines(
            prepared_lines,
            lam1=lam1,
            lamB=lamB,
            Nd=1,
            LW_in=thisLW_in,
            workers=workers,
            executor=executor
        )

        # Fit again using median width if width is not supplied
        if LW_in is None:
            median_width = np.nanmedian(line_data_in["LW_line"])
            if not np.isfinite(median_width) or median_width <= 0:
                raise RuntimeError(
                    f"Invalid median interface width for interface {ndx}: "
                    f"{median_width}"
                )
            thisLW_in = np.asarray([median_width])
            line_data_in = fit_interface_lines(
                prepared_lines,
                lam1=lam1,
                lamB=lamB,
                Nd=1,
                LW_in=thisLW_in,
                workers=workers,
                executor=executor
            )

        line_data_all.append(line_data_in)
        LW_median[ndx] = np.nanmedian(line_data_in["LW_line"])

        # compute area
        area[ndx] = np.abs(polygon_area(
            line_data_in["xs_line"],line_data_in["ys_line"]))

        # mask
        if ndx < Nd - 1:
            mask_line = line_data_in["mask_line"]
            x_line = line_data_in["x_line"]
            y_line = line_data_in["y_line"]
            tree = cKDTree(np.column_stack([x_line.ravel(), y_line.ravel()]))
            nearest_idx = tree.query(np.column_stack([x.ravel(), y.ravel()]))[1]
            mask = mask_line.ravel()[nearest_idx].reshape(x.shape)
    
    # sort by area
    area_idxSort = np.argsort(area)
    line_data_all = [line_data_all[i] for i in area_idxSort]
    LW_median = LW_median[area_idxSort]

    line_data_overlap = []
    if Nd > 1:
        for ndx in range(Nd-1):
            x_line_in = line_data_all[ndx]["x_line"][:,-1]
            y_line_in = line_data_all[ndx]["y_line"][:,-1]
            x_line_out = line_data_all[ndx+1]["x_line"][:,0]
            y_line_out = line_data_all[ndx+1]["y_line"][:,0]
            a_in = np.abs(polygon_area(x_line_in,y_line_in))
            a_out = np.abs(polygon_area(x_line_out,y_line_out))
            if a_in > a_out:
                x_line_in = line_data_all[ndx]["x_line"][:,0]
                y_line_in = line_data_all[ndx]["y_line"][:,0]
                xs_line = line_data_all[ndx+1]["xs_line"][:,0]
                ys_line = line_data_all[ndx+1]["ys_line"][:,0]
                xy_in = np.concatenate((x_line_in[:,None],y_line_in[:,None]),
                                        axis=1)
                xy_s  = np.concatenate((xs_line[:,None]  , ys_line[:,None]) ,
                                        axis=1)
                tree = cKDTree(xy_in)
                dist, _ = tree.query(xy_s)
                dx = x[0,1] - x[0,0]
                dy = y[1,0] - y[0,0]
                dt = min(dx, dy)
                Nt_in = int(np.ceil(dist.max() / dt))
                line_data_in = computeInterfaceLines_inner(
                    q,
                    x,
                    y,
                    s1,
                    s1_norm,
                    lam1=lam1,
                    lamB=lamB,
                    Nd=2,
                    Nt_in=Nt_in,
                    LW_in=LW_median[ndx:ndx+2],
                    workers=workers,
                    executor=executor
                )
                line_data_overlap.append(line_data_in)
        if len(line_data_overlap) > 0:
            line_data_all = line_data_overlap
    

    return line_data_all


def computeInterfaceLines_inner(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s1: np.ndarray,
    s1_norm: np.ndarray,
    lam1: float,
    lamB: float,
    Nd: int = 1,
    Nt_in: int = 10,
    Nt_out: int = 10,
    LW_in: np.ndarray | None = None,
    workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
):
    """Prepare and fit level sets along normals to one 2D contour.

    Input:
    - q: Nv by Ny by Nx array of fields to fit
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - s1: Ny by Nx by 2 array containing the tracking-field gradient
    - s1_norm: Ny by Nx array containing the gradient magnitude
    - lam1: regularization parameter to promote smoothness
    - lamB: regularization parameter to enforce boundary conditions
    - Nd: number of interfaces to fit on each normal line
    - Nt_in: approximate number of grid spacings inside the contour
    - Nt_out: approximate number of grid spacings outside the contour
    - LW_in: optional length-Nd array of prescribed interface widths
    - workers: number of parallel fitting workers, or None for the default
    - executor: optional executor to reuse for parallel fitting

    Output:
    - line_data: dictionary containing geometry and fitted normal-line data
    """


    prepared_lines = prepare_interface_lines(
        q,
        x,
        y,
        s1,
        s1_norm,
        Nt_in=Nt_in,
        Nt_out=Nt_out,
    )
    return fit_interface_lines(
        prepared_lines,
        lam1=lam1,
        lamB=lamB,
        Nd=Nd,
        LW_in=LW_in,
        workers=workers,
        executor=executor,
    )


def prepare_interface_lines(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s1: np.ndarray,
    s1_norm: np.ndarray,
    Nt_in: int =10,
    Nt_out: int = 10,
):
    """Construct and sample lines normal to a 2D interface contour.

    Input:
    - q: Nv by Ny by Nx array of fields to sample
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - s1: Ny by Nx by 2 array containing the tracking-field gradient
    - s1_norm: Ny by Nx array containing the gradient magnitude
    - Nt_in: approximate number of grid spacings inside the contour
    - Nt_out: approximate number of grid spacings outside the contour

    Output:
    - prepared_lines: dictionary containing the contour, normals, sampled lines,
      field values, and gradients
    """


    Ny, Nx = x.shape
    Nv = q.shape[0]
    assert(s1.shape == (Ny,Nx,2))
    assert(x.shape == (Ny,Nx))
    assert(y.shape == (Ny,Nx))
    assert(q.shape == (Nv,Ny,Nx))

    # get domain maximum of first derivative
    ldxMax = np.argmax(s1_norm)

    contour_ldx = get_contour(ldxMax,s1,s1_norm)
    xp = x.flat[contour_ldx]
    yp = y.flat[contour_ldx]
    Np = len(contour_ldx)

    # interpolate
    grid_x = x[0,:]
    grid_y = y[:,0]
    interp_fields = np.concatenate([np.moveaxis(q, 0, -1), s1[:, :, :2]],
                                   axis=-1)
    F = RegularGridInterpolator((grid_y, grid_x),
                                interp_fields,
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

    # interpolate first derivative of this variable on contour
    yxp_pts = np.column_stack([yp,xp])
    xyp_pts = np.column_stack([xp,yp])
    interp_values = F(yxp_pts)
    s11 = interp_values[:,Nv  ]
    s12 = interp_values[:,Nv+1]

    # compute normal vector
    n = np.column_stack([s11,s12])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    n = flip_normal_sign(xyp_pts,n)

    # get grid spacing
    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]
    assert(np.max(np.abs(x[:,1:]-x[:,:-1]-dx))<1e-12)
    assert(np.max(np.abs(y[1:,:]-y[:-1,:]-dy))<1e-12)
    dt = min(dx, dy)

    # set width of interpolation (in shock normal direction)
    t1 = -Nt_in*dt
    t2 = Nt_out*dt
    NtMax = int(np.ceil((t2 - t1) / dt))

    # get interpolation regions
    x_line = np.zeros((Np,NtMax))
    y_line = np.zeros((Np,NtMax))
    t_line = np.zeros((Np,NtMax))
    for ldx in range(Np):
        x_line[ldx],y_line[ldx],t_line[ldx] = createShockNormalLine(
            xp[ldx],yp[ldx],n[ldx],t1,t2,NtMax)

    # interpolate solution and gradients
    G_line = np.zeros((Np,2,NtMax))
    pts_line = np.column_stack([y_line.ravel(),x_line.ravel()])
    interp_values = F(pts_line)
    q_line = np.zeros((Np,NtMax,Nv))
    for vdx in range(Nv):
        q_line[:,:,vdx] = interp_values[:,vdx].reshape(Np,NtMax)
    G_line[:,0,:] = interp_values[:,Nv  ].reshape(Np,NtMax)
    G_line[:,1,:] = interp_values[:,Nv+1].reshape(Np,NtMax)

    # compute normal and tangential velocity
    if Nv == 2:
        q_line_fit = q_line[:,:,0] * n[:,0,None] + q_line[:,:,1] * n[:,1,None]
        q_line_tan = -q_line[:,:,0] * n[:,1,None] + q_line[:,:,1] * n[:,0,None]
    else:
        q_line_fit = q_line[:,:,0]
        q_line_tan = np.zeros_like(q_line_fit)

    return {
        "xp": xp,
        "yp": yp,
        "n": n,
        "x_line": x_line,
        "y_line": y_line,
        "t_line": t_line,
        "q_line": q_line,
        "q_line_fit": q_line_fit,
        "q_line_tan": q_line_tan,
        "G_line": G_line,
    }


def fit_interface_lines(
    prepared_lines,
    lam1: float,
    lamB: float,
    Nd: int = 1,
    LW_in: np.ndarray | None = None,
    workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
):
    """Fit one-dimensional level sets to prepared interface-normal lines.

    Input:
    - prepared_lines: dictionary returned by prepare_interface_lines
    - lam1: regularization parameter to promote smoothness
    - lamB: regularization parameter to enforce boundary conditions
    - Nd: number of interfaces to fit on each line
    - LW_in: optional length-Nd array of prescribed interface widths
    - workers: number of parallel fitting workers, or None for the default
    - executor: optional executor to reuse for parallel fitting

    Output:
    - line_data: dictionary containing the prepared geometry and fitted line data
    """


    if LW_in is not None:
        LW_in = np.asarray(LW_in)
        assert LW_in.shape == (Nd,)

    xp = prepared_lines["xp"]
    yp = prepared_lines["yp"]
    n = prepared_lines["n"]
    x_line = prepared_lines["x_line"]
    y_line = prepared_lines["y_line"]
    t_line = prepared_lines["t_line"]
    q_line = prepared_lines["q_line"]
    q_line_fit = prepared_lines["q_line_fit"]
    q_line_tan = prepared_lines["q_line_tan"]
    G_line = prepared_lines["G_line"]

    Np, NtMax = t_line.shape
    Nv = q_line.shape[2]

    # compute level set functions
    LF_line_fit = np.zeros((Np,Nd+1,NtMax))
    LS_line = np.zeros((Np,Nd,NtMax))
    LW_line = np.zeros((Np,Nd))
    ys_line = np.zeros((Np,Nd))
    xs_line = np.zeros((Np,Nd))
    mask_line = np.zeros((Np,NtMax),dtype='bool')


    def fit_lines(active_executor):
        futures = []
        for ldx in range(Np):
            future = active_executor.submit(
                computeShockNormalLine,
                xp[ldx],
                yp[ldx],
                n[ldx],
                t_line[ldx],
                q_line_fit[ldx],
                G_line[ldx],
                lam1=lam1,
                lamB=lamB,
                Nd=Nd,
                LW_in=LW_in,
            )
            futures.append((ldx, future))

        for ldx, future in futures:
            (
                LF_line_fit[ldx],
                LS_line[ldx],
                LW_line[ldx],
                xs_line[ldx],
                ys_line[ldx],
                mask_line[ldx],
            ) = future.result()

    if executor is None:
        with ProcessPoolExecutor(max_workers=workers) as local_executor:
            fit_lines(local_executor)
    else:
        fit_lines(executor)

    if LW_in is not None:
        LW_line[:] = LW_in[np.newaxis, :]

    LW_out = np.nanmedian(LW_line, axis=0)

    if Nv == 2:
        LF_line = np.zeros((Np, 2, NtMax, 2))
        nx = n[:,None,None,0]
        ny = n[:,None,None,1]
        ut = q_line_tan[:,None,:]
        LF_line[:,:,:,0] = LF_line_fit * nx - ut * ny
        LF_line[:,:,:,1] = LF_line_fit * ny + ut * nx
    else:
        LF_line = LF_line_fit.copy()[:,:,:,None]

    return {
        "x_line": x_line,
        "y_line": y_line,
        "q_line": q_line,
        "q_line_fit": q_line_fit,
        "q_line_tan": q_line_tan,
        "LF_line": LF_line,
        "LS_line": LS_line,
        "xs_line": xs_line,
        "ys_line": ys_line,
        "mask_line": mask_line,
        "LW_line": LW_line,
        "LW_out": LW_out,
    }


def createShockNormalLine(
    x0: float,
    y0: float,
    n: np.ndarray,
    t1: float,
    t2: float,
    Nt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parameterize a line through (x0, y0) in the normal direction n.

    Input:
    - x0: x-coordinate of starting point
    - y0: y-coordinate of starting point
    - n: length-2 normal vector [nx, ny]
    - t1: first line parameter
    - t2: last line parameter
    - Nt: number of line samples

    Output:
    - x_line: Nt array of x-coordinates
    - y_line: Nt array of y-coordinates
    - t_line: Nt array of t-coordinates
    """
    n = np.asarray(n, dtype=float).reshape(2)
    n /= np.linalg.norm(n)
    t_line = np.linspace(t1, t2, Nt).reshape(Nt)
    x_line = x0 + t_line * n[0]
    y_line = y0 + t_line * n[1]

    return x_line, y_line, t_line


def computeShockNormalLine(
    x0: float,
    y0: float,
    n: np.ndarray,
    t_line: np.ndarray,
    q_line_fit: np.ndarray,
    G_line: np.ndarray,
    lam1: float,
    lamB: float,
    Nd: int = 1,
    LW_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 1D level set representation along an interface-normal line.

    Input:
    - x0: starting x-coordinate
    - y0: starting y-coordinate
    - n: length-2 normal vector
    - t_line: Nt line parameters used to compute x(t) and y(t)
    - q_line_fit: Nt field values sampled along the line
    - G_line: 2 by Nt gradient components sampled along the line
    - lam1: regularization parameter to promote smoothness
    - lamB: regularization parameter to enforce boundary conditions
    - Nd: number of interfaces to fit
    - LW_in: optional length-Nd array of prescribed interface widths

    Output:
    - LF: (Nd + 1) by Nt array of level set extensions
    - LS: Nd by Nt array of level set functions
    - LW: length-Nd array of fitted interface widths
    - xs: length-Nd array of interface x-coordinates
    - ys: length-Nd array of interface y-coordinates
    - mask: length-Nt Boolean mask for the fitted interface region
    """
    
    # get normal vector
    n = np.asarray(n, dtype=float).reshape(2)
    n /= np.linalg.norm(n)

    # get sizes
    Nt = t_line.shape[0]
    assert(G_line.shape[0]==2)
    assert(G_line.shape[1]==Nt)
    assert(q_line_fit.shape[0]==Nt)
    assert(q_line_fit.ndim==1)
    if LW_in is not None:
        assert(LW_in.shape[0]==Nd)

    # directional derivative along n
    s1_line_pm = np.sum(G_line*n[None,:,None],axis=1).flatten()
    s1_line = np.abs(s1_line_pm)

    # local maxima
    ldxMax_s1_all = argrelextrema(s1_line, np.greater)[0]
    assert(len(ldxMax_s1_all)>0)
    idxSort = np.argsort(s1_line[ldxMax_s1_all])[::-1]
    ldxLS = np.sort(ldxMax_s1_all[idxSort[:Nd]])

    # compute minima (helper function)
    ldxMin = levelSet1d.chooseIdxMin(s1_line_pm,ldxLS)
    assert np.all(np.diff(ldxMin) >= 0), "Min indices not sorted"

    # fit
    ldxMinMat = ldxMin.reshape([Nd,2])
    LF, LS, LW, ts, _ = levelSet1d_optim.train_inner(
        t_line,
        q_line_fit,
        s1_line_pm,
        ldxMinMat,
        Nd,
        lam1=lam1,
        lamB=lamB,
        LW_in=LW_in
    )
    xs = x0 + ts * n[0]
    ys = y0 + ts * n[1]
    
    # get local min in x and y
    mask = np.zeros(Nt,dtype='bool')
    mask[ldxMin[0]:ldxMin[1]]=1
    
    return LF, LS, LW, xs, ys, mask


def compute_centroid(
    xy_pts: np.ndarray,
) -> np.ndarray:
    """Compute the area centroid of a closed polygonal contour.

    Input:
    - xy_pts: Np by 2 array of ordered contour coordinates
    
    Output:
    - centroid: length-2 array [x_center, y_center]
    """
    x = xy_pts[:,0]
    y = xy_pts[:,1]
    x_next = np.roll(x,-1)
    y_next = np.roll(y,-1)
    cross = x * y_next - x_next * y
    A = 0.5 * np.sum(cross)
    Cx = np.sum((x + x_next) * cross) / (6 * A)
    Cy = np.sum((y + y_next) * cross) / (6 * A)
    return np.array([Cx,Cy])


def flip_normal_sign(
    xy_pts: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    """Orient all contour normals away from the contour centroid.

    Input:
    - xy_pts: Np by 2 array of points on the contour
    - n: Np by 2 array of consistently inward- or outward-facing normals

    Output:
    - n_outward: Np by 2 array of outward-facing normal vectors
    """
    xyc = compute_centroid(xy_pts)
    is_outward = np.sum(n*(xyc-xy_pts), axis=1) < 0
    if not (np.all(is_outward) or np.all(is_outward==False)):
        raise ValueError("Normal direction not defined correctly")
    if np.all(is_outward==False):
        n = -n
    return n


def trace_contour(
    tangent_sign: int,
    ldxMax: int,
    max_steps: int,
    s1: np.ndarray,
    s1_norm: np.ndarray,
) -> list[int]:
    """Trace an interface through neighboring gradient-magnitude maxima.

    Input:
    - tangent_sign: +1 or -1 (used to choose starting direction)
    - ldxMax: flattened index of the starting gradient maximum
    - max_steps: maximum number of steps along contour
    - s1: Ny by Nx by 2 gradient array
    - s1_norm: Ny by Nx gradient-magnitude array

    Output:
    - contour_ldx: flattened grid indices visited along the contour
    """
    contour_ldx = []
    visited = {ldxMax}
    this_ldx = ldxMax
    contour_ldx.append(ldxMax)

    for _ in range(max_steps):
        next_ldx = get_quadrant_ldx_max(
            this_ldx, s1, s1_norm, tangent_sign=tangent_sign
        )
        if next_ldx == this_ldx:
            next_ldx = get_quadrant_ldx_max(
                this_ldx, s1, s1_norm, tangent_sign=tangent_sign
            )

        if next_ldx == this_ldx or next_ldx in visited:
            break

        contour_ldx.append(next_ldx)
        visited.add(next_ldx)
        this_ldx = next_ldx

    return contour_ldx


def get_contour(
    ldxMax: int,
    s1: np.ndarray,
    s1_norm: np.ndarray,
    max_steps: int | None = None,
) -> list[int]:
    """Trace a contour beginning at a gradient-magnitude maximum.

    Input:
    - ldxMax: flattened index of the starting gradient maximum
    - s1: Ny by Nx by 2 gradient array
    - s1_norm: Ny by Nx gradient-magnitude array
    - max_steps: maximum number of contour steps, or None for the grid size

    Output:
    - contour_ldx: flattened grid indices visited along the contour
    """
    if max_steps is None:
        max_steps = s1_norm.size
    return trace_contour(1,ldxMax,max_steps,s1,s1_norm)


def get_quadrant_ldx_max(
    ldxMax: int,
    s1: np.ndarray,
    s1_norm: np.ndarray,
    tangent_sign: int = 1,
) -> int:
    """Select the strongest neighboring point in a tangent-directed quadrant.

    Input:
    - ldxMax: flattened index of the current point
    - s1: Ny by Nx by 2 gradient array
    - s1_norm: Ny by Nx gradient-magnitude array
    - tangent_sign: tangent direction, normally +1 or -1

    Output:
    - next_ldx: flattened index of the strongest eligible neighbor, or ldxMax
    """
    # get normal direction at max sensor location
    n = np.array([
        s1[:, :, 0].flat[ldxMax],
        s1[:, :, 1].flat[ldxMax],
    ], dtype=float)

    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        return ldxMax

    n /= n_norm

    # tangent direction
    t = tangent_sign * np.array([-n[1], n[0]])

    j, i = np.unravel_index(ldxMax, s1_norm.shape)

    tx, ty = t

    dj = int(np.sign(ty))
    di = int(np.sign(tx))

    quadrant_nodes = []

    for off in [(dj, 0), (0, di), (dj, di)]:
        jj = j + off[0]
        ii = i + off[1]

        if 0 <= jj < s1.shape[0] and 0 <= ii < s1.shape[1]:
            quadrant_nodes.append((jj, ii))

    if len(quadrant_nodes) == 0:
        return ldxMax

    quadrant_ldx = np.array([
        np.ravel_multi_index(node, s1_norm.shape)
        for node in quadrant_nodes
    ])

    quadrant_ldx_max = quadrant_ldx[np.argmax(s1_norm.flat[quadrant_ldx])]

    return quadrant_ldx_max


def interpLineToGrid(
    line_data,
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate fitted interface-normal line data onto the 2D grid.

    Input:
    - line_data: list of fitted line-data dictionaries
    - q: Nv by Ny by Nx array of fields
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates

    Output:
    - LF: Nv by (Nd + 1) by Ny by Nx array of level set extensions
    - LS: Nd by Ny by Nx array of shared level set functions
    - LW_out: length-Nd array of fitted interface widths
    """

    # get size
    Ny, Nx = x.shape
    Nv = q.shape[0]
    assert(y.shape==(Ny,Nx))
    assert(q.shape==(Nv,Ny,Nx))
    
    # verify line_data
    assert(type(line_data)==list)
    Nl = len(line_data)

    Nd = 0
    # get number of discontinuities
    for ldx in range(Nl):
        thisNd = line_data[ldx]["LS_line"].shape[1]
        Nd += thisNd

    # verify sizes
    LW_out = np.zeros(Nd)
    for ldx in range(Nl):
        Np, NtMax = line_data[ldx]["x_line"].shape
        thisNd = line_data[ldx]["LS_line"].shape[1]
        assert(line_data[ldx]["x_line"].shape==(Np,NtMax))
        assert(line_data[ldx]["y_line"].shape==(Np,NtMax))
        assert(line_data[ldx]["LF_line"].shape==(Np,thisNd+1,NtMax,Nv))
        assert(line_data[ldx]["LS_line"].shape==(Np,thisNd,NtMax))
        assert(line_data[ldx]["xs_line"].shape==(Np,thisNd))
        assert(line_data[ldx]["ys_line"].shape==(Np,thisNd))
        assert(len(line_data[ldx]["LW_out"])==thisNd)
        LW_out[np.arange(0,thisNd)+ldx] = line_data[ldx]["LW_out"]

    # assemble LF and LS
    LF = np.full((Nd+1,Ny,Nx,Nv),np.nan)
    LS = np.full((Nd  ,Ny,Nx),np.nan)
    dCount = 0
    for ldx in range(Nl):

        # extract data
        x_line = line_data[ldx]["x_line"]
        y_line = line_data[ldx]["y_line"]
        LF_line = line_data[ldx]["LF_line"]
        LS_line = line_data[ldx]["LS_line"]
        xs_line = line_data[ldx]["xs_line"]
        ys_line = line_data[ldx]["ys_line"]

        # get sizes
        Np, NtMax = line_data[ldx]["x_line"].shape
        thisNd = line_data[ldx]["LS_line"].shape[1]

        # permute line objects
        x_line = np.permute_dims(x_line,(1,0))
        y_line = np.permute_dims(y_line,(1,0))
        LF_line = np.permute_dims(LF_line,(1,2,0,3))
        LS_line = np.permute_dims(LS_line,(1,2,0))

        # handle boundary points
        xb_line = x_line[[0, -1],:]
        yb_line = y_line[[0, -1],:]
        ldxB = np.ravel_multi_index(
            (np.array([[0], [NtMax - 1]]), np.arange(Np)[None, :]),
            x_line.shape
        )
        assert(np.max(np.abs(x_line.ravel()[ldxB]-xb_line)) < 1e-12)
        assert(np.max(np.abs(y_line.ravel()[ldxB]-yb_line)) < 1e-12)

        # KNN for nearest-neighbour interpolation
        tree = cKDTree(np.column_stack([x_line.ravel(),y_line.ravel()]))
        knnIdx = tree.query(
            np.column_stack([x.ravel(),y.ravel()]))[1].reshape((Ny,Nx))
        knnBool = np.zeros((2,Ny,Nx),dtype=bool)
        for i in range(2):
            knnBool[i] = np.isin(knnIdx.ravel(),ldxB[i]).reshape((Ny,Nx))

        # determine boundaries
        outPoly = np.zeros((2,Ny,Nx),dtype='bool')
        outPoly[0][knnBool[0]] =  Path(
            np.column_stack([xb_line[0],yb_line[0]])
        ).contains_points(
            np.column_stack([x[knnBool[0]],y[knnBool[0]]]), radius=-1e-12
        )
        outPoly[1][knnBool[1]] = ~Path(
            np.column_stack([xb_line[1],yb_line[1]])
        ).contains_points(
            np.column_stack([x[knnBool[1]],y[knnBool[1]]]), radius=-1e-12
        )

        # prepare for scattered interpolation
        inPoly = ~(outPoly[0] | outPoly[1])

        xy_line = np.column_stack([x_line.ravel(),y_line.ravel()])
        xInPoly = x[inPoly]
        yInPoly = y[inPoly]
        xy_eval = np.column_stack((xInPoly,yInPoly))

        # extract field values for RBF
        field_vals = np.permute_dims(
            LF_line.reshape(thisNd+1,Np*NtMax,Nv),(1,0,2)
        )
        field_vals_interp = field_vals.reshape(
            (field_vals.shape[0],(thisNd+1)*Nv)
        )
        rbf = RBFInterpolator(
            xy_line,
            field_vals_interp,
            kernel="multiquadric",
            epsilon=1.0,
            smoothing=1e-3,
            neighbors=32,
        )
        field_eval = rbf(xy_eval).reshape(-1,thisNd+1,Nv)

        for ddx in range(thisNd+1):

            # handle level set function (LS)
            if ddx < thisNd:
                LS[dCount+ddx] = signed_distance_function(
                    x, y, xs_line[:,ddx], ys_line[:,ddx])
        
            for vdx in range(Nv):
                # interpolate LF using RBF
                thisLF = np.full((Ny, Nx), np.nan)
                thisLF[inPoly] = field_eval[:,ddx,vdx]

                # keep original function values
                if dCount + ddx == 0:
                    thisLF[knnBool[0]] = q[vdx][knnBool[0]]
                elif dCount + ddx == Nd:
                    thisLF[knnBool[1]] = q[vdx][knnBool[1]]

                # extend
                mask = np.isnan(thisLF)
                if np.any(mask):
                    thisLF[mask] = field_vals[knnIdx[mask],ddx,vdx]

                    for _ in range(100):
                        avg = np.empty_like(thisLF)
                        avg[:] = thisLF

                        avg[1:-1, 1:-1] = (
                            thisLF[:-2, 1:-1] +
                            thisLF[2:, 1:-1] +
                            thisLF[1:-1, :-2] +
                            thisLF[1:-1, 2:]
                        ) * 0.25

                        avg[0, 1:-1] = (
                            thisLF[1, 1:-1] + thisLF[0, :-2] + thisLF[0, 2:]
                        ) / 3.0
                        avg[-1, 1:-1] = (
                            thisLF[-2, 1:-1] + thisLF[-1, :-2] + thisLF[-1, 2:]
                        ) / 3.0
                        avg[1:-1, 0] = (
                            thisLF[:-2, 0] + thisLF[2:, 0] + thisLF[1:-1, 1]
                        ) / 3.0
                        avg[1:-1, -1] = (
                            thisLF[:-2, -1] + thisLF[2:, -1] + thisLF[1:-1, -2]
                        ) / 3.0

                        avg[0, 0] = 0.5 * (
                            thisLF[1, 0] + thisLF[0, 1]
                        )
                        avg[0, -1] = 0.5 * (
                            thisLF[1, -1] + thisLF[0, -2]
                        )
                        avg[-1, 0] = 0.5 * (
                            thisLF[-2, 0] + thisLF[-1, 1]
                        )
                        avg[-1, -1] = 0.5 * (
                            thisLF[-2, -1] + thisLF[-1, -2]
                        )

                        delta = np.max(
                            np.abs(avg[mask] - thisLF[mask])
                        ) / np.max(np.abs(q[vdx]))
                        thisLF[mask] = avg[mask]

                        if delta < 1e-6:
                            break


                # store LF
                LF[dCount+ddx,:,:,vdx] = thisLF


        # update interface count
        dCount += thisNd


    # permute LF
    LF = np.permute_dims(LF,(3,0,1,2))    


    return LF, LS, LW_out


def polygon_area(
    xs_line: np.ndarray,
    ys_line: np.ndarray,
) -> float:
    """Compute the signed area enclosed by a polygonal contour.

    Input:
    - xs_line: length-Np array of contour x-coordinates
    - ys_line: length-Np array of contour y-coordinates

    Output:
    - area: contour area
    """
    
    x = np.asarray(xs_line).ravel()
    y = np.asarray(ys_line).ravel()
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def signed_distance_function(
    x: np.ndarray,
    y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Approximate the signed distance (measure distance to contour)

    Input:
    - x: Ny by Nx array of x-coordinates on grid
    - y: Ny by Nx array of y-coordinates on grid
    - xs: length-Np array of contour x-coordinates
    - ys: length-Np array of contour y-coordinates
    - k: number of nearby contour segments tested per grid point

    Output:
    - sdf: Ny by Nx signed-distance array, negative inside the contour
    """

    # verify sizes
    assert(xs.ndim==1)
    assert(ys.ndim==1)
    Np = xs.shape[0]
    assert(ys.shape[0]==Np)
    assert(x.ndim==2)
    assert(y.ndim==2)
    Ny, Nx = x.shape
    assert(y.shape==(Ny,Nx))

    # Flatten grid points
    P = np.column_stack((x.ravel(), y.ravel()))

    # get consecutive points on contour and compute their difference
    A = np.column_stack([xs[ :-1], ys[ :-1]])
    B = np.column_stack([xs[1:  ], ys[1:  ]])
    diffAB = B - A

    # compute closest midpoint to each point P
    tree = cKDTree(0.5 * (A + B))
    _, seg_idx = tree.query(P, k=min(k, len(A)))
    if seg_idx.ndim == 1:
        seg_idx = seg_idx[:, None]

    # compute shortest distance to contour
    dmin = np.full(P.shape[0], np.inf)
    for j in range(seg_idx.shape[1]):
        idx = seg_idx[:, j]

        # current point
        Aj = A[idx]

        # project onto contour
        ABj = diffAB[idx]
        AP = P - Aj
        denom = np.sum(ABj**2, axis=1)
        t = np.sum(AP * ABj, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        C = Aj + t[:, None] * ABj

        # calculate the shortest distance to the contour
        d = np.linalg.norm(P - C, axis=1)

        # take minimum value (smallest of d and current dmin)
        dmin = np.minimum(dmin, d)

    # determine whether a point is inside or outside the contour
    inside = Path(np.column_stack([xs, ys])).contains_points(P)

    # assign negative value if inside the contour
    dmin[inside] *= -1.0

    return dmin.reshape((Ny,Nx))


def eval(
    LF: np.ndarray,
    LS: np.ndarray,
    LW: np.ndarray,
    NdVec: np.ndarray,
) -> np.ndarray:
    """Reconstruct physical variables from their level set representations.

    Input:
    - LF: Nv by (Nd + 1) by Ny by Nx array of level set extensions
    - LS: Nv by Nd by Ny by Nx array of level set functions
    - LW: Nv by Nd array of interface widths
    - NdVec: length-Nv array containing the number of events per field

    Output:
    - q: Nv by Ny by Nx reconstructed fields
    """

    # verify number of variables
    Nv = LF.shape[0]
    assert(len(NdVec)==Nv)
    assert(LS.shape[0]==Nv)
    assert(LW.shape[0]==Nv)

    # verify number of levels
    Nd = np.max(NdVec)
    assert(LF.shape[1]==Nd+1)
    assert(LS.shape[1]==Nd)
    assert(LW.shape[1]==Nd)

    # verify grid size
    Ny = LF.shape[2]
    Nx = LF.shape[3]
    assert(LS.shape[2]==Ny)
    assert(LS.shape[3]==Nx)
    
    q = np.zeros((Nv,Ny,Nx))
    for vdx in range(Nv):
        thisNd = NdVec[vdx]
        q[vdx] = evalInner(LF[vdx,:thisNd+1],LS[vdx,:thisNd],LW[vdx,:thisNd])

    return q


def evalInner(
    LF: np.ndarray,
    LS: np.ndarray,
    LW: np.ndarray,
) -> np.ndarray:
    """Reconstruct one field from its level set representation.

    Input:
    - LF: (Nd + 1) by Ny by Nx array of level set extensions
    - LS: Nd by Ny by Nx array of level set functions
    - LW: length-Nd array of interface widths

    Output:
    - q: Ny by Nx reconstructed field
    """
    
    # verify number of levels
    Nd = LS.shape[0]
    assert(LF.shape[0]==Nd+1)
    assert(len(LW)==Nd)

    # verify grid
    Ny = LF.shape[1]
    Nx = LF.shape[2]
    assert(LS.shape[1]==Ny)
    assert(LS.shape[2]==Nx)

    # compute Heaviside function
    He = (np.tanh(LS/LW.reshape(Nd,1,1)) + 1) / 2

    tanh_member = np.zeros((Nd+1,Ny,Nx,))
    for fdx in range(Nd+1):
        mask = np.ones((Ny,Nx))
        for jdx in range(Nd):
            if jdx >= fdx:
                mask *= (1 - He[jdx])
            else:
                mask *= He[jdx]
        tanh_member[fdx] = mask

    # Sum contributions
    q = np.sum(tanh_member*LF,axis=0)

    return q


def EnKF(
        prim: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        y_obs: np.ndarray,
        sigma_obs: np.ndarray,
        NdVec: np.ndarray,
        OBS_X: np.ndarray,
        OBS_Y: np.ndarray,
        rng: np.random,
        lam1: float = 1e-2,
        lamB: float = 1e-2,
        alignVecLF: np.ndarray | None = None,
        workers : int | None = None,
        sharp_bool : bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
    """Apply an ensemble Kalman filter update in level set space.

    1) Compute level set representations
    2) Registration of level set functions
    3) Registration of rarefaction waves
    4) EnKF in level set space
    5) Invert registration
    6) Transform back to physical space

    Input:
    - prim: Nv by Ny by Nx forecast, optionally with Ne ensemble members
    - x: Ny by Nx array of x-coordinates
    - y: Ny by Nx array of y-coordinates
    - y_obs: length-Nobs array of observation values
    - sigma_obs: length-Nobs array of observation-error standard deviations
    - NdVec: length-Nv array containing the number of events per variable
    - OBS_X: array containing Nobs sensor x-coordinates
    - OBS_Y: array containing Nobs sensor y-coordinates
    - rng: NumPy random-number generator
    - lam1: regularization parameter for smoothness
    - lamB: regularization parameter for enforcing boundary conditions
    - alignVecLF: optional length-(Nd + 1) Boolean mask selecting level set
      extensions for registration
    - workers: number of parallel fitting workers, or None for the default
    - sharp_bool: whether to use the sharpest median width for each variable

    Output:
    - prim_a: Nv by Ny by Nx by Ne analysis ensemble
    - prim_f_LS: Nv by Ny by Nx by Ne reconstructed forecast ensemble
    """

    Nd = int(np.max(NdVec))
    Ny,Nx = x.shape
    assert(y.shape==x.shape)
    Nv = prim.shape[0]
    assert(prim.shape[1]==Ny)
    assert(prim.shape[2]==Nx)
    assert(len(NdVec)==Nv)
    if alignVecLF is None:
        alignVecLF = np.zeros(Nd+1)
    assert(len(alignVecLF)==Nd+1)
    if prim.ndim == 3:
        prim = prim[:,:,:,np.newaxis]
    Ne = prim.shape[3]

    # verify observation sizes
    y_obs = np.asarray(y_obs).ravel()
    sigma_obs = np.asarray(sigma_obs).ravel()
    OBS_X = np.asarray(OBS_X)
    OBS_Y = np.asarray(OBS_Y)

    n_obs = y_obs.size
    if not (
        sigma_obs.size == n_obs
        and OBS_X.size == n_obs
        and OBS_Y.size == n_obs
    ):
        raise ValueError(
            "y_obs, sigma_obs, OBS_X, and OBS_Y must contain the same "
            f"number of observations; got {n_obs}, "
            f"{np.asarray(sigma_obs).size}, {np.asarray(OBS_X).size}, "
            f"and {np.asarray(OBS_Y).size}"
        )

    # intialize
    LF_f   = np.zeros((Nv,Nd+1,Ny,Nx,Ne))
    LS_f   = np.zeros((Nv,Nd  ,Ny,Nx,Ne))

    # 1. Fit level sets, reusing one worker pool across all members.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        print("Fitting ensemble {} of {}".format(1, Ne))
        LF_f[:,:,:,:,0],LS_f[:,:,:,:,0],LW0 = train(
            prim[:,:,:,0],x,y,NdVec,lam1=lam1,lamB=lamB,
            workers=workers,executor=executor)
        for i in range(1,Ne):
            print("Fitting ensemble {} of {}".format(i+1, Ne))
            LF_f[:,:,:,:,i],LS_f[:,:,:,:,i],_ = train(
                prim[:,:,:,i],x,y,NdVec,lam1=lam1,lamB=lamB,LW_in=LW0,
                workers=workers,executor=executor)

    # fitting complete
    print("Fitting complete")

    if sharp_bool:
        LW = LW0.copy()
        LW[LW < 1e-12] = np.nan
        LW_min = np.nanmin(LW, axis=1)
        LW = np.broadcast_to(LW_min[:, None], LW.shape).copy()
        LW[np.isnan(LW0)] = np.nan
    else:
        LW = LW0

    # 2 Get Observations from Level Set
    primLS = np.zeros_like(prim)
    y_preds = np.zeros((len(y_obs),Ne))
    obs_yx = np.column_stack((OBS_Y.ravel(),OBS_X.ravel()))
    for i in range(Ne):
        primLS[:,:,:,i] = eval(
            LF_f[:,:,:,:,i],
            LS_f[:,:,:,:,i],
            LW,
            NdVec
        )
        p_i = primLS[3,:,:,i]
        p_interp = RegularGridInterpolator(
            (y[:,0],x[0,:]),
            p_i,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        y_preds[:,i] = p_interp(obs_yx)

    # 3. Registration (LS)
    r_LS_f = np.full((Nv,Nd,3,Ne), np.nan)
    LS_align_f = LS_f.copy()
    for ndx in range(Nd):
        for vdx in range(Nv):
            LS0 = LS_f[vdx,ndx,:,:,0]

            if np.all(np.isnan(LS0)):
                continue
            
            # iterate through ensemble members
            for i in range(Ne):

                # get level set function
                LSi = LS_f[vdx,ndx,:,:,i]

                # compute center
                ci, ri = align2d_LS.ls_center(LSi,x,y)

                # store parameters of transformation
                r_LS_f[vdx,ndx,0:2,i] = ci
                r_LS_f[vdx,ndx,2,i] = ri
                if i == 0:
                    c0 = ci.copy()

                # translate the circle
                LS_align_f[vdx,ndx,:,:,i] = align2d_LS.warp_radial_2d(
                    LSi,x,y,c0,ci
                )
                

    # 4. Registration (LF)
    r_LF_f = np.full((Nd+1,2,Ne), np.nan)
    LF_align_f = LF_f.copy()
    for ndx in range(Nd+1):
        if alignVecLF[ndx]:
            for i in range(Ne):

                # get center of circle and radii
                if ndx < Nd:
                    c0 = r_LS_f[0,ndx,0:2,0]
                    r0 = r_LS_f[0,ndx,2,0]
                    ci = r_LS_f[0,ndx,0:2,i]
                    ri = r_LS_f[0,ndx,2,i]
                else:
                    c0 = r_LS_f[0,ndx-1,0:2,0]
                    r0 = r_LS_f[0,ndx-1,2,0]
                    ci = r_LS_f[0,ndx-1,0:2,i]
                    ri = r_LS_f[0,ndx-1,2,i]
                
                # align
                (
                    ai,
                    bi,
                    LF_align_f[:,ndx,:,:,i],
                ) = align2d.align_affine_profile_2d(
                    LF_f[:,ndx,:,:,0],
                    LF_f[:,ndx,:,:,i],
                    x,
                    y,
                    c0,
                    ci,
                    r0,
                    ri,
                )
                r_LF_f[ndx,0,i] = ai
                r_LF_f[ndx,1,i] = bi
    
    # 5. Create state vector
    X_LF_f = LF_align_f.reshape((Nv*(Nd+1)*Ny*Nx,Ne))
    X_LS_f = LS_align_f.reshape((Nv*(Nd+0)*Ny*Nx,Ne))
    X_r_LF_f = r_LF_f.reshape((2*(Nd+1),Ne))
    X_r_LS_f = r_LS_f.reshape((Nv*Nd*3,Ne))
    state_f_all = np.vstack([
        X_LF_f,
        X_LS_f,
        X_r_LF_f,
        X_r_LS_f
    ])
    state_mask = ~np.isnan(state_f_all[:,0])
    assert(np.all(np.isnan(state_f_all[~state_mask,1:])))
    state_f = state_f_all[state_mask,:]
    assert(np.any(np.isnan(state_f)) == 0)

    # Split row counts
    n_LF   = Nv * (Nd + 1) * Ny * Nx
    n_LS   = Nv * Nd       * Ny * Nx
    n_r_LF = 2 * (Nd + 1)
    n_r_LS = Nv * Nd * 3
    i0 = 0
    i1 = i0 + n_LF
    i2 = i1 + n_LS
    i3 = i2 + n_r_LF
    i4 = i3 + n_r_LS

    # 6. EnKF Update
    obs_ens = y_obs[:, None] + rng.normal(
        0, sigma_obs[:, None], (len(y_obs), Ne)
    )
    state_a = utils_da.analysis_stochastic(
        state_f, y_preds, obs_ens, np.diag(sigma_obs**2), block_size = 1000
    )

    # 7. Extract updated state
    state_a_all = np.full((state_mask.size,Ne),np.nan)
    state_a_all[state_mask] = state_a
    X_LF_a   = state_a_all[i0:i1]
    X_LS_a   = state_a_all[i1:i2]
    X_r_LF_a = state_a_all[i2:i3]
    X_r_LS_a = state_a_all[i3:i4]
    LF_align_a = X_LF_a.reshape((Nv,Nd+1,Ny,Nx,Ne))
    LS_align_a = X_LS_a.reshape((Nv,Nd,  Ny,Nx,Ne))
    r_LF_a     = X_r_LF_a.reshape((Nd+1,2,Ne))
    r_LS_a     = X_r_LS_a.reshape((Nv,Nd,3,Ne))

    # 8. Invert the registration (LF)
    LF_a = LF_align_a.copy()
    for ndx in range(Nd + 1):
        if alignVecLF[ndx]:
            for i in range(Ne):
                ai = r_LF_a[ndx,0,i]
                bi = r_LF_a[ndx,1,i]
                if not np.isfinite(ai) or not np.isfinite(bi) or ai <= 0:
                    raise RuntimeError(
                        f"Invalid analyzed LF warp: a={ai}, b={bi}, "
                        f"ndx={ndx}, ensemble={i}"
                    )
                if ndx < Nd:
                    c0 = r_LS_f[0,ndx,0:2,0]
                    ci = r_LS_a[0,ndx,0:2,i]
                else:
                    c0 = r_LS_f[0,ndx-1,0:2,0]
                    ci = r_LS_a[0,ndx-1,0:2,i]
                LF_a[:,ndx,:,:,i] = align2d.inverse_warp_radial_2d(
                    LF_align_a[:,ndx,:,:,i],ai,bi,x,y,c0,ci
                )

    # 9. Invert the registration (LS)
    LS_a = LS_align_a.copy()
    for vdx in range(Nv):
        for ndx in range(Nd):
            for i in range(Ne):
                if np.any(np.isnan(r_LS_a[vdx,ndx,:,i])):
                    if np.any(np.isnan(r_LS_a[vdx,ndx,:,0])):
                        continue
                    else:
                        raise RuntimeError(
                            f"Problem with registration for level set function: "
                            f"vdx={vdx},ndx={ndx},ensemble={i}"
                        )
                c0 = r_LS_f[vdx,ndx,0:2,0]
                ci = r_LS_a[vdx,ndx,0:2,i]
                LS_a[vdx,ndx,:,:,i] = align2d_LS.inverse_warp_radial_2d(
                    LS_align_a[vdx,ndx,:,:,i],x,y,c0,ci
                )

    # 10. Evaluate the level set function
    prim_a = np.zeros((Nv,Ny,Nx,Ne))
    for i in range(Ne):
        prim_a[:,:,:,i] = eval(
            LF_a[:,:,:,:,i],LS_a[:,:,:,:,i],LW,NdVec
        )
    
    return prim_a,primLS
