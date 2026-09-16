# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT
#
# Based on code originally developed by Xuhui Zhou for the Neural EnKF.
# Modified by Michael Sleeman for the Level Set EnKF.


from __future__ import annotations
import re
import pyvista as pv
from pathlib import Path
from typing import Union,Dict
import numpy as np 

PathLike = Union[str, Path]


# ------------------------------------------------------------
# Read VTK and input.st files to obtain the solution data
# ------------------------------------------------------------
def load_solution(
    vtr_file: Path,
    input_st_file: Path,
) -> Dict[str, np.ndarray]:
    """
    Load a 2D solution and append its farfield boundary states.

    Interior density, velocity, and pressure are read from the VTR file. The
    outer rows and columns are filled using the state in the Farfield block of
    input.st.

    Input:
    - vtr_file: path to the VTR solution file
    - input_st_file: path to input.st containing the boundary conditions

    Output:
    - solution: dictionary with keys:
      - x: (Ny + 2) by (Nx + 2) array of x-coordinates
      - y: (Ny + 2) by (Nx + 2) array of y-coordinates
      - rho: (Ny + 2) by (Nx + 2) array of density values
      - u: (Ny + 2) by (Nx + 2) array of x-velocity values
      - v: (Ny + 2) by (Nx + 2) array of y-velocity values
      - p: (Ny + 2) by (Nx + 2) array of pressure values
    """
    # ---------- 1) Read VTR point data ----------
    sol = pv.read(str(vtr_file))

    # Arrays are defined at grid points (point_data)
    try:
        Nx = sol.dimensions[0]
        Ny = sol.dimensions[1]
        x_inner = sol.points[:, 0]
        y_inner = sol.points[:, 1]
        rho_inner = np.asarray(sol.point_data["density"])
        u_inner   = np.asarray(sol.point_data["velocity"])[:, 0]
        v_inner   = np.asarray(sol.point_data["velocity"])[:, 1]
        p_inner   = np.asarray(sol.point_data["pressure"])
    except KeyError as e:
        raise KeyError(f"Missing array in VTR: {e}. "
                       f"Available point_data: {list(sol.point_data.keys())}") from e

    # ---------- 2) Parse input.st for boundary values ----------
    text = Path(input_st_file).read_text()

    def get_block_value(label: str, key: str) -> float:
        """Extract numeric value of `key` from an `under <label> { ... }` block."""
        block_re = re.compile(rf"under\s+{label}\s*\{{(.*?)\}}", re.S)
        blk = block_re.search(text)
        if not blk:
            raise ValueError(f"Block '{label}' not found in {input_st_file}")
        body = blk.group(1)

        field_re = re.compile(rf"\b{key}\s*=\s*([-+0-9.eE]+)")
        m = field_re.search(body)
        if not m:
            raise ValueError(f"Key '{key}' not found in block '{label}'")
        return float(m.group(1))

    # Farfield boundary
    r2 = get_block_value("Farfield",  "Density")
    u2 = get_block_value("Farfield",  "VelocityX")
    v2 = get_block_value("Farfield",  "VelocityY")
    p2 = get_block_value("Farfield",  "Pressure")

    # ---------- 3) Stitch boundaries + inner domain ----------
    x = np.zeros((Ny+2,Nx+2))
    y = np.zeros((Ny+2,Nx+2))
    x[1:-1,1:-1] = x_inner.reshape(Ny,Nx)
    x[0,:] = x[1,:]
    x[-1,:] = x[-2,:]
    x[:,0] = 0.0
    x[:,-1] = 2.0
    y[1:-1,1:-1] = y_inner.reshape(Ny,Nx)
    y[:,0] = y[:,1]
    y[:,-1] = y[:,-2]
    y[0,:] = 0.0
    y[-1,:] = 2.0

    rho = np.full((Ny+2, Nx+2), r2)
    rho[1:-1, 1:-1] = rho_inner.reshape(Ny,Nx)
    u = np.full((Ny+2, Nx+2), u2)
    u[1:-1, 1:-1] = u_inner.reshape(Ny,Nx)
    v = np.full((Ny+2, Nx+2), v2)
    v[1:-1, 1:-1] = v_inner.reshape(Ny,Nx)
    p = np.full((Ny+2, Nx+2), p2)
    p[1:-1, 1:-1] = p_inner.reshape(Ny,Nx)
    
    return dict(x=x, y=y, rho=rho, u=u, v=v, p=p)

# ------------------------------------------------------------
# Write VTK (and boundary conditions) with updated fields 
# ------------------------------------------------------------
def write_vtk_analysis(
    *,
    template_vtr: Path,
    input_st_path: Path,
    x_full: np.ndarray,
    y_full: np.ndarray,
    rho_full: np.ndarray,
    u_full: np.ndarray,
    v_full: np.ndarray,
    p_full: np.ndarray,
    out_vtr_path: Path,
) -> Path:
    """
    Write analyzed 2D fields to a VTR file and update the farfield state.

    The interior values are written to a copy of the template VTR. The state
    at the first array entry is written to the Farfield block of input.st.

    Input:
    - template_vtr: VTR file providing the grid and point-data structure
    - input_st_path: input.st file whose Farfield state will be updated
    - x_full: (Ny + 2) by (Nx + 2) x-coordinates including the boundary
    - y_full: (Ny + 2) by (Nx + 2) y-coordinates including the boundary
    - rho_full: (Ny + 2) by (Nx + 2) density values including the boundary
    - u_full: (Ny + 2) by (Nx + 2) x-velocity values including the boundary
    - v_full: (Ny + 2) by (Nx + 2) y-velocity values including the boundary
    - p_full: (Ny + 2) by (Nx + 2) pressure values including the boundary
    - out_vtr_path: path at which to write the updated VTR file

    Output:
    - out_vtr_path: path to the written VTR file

    Assumptions:
    - The template contains Ny by Nx interior points.
    - Its point data include density, three-component velocity, and pressure.
    - Each full field contains one layer of farfield boundary values around
      the interior domain.
    """
    template_vtr = Path(template_vtr)
    input_st_path = Path(input_st_path)
    out_vtr_path = Path(out_vtr_path)

    if not template_vtr.is_file():
        raise FileNotFoundError(f"Template VTR not found: {template_vtr}")
    if not input_st_path.is_file():
        raise FileNotFoundError(f"input.st not found: {input_st_path}")

    # ---- Split boundaries vs inner domain (no checks on coordinates per your request)
    rho2, u2, v2, p2 = float(rho_full[0,0]), float(u_full[0,0]), float(v_full[0,0]), float(p_full[0,0])

    rho_inner = np.asarray(rho_full[1:-1,1:-1], dtype=np.float64)
    u_inner   = np.asarray(u_full[1:-1,1:-1],   dtype=np.float64)
    v_inner   = np.asarray(v_full[1:-1,1:-1],   dtype=np.float64)
    p_inner   = np.asarray(p_full[1:-1,1:-1],   dtype=np.float64)

    # ---- Load template VTR and update arrays
    grid = pv.read(str(template_vtr))

    # density
    if "density" not in grid.point_data:
        raise KeyError(f"'density' array not found in {template_vtr}")
    if grid.point_data["density"].size != rho_inner.size:
        raise ValueError(f"density size mismatch: VTR={grid.point_data['density'].size} vs inner={rho_inner.size}")
    grid.point_data["density"] = rho_inner.ravel()

    # velocity (set Vy=Vz=0, Vx=u_inner)
    if "velocity" not in grid.point_data:
        raise KeyError(f"'velocity' array not found in {template_vtr}")
    vel = np.zeros((rho_inner.size, 3), dtype=np.float64)
    vel[:, 0] = u_inner.ravel()
    vel[:, 1] = v_inner.ravel()
    grid.point_data["velocity"] = vel

    # pressure
    if "pressure" not in grid.point_data:
        raise KeyError(f"'pressure' array not found in {template_vtr}")
    if grid.point_data["pressure"].size != p_inner.size:
        raise ValueError(f"pressure size mismatch: VTR={grid.point_data['pressure'].size} vs inner={p_inner.size}")
    grid.point_data["pressure"] = p_inner.ravel()

    # ---- Save updated VTR
    out_vtr_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out_vtr_path))

    # ---- Update input.st boundaries (Inlet @ x=0, Inlet2 @ x=1)
    text = input_st_path.read_text()

    def _set_field(body: str, key: str, value: float) -> str:
        # Replace `key = number` with new value (scientific notation ok)
        pat = re.compile(rf"(\b{key}\s*=\s*)([-+0-9.eE]+)")
        return pat.sub(lambda m: f"{m.group(1)}{value:.8e}", body, count=1)

    def _replace_block_value(full_text: str, label: str, updates: dict) -> str:
        # Find `under <label> { ... }` block and replace specified fields
        block_re = re.compile(rf"(under\s+{label}\s*\{{)(.*?)(\}})", re.S)
        m = block_re.search(full_text)
        if not m:
            raise ValueError(f"Block '{label}' not found in {input_st_path}")
        start, body, end = m.group(1), m.group(2), m.group(3)
        for k, v in updates.items():
            body = _set_field(body, k, float(v))
        return full_text[:m.start()] + start + body + end + full_text[m.end():]

    # Farfield
    text = _replace_block_value(
        text, "Farfield",
        {"Density": rho2, "VelocityX": u2, "VelocityY": v2, "Pressure": p2}
    )
    
    input_st_path.write_text(text)

    return out_vtr_path
