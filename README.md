# LevelSetEnKFpy

This code implements the level set ensemble Kalman filter (EnKF), which enables
data assimilation in compressible flows with shocks and other sharp
discontinuities. The repository fits the level set representation and applies
the EnKF to the resulting level set representation using Python code, while it
uses M2C, an open source solver for compressible flows (Developer’s repository
link:
https://github.com/kevinwgy/m2c; https://doi.org/10.1016/j.cpc.2026.110023).

We apply the level set EnKF to the following flows:

- Sod's shock tube (1D)
- Toro's shock tube (1D)
- Shu-Osher shock-entropy problem (1D)
- Blast wave problem (2D)


## How it works

For each experiment, the code copies a case template into one directory per
ensemble member, runs M2C to advance the flow, reads the resulting VTK files,
performs a level set fit, followed by the EnKF analysis step, and writes the
updated state back for the next forecast. Summary data are written as HDF5 files
in the repository root.

Experiment settings such as ensemble size, observation locations, noise, random
seed, assimilation schedule, and regularization parameters are defined near the
top of each entry-point script.


## Requirements

The code requires:

- M2C (https://github.com/kevinwgy/m2c)
- MPI
- Python 3
- NumPy
- SciPy
- h5py
- PyVista
- scikit-image
- Matplotlib
- CMake, a C++ compiler, and Make
- MATLAB for the optional plotting scripts under `Plot/`


## Plotting

MATLAB scripts under `Plot/Plot1D` and `Plot/Plot2D` read the generated HDF5
files. Run them from their respective directories so their relative paths point
to the repository root.


## License and third-party code

This is a mixed-license repository:

- The Level Set EnKF code is authored by Michael Sleeman and is available under
  the [MIT License](LICENSE.txt).
- Some of the code in this repository is based on the Neural EnKF code authored
  by Xuhui Zhou. The Neural EnKF is authorized for distribution under the MIT
  License. In accordance with this license, some of this code has been modified
  by Michael Sleeman for the Level Set EnKF. Each file indicates whether it was
  originally written by Xuhui Zhou and modified by Michael Sleeman, or whether
  it was originally authored by Michael Sleeman.
- The M2C-derived `UserDefinedState.cpp` files are copyright the Multiphysics
  Modeling and Computation (M2C) Lab and are available under the
  [GNU General Public License, version 3 only](https://www.gnu.org/licenses/gpl-3.0.txt).
- M2C itself is an external dependency and is not distributed as part of this
  repository. Its own license applies when it is obtained or used.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the affected files and
attribution details. The MIT License does not apply to, replace, or override the
GPLv3 terms for the M2C-derived files.
