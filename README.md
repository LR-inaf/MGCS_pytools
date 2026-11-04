# MGCS_pytools

**Scientific Python tools for the Missing Globular Cluster Survey ([D. Massari et al. 2025](https://doi.org/10.1051/0004-6361/202554007))**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/LR-inaf/MGCS_pytools)

`MGCS_pytools` is a professional Python library providing vectorized and JIT-accelerated statistical tools for kinematic proper motion analysis, adaptive Voronoi Color-Magnitude Diagram (CMD) decontamination, and iterative differential reddening correction, developed for the *Missing Globular Cluster Survey* (MGCS) HST Treasury program.

---

## Key Features & Architecture

The library is structured into three core subpackages plus an offline documentation utility:

1. **`mgcs_pytools.mcmc` — Kinematic MCMC Mixture Modeling**
   - Fits multi-component 2D Gaussian Mixture Models (GMM) convolved with per-star observational error ellipses on Vector Point Diagrams (VPD).
   - Powered by [`emcee`](https://emcee.readthedocs.io/) with optional `Numba` JIT acceleration (`bivariate_normal_pdf`, `MCMCEnsembleSampler`, `posterior`, `get_component_membership_from_mcmc_chain`).

2. **`mgcs_pytools.statistical_membership` — Photometric CMD Decontamination & Differential Reddening**
   - Decontaminates cluster CMDs from Galactic field star contamination using adaptive **Centroidal Voronoi Tessellations (CVT)** with Lloyd relaxation and cell dilation (`CVTGrid`).
   - Combines statistical subtraction (`compute_statistical_membership`, `get_membership`, `spatial_groupby`) with iterative **differential reddening ($\Delta E(B-V)$)** estimation (`compute_differential_reddening`, `ROISelector`).
   - Outputs the enriched photometric catalog with posterior membership probabilities, differential-reddening-corrected magnitudes, and $\Delta E(B-V)$ map values.

3. **`mgcs_pytools.utils.plotting` — Astrophysical Diagnostic Plotting**
   - Publication-ready diagnostic figures: Vector Point Diagrams with marginal component distributions, MCMC walker trace and corner plots, CMD Voronoi tessellations, differential reddening maps, and 4-panel membership overviews.

4. **`mgcs_pytools.docs` — Bundled Offline Documentation**
   - Includes pre-built Sphinx ReadTheDocs HTML documentation directly inside the package distribution, accessible offline via CLI (`mgcs-docs`) or Python API (`mgcs_pytools.open_docs()`).

---

## Installation

### Standard Installation (PyPI)
Requires **Python >= 3.10**:

```bash
pip install mgcs-pytools
```

### Optional Dependencies
Enable **Numba JIT acceleration** for up to ~40x faster MCMC likelihood evaluation:

```bash
pip install "mgcs-pytools[numba]"
```

For building documentation or contributing to development:

```bash
pip install "mgcs-pytools[docs,dev]"
```

---

## Quickstart

### 1. Kinematic MCMC Mixture Modeling
Fit a 2D Gaussian Mixture Model to proper motion data convolved with measurement errors:

```python
import numpy as np
from mgcs_pytools.mcmc import MCMCEnsembleSampler

# Input array: [pm_ra, pm_dec, epm_ra, epm_dec] of shape (N, 4)
N = 500
data = np.column_stack(
    [
        np.random.normal(-3.0, 0.8, N),
        np.random.normal(1.5, 0.8, N),
        np.full(N, 0.15),
        np.full(N, 0.15),
    ]
)

# Parameter bounds for 1 component (mu1, mu2, sig1, sig2, rho) + weight
qmin = np.array([-10.0, -10.0, 0.05, 0.05, -0.99, 0.0])
qmax = np.array([10.0, 10.0, 5.0, 5.0, 0.99, 1.0])

sampler = MCMCEnsembleSampler(
    data=data,
    nwalkers=32,
    nsteps=500,
    ncomp=1,
    par_per_comp=5,
    qmin=qmin,
    qmax=qmax,
    pmodel="bivariate_normal",
)
sampler.run_emcee()

# Retrieve median parameters (50th percentile)
params = sampler.get_model_parameters(sampler.sampler, which="percentiles", flat=True)
print("Median parameters:", params[1])
```

### 2. Photometric Membership & Differential Reddening
Run the iterative CMD decontamination and differential reddening pipeline (see [`boilerplate.py`](boilerplate.py) for a complete template):

```python
from mgcs_pytools.statistical_membership import compute_statistical_membership
from mgcs_pytools.utils import plotting as utplot

dr_params = {
    "rband1": 3.1,  # Extinction coefficient in band 1
    "rband2": 2.5,  # Extinction coefficient in band 2
    "TO_mag": 17.5,  # Turn-off magnitude
    "TO_color": 0.9,  # Turn-off color
    "nref": 25,  # Number of nearest reference stars
    "ord_step": 0.4,  # Fiducial ridge line bin step
}

df_output = compute_statistical_membership(
    df_cluster=df_cluster,
    df_field=df_field,
    target_mag_col=["mag_b", "mag_v"],
    target_mag_col_error=["mag_b_err", "mag_v_err"],
    field_mag_col=["mag_b", "mag_v"],
    dr_params=dr_params,
    parallel_fov=0.16,
    racol="RA",
    deccol="DEC",
    member_threshold=0.8,
    membership_method="analytical",
)

# Visualize 4-panel membership overview
fig = utplot.plot_membership_overview(
    btarget=df_output["mag_b_drcorr"],
    vtarget=df_output["mag_v_drcorr"],
    bfield=df_field["mag_b"],
    vfield=df_field["mag_v"],
    membership=df_output["membership"],
)
```

---

## Offline Documentation

Full API reference and step-by-step tutorials (with visual examples) are bundled inside the package and can be opened offline at any time:

```bash
# Via CLI script
mgcs-docs

# Print path to index.html without opening a browser
mgcs-docs --path

# Rebuild HTML documentation from source (requires [docs] extras)
mgcs-docs --build
```

Or directly from Python:

```python
import mgcs_pytools

mgcs_pytools.open_docs()
```

---

## Development & Verification

This repository uses [`uv`](https://docs.astral.sh/uv/) for dependency management and quality gates:

```bash
# 1. Code formatting check
uv run ruff format --check .

# 2. Linter
uv run ruff check .

# 3. Static type checking
uv run mypy .

# 4. Unit test suite
uv run pytest -v
```

---

## Contributing & Issue Reporting

Contributions are very welcome! To contribute:
1. Clone this repository and create a feature branch.
2. Implement your changes and ensure all verification checks (`ruff`, `mypy`, `pytest`) pass.
3. Open a Pull Request.

If you encounter any bugs or have feature requests, please open an issue on the [GitHub Issue Tracker](https://github.com/LR-inaf/MGCS_pytools/issues).

---

## Contact & Citation

For questions or inquiries, please contact:
- **Luca Rosignoli** — [luca.rosignoli@inaf.it](mailto:luca.rosignoli@inaf.it)

If you use `MGCS_pytools` in your scientific publication, please cite the Missing Globular Cluster Survey paper:
> D. Massari et al. (2025), *A&A*, [doi:10.1051/0004-6361/202554007](https://doi.org/10.1051/0004-6361/202554007)
