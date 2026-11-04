MGCS_pytools Documentation
==========================

**MGCS_pytools** is a Python library developed for the *Missing Globular Cluster Survey* (`D. Massari et al. 2024 <https://doi.org/10.1051/0004-6361/202554007>`_).
It provides robust, vectorized, and accelerated statistical tools for:

1. **Kinematic MCMC Mixture Modeling** (:mod:`mgcs_pytools.mcmc`):
   Fitting multi-component 2D Gaussian Mixture Models (GMM) convolved with per-star observational uncertainties on Proper Motion distributions (Vector Point Diagrams). Accelerated via `Numba` JIT and `emcee`.

2. **Statistical Photometric Membership Decontamination** (:mod:`mgcs_pytools.statistical_membership`):
   Decontaminating cluster Color-Magnitude Diagrams (CMD) from field star contamination using adaptive Centroidal Voronoi Tessellations (CVT), Lloyd relaxation, and iterative differential reddening (:math:`\Delta E(B-V)`) map corrections.

3. **Astrophysical Diagnostic Plotting** (:mod:`mgcs_pytools.utils.plotting`):
   Production-ready publication figures: Vector Point Diagrams with marginal model distributions, MCMC trace and corner plots, CMD Voronoi tessellations, differential reddening maps, and spatial membership diagnostics.

Offline Documentation Access
----------------------------
This documentation is bundled inside the Python distribution package (`.whl`).
You can open and view it offline at any time:

.. code-block:: bash

   # From the command line
   mgcs-docs

   # Or via Python module
   python -m mgcs_pytools.docs

Or programmatically in Python:

.. code-block:: python

   import mgcs_pytools

   mgcs_pytools.open_docs()

User Guide & Tutorials
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Workflows

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
