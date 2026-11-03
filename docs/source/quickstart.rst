Quickstart Guide
================

This quickstart guide demonstrates basic usage of the kinematic MCMC modeling and statistical decontamination modules.

1. Kinematic MCMC Mixture Modeling
----------------------------------
Fit a 2D Gaussian Mixture Model to proper motion data convolved with measurement errors:

.. code-block:: python

   import numpy as np
   from mgcs_pytools.mcmc import MCMCEnsembleSampler
   from mgcs_pytools.utils.plotting import plot_vpd_and_mdist_component

   # Synthetic data: [pm_ra, pm_dec, epm_ra, epm_dec]
   N = 500
   pm_ra = np.random.normal(loc=-3.0, scale=0.8, size=N)
   pm_dec = np.random.normal(loc=1.5, scale=0.8, size=N)
   epm_ra = np.full(N, 0.15)
   epm_dec = np.full(N, 0.15)
   data = np.column_stack([pm_ra, pm_dec, epm_ra, epm_dec])

   # Setup parameter bounds for 1 component (mu1, mu2, sig1, sig2, rho) + weight
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

   # Retrieve best-fit parameters (50th percentile)
   params = sampler.get_model_parameters(sampler.sampler, which="percentiles", flat=True)
   print("Median parameters:", params[1])

2. Voronoi CVT Photometric Decontamination
------------------------------------------
Compute an adaptive Centroidal Voronoi Tessellation (CVT) grid over a Color-Magnitude Diagram:

.. code-block:: python

   import numpy as np
   from mgcs_pytools.statistical_membership import CVTGrid
   from mgcs_pytools.utils.plotting import show_cmds_with_voronoi

   # Color (B - V) and Magnitude (V)
   color = np.random.uniform(0.5, 2.5, 800)
   mag = np.random.uniform(14.0, 22.0, 800)
   points = np.column_stack([color, mag])

   # Generate a Lloyd-relaxed CVT grid
   cvt = CVTGrid(points, which="dilation", iter=3, target_median=5)
   print(f"Constructed {len(cvt.grid)} Voronoi cells")

3. Viewing Offline Documentation
--------------------------------
You can open this complete documentation in your browser at any time:

.. code-block:: python

   import mgcs_pytools

   mgcs_pytools.open_docs()
