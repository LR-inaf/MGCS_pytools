"""Example: Kinematic Proper Motion Mixture Modeling with MCMCEnsembleSampler.

This script demonstrates fitting a two-component bivariate normal mixture
(representing cluster stars + field stars) to observational PM data with errors.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mgcs_pytools.mcmc import MCMCEnsembleSampler, get_component_membership_from_mcmc_chain
from mgcs_pytools.utils import plotting as utplot


def main():
    np.random.seed(42)

    # 1. Synthesize Mock Observational Data: Cluster (N=200) + Field (N=300)
    # Component 1 (Cluster): centered at (0.0, 0.0), dispersion 0.3 mas/yr
    pm_ra_cluster = np.random.normal(0.0, 0.3, 200)
    pm_dec_cluster = np.random.normal(0.0, 0.3, 200)

    # Component 2 (Field): centered at (2.0, 1.5), dispersion 1.2 mas/yr
    pm_ra_field = np.random.normal(2.0, 1.2, 300)
    pm_dec_field = np.random.normal(1.5, 1.2, 300)

    pm_ra = np.concatenate([pm_ra_cluster, pm_ra_field])
    pm_dec = np.concatenate([pm_dec_cluster, pm_dec_field])

    # Measurement uncertainties: median 0.1 mas/yr
    epm_ra = np.random.uniform(0.08, 0.12, len(pm_ra))
    epm_dec = np.random.uniform(0.08, 0.12, len(pm_dec))

    # Data array shape (N, 4): [pm_ra, pm_dec, epm_ra, epm_dec]
    data = np.column_stack([pm_ra, pm_dec, epm_ra, epm_dec])

    # 2. Setup Prior Bounds: 2 components * 5 parameters + 1 weight = 11 parameters
    # Parameters per component: [mu1, mu2, sigma1, sigma2, rho]
    # Parameter 11: weight w_1 (fraction of stars in component 1)
    qmin = np.array([-2.0, -2.0, 0.1, 0.1, -0.9, 0.5, 0.5, 0.5, 0.5, -0.9, 0.1])
    qmax = np.array([1.0, 1.0, 1.0, 1.0, 0.9, 3.5, 3.0, 2.5, 2.5, 0.9, 0.8])

    sampler = MCMCEnsembleSampler(
        data=data,
        nwalkers=24,
        nsteps=60,
        ncomp=2,
        par_per_comp=5,
        qmin=qmin,
        qmax=qmax,
        p0=None,
        pmodel="bivariate_normal",
    )

    print("Running MCMC ensemble sampler (60 steps)...")
    sampler.run_emcee()

    # 3. Diagnostics and Results
    flat_chain = sampler.get_chain(flat=True, discard=10)
    print(f"Flat chain shape after burn-in: {flat_chain.shape}")

    # Compute individual star kinematic memberships
    memberships = get_component_membership_from_mcmc_chain(
        flat_chain, data, ncomp=2, par_per_comp=5
    )
    cluster_prob = memberships[0]
    print(
        f"Mean estimated cluster membership probability: {cluster_prob[:200].mean():.2f} "
        f"(true cluster stars) vs {cluster_prob[200:].mean():.2f} (true field stars)"
    )

    # 4. Generate Diagnostic Figures (headless save)
    fig_traces = utplot.traceplots(sampler.get_chain())
    fig_traces.savefig("mcmc_traces_demo.png", dpi=100)
    plt.close(fig_traces)

    fig_corner = utplot.plot_corner(flat_chain[:, [0, 1, 2, 5, 6, 7, 10]])
    fig_corner.savefig("mcmc_corner_demo.png", dpi=100)
    plt.close(fig_corner)

    fig_vpd = utplot.plot_vpd_and_mdist_component(
        data[:, 0], data[:, 1], flat_chain, ncomp=2, membership=cluster_prob
    )
    fig_vpd.savefig("mcmc_vpd_demo.png", dpi=100)
    plt.close(fig_vpd)

    print("MCMC mixture analysis completed successfully! Demo plots saved.")


if __name__ == "__main__":
    main()
