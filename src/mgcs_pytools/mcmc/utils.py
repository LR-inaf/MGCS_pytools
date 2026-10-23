"""Utility functions for MCMC chains, priors, and posterior probability distributions."""

import numpy as np
from numpy.typing import NDArray

try:
    import numba

    @numba.njit(parallel=True, fastmath=True)
    def _numba_weighted_mvar_pdf_fused(
        x1: NDArray,
        x2: NDArray,
        x1e: NDArray,
        x2e: NDArray,
        ch: NDArray,
    ) -> NDArray:
        """Parallel fused kernel for evaluating weighted bivariate normal PDF across chain samples."""
        Ndata = len(x1)
        Nchain = len(ch)
        out = np.empty((Ndata, Nchain), dtype=np.float64)
        two_pi = 2.0 * np.pi

        mu1 = ch[:, 0]
        mu2 = ch[:, 1]
        s1 = ch[:, 2]
        s2 = ch[:, 3]
        rho = ch[:, 4]
        w = ch[:, -1]

        for i in numba.prange(Ndata):
            x1_val = x1[i]
            x2_val = x2[i]
            e1_sq = x1e[i] ** 2
            e2_sq = x2e[i] ** 2
            for j in range(Nchain):
                s1_obs = np.sqrt(s1[j] ** 2 + e1_sq)
                s2_obs = np.sqrt(s2[j] ** 2 + e2_sq)
                r_eff = rho[j] * (s1[j] * s2[j]) / (s1_obs * s2_obs)
                one_minus_r2 = 1.0 - r_eff * r_eff
                det = s1_obs * s2_obs * np.sqrt(one_minus_r2)
                inv_norm = 1.0 / (two_pi * det)
                denom = 2.0 * one_minus_r2
                dx1 = x1_val - mu1[j]
                dx2 = x2_val - mu2[j]
                z = (
                    (dx1 / s1_obs) ** 2
                    + (dx2 / s2_obs) ** 2
                    - 2.0 * r_eff * dx1 * dx2 / (s1_obs * s2_obs)
                ) / denom
                out[i, j] = w[j] * (inv_norm * np.exp(-z))
        return out

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _bivar_norm_broadcasting_for_chain(
    x1: NDArray,
    x2: NDArray,
    mu1: NDArray,
    mu2: NDArray,
    sigma1: NDArray,
    sigma2: NDArray,
    rho: NDArray,
) -> NDArray:
    """Multivariate normal PDF (2D) with NumPy broadcasting for MCMC chain.

    Parameters
    ----------
    x1 : NDArray
        First observable (Ndata,).
    x2 : NDArray
        Second observable (Ndata,).
    mu1 : NDArray
        Mean of first variable (Nchain,).
    mu2 : NDArray
        Mean of second variable (Nchain,).
    sigma1 : NDArray
        Standard deviation of first variable (Ndata, Nchain).
    sigma2 : NDArray
        Standard deviation of second variable (Ndata, Nchain).
    rho : NDArray
        Effective correlation coefficient (Ndata, Nchain).

    Returns
    -------
    NDArray
        PDF evaluations with shape (Ndata, Nchain).
    """
    dx1 = x1[:, None] - mu1[None, :]
    dx2 = x2[:, None] - mu2[None, :]

    det = sigma1 * sigma2 * np.sqrt(1 - rho**2)
    norm = 1.0 / (2.0 * np.pi * det)

    z = ((dx1 / sigma1) ** 2 + (dx2 / sigma2) ** 2 - 2 * rho * dx1 * dx2 / (sigma1 * sigma2)) / (
        2 * (1 - rho**2)
    )

    return norm * np.exp(-z)


def _generate_weighted_mvar_pdf(
    ch: NDArray, x1: NDArray, x2: NDArray, x1e: NDArray, x2e: NDArray
) -> NDArray:
    """Evaluate weighted bivariate normal PDF across chain samples and observations.

    Uses a parallel Numba kernel when available (>40x speedup), otherwise falls back
    to pure NumPy broadcasting.

    Parameters
    ----------
    ch : NDArray
        Chain parameters of shape (Nchain, 6): [mu1, mu2, s1, s2, rho, weight].
    x1 : NDArray
        First observable (Ndata,).
    x2 : NDArray
        Second observable (Ndata,).
    x1e : NDArray
        First observable error (Ndata,).
    x2e : NDArray
        Second observable error (Ndata,).

    Returns
    -------
    NDArray
        Weighted bivariate normal PDF of shape (Ndata, Nchain).
    """
    if HAS_NUMBA:
        return _numba_weighted_mvar_pdf_fused(x1, x2, x1e, x2e, ch)

    mu1, mu2 = ch[:, 0], ch[:, 1]
    s1, s2 = ch[:, 2], ch[:, 3]
    rho = ch[:, 4]
    w = ch[:, -1]

    sigma1_obs = np.sqrt(s1[None, :] ** 2 + x1e[:, None] ** 2)
    sigma2_obs = np.sqrt(s2[None, :] ** 2 + x2e[:, None] ** 2)

    rho_eff = rho[None, :] * (s1[None, :] * s2[None, :]) / (sigma1_obs * sigma2_obs)

    pdf = _bivar_norm_broadcasting_for_chain(x1, x2, mu1, mu2, sigma1_obs, sigma2_obs, rho_eff)

    return w[None, :] * pdf


def get_component_membership_from_mcmc_chain(
    ch: NDArray, data: NDArray, ncomp: int, par_per_comp: int, batch_size: int = 1000
) -> NDArray:
    """Compute posterior star-by-star kinematic component membership probabilities from MCMC chain.

    Parameters
    ----------
    ch : NDArray
        MCMC chain parameters with shape (Nchain, Ndim).
    data : NDArray
        Observational data array of shape (Ndata, 4): [pm_ra, pm_dec, epm_ra, epm_dec].
    ncomp : int
        Number of mixture components.
    par_per_comp : int
        Number of parameters per component (typically 5: mu1, mu2, s1, s2, rho).
    batch_size : int, optional
        Batch size across chain samples to limit peak memory usage, by default 1000.

    Returns
    -------
    NDArray
        Component membership probabilities with shape (ncomp, Ndata).
    """
    x1, x2, x1e, x2e = data.T
    n_chain = ch.shape[0]

    membership_sum = np.zeros((ncomp, x1.shape[0]))

    for batch_start in range(0, n_chain, batch_size):
        batch_end = min(batch_start + batch_size, n_chain)
        batch_ch = ch[batch_start:batch_end]

        if ncomp > 1:
            w = batch_ch[:, -(ncomp - 1) :]
        else:
            w = np.ones((len(batch_ch), 1))

        pdf = np.empty((ncomp, x1.shape[0], batch_ch.shape[0]))
        for i in range(ncomp):
            c_ch = np.empty((batch_ch.shape[0], par_per_comp + 1))
            c_ch[:, :par_per_comp] = batch_ch[:, (i * par_per_comp) : (1 + i) * par_per_comp]

            if ncomp > 1 and i < ncomp - 1:
                c_ch[:, -1] = w[:, i]
            elif ncomp > 1 and i == ncomp - 1:
                c_ch[:, -1] = 1.0 - np.sum(w, axis=1)
            elif ncomp == 1:
                c_ch[:, -1] = w[:, 0]

            pdf[i] = _generate_weighted_mvar_pdf(c_ch, x1, x2, x1e, x2e)

        denom = pdf.sum(axis=0)
        membership_sum += np.sum(pdf / denom[None, :], axis=2)

    return membership_sum / n_chain
