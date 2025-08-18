from matplotlib.pylab import det
import numpy as np
from numpy.typing import NDArray


def bivar_norm(
    x1: NDArray,
    x2: NDArray,
    mu1: NDArray,
    mu2: NDArray,
    sigma1: NDArray,
    sigma2: NDArray,
    rho: NDArray,
) -> NDArray:
    """Bivariate normal distribution

    Parameters:
        x1: NDArray
            First variable.
        x2: NDArray
            Second variable.
        mu1: NDArray
            Mean of the first variable.
        mu2: NDArray
            Mean of the second variable.
        sigma1: NDArray
            Standard deviation of the first variable.
        sigma2: NDArray
            Standard deviation of the second variable.
        rho: NDArray
            Correlation coefficient.

    Returns:
        PDF: NDArray
            Probability density function values.
    """
    dx1 = x1 - mu1
    dx2 = x2 - mu2

    det = sigma1 * sigma2 * np.sqrt(1 - rho**2)
    norm = 1.0 / (2.0 * np.pi * det)

    z = (
        (dx1 / sigma1) ** 2
        + (dx2 / sigma2) ** 2
        - 2 * rho * dx1 * dx2 / (sigma1 * sigma2)
    ) / (2 * (1 - rho**2))

    return norm * np.exp(-z)


def _logl(data, prm, component, stride=5):

    #   unpack data
    pm_ra = data[:, 0]
    pm_dec = data[:, 1]
    epm_ra = data[:, 2]
    epm_dec = data[:, 3]

    if component > 1:
        w = prm[-(component - 1) :]
    else:
        w = prm[-1]

    # create a model for each component
    c_dist = np.zeros(pm_ra.shape)
    for i in range(component):
        mu1, mu2 = prm[0 + i * stride], prm[1 + i * stride]
        s1, s2 = prm[2 + i * stride], prm[3 + i * stride]
        rho = prm[4 + i * stride]

        sigma1_obs = np.sqrt(s1**2 + epm_ra**2)
        sigma2_obs = np.sqrt(s2**2 + epm_dec**2)
        # rho_eff
        rho_eff = rho * (s1 * s2) / (sigma1_obs * sigma2_obs)

        if component > 1 and i < component - 1:
            c_dist += w[i] * bivar_norm(
                pm_ra, pm_dec, mu1, mu2, sigma1_obs, sigma2_obs, rho_eff
            )
        elif component == 1:
            c_dist += w * bivar_norm(
                pm_ra, pm_dec, mu1, mu2, sigma1_obs, sigma2_obs, rho_eff
            )
        else:
            c_dist += (1 - w.sum()) * bivar_norm(
                pm_ra, pm_dec, mu1, mu2, sigma1_obs, sigma2_obs, rho_eff
            )

    # print(c_dist.shape)
    if np.any(np.isnan(np.log(c_dist))):
        print(c_dist)
        print("crashing parameters:", prm)
        return np.nan

    return np.log(c_dist).sum()


def _lnprior(prm, qmin, qmax, ncomp) -> float:
    if ncomp > 1:
        ws = prm[-(ncomp - 1) :]
        if (prm < qmin).any() or (prm > qmax).any() or 1 - ws.sum() < 0.0:
            return -np.inf
        else:
            return 0.0
    else:
        if (prm < qmin).any() or (prm > qmax).any():
            return -np.inf
        else:
            return 0.0
