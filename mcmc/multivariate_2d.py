import numpy as np
from numpy.typing import NDArray


def mvar_norm(
    x1: NDArray, x2: NDArray, mu1: NDArray, mu2: NDArray, cov: NDArray
) -> NDArray:
    """Multivariate normal distribution function.

    Parameters
    ----------
    x1 : `NDArray`
        Input along first dimension.
    x2 : `NDArray`
        Input along second dimension.
    mu1 : `NDArray`
        Mean value along first dimension.
    mu2 : `NDArray`
        Mean value along first dimension.
    cov : `NDArray`
        Covariance matrix.

    Returns
    -------
    `NDArray`
        Probability density function.
    """

    diff = np.array([x1 - mu1, x2 - mu2]).T
    minv = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    prod1 = np.matmul(minv, diff[:, :, np.newaxis])
    prod2 = np.matmul(diff[:, np.newaxis, :], prod1)

    return norm * np.exp(-0.5 * prod2.T)
