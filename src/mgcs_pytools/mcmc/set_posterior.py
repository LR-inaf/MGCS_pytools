"""Log-posterior function used as the target distribution of the MCMC sampler."""

import numpy as np
from numpy.typing import NDArray

from .bivariate_normal import _lnprior, _logl


def posterior(
    prm: NDArray,
    data: NDArray,
    qmin: NDArray,
    qmax: NDArray,
    component: int,
    pmodel: str = "bivariate_normal",
    fix_params: NDArray | None = None,
    lnk_params: NDArray | None = None,
    stride: int = 5,
) -> float:
    """Posterior function for the MCMC.

    Parameters
    ----------
    prm : NDArray
        Parameters to fit.
    data : NDArray
        Data to fit.
    qmin : NDArray
        Lower limits for priors.
    qmax : NDArray
        Upper limits for priors.
    component : int
        Number of components to fit.
    pmodel : str, optional
        Model to use for the posterior. Default is ``"bivariate_normal"``.
        Currently, only ``"bivariate_normal"`` is supported.
    fix_params : NDArray or None, optional
        Array containing the fixed parameters; ``NaN`` entries are left free.
        If None, no parameters will be fixed. Default is None.
    lnk_params : NDArray or None, optional
        Array of ``(to_link, which)`` index pairs: parameter ``which`` is set equal to
        parameter ``to_link``. If None, no parameters will be linked together.
        Default is None.
    stride : int, optional
        Number of parameters per component. Default is 5.

    Returns
    -------
    float
        Logarithm of the posterior probability (``-inf`` outside the prior bounds).

    Raises
    ------
    ValueError
        If ``pmodel`` is not supported.
    """
    if pmodel != "bivariate_normal":
        raise ValueError(f"Model {pmodel} is not supported.")

    lp = _lnprior(prm, qmin, qmax, component)

    if fix_params is not None:
        prm_new = np.where(np.isnan(fix_params), prm, fix_params)
    else:
        prm_new = prm.copy()

    if lnk_params is not None:
        for to_link, which in lnk_params:
            prm_new[which] = prm_new[to_link]

    output = -np.inf if not np.isfinite(lp) else lp + _logl(data, prm_new, component, stride)
    return output
