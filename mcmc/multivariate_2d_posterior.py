import numpy as np
from numpy.typing import NDArray
from .multivariate_2d import mvar_norm


def _logl(data, prm, component, stride=5) -> float:

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

        cprm = prm[(i * stride) : (i + 1) * stride]
        sigma_ra = np.sqrt(cprm[2] * cprm[2] + epm_ra * epm_ra)
        sigma_dec = np.sqrt(cprm[3] * cprm[3] + epm_dec * epm_dec)
        cov = np.array(
            [
                np.array(
                    [
                        [c_si_ra * c_si_ra, cprm[-1] * c_si_ra * c_si_dec],
                        [cprm[-1] * c_si_ra * c_si_dec, c_si_dec * c_si_dec],
                    ]
                )
                for c_si_ra, c_si_dec in zip(sigma_ra, sigma_dec)
            ]
        )

        if component > 1 and i < component - 1:
            c_dist += w[i] * mvar_norm(pm_ra, pm_dec, cprm[0], cprm[1], cov)[0, 0, :]
        elif component == 1:
            c_dist += w * mvar_norm(pm_ra, pm_dec, cprm[0], cprm[1], cov)[0, 0, :]
        else:
            c_dist += (1 - w.sum()) * mvar_norm(pm_ra, pm_dec, cprm[0], cprm[1], cov)[
                0, 0, :
            ]

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
