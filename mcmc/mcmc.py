import numpy as np
from numpy.typing import NDArray

import emcee
from multiprocessing import Pool

from .set_posterior import posterior


class MyEmceeES:
    """Class to run MCMC with emcee.
    This class is particularly design to use multivariate fuctions as posterior,
    but it can be genewralized to work with any function.
    It can managed one or more gaussian components
    and their parameters can be either linked of freezed.
    Parameters:
        data: `NDArray`
            Data to fit.
        posterior: `callable`
            Posterior function to use in the mcmc to compute the loglike.
        nwalkers: `int`
            Number of walkers initialized in the MCMC.
        nsteps: `int`
            Number of steps to run.
        ncomp: `int`
            Number of model components to use.
        par_per_comp: `int`
            Number of parameters per component.
        qmin: `NDArray`
            Lower limits for priors.
        qmax: `NDArray`
            Upper limits for priors.
        p0: `NDArray`  or `None`, optional
            Initial guess for the parameters.
            If `None`, uniform distribution between qmin and qmax will be used. Defult is `None`.
        fix_params: `NDArray` or `None`, optional
            Array containing the fixed parameters. If None, no parameters will be fixed. Defaul is `None`
        lnk_params: `NDArray` or `None`, optional
            Array containing the linked parameters. If None, no parameters will be linked together.
            Defaul is `None`
    """

    def __init__(
        self,
        data: NDArray,
        nwalkers: int,
        nsteps: int,
        ncomp: int,
        par_per_comp: int,
        qmin: NDArray,
        qmax: NDArray,
        pmodel: str = "bivariate_normal",
        p0: NDArray | None = None,
        fix_params: NDArray | None = None,
        lnk_params: NDArray | None = None,
    ):
        self.data = data
        self.pmodel = pmodel
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.ncomp = ncomp
        self.par_per_comp = par_per_comp
        self.ndim = (
            (ncomp * par_per_comp) + ncomp - 1
            if self.ncomp > 1
            else (ncomp * par_per_comp) + ncomp
        )
        self.qmin = qmin
        self.qmax = qmax
        self.fix_params = fix_params
        self.lnk_params = lnk_params
        self.chain = None

        self.p0 = (
            np.array(
                [np.random.uniform(self.qmin, self.qmax) for _ in range(self.nwalkers)]
            )
            if p0 is None
            else p0
        )

        self.p0[:, -2] = np.random.uniform(0.3, 0.6, size=self.nwalkers)
        self.p0[:, -1] = np.random.uniform(0.0, 0.3, size=self.nwalkers)

        self.bpars = None
        self.sampler = None

    def run_emcee(self, parallel=False):
        """Run the MCMC using emcee.
        Parameters:
            parallel: `bool`, optional
                If True, use multiprocessing to run the MCMC on multiple processor.
                If False, run the MCMC in a single process. Default is False.
        """

        if parallel:
            with Pool() as pool:
                self.sampler = emcee.EnsembleSampler(
                    nwalkers=self.nwalkers,
                    ndim=self.ndim,
                    log_prob_fn=posterior,
                    kwargs={
                        "data": self.data,
                        "qmin": self.qmin,
                        "qmax": self.qmax,
                        "component": self.ncomp,
                        "stride": self.par_per_comp,
                        "fix_params": self.fix_params,
                        "lnk_params": self.lnk_params,
                        "pmodel": self.pmodel,
                    },
                    pool=pool,
                    moves=[
                        (emcee.moves.DEMove(), 0.8),
                        (emcee.moves.DESnookerMove(), 0.2),
                    ],
                )
                self.sampler.run_mcmc(self.p0, self.nsteps, progress=True)

        else:
            self.sampler = emcee.EnsembleSampler(
                nwalkers=self.nwalkers,
                ndim=self.ndim,
                log_prob_fn=posterior,
                kwargs={
                    "data": self.data,
                    "qmin": self.qmin,
                    "qmax": self.qmax,
                    "component": self.ncomp,
                    "stride": self.par_per_comp,
                    "fix_params": self.fix_params,
                    "lnk_params": self.lnk_params,
                    "pmodel": self.pmodel,
                },
                moves=[
                    (emcee.moves.DEMove(), 0.8),
                    (emcee.moves.DESnookerMove(), 0.2),
                ],
            )
            self.sampler.run_mcmc(self.p0, self.nsteps, progress=True)

    def get_chain(self, **kwargs) -> NDArray:
        """Get the chain of samples.
        Parameters:
            kwargs:
                Parameters of get_chain emcee's method (*thin*, *flat*, *discard*).
        Returns:
            chain: `NDArray`
                Chain of samples.
        """
        return self.sampler.get_chain(**kwargs)

    def get_chpars(self, which, percentile=[16, 50, 84], **kwargs) -> NDArray:
        """Obtain the parameters of the max lnprob model

        Parameters:
            which: `str`
                Which parameter set to return.
                *median* for the median of the parameters. Also the *quantiles* will be pass as error limits
                *maxln* for the parameters of the max loglike.

            percentiles: `list`, optional
                Percentiles to use, defualt are [16, 50, 84].

            kwargs:
                Parameters of get_chain and get_log_prob emcee methods (*thin*, *flat*, *discard*).

        Returns:
            pars: `NDArray`
                Parameters set from the chain.
        """
        if which == "maxln":
            maxln = self.sampler.get_log_prob(**kwargs).argmax()
            return self.sampler.get_chain(**kwargs)[maxln]
        else:
            return np.percentile(self.sampler.get_chain(**kwargs), percentile, axis=0)
