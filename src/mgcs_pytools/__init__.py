"""
Python package for tools used for the Kinematic and photometric analysis of MGCS GCs.

Subpackages:
- mcmc: Tools Proper Motions (PMs) analysis based on Markov Chain Monte Carlo (MCMC) methods.
- statistical_membership: method to perform statistical membership based on photometry.
- plotting: Tools for plotting the results of the analysis.
"""

submodules = __all__ = ["mcmc", "statistical_membership", "plotting"]


def __dir__():
    return submodules
