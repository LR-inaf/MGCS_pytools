"""MGCS_pytools: Scientific Python tools for the Missing Globular Cluster Survey (MGCS).

Subpackages:

- mcmc: Kinematic Markov Chain Monte Carlo (MCMC) mixture modeling of Proper Motions.
- statistical_membership: Voronoi CVT spatial membership & differential reddening correction.
- utils: Visualization and plotting utilities.
"""

import os
import tempfile

# Ensure safe fallback config directories for astropy and matplotlib in sandbox/headless environments
if "ASTROPY_CONFIG_DIR" not in os.environ:
    _default_astropy = os.path.join(tempfile.gettempdir(), "astropy_mgcs")
    os.makedirs(_default_astropy, exist_ok=True)
    os.environ["ASTROPY_CONFIG_DIR"] = _default_astropy
    os.environ["ASTROPY_CACHE_DIR"] = _default_astropy

if "MPLCONFIGDIR" not in os.environ:
    _default_mpl = os.path.join(tempfile.gettempdir(), "matplotlib_mgcs")
    os.makedirs(_default_mpl, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _default_mpl

from typing import Any

from . import mcmc, statistical_membership, utils
from .utils import plotting

__version__ = "0.3.0"

__all__ = [
    "mcmc",
    "statistical_membership",
    "utils",
    "plotting",
    "open_docs",
    "get_docs_path",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazy-load documentation helpers on demand."""
    if name in ("open_docs", "get_docs_path"):
        from . import docs

        return getattr(docs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
