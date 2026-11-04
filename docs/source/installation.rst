Installation
============

Requirements
------------
- Python >= 3.10
- Key scientific dependencies: ``numpy``, ``scipy``, ``pandas``, ``shapely``, ``matplotlib``, ``astropy``, ``emcee``, ``corner``, ``scikit-learn``, ``h5py``, ``seaborn``, ``tqdm``.

Installing via PyPI
-------------------
You can install the stable release from PyPI:

.. code-block:: bash

   pip install mgcs-pytools

Installing from a Local Wheel
-----------------------------
When installing from a pre-built wheel (which includes the offline HTML documentation):

.. code-block:: bash

   pip install dist/mgcs_pytools-0.3.0-py3-none-any.whl

Optional Dependencies
---------------------
**Numba JIT Acceleration**
To enable high-performance accelerated kernels for evaluating MCMC mixture PDFs (up to 40x speedup):

.. code-block:: bash

   pip install "mgcs-pytools[numba]"

**Documentation Tools**
If you wish to modify and rebuild the Sphinx documentation locally:

.. code-block:: bash

   pip install "mgcs-pytools[docs]"

**Development Tools**
For running the test suite, linting, and type checking:

.. code-block:: bash

   pip install "mgcs-pytools[dev]"

Verification
------------
Check that the package is correctly installed:

.. code-block:: python

   import mgcs_pytools

   print(mgcs_pytools.__doc__)
