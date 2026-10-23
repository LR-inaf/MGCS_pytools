import os

import matplotlib

# Set matplotlib backend to headless Agg
matplotlib.use("Agg")

# Configure astropy and matplotlib directories inside workspace so sandbox permissions are respected
_ws_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_astropy_dir = os.path.join(_ws_dir, ".astropy_config")
_mpl_dir = os.path.join(_ws_dir, ".mpl_config")

os.makedirs(_astropy_dir, exist_ok=True)
os.makedirs(_mpl_dir, exist_ok=True)

os.environ["ASTROPY_CONFIG_DIR"] = _astropy_dir
os.environ["ASTROPY_CACHE_DIR"] = _astropy_dir
os.environ["MPLCONFIGDIR"] = _mpl_dir
