# Example pipeline boilerplate for MGCS_pytools
import os

import astropy.units as u  # noqa: F401
import numpy as np
import pandas as pd  # noqa: F401
from astropy.coordinates import SkyCoord  # noqa: F401

from mgcs_pytools.statistical_membership import compute_statistical_membership  # noqa: F401
from mgcs_pytools.utils import plotting as utplot  # noqa: F401

# 1. Import cluster and parallel field catalogs
CATALOG_PATH_CLUSTER = os.path.abspath("path_to_cluster_catalog.csv")
CATALOG_PATH_FIELD = os.path.abspath("path_to_field_catalog.csv")

# df_cluster = pd.read_csv(CATALOG_PATH_CLUSTER)
# df_field = pd.read_csv(CATALOG_PATH_FIELD)

# 2. Setup parameters for statistical membership
field_fov = 0.16  # Parallel field FoV in deg^2
cluster_mag = ["mag_b", "mag_v"]  # Magnitude columns (band1, band2)
cluster_mag_error = ["mag_b_err", "mag_v_err"]  # Photometric error columns
field_mag = ["mag_b", "mag_v"]  # Field magnitude columns
racol = "RA"  # Right Ascension column name
deccol = "DEC"  # Declination column name

dr_params = {
    "rband1": 1.32,  # A(lambda)/A(V) extinction ratio in band1 (multiplied by R_V=3.1 internally)
    "rband2": 1.00,  # A(lambda)/A(V) extinction ratio in band2 (multiplied by R_V=3.1 internally)
    "TO_mag": 17.5,  # Turn-off magnitude
    "TO_color": 0.9,  # Turn-off color
    "nref": 25,  # Number of reference stars
    "ord_step": 0.4,  # Bin size in ordinate to calculate the fiducial ridge line
}

# Region of interest (ROI) for reference stars in the rotated CMD frame
roi = np.array([[-1.0, 16.0], [2.5, 16.0], [2.5, 21.0], [-1.0, 21.0]])

min_field_stars = 100  # Minimum expected field stars in each annular sub-FoV
member_threshold = 0.8  # Membership probability threshold for reference stars
membership_iter = 100  # Iterations for statistical subtraction
min_star_per_cell = 5  # Minimum stars per Voronoi cell for dilation

# 3. Run the statistical membership decontamination pipeline
# df_output = compute_statistical_membership(
#     df_cluster=df_cluster,
#     df_field=df_field,
#     target_mag_col=cluster_mag,
#     target_mag_col_error=cluster_mag_error,
#     field_mag_col=field_mag,
#     dr_params=dr_params.copy(),
#     parallel_fov=field_fov,
#     racol=racol,
#     deccol=deccol,
#     roi=roi,
#     minstars=min_field_stars,
#     membership_iter=membership_iter,
#     member_threshold=member_threshold,
#     min_star_per_cell=min_star_per_cell,
#     membership_method="analytical",
#     plot_dred=False,
#     plot_voronoi=False,
#     which_voronoi="dilation",
#     do_dilation=True,
# )

# In df_output:
# - df_output["membership"]: posterior membership probability [0, 1]
# - df_output[cluster_mag[0] + "_drcorr"]: corrected magnitude 1
# - df_output[cluster_mag[1] + "_drcorr"]: corrected magnitude 2
# - df_output["delta_ebv"]: differential reddening correction map value

# 4. Visualization Utilities
# fig = utplot.plot_membership_overview(
#     btarget=df_output[cluster_mag[0] + "_drcorr"],
#     vtarget=df_output[cluster_mag[1] + "_drcorr"],
#     bfield=df_field[field_mag[0]],
#     vfield=df_field[field_mag[1]],
#     membership=df_output["membership"],
# )

# fig = utplot.plot_decontamination_snapshot(
#     df_output[cluster_mag[0] + "_drcorr"],
#     df_output[cluster_mag[1] + "_drcorr"],
#     df_field[field_mag[0]],
#     df_field[field_mag[1]],
#     df_output["membership"],
#     member_threshold=0.8,
# )

# fig = utplot.plot_spatial_membership(
#     df_output[racol],
#     df_output[deccol],
#     df_output["membership"],
#     member_threshold=0.8,
# )

# cluster_center = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg)
# fig = utplot.plot_differential_reddening_map(
#     df_output[racol],
#     df_output[deccol],
#     cluster_center.ra.deg,
#     cluster_center.dec.deg,
#     df_output["delta_ebv"],
# )
