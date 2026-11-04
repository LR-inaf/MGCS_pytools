"""Statistical membership: CMD decontamination and differential reddening correction.

The decontamination relies on an adaptive Voronoi (CVT) grid in the colour-magnitude
diagram, combined with a differential reddening correction.
"""

from .cvt_grid import CVTGrid, cvtGrid
from .dr_correction import ROISelector, compute_differential_reddening
from .membership import compute_statistical_membership, get_membership, spatial_groupby
from .region_tools import get_pts_in_regions, get_regions_count

__all__ = [
    "compute_statistical_membership",
    "spatial_groupby",
    "get_membership",
    "compute_differential_reddening",
    "ROISelector",
    "CVTGrid",
    "cvtGrid",
    "get_regions_count",
    "get_pts_in_regions",
]
