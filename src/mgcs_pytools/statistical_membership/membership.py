"""Statistical membership of cluster stars via CMD decontamination.

Cluster stars are grouped in concentric spatial regions, corrected for differential
reddening and compared with a field CMD on a Voronoi grid to estimate the membership
probability of each star.
"""

import logging
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from pandas import DataFrame
from scipy.spatial import ConvexHull
from shapely import Point, Polygon

from mgcs_pytools.utils import plotting as utplot

from .cvt_grid import cvtGrid
from .dr_correction import compute_differential_reddening
from .region_tools import get_pts_in_regions, get_regions_count

logger = logging.getLogger(__name__)


def spatial_groupby(
    df_cluster: DataFrame,
    df_field: DataFrame,
    racol: str,
    deccol: str,
    minstars: int = 200,
) -> tuple[list[DataFrame], list[Polygon], Polygon]:
    """Group stars in the cluster dataframe by their spatial location.

    The convex hull of the cluster star positions is used as the FoV and is split in
    concentric regions (a central disc followed by annuli) around the center of the
    RA/DEC range. Each region has the area expected to contain about ``minstars`` field
    stars, given the field star density estimated from ``df_field``; the last region
    absorbs the remainder. Coordinates and areas are in degrees.

    Parameters
    ----------
    df_cluster : DataFrame
        DataFrame containing the cluster stars.
    df_field : DataFrame
        DataFrame containing the field stars.
    racol : str
        Name of the column containing the RA of the stars.
    deccol : str
        Name of the column containing the DEC of the stars.
    minstars : int, optional
        Target number of field stars in each region (sets the area of the regions).
        Default is 200.

    Returns
    -------
    groups : list of DataFrame
        List of DataFrames, each containing the cluster stars in a region.
    subfovs : list of Polygon
        List of Polygons, each containing the area of a region.
    fov : Polygon
        Convex hull of the cluster star positions, used as the FoV.
    """
    from scipy.optimize import root_scalar

    # Calculate the field star density and the min area for at least min stars
    ra_center_field = df_field[racol].min() + (df_field[racol].max() - df_field[racol].min()) / 2
    dec_center_field = (
        df_field[deccol].min() + (df_field[deccol].max() - df_field[deccol].min()) / 2
    )

    field_center = SkyCoord(ra_center_field, dec_center_field, unit=u.deg)

    center_sep = field_center.separation(
        SkyCoord(
            df_field[racol],
            df_field[deccol],
            unit="deg",
        )
    ).arcsec

    df_field = df_field.copy()
    df_field.loc[:, "center_sep"] = center_sep

    area_conversion = 3600**2
    star_density = df_field.shape[0] / (np.pi * df_field["center_sep"].max() ** 2)
    min_area = minstars / star_density
    min_area_deg2 = min_area / area_conversion

    logger.info(f"Star density: {star_density:.2f} stars / arcsec^2")
    logger.info(f"Min area for at least {minstars} stars: {min_area:.2f} arcsec^2")

    # groupby the cluster dataframe in bins with at least minstars
    ra_center = df_cluster[racol].min() + (df_cluster[racol].max() - df_cluster[racol].min()) / 2
    dec_center = (
        df_cluster[deccol].min() + (df_cluster[deccol].max() - df_cluster[deccol].min()) / 2
    )

    # make the FoV polygon from coordinates
    hull = ConvexHull(df_cluster.loc[:, [racol, deccol]].values)
    fov = Polygon(hull.points[hull.vertices])

    center_pt = Point(ra_center, dec_center)
    total_fov_area = fov.area

    r_boundaries = [0.0]
    current_r_in = 0.0
    current_area_in = 0.0

    # max possible radius
    max_r = np.max(np.linalg.norm(hull.points - np.array([ra_center, dec_center]), axis=1))
    max_r += 1e-5

    def get_fov_intersect_area(r: float) -> float:
        return center_pt.buffer(r).intersection(fov).area

    while True:
        if total_fov_area - current_area_in < min_area_deg2:
            r_boundaries.append(max_r)
            break

        def area_diff(r: float) -> float:
            return get_fov_intersect_area(r) - current_area_in - min_area_deg2

        try:
            res = root_scalar(area_diff, bracket=[current_r_in, max_r], method="brentq")
            r_out = res.root
            r_boundaries.append(r_out)
            current_r_in = r_out
            current_area_in = get_fov_intersect_area(r_out)
        except ValueError:
            r_boundaries.append(max_r)
            break

    subfovs = []
    for i in range(1, len(r_boundaries)):
        r_in = r_boundaries[i - 1]
        r_out = r_boundaries[i]

        if r_in == 0.0:
            subf = center_pt.buffer(r_out).intersection(fov)
        elif i == len(r_boundaries) - 1:
            subf = fov.difference(center_pt.buffer(r_in))
        else:
            annulus = center_pt.buffer(r_out).difference(center_pt.buffer(r_in))
            subf = annulus.intersection(fov)

        subfovs.append(subf)

    # Compute radial separation of all stars to cluster center
    # the Euclidean metric was used in Shapely buffers (r was in degrees)
    radii = np.sqrt((df_cluster[racol] - ra_center) ** 2 + (df_cluster[deccol] - dec_center) ** 2)

    groups = []
    for i in range(1, len(r_boundaries)):
        r_in = r_boundaries[i - 1]
        r_out = r_boundaries[i]

        if i == len(r_boundaries) - 1:
            mask = (radii >= r_in) & (radii <= r_out)
        else:
            mask = (radii >= r_in) & (radii < r_out)

        groups.append(df_cluster[mask].copy())

    return groups, subfovs, fov


def get_membership(
    cell_counts: list[tuple[int, int]],
    cell_points: list[tuple[int, list[int]]],
    common_regions: list[int],
    iter: int = 1000,
    fov_ratio: float = 1.0,
    method: str = "analytical",
) -> np.ndarray:
    """Get the membership of each star.

    Parameters
    ----------
    cell_counts : list of tuple of (int, int)
        List of tuples containing the region id and the number of field stars
        in that region.
    cell_points : list of tuple of (int, list of int)
        List of tuples containing the region id and the positional indices of the
        cluster stars in that region.
    common_regions : list of int
        List of region ids to evaluate (if a region is missing from ``cell_counts``,
        its field star count defaults to 0).
    iter : int, optional
        Number of iterations to perform if using montecarlo. Default is 1000.
    fov_ratio : float, optional
        Ratio between the sub-FoV area and the area covered by the field sample
        (used to rescale the field counts). Default is 1.0.
    method : {"analytical", "montecarlo"}, optional
        Method to compute probability. Default is "analytical".

    Returns
    -------
    NDArray
        Array of shape (N, 2) containing the star index and its membership
        probability.

    Raises
    ------
    ValueError
        If ``method`` is not ``"analytical"`` or ``"montecarlo"``.
    """
    if method not in ["analytical", "montecarlo"]:
        raise ValueError("method must be 'analytical' or 'montecarlo'")

    cell_counts_dict = dict(cell_counts)
    cell_points_dict = dict(cell_points)

    pts_all = []
    target_counts = []
    field_counts_rescaled = []

    for rid in common_regions:
        if rid not in cell_points_dict:
            continue
        pts = cell_points_dict[rid]
        target_counts.append(len(pts))
        field_counts_rescaled.append(cell_counts_dict.get(rid, 0) * fov_ratio)
        pts_all.extend(pts)

    if len(pts_all) == 0:
        return np.empty((0, 2), dtype=float)

    pts_all_arr = np.array(pts_all)
    target_counts_arr = np.array(target_counts)
    field_counts_rescaled_arr = np.array(field_counts_rescaled)

    if method == "analytical":
        membership_probs = np.zeros(len(pts_all_arr))

        idx_offset = 0
        for i in range(len(target_counts_arr)):
            tc = target_counts_arr[i]
            if tc > 0:
                prob = max(0.0, 1.0 - field_counts_rescaled_arr[i] / tc)
                membership_probs[idx_offset : idx_offset + tc] = prob
            idx_offset += tc

        return np.column_stack((pts_all_arr, membership_probs))

    # montecarlo method
    p = field_counts_rescaled_arr % 1
    bias = field_counts_rescaled_arr.astype(int)

    N_cells = len(target_counts_arr)
    N_stars = len(pts_all_arr)
    star_retention_counts = np.zeros(N_stars, dtype=int)

    rands = np.random.rand(iter, N_cells)
    extra_field_stars = (rands < p).astype(int)
    simulated_field_counts = bias + extra_field_stars

    idx_offsets = (
        np.concatenate(([0], np.cumsum(target_counts_arr)[:-1]))
        if len(target_counts_arr) > 0
        else np.array([])
    )

    for i in range(iter):
        cell_field_counts = simulated_field_counts[i]

        for cell_idx in range(N_cells):
            tc = target_counts_arr[cell_idx]
            if tc > 0:
                fc = min(cell_field_counts[cell_idx], tc)
                keep = tc - fc

                if keep > 0:
                    keep_idx = np.random.choice(tc, keep, replace=False)
                    global_idx = idx_offsets[cell_idx] + keep_idx
                    star_retention_counts[global_idx] += 1

    membership_probs = star_retention_counts / iter
    return np.column_stack((pts_all_arr, membership_probs))


def compute_statistical_membership(
    df_cluster: DataFrame,
    df_field: DataFrame,
    target_mag_col: list[str],
    target_mag_col_error: list[str],
    field_mag_col: list[str],
    dr_params: dict[str, Any],
    parallel_fov: float,
    member_threshold: float = 0.8,
    minstars: int = 200,
    racol: str = "RA",
    deccol: str = "DEC",
    min_star_per_cell: int = 3,
    membership_iter: int = 1000,
    roi: Any = None,
    plot_dred: bool = False,
    plot_voronoi: bool = False,
    which_voronoi: str = "dilation",
    do_dilation: bool = True,
    membership_method: str = "analytical",
    max_iter: int = 20,
) -> DataFrame:
    """Compute the statistical membership of stars in a cluster field.

    The input dataframes are not modified. The procedure iterates differential reddening
    correction and CMD decontamination until the median reddening correction falls below
    the median photometric error (one more iteration is then performed).

    Parameters
    ----------
    df_cluster : DataFrame
        DataFrame containing the cluster stars.
    df_field : DataFrame
        DataFrame containing the field stars.
    target_mag_col : list of str
        List containing the names of the columns with the magnitudes
        to use for the target CMD (band1, band2).
    target_mag_col_error : list of str
        List containing the names of the columns with the magnitude errors
        to use for the target CMD (band1, band2).
    field_mag_col : list of str
        List containing the names of the columns with the magnitudes
        to use for the field CMD (band1, band2).
    dr_params : dict
        Dictionary containing input parameters for the differential reddening
        correction. It must contain the following keys:

        - ``'rband1'``: the A(lambda)/A(V) extinction ratio of band1,
        - ``'rband2'``: the A(lambda)/A(V) extinction ratio of band2,
        - ``'TO_mag'``: the magnitude of the Turn-off,
        - ``'TO_color'``: the color of the Turn-off.

        Optional keys are ``'nref'`` (or ``'n_reference_stars'``), the number of
        reference stars (default 50), and ``'ord_step'``, the bin size in the
        'ordinate' to calculate the ridge line (default 0.1).

        The keys ``'band1'``, ``'band2'``, ``'xcol'`` and ``'ycol'`` are set by this
        function (the dictionary is modified in place).
    parallel_fov : float
        Area of the parallel (field) FoV in deg^2.
    member_threshold : float, optional
        Membership threshold to consider a star as a reference member of the cluster.
        Default is 0.8.
    minstars : int, optional
        Target number of field stars in each spatial region (see
        :func:`spatial_groupby`). Default is 200.
    racol : str, optional
        Name of the column containing the RA of the stars. Default is ``"RA"``.
    deccol : str, optional
        Name of the column containing the DEC of the stars. Default is ``"DEC"``.
    min_star_per_cell : int, optional
        Target median number of stars per Voronoi cell (``target_median`` of
        :class:`CVTGrid`). Default is 3.
    membership_iter : int, optional
        Number of iterations to perform for the membership calculation
        (used by the ``"montecarlo"`` method). Default is 1000.
    roi : list of tuple of float or shapely.Polygon or None, optional
        Region of interest for the differential reddening correction, given as the
        polygon vertices (or the polygon) in the rotated (abscissa, ordinate) CMD plane.
        If None, an interactive selection window is opened. Default is None.
    plot_dred : bool, optional
        If True, plot the differential reddening correction results.
        Default is False.
    plot_voronoi : bool, optional
        If True, plot the Voronoi grid used for the membership calculation.
        Default is False.
    which_voronoi : {"dilation", "wkmeans"}, optional
        Type of Voronoi grid to use: ``"dilation"`` (Lloyd-relaxed CVT, optionally
        dilated) or ``"wkmeans"`` (CVT seeded with weighted k-means).
        Default is ``"dilation"``.
    do_dilation : bool, optional
        If True, merge neighbouring Voronoi cells until the median number of stars per
        cell reaches ``min_star_per_cell`` (only for ``which_voronoi="dilation"``).
        Default is True.
    membership_method : {"analytical", "montecarlo"}, optional
        Method to compute the membership probability. Default is ``"analytical"``.
    max_iter : int, optional
        Maximum number of differential reddening and decontamination iterations.
        Default is 20.

    Returns
    -------
    DataFrame
        Copy of ``df_cluster`` with the columns ``membership``, ``delta_ebv`` and the
        differential-reddening corrected magnitudes (``<band>_drcorr``) added.
    """
    # make a copy to preserve the original dataframe
    df_cluster = df_cluster.copy()
    df_field = df_field.copy()

    groups, subfovs, instrument_fov = spatial_groupby(
        df_cluster,
        df_field,
        racol=racol,
        deccol=deccol,
        minstars=minstars,
    )

    ra_center = df_cluster[racol].min() + (df_cluster[racol].max() - df_cluster[racol].min()) / 2
    dec_center = (
        df_cluster[deccol].min() + (df_cluster[deccol].max() - df_cluster[deccol].min()) / 2
    )

    fov_ratio = [subfov.area / parallel_fov for subfov in subfovs]

    field_pts = np.column_stack(
        (
            df_field[field_mag_col[0]].values - df_field[field_mag_col[1]].values,
            df_field[field_mag_col[0]].values,
        )
    )

    # initialize the cluster columns for membership and differential reddening
    dr_band1_corr = target_mag_col[0] + "_drcorr"
    dr_band2_corr = target_mag_col[1] + "_drcorr"
    df_cluster[dr_band1_corr] = df_cluster[target_mag_col[0]]
    df_cluster[dr_band2_corr] = df_cluster[target_mag_col[1]]

    dr_params["band1"] = dr_band1_corr
    dr_params["band2"] = dr_band2_corr

    dr_params["xcol"] = racol
    dr_params["ycol"] = deccol

    df_cluster["delta_ebv"] = 0.0
    df_cluster["membership"] = 1.0

    it = 0
    it_stop = None
    while True:
        logger.info(f"ITERATION {it + 1}")

        # reddening differential correction
        logger.info("Compute differential reddening..")
        if it == 0:
            logger.info("First iteration, correction threshold skipped")
            calc_correction_threshold = False
        else:
            logger.info("Do calculation of the correction threshold")
            calc_correction_threshold = True

        b1_corr, b2_corr, EBV, poli = compute_differential_reddening(
            df_cluster,
            params=dr_params,
            member_threshold=member_threshold,
            calc_correction_threshold=calc_correction_threshold,
            roi=roi,
            plot=plot_dred,
        )

        roi = poli
        df_cluster[dr_band1_corr] = b1_corr
        df_cluster[dr_band2_corr] = b2_corr
        df_cluster["delta_ebv"] += EBV

        if plot_dred:
            # plot new cmd
            _, ax = plt.subplots(layout="constrained")

            utplot.plot_cmd(
                df_cluster[dr_band1_corr],
                df_cluster[dr_band2_corr],
                ax,
            )
            # ax.set(xlim=(0.5, 3.5), ylim=(26.5, 14))

            # plot reddening map
            _ = utplot.plot_reddening_map(
                df_cluster[racol],
                df_cluster[deccol],
                ra_center,
                dec_center,
                EBV,
            )
            # plt.show(block=False)
            plt.draw()
            plt.pause(1.0)

        logger.info("...differential reddening completed")
        logger.info(
            f"EBV stat (mean, min, max): {np.median(EBV):.6f}, {EBV.min():.6f}, {EBV.max():.6f}"
        )

        logger.info(
            f"Number of stars with null ebv: {np.sum(EBV == 0.0) / EBV.shape[0] * 100:.2f}%"
            f" ({np.sum(EBV == 0.0)} over {EBV.shape[0]})"
        )
        photometric_error = float(
            np.median(
                (
                    df_cluster[target_mag_col_error[0]] ** 2
                    + df_cluster[target_mag_col_error[1]] ** 2
                )
                ** 0.5
            )
        )
        logger.info(f"Mean photometric error: {photometric_error:.6f}")

        logger.info("Start decontamination process..")
        for i, (group, fovr) in enumerate(zip(groups, fov_ratio)):
            # updating the membership of the groups
            group.loc[:, "membership"] = df_cluster.loc[group.index, "membership"]
            group.loc[:, dr_band1_corr] = df_cluster.loc[group.index, dr_band1_corr]
            group.loc[:, dr_band2_corr] = df_cluster.loc[group.index, dr_band2_corr]

            logger.info(f"\t#### GROUP N.{i + 1} ####")

            # Voronoi grid creation
            xg = group[dr_band1_corr].values - group[dr_band2_corr].values
            yg = group[dr_band1_corr].values
            cluster_pts = np.column_stack((xg, yg))

            # dilate regions to have at least n stars in each cell
            if do_dilation:
                if min_star_per_cell > group.shape[0]:
                    logger.info(
                        f"\tNot enough stars in this region "
                        f"({group.shape[0]}, requested {min_star_per_cell})"
                    )
                    dilate = False
                else:
                    dilate = True
            else:
                dilate = False

            cvt_grid = cvtGrid(
                cluster_pts,
                iter=1,
                dilate=dilate,
                which=which_voronoi,
                target_median=min_star_per_cell,
            )

            regions = (
                cvt_grid.dilated_grid
                if (dilate and cvt_grid.dilated_grid is not None)
                else cvt_grid.grid
            )
            cluster_ids = group.index.values

            # count field stars in cells
            cell_field_counts = get_regions_count(regions, field_pts)
            points_in_regions = get_pts_in_regions(regions, cluster_pts)

            if plot_voronoi:
                _ = utplot.plot_cmd_and_vorgrid(
                    group[dr_band1_corr].values,
                    group[dr_band2_corr].values,
                    regions,
                    points_in_regions,
                )
                # plt.show(block=False)
                plt.draw()
                plt.pause(1.0)

            active_regions = [el[0] for el in points_in_regions]

            # add membership
            membership = get_membership(
                cell_field_counts,
                points_in_regions,
                active_regions,
                membership_iter,
                fovr,
                method=membership_method,
            )

            if len(membership) > 0:
                logger.info(
                    f"\tMembership (mean, min, max): "
                    f"({membership[:, 1].mean():.4f},"
                    f" {membership[:, 1].min():.4f},"
                    f" {membership[:, 1].max():.4f})\n\n"
                )

                df_cluster.loc[cluster_ids[membership[:, 0].astype(int)], "membership"] = (
                    membership[:, 1]
                )

        if it == it_stop or it + 1 >= max_iter:
            break

        if abs(float(np.median(EBV))) < abs(photometric_error) and it > 0:
            logger.info("Differential reddening converged, stopping iterations.")
            logger.info("Next iteration will be the last one.")
            it_stop = it + 1

        it += 1

    return df_cluster


# Backward compatibility alias
do_statistical_membership = compute_statistical_membership
