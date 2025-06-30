import numpy as np
from math import modf

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from shapely import (
    Polygon,
    Point,
    union,
    difference,
    intersection,
)

from scipy.spatial import ConvexHull

from .dr_correction import compute_differential_reddening
from .cvt_grid import cvtGrid
from .region_tools import get_regions_count, get_pts_in_regions

import sys

sys.path.append("..")  # add parent directory to path
from utils import plotting as utplot


def spatial_groupby(df_cluster, df_field, racol, deccol, minstars=200):

    # Calculate the field star density and the min area for at least min stars
    ra_center = (
        df_field[racol].min() + (df_field[racol].max() - df_field[racol].min()) / 2
    )
    dec_center = (
        df_field[deccol].min() + (df_field[deccol].max() - df_field[deccol].min()) / 2
    )

    field_center = SkyCoord(ra_center, dec_center, unit=u.deg)

    df_field.loc[:, "center_sep"] = field_center.separation(
        SkyCoord(
            df_field[racol],
            df_field[deccol],
            unit="deg",
        )
    ).arcsec

    area_conversion = 3600**2
    star_density = df_field.shape[0] / (np.pi * df_field["center_sep"].max() ** 2)
    min_stars = minstars
    min_area = min_stars / star_density

    print(
        f"Star density: {star_density:.2f} stars / arcsec^2",
        f"Min area for at least {min_stars} stars: {min_area:.2f} arcsec^2",
    )

    # groupby the cluster datafram in bin with at least minstars
    ra_center = (
        df_cluster[racol].min()
        + (df_cluster[racol].max() - df_cluster[racol].min()) / 2
    )
    dec_center = (
        df_cluster[deccol].min()
        + (df_cluster[deccol].max() - df_cluster[deccol].min()) / 2
    )

    # make the FoV polygon from coordinates
    hull = ConvexHull(df_cluster.loc[:, [racol, deccol]].values)
    fov = Polygon(hull.points[hull.vertices])

    dr = 1 / 3600
    r = dr
    subfovs = [Point(ra_center, dec_center)]

    # iterating by increasing the annulus radius
    while True:

        # make annulus by subtraction wrt the last subfov
        subfov = Point(ra_center, dec_center).buffer(r)

        if subfov.contains(fov):
            break  # max fov exceeded
        elif (
            intersection(difference(subfov, subfovs[-1]), fov).area
            > min_area / area_conversion
        ):
            pol = intersection(difference(subfov, subfovs[-1]), fov)
            subfovs.append(subfov)
        r += dr

    subfovs[-1] = union(intersection(fov, subfovs[-1]), difference(fov, subfovs[-1]))
    subfovs = [difference(subfovs[i], subfovs[i - 1]) for i in range(1, len(subfovs))]

    # TODO: This part need to be change using pandas groupby
    groups = []
    for subf in subfovs:
        group = df_cluster[
            df_cluster.apply(lambda x: Point(x[racol], x[deccol]).within(subf), axis=1)
        ].copy()
        groups.append(group)

    return groups, subfovs, fov


def get_membership(cell_counts, cell_points, common_regions, iter=1000, fov_ratio=1.0):
    """
    Get the membership of each stars thorugh iterative random extractions."""

    cell_counts_points = [
        (
            [count for rrid, count in cell_counts if rrid == rid][0],
            [pts for rrid, pts in cell_points if rrid == rid][0],
        )
        for rid in common_regions
    ]

    # initialize extractions vector
    extractions = np.full(
        iter * np.sum([count for count, _ in cell_counts_points]),
        fill_value=-1,
        dtype=int,
    )

    c = 0
    for _ in range(iter):
        for count, pts in cell_counts_points:
            # rescaling the counts
            new_count = count * fov_ratio
            if new_count < 1.0:
                new_count = np.random.choice(
                    np.array([0, 1]), p=[1 - new_count, new_count]
                )
            else:
                p, bias = modf(new_count)
                new_count = int(bias) + np.random.choice(np.array([0, 1]), p=[1 - p, p])

            if len(pts) > new_count:
                extractions[c : c + new_count] = np.random.choice(
                    pts, new_count, replace=False
                )
                c += new_count
            else:
                extractions[c : c + (len(pts))] = list(pts)
                c += len(pts)

    extractions = extractions[extractions != -1]
    extractions = sorted(extractions)
    unique_id = np.unique(extractions)
    bincount = np.bincount(extractions)
    bincount = bincount[bincount > 0]

    membership = np.vstack((unique_id, bincount)).T
    res = [1 - count / iter for _, count in membership]

    membership = np.array(membership).astype(float)
    membership[:, 1] = res

    return membership


def do_statistical_membership(
    df_cluster_input,
    df_field_input,
    field_mag_col,
    dr_params,
    member_threshold=0.8,
    minstars=200,
    racol="RA",
    deccol="DEC",
    process_iter=3,
    min_star_per_cell=3,
    memebership_iter=1000,
    fov_ratio=1.0,
    roi=None,
    plot_dred=False,
    plot_voronoi=False,
    which_voronoi="dilation",
    do_dilation=True,
):
    # make a copy to preserve the original dataframe
    df_cluster = df_cluster_input.copy()
    df_field = df_field_input.copy()

    groups, subfovs, instrument_fov = spatial_groupby(
        df_cluster,
        df_field,
        racol=racol,
        deccol=deccol,
        minstars=minstars,
    )

    ra_center = (
        df_cluster[racol].min()
        + (df_cluster[racol].max() - df_cluster[racol].min()) / 2
    )
    dec_center = (
        df_cluster[deccol].min()
        + (df_cluster[deccol].max() - df_cluster[deccol].min()) / 2
    )

    fov_ratio = [subfov.area / instrument_fov.area for subfov in subfovs]

    field_pts = [
        Point(row[field_mag_col[0]] - row[field_mag_col[1]], row[field_mag_col[0]])
        for _, row in df_field.iterrows()
    ]

    # initialize the cluster columns for membership and differential reddening
    dr_band1_corr = dr_params["band1"] + "_drcorr"
    dr_band2_corr = dr_params["band2"] + "_drcorr"
    df_cluster[dr_band1_corr] = df_cluster[dr_params["band1"]]
    df_cluster[dr_band2_corr] = df_cluster[dr_params["band2"]]

    dr_params["band1"] = dr_band1_corr
    dr_params["band2"] = dr_band2_corr

    df_cluster["delta_ebv"] = 0.0
    df_cluster["membership"] = 1.0

    for it in range(process_iter):
        print(f"ITERATION {it+1}")

        # reddening differential correction
        print("compute differential reddening..")
        print(df_cluster["membership"].describe())
        if it == 0:
            print("First iteration, correction threshold skipped")
            calc_corr_threhsold = False
        else:
            print(f"do calcuation of the Correction threshold")
            calc_corr_threhsold = True

        b1_corr, b2_corr, EBV, poli = compute_differential_reddening(
            df_cluster,
            params=dr_params,
            member_threshold=member_threshold,
            calc_corr_threhsold=calc_corr_threhsold,
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
                ax,
                df_cluster,
                [dr_band1_corr, dr_band2_corr],
                color="black",
                inverty=True,
            )
            ax.set(xlim=(0.5, 3.5), ylim=(26.5, 14))

            # plot reddening map
            _ = utplot.plot_reddening_map(
                df_cluster["ra(1)"],
                df_cluster[deccol],
                ra_center,
                dec_center,
                EBV,
            )
            # plt.show(block=False)
            plt.draw()
            plt.pause(1.0)

        print("...differential reddening completed")
        print(f"EBV stat: {EBV.mean():.4f}, {EBV.min():.4f}, {EBV.max():.4f}")

        print("Start decontamionation process..")
        for i, (group, fovr) in enumerate(zip(groups, fov_ratio)):

            # updating the membership of the groups
            # !!! this can be avoided if pandas groupby is used !!!
            group.loc[:, "membership"] = df_cluster.loc[group.index, "membership"]
            group.loc[:, dr_band1_corr] = df_cluster.loc[group.index, dr_band1_corr]
            group.loc[:, dr_band2_corr] = df_cluster.loc[group.index, dr_band2_corr]

            print(f"\t#### GROUP N.{i + 1} ####")

            # Voronoi grid creation
            xg = group[dr_band1_corr] - group[dr_band2_corr]
            yg = group[dr_band1_corr]
            cluster_pts = [Point(x, y) for x, y in zip(xg, yg)]

            # dilate regions to have at least n stars in each cell
            if do_dilation:
                if min_star_per_cell > group.shape[0]:
                    print(
                        f"\tNot enough stars in this region "
                        f"({group.shape[0]}, requested {min_star_per_cell})"
                    )
                    dilate = False
                else:
                    dilate = True
            else:
                dilate = False

            # cvt_grid = cvtGrid(points, iter=1, dilate=dilate)

            cvt_grid = cvtGrid(
                np.array([xg.values, yg.values]).T,
                iter=1,
                dilate=dilate,
                which=which_voronoi,
                target_median=min_star_per_cell,
            )

            regions = cvt_grid.grid if not dilate else cvt_grid.dilated_grid
            cluster_ids = group.index.values

            # count field stars in cells
            cell_field_counts = get_regions_count(regions, field_pts)
            points_in_reigons = get_pts_in_regions(regions, cluster_pts)

            if plot_voronoi:
                _ = utplot.plot_cmd_and_vorgrid(
                    group,
                    [dr_band1_corr, dr_band2_corr],
                    regions,
                    points_in_reigons,
                )
                # plt.show(block=False)
                plt.draw()
                plt.pause(1.0)

            common_regions = set([el[0] for el in cell_field_counts]).intersection(
                set([el[0] for el in points_in_reigons])
            )

            # add membership
            membership = get_membership(
                cell_field_counts,
                points_in_reigons,
                common_regions,
                memebership_iter,
                fovr,
            )

            print(
                f"\tMembership (mean, min, max): "
                f"({membership[:, 1].mean():.4f},"
                f" {membership[:, 1].min():.4f},"
                f" {membership[:, 1].max():.4f})\n\n"
            )

            df_cluster.loc[cluster_ids[membership[:, 0].astype(int)], "membership"] = (
                membership[:, 1]
            )

    return df_cluster
