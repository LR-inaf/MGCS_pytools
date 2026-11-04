"""Utility functions for spatial polygon regions and point queries."""

from typing import Sequence

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import Point, Polygon, STRtree


def get_regions_count(
    regions: Sequence[Polygon] | NDArray,
    points: Sequence[Point] | NDArray,
) -> list[tuple[int, int]]:
    """Get the count of points falling within each polygon region.

    Parameters
    ----------
    regions : Sequence[Polygon] or NDArray
        Polygonal regions to query.
    points : Sequence[Point] or NDArray
        Points to count, either as a sequence of Shapely Point objects or as an
        (N, 2) NumPy array of (x, y) coordinates.

    Returns
    -------
    list of tuple of (int, int)
        List of (region_index, count) tuples for regions containing at least one point.
    """
    if len(regions) == 0:
        return []

    # Fast vectorized conversion if numpy array or raw coordinates
    if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2:
        pts_geom = shapely.points(points[:, 0], points[:, 1])
    else:
        # Check if already shapely Point instances, otherwise convert
        pts_list = list(points)
        if len(pts_list) == 0:
            return []
        if all(isinstance(pt, Point) for pt in pts_list):
            pts_geom = pts_list
        else:
            coords = np.array(
                [[pt.x, pt.y] if isinstance(pt, Point) else [pt[0], pt[1]] for pt in pts_list],
                dtype=float,
            )
            pts_geom = shapely.points(coords[:, 0], coords[:, 1])

    tree = STRtree(regions)
    _, region_ids = tree.query(pts_geom, predicate="intersects")

    if len(region_ids) == 0:
        return []

    unique_regs, counts = np.unique(region_ids, return_counts=True)
    return [(int(r), int(c)) for r, c in zip(unique_regs, counts)]


def get_pts_in_regions(
    regions: Sequence[Polygon] | NDArray,
    points: Sequence[Point] | NDArray,
) -> list[tuple[int, list[int]]]:
    """Get the point indices assigned to each region.

    Parameters
    ----------
    regions : Sequence[Polygon] or NDArray
        Polygonal regions to query.
    points : Sequence[Point] or NDArray
        Points to partition, either as a sequence of Shapely Point objects or
        as an (N, 2) array of coordinates.

    Returns
    -------
    list of tuple of (int, list of int)
        List of (region_index, list_of_point_indices) tuples.
    """
    if len(regions) == 0:
        return []

    if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2:
        pts_geom = shapely.points(points[:, 0], points[:, 1])
    else:
        pts_list = list(points)
        if len(pts_list) == 0:
            return []
        if all(isinstance(pt, Point) for pt in pts_list):
            pts_geom = pts_list
        else:
            coords = np.array(
                [[pt.x, pt.y] if isinstance(pt, Point) else [pt[0], pt[1]] for pt in pts_list],
                dtype=float,
            )
            pts_geom = shapely.points(coords[:, 0], coords[:, 1])

    tree = STRtree(regions)
    id_pts, id_region = tree.query(pts_geom, predicate="intersects")

    if len(id_region) == 0:
        return []

    # Vectorized sorting and splitting to group point indices by region ID
    sort_order = np.argsort(id_region)
    sorted_regions = id_region[sort_order]
    sorted_pts = id_pts[sort_order]

    unique_regs, split_indices = np.unique(sorted_regions, return_index=True)
    grouped_pts = np.split(sorted_pts, split_indices[1:])

    return [(int(reg), pts.tolist()) for reg, pts in zip(unique_regs, grouped_pts)]
