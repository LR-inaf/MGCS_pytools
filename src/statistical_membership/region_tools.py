import numpy as np
from shapely import Polygon, STRtree, Point


def get_regions_count(
    regions: list[Polygon], points: list[Point]
) -> list[tuple[int, int]]:
    """Get the count of points in each region.

    Args:
        regions: list of shapely Polygons
            List of regions to check.
        points: list of shapely Points
            List of points to check.

    Returns:
        list of tuples
            List of tuples with region index and count of points in that region.
    """

    k3_regs = STRtree(regions)
    # check if points are shapely Points object and convert if necessary
    if not all(isinstance(pt, Point) for pt in points):
        points = [Point(*pt) for pt in points if not isinstance(pt, Point)]

    _, region_ids = k3_regs.query(points, predicate="intersects")

    regions_count = [(reg, list(region_ids).count(reg)) for reg in set(region_ids)]
    return regions_count


def get_pts_in_regions(
    regions: list[Polygon], points: list[Point]
) -> list[tuple[int, list[Point]]]:
    """Get the points in each region.

    Args:
        regions: list of shapely Polygons
            List of regions to check.
        points: list of shapely Points
            List of points to check.

    Returns:
        list of tuples
            List of tuples with region index and list of points in that region.
    """

    k3_pts = STRtree(regions)
    id_pts, id_region = k3_pts.query(points, predicate="intersects")
    pts_in_regions = [
        (reg, id_pts[np.where(id_region == reg)[0]]) for reg in np.unique(id_region)
    ]
    return pts_in_regions
