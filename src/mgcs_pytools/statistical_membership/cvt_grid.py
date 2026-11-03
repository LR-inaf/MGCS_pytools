"""Centroidal Voronoi Tessellation (CVT) and grid dilation algorithms."""

import sys
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Voronoi
from shapely import Polygon, STRtree, union
from sklearn.cluster import KMeans

from .region_tools import get_regions_count


def _bounded_vor(points: NDArray, bounding_box: Sequence[float] | NDArray) -> Voronoi:
    """Compute bounded Voronoi tessellation via point reflection.

    Parameters
    ----------
    points : NDArray
        Points to compute the Voronoi tessellation (N, 2).
    bounding_box : Sequence[float] or NDArray
        Bounding box [xmin, xmax, ymin, ymax].

    Returns
    -------
    Voronoi
        Bounded Voronoi tessellation object with filtered_points and filtered_regions.
    """
    eps = sys.float_info.epsilon
    xmin, xmax, ymin, ymax = (
        bounding_box[0],
        bounding_box[1],
        bounding_box[2],
        bounding_box[3],
    )

    points_center = points
    # Vectorized point reflections
    points_left = np.column_stack((2.0 * xmin - points_center[:, 0], points_center[:, 1]))
    points_right = np.column_stack((2.0 * xmax - points_center[:, 0], points_center[:, 1]))
    points_down = np.column_stack((points_center[:, 0], 2.0 * ymin - points_center[:, 1]))
    points_up = np.column_stack((points_center[:, 0], 2.0 * ymax - points_center[:, 1]))

    all_points = np.vstack((points_center, points_left, points_right, points_down, points_up))

    vor = Voronoi(all_points)

    # Filter regions that are completely within bounding box
    regions = []
    for region in vor.regions:
        if not region or -1 in region:
            continue
        v = vor.vertices[region]
        if (
            np.all(v[:, 0] >= xmin - eps)
            and np.all(v[:, 0] <= xmax + eps)
            and np.all(v[:, 1] >= ymin - eps)
            and np.all(v[:, 1] <= ymax + eps)
        ):
            regions.append(region)

    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def _centroid_region(vertices: NDArray) -> NDArray:
    """Compute the centroid of a 2D closed polygon using NumPy vectorization.

    Parameters
    ----------
    vertices : NDArray
        Vertices of the polygon of shape (N, 2), where the last vertex equals the first.

    Returns
    -------
    NDArray
        Centroid of the polygon with shape (1, 2).
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Vectorized cross-product terms for consecutive edges
    s = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * np.sum(s)

    if abs(signed_area) < 1e-12:
        return np.mean(vertices[:-1], axis=0, keepdims=True)

    factor = 1.0 / (6.0 * signed_area)
    cx = factor * np.sum((x[:-1] + x[1:]) * s)
    cy = factor * np.sum((y[:-1] + y[1:]) * s)

    return np.array([[cx, cy]])


def _cvt_vor(
    init_points: NDArray, iters: int, padx: float = 0.01, pady: float = 0.01
) -> Tuple[Voronoi, NDArray]:
    """Compute Centroidal Voronoi Tessellation (CVT) through Lloyd iterations.

    Parameters
    ----------
    init_points : NDArray
        Initial points to compute the CVT (N, 2).
    iters : int
        Number of Lloyd relaxation iterations.
    padx : float, optional
        Padding in x direction. Default is 0.01.
    pady : float, optional
        Padding in y direction. Default is 0.01.

    Returns
    -------
    tuple of (Voronoi, NDArray)
        Relaxed Voronoi tessellation and cell centroids.
    """
    points = init_points
    centroids = points

    for i in range(iters):
        bbox = [
            float(points[:, 0].min() - padx),
            float(points[:, 0].max() + padx),
            float(points[:, 1].min() - pady),
            float(points[:, 1].max() + pady),
        ]
        vor = _bounded_vor(points, bbox)

        if i == iters - 1 and iters > 1:
            break

        centroid_list = []
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            centroid = _centroid_region(vertices)
            centroid_list.append(centroid[0])

        if len(centroid_list) > 0:
            points = np.array(centroid_list)
            centroids = points

    return vor, np.array(centroids)


def _aggregate_regions(regions: NDArray) -> NDArray:
    """Aggregate adjacent regions based on area ordering and spatial adjacency.

    Parameters
    ----------
    regions : NDArray
        Array of Shapely Polygon regions.

    Returns
    -------
    NDArray
        Array of aggregated Polygon regions.
    """
    if len(regions) == 0:
        return np.array([])

    joint_regions = []
    touched_set: set[int] = set()

    # Sort regions ascending by area
    sorted_indices = np.argsort([r.area for r in regions])
    sorted_regions = np.array(regions)[sorted_indices]

    ktree_regions = STRtree(sorted_regions)

    for i, reg in enumerate(sorted_regions):
        if i in touched_set:
            continue

        neighbors = ktree_regions.query(reg, predicate="touches")
        valid_neighbors = [idx for idx in neighbors if idx not in touched_set]

        # Filter out regions that are excessively larger (> 10x area)
        filtered_neighbors = [
            idx for idx in valid_neighbors if sorted_regions[idx].area <= 10.0 * reg.area
        ]

        touched_set.add(i)
        touched_set.update(filtered_neighbors)

        joint_reg = reg
        for idx in filtered_neighbors:
            joint_reg = union(joint_reg, sorted_regions[idx])

        joint_regions.append(joint_reg)

    return np.array(joint_regions)


def _dilate_grid(regions: NDArray, points: NDArray, target_median: float) -> NDArray:
    """Dilate grid by aggregating regions until a target median points per region is reached.

    Parameters
    ----------
    regions : NDArray
        Array of Polygon regions.
    points : NDArray
        Point coordinates (N, 2).
    target_median : float
        Target median points per region.

    Returns
    -------
    NDArray
        Dilated Polygon regions array.
    """
    joint_regions = np.array(regions)

    while True:
        joint_regions = _aggregate_regions(joint_regions)
        counts = get_regions_count(joint_regions, points)
        if len(counts) == 0:
            break
        counts_arr = np.array(counts)
        if np.median(counts_arr[:, 1]) >= target_median:
            break

    return joint_regions


def generate_weights(xy: NDArray) -> NDArray:
    """Generate weights for points based on 2D kernel density estimation.

    Parameters
    ----------
    xy : NDArray
        Array of points of shape (N, 2).

    Returns
    -------
    NDArray
        Interpolated density weights for each point.
    """
    hist, xedges, yedges = np.histogram2d(x=xy[:, 0], y=xy[:, 1], bins=200, density=True)
    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    ycenters = 0.5 * (yedges[1:] + yedges[:-1])

    interp = RegularGridInterpolator((xcenters, ycenters), hist, bounds_error=False, fill_value=0.0)
    weights = interp(xy)
    return np.nan_to_num(weights, nan=0.0)


def generate_centroids(
    pos: NDArray,
    points_per_bin: int | None = None,
    nbins: int | None = None,
    weights: NDArray | None = None,
) -> NDArray:
    """Generate centroids using weighted k-means clustering.

    Parameters
    ----------
    pos : NDArray
        Coordinates array of shape (N, 2).
    points_per_bin : int, optional
        Target points per bin.
    nbins : int, optional
        Number of clusters to generate.
    weights : NDArray, optional
        Sample weights for clustering.

    Returns
    -------
    NDArray
        Cluster centroids of shape (nbins, 2).
    """
    if points_per_bin is None and nbins is None:
        raise ValueError("Either points_per_bin or nbins must be specified.")

    if points_per_bin is not None:
        nbins = max(1, pos.shape[0] // points_per_bin)

    assert nbins is not None
    kmeans = KMeans(n_clusters=nbins, random_state=0)
    kmeans.fit(pos, sample_weight=weights)
    return kmeans.cluster_centers_


class CVTGrid:
    """Compute Centroidal Voronoi Tessellation (CVT) grid for spatial point distributions.

    Parameters
    ----------
    points : NDArray
        Points to tessellate (N, 2).
    which : {"dilation", "wkmeans"}, optional
        Tessellation mode: "dilation" or "wkmeans". Default is "dilation".
    iter : int, optional
        Lloyd relaxation iterations. Default is 3.
    padx : float, optional
        X-axis bounding box padding. Default is 0.01.
    pady : float, optional
        Y-axis bounding box padding. Default is 0.01.
    dilate : bool, optional
        Whether to dilate cells to meet minimum star counts. Default is False.
    target_median : int, optional
        Median number of stars per cell threshold for dilation. Default is 3.

    Attributes
    ----------
    which : str
        Tessellation mode used.
    points : NDArray
        Input points that were tessellated.
    cvor : scipy.spatial.Voronoi
        Underlying Voronoi diagram.
    cpoints : NDArray
        Generators / cell centroids.
    grid : NDArray
        Array of Shapely Polygon objects defining the Voronoi cells.
    dilated_grid : list of Polygon or None
        Merged/dilated Voronoi cells if dilation was requested, None otherwise.

    Raises
    ------
    ValueError
        If ``which`` is not ``"dilation"`` or ``"wkmeans"``.
    """

    def __init__(
        self,
        points: NDArray,
        which: str = "dilation",
        iter: int = 3,
        padx: float = 0.01,
        pady: float = 0.01,
        dilate: bool = False,
        target_median: int = 3,
    ) -> None:
        self.which = which
        self.points = points

        if which == "dilation":
            self.cvor, self.cpoints = _cvt_vor(points, iter, padx, pady)
            self.grid = np.array(
                [Polygon(self.cvor.vertices[reg]) for reg in self.cvor.filtered_regions]
            )
            self.dilated_grid = _dilate_grid(self.grid, points, target_median) if dilate else None
        elif which == "wkmeans":
            weights = generate_weights(points)
            centroids = generate_centroids(points, points_per_bin=target_median, weights=weights)
            self.cvor, self.cpoints = _cvt_vor(centroids, iter, padx, pady)
            self.grid = np.array(
                [Polygon(self.cvor.vertices[reg]) for reg in self.cvor.filtered_regions]
            )
            self.dilated_grid = None
        else:
            raise ValueError(f"Unknown which={which!r}. Must be 'dilation' or 'wkmeans'.")


# Backward-compatibility alias
cvtGrid = CVTGrid
