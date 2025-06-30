import numpy as np
import sys
from scipy.spatial import Voronoi
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans

from shapely import (
    Polygon,
    STRtree,
    union,
)

from .region_tools import get_regions_count, get_pts_in_regions


def _bounded_vor(points, bounding_box):
    eps = sys.float_info.epsilon

    # Mirror points
    points_center = points
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(
        points_center,
        np.append(
            np.append(points_left, points_right, axis=0),
            np.append(points_down, points_up, axis=0),
            axis=0,
        ),
        axis=0,
    )
    # Compute Voronoi
    vor = Voronoi(points)

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (
                    bounding_box[0] - eps <= x
                    and x <= bounding_box[1] + eps
                    and bounding_box[2] - eps <= y
                    and y <= bounding_box[3] + eps
                ):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def _centroid_region(vertices):
    # Polygon's signed area
    A = 0

    # Centroid's x
    C_x = 0

    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1]
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y

    return np.array([[C_x, C_y]])


def _cvt_vor(init_points, iters, padx=0.01, pady=0.01):
    points = init_points

    for i in range(iters):

        vor = _bounded_vor(
            points,
            [
                points[:, 0].min() - padx,
                points[:, 0].max() + padx,
                points[:, 1].min() - pady,
                points[:, 1].max() + pady,
            ],
        )

        if i == iters - 1 and iters > 1:
            break
        centroids = []
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            centroid = _centroid_region(vertices)
            centroids.append(list(centroid[0, :]))

        points = np.array(centroids)
    return vor, np.array(centroids)


def _aggregate_regions(regions):

    joint_regions = []

    old_touched_regions = []
    new_touched_regions = []

    # sort regions for their area
    regions = sorted(regions, key=lambda x: Polygon(x).area, reverse=False)
    regions = np.array(regions)
    for i, reg in enumerate(regions):

        if i in old_touched_regions:
            continue

        ktree_regions = STRtree(regions)

        new_touched_regions = ktree_regions.query(reg, predicate="touches")

        new_touched_regions = np.delete(
            new_touched_regions,
            np.where(np.isin(new_touched_regions, old_touched_regions)),
        )

        old_touched_regions = np.append(old_touched_regions, new_touched_regions)
        old_touched_regions = np.append(old_touched_regions, i)

        joint_reg = reg
        for touched_reg in regions[new_touched_regions]:
            joint_reg = union(joint_reg, touched_reg)

        joint_regions.append(joint_reg)

    return np.array(joint_regions)


def _dilate_grid(regions, points, target_median):

    joint_regions = np.array(regions)

    c = 0
    while True:
        joint_regions = _aggregate_regions(joint_regions)
        pts_in_regions = np.array(get_regions_count(joint_regions, points))
        if np.median(pts_in_regions[:, 1]) >= target_median:
            break
        c += 1
        print(c)

    return joint_regions


def generate_weights(XY):
    #       compute 2d map to evaluate weights
    hist, xedges, yedges = np.histogram2d(
        x=XY[:, 0], y=XY[:, 1], bins=200, density=True
    )
    xedges = 0.5 * (xedges[1:] + xedges[:-1])
    yedges = 0.5 * (yedges[1:] + yedges[:-1])

    # 	interpolate the grid
    interp = RegularGridInterpolator(
        (xedges, yedges), hist, bounds_error=False, fill_value=np.nan
    )  # from scipy
    weights = interp((XY[:, 0], XY[:, 1]))
    return weights


def generate_centroids(pos, points_per_bin=None, nbins=None, weights=None):
    if points_per_bin is None and nbins is None:
        raise ValueError("points_per_bin or nbins must be specificied")

    if points_per_bin is not None:
        nbins = pos.shape[0] // points_per_bin

    # 	Perform KMeans clustering to divide the points into nbins clusters
    kmeans = KMeans(n_clusters=nbins, random_state=0)
    kmeans.fit(pos) if weights is None else kmeans.fit(pos, sample_weight=weights)
    # labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return centroids


# class cvtGrid:
# def __init__(
# self, points, iter=3, padx=0.01, pady=0.01, dilate=False, target_median=3
# ):
# self.cvor, self.cpoints = _cvt_vor(points, iter, padx, pady)
#
# self.grid = np.array(
# [Polygon(self.cvor.vertices[reg]) for reg in self.cvor.filtered_regions]
# )
#
# self.dilated_grid = (
# None if not dilate else _dilate_grid(self.grid, points, target_median)
# )
class cvtGrid:
    def __init__(
        self,
        points,
        which="dilation",
        iter=3,
        padx=0.01,
        pady=0.01,
        dilate=False,
        target_median=3,
    ):
        if which == "dilation":
            self.cvor, self.cpoints = _cvt_vor(points, iter, padx, pady)

            self.grid = np.array(
                [Polygon(self.cvor.vertices[reg]) for reg in self.cvor.filtered_regions]
            )

            self.dilated_grid = (
                None if not dilate else _dilate_grid(self.grid, points, target_median)
            )
        elif which == "wkmeans":
            weights = generate_weights(points)
            weights = np.where(np.isnan(weights), 0.0, weights)
            centroids = generate_centroids(
                points, points_per_bin=target_median, weights=weights
            )

            self.cvor, self.cpoints = _cvt_vor(centroids, iter, padx, pady)
            self.grid = np.array(
                [Polygon(self.cvor.vertices[reg]) for reg in self.cvor.filtered_regions]
            )
