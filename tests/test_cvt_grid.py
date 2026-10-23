import numpy as np

from mgcs_pytools.statistical_membership.cvt_grid import (
    CVTGrid,
    _centroid_region,
    generate_centroids,
    generate_weights,
)


def test_centroid_region_square():
    # Square [0, 2] x [0, 2]
    vertices = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    c = _centroid_region(vertices)
    np.testing.assert_allclose(c, [[1.0, 1.0]], atol=1e-6)


def test_centroid_region_triangle():
    # Triangle (0,0), (3,0), (0,3) -> centroid should be (1, 1)
    vertices = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 0.0],
        ]
    )
    c = _centroid_region(vertices)
    np.testing.assert_allclose(c, [[1.0, 1.0]], atol=1e-6)


def test_cvt_grid_dilation():
    np.random.seed(42)
    points = np.random.uniform(0, 10, size=(50, 2))

    grid_obj = CVTGrid(points, which="dilation", iter=2, dilate=False)
    assert len(grid_obj.grid) > 0
    assert grid_obj.dilated_grid is None

    # Test with dilation
    grid_dilated = CVTGrid(points, which="dilation", iter=2, dilate=True, target_median=3)
    assert grid_dilated.dilated_grid is not None
    assert len(grid_dilated.dilated_grid) > 0


def test_cvt_grid_wkmeans():
    np.random.seed(42)
    points = np.random.uniform(0, 10, size=(60, 2))

    weights = generate_weights(points)
    assert len(weights) == 60
    assert not np.isnan(weights).any()

    centroids = generate_centroids(points, points_per_bin=10)
    assert centroids.shape == (6, 2)

    grid_wkmeans = CVTGrid(points, which="wkmeans", iter=2, target_median=10)
    assert len(grid_wkmeans.grid) > 0
