import numpy as np
import pytest
from shapely import Point, Polygon

from mgcs_pytools.statistical_membership.region_tools import (
    get_pts_in_regions,
    get_regions_count,
)


@pytest.fixture
def sample_grid():
    # 2x2 unit box grid: [0,1]x[0,1], [1,2]x[0,1], [0,1]x[1,2], [1,2]x[1,2]
    r0 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r1 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    r2 = Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
    r3 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    return [r0, r1, r2, r3]


def test_get_regions_count_with_shapely_points(sample_grid):
    points = [
        Point(0.5, 0.5),  # r0
        Point(0.2, 0.8),  # r0
        Point(1.5, 0.5),  # r1
        Point(0.5, 1.5),  # r2
        Point(1.5, 1.5),  # r3
        Point(1.8, 1.8),  # r3
    ]
    counts = dict(get_regions_count(sample_grid, points))
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[3] == 2


def test_get_regions_count_with_numpy_coords(sample_grid):
    coords = np.array(
        [
            [0.5, 0.5],
            [0.2, 0.8],
            [1.5, 0.5],
            [1.5, 1.5],
        ]
    )
    counts = dict(get_regions_count(sample_grid, coords))
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[3] == 1
    assert 2 not in counts  # region 2 has 0 points


def test_get_pts_in_regions(sample_grid):
    points = np.array(
        [
            [0.5, 0.5],  # idx 0 in r0
            [1.5, 0.5],  # idx 1 in r1
            [0.8, 0.2],  # idx 2 in r0
            [1.2, 1.2],  # idx 3 in r3
        ]
    )
    res = dict(get_pts_in_regions(sample_grid, points))
    assert sorted(res[0]) == [0, 2]
    assert res[1] == [1]
    assert res[3] == [3]


def test_empty_inputs(sample_grid):
    assert get_regions_count([], []) == []
    assert get_regions_count(sample_grid, []) == []
    assert get_pts_in_regions([], []) == []
    assert get_pts_in_regions(sample_grid, []) == []
