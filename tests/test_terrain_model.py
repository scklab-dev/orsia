"""Check the application-owned terrain's common interpolant, support and ownership."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_geometry import intersect_terrain, surface_points

from notebooks.support.terrain import SplineTerrain


def grid():
    x = np.array([-10.0, -6.0, -1.0, 3.0, 10.0])
    y = np.array([-8.0, -4.0, -2.0, 0.0, 2.0, 8.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x, y, 2 + 0.03 * xx**2 + 0.02 * yy**2 + 0.01 * xx * yy


def test_terrain_uses_one_interpolant_at_interior_edges_and_corners():
    terrain = SplineTerrain(*grid())
    xy = np.array([[x, y] for x in (-10.0, -1.7, 10.0) for y in (-8.0, 2.3, 8.0)])
    x, y = xy.T
    assert_allclose(
        terrain.height(xy), 2 + 0.03 * x**2 + 0.02 * y**2 + 0.01 * x * y, rtol=1e-11, atol=1e-11
    )
    expected = np.column_stack((0.06 * x + 0.01 * y, 0.04 * y + 0.01 * x))
    assert_allclose(terrain.gradient(xy), expected, atol=1e-11)
    points, normals = surface_points(xy, height=terrain.height, gradient=terrain.gradient)
    assert_allclose(points[:, 2], terrain.height(xy))
    expected_normal = np.column_stack((-expected, np.ones(len(xy))))
    expected_normal /= np.linalg.norm(expected_normal, axis=1, keepdims=True)
    assert_allclose(normals, expected_normal, atol=1e-11)
    for shape in ((2,), (2, 3, 2), (0, 2), (2, 0, 2)):
        query = np.zeros(shape)
        height, slope = terrain.height(query), terrain.gradient(query)
        assert height.shape == shape[:-1] and slope.shape == shape
        assert height.flags.writeable and slope.flags.writeable
        assert not np.shares_memory(height, terrain.heights_m)
        assert not np.shares_memory(slope, query)
    target = np.array([0.0, 0.0, 2.0])
    direction = np.array([0.6, 0.0, -0.8])
    hit, actual = intersect_terrain(
        target - 5.3 * direction,
        direction,
        height=terrain.height,
        gradient=terrain.gradient,
        horizontal_bounds_m=terrain.horizontal_bounds_m,
        distance_nodes_m=np.linspace(0, 8, 9),
        distance_tolerance_m=1e-8,
        height_tolerance_m=1e-9,
        minimum_incidence_cosine=0.1,
    )
    assert bool(hit)
    assert_allclose(actual, target, atol=2e-9)


def test_terrain_snapshots_readonly_grid_and_returns_independent_values():
    arrays = grid()
    for values in arrays:
        values.flags.writeable = False
    terrain = SplineTerrain(*arrays)
    expected = terrain.height(np.array([1.0, 2.0]))
    for source, snapshot in zip(arrays, (terrain.x_m, terrain.y_m, terrain.heights_m), strict=True):
        assert not np.shares_memory(source, snapshot)
        with pytest.raises(ValueError):
            snapshot.flags.writeable = True
        source.flags.writeable = True
        source[:] = 0
    assert_allclose(terrain.height(np.array([1.0, 2.0])), expected)
    output = terrain.height(np.array([1.0, 2.0]))
    output[...] = -1
    assert_allclose(terrain.height(np.array([1.0, 2.0])), expected)
    assert terrain.horizontal_bounds_m == (-10, 10, -8, 8)


@pytest.mark.parametrize(
    "query",
    [
        np.array(1.0),
        np.zeros(3),
        np.array([np.nan, 0]),
        np.array([-10.0001, 0]),
        np.array([0, 8.0001]),
    ],
)
def test_terrain_rejects_unsupported_queries(query):
    terrain = SplineTerrain(*grid())
    for operation in (terrain.height, terrain.gradient):
        with pytest.raises(ValueError):
            operation(query)


@pytest.mark.parametrize(
    "kind",
    ["short", "reversed", "nonfinite_axis", "axis_shape", "height_shape", "nonfinite_height"],
)
def test_terrain_rejects_invalid_grids(kind):
    x, y, heights = grid()
    if kind == "short":
        x, heights = x[:3], heights[:3]
    elif kind == "reversed":
        x = x[::-1]
    elif kind == "nonfinite_axis":
        x[0] = np.nan
    elif kind == "axis_shape":
        x = x[None, :]
    elif kind == "height_shape":
        heights = heights.T
    else:
        heights[0, 0] = np.nan
    with pytest.raises(ValueError):
        SplineTerrain(x, y, heights)
