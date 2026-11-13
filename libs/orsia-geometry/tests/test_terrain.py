"""Analytic affine-terrain and ragged horizontal-layer geometry references."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import affine_surface_points, intersect_affine_terrain, segment_layer_paths


def test_plane_round_trip_and_supported_misses():
    xy = np.array([[2.0, 3.0], [-5.0, 7.0]])
    slopes = np.array([0.2, -0.1])
    points, normal = affine_surface_points(xy, height_m=10.0, slopes=slopes)
    origins = points + 100 * normal
    directions = np.broadcast_to(-normal, origins.shape)
    hit, actual = intersect_affine_terrain(
        origins, directions, height_m=10, slopes=slopes, minimum_incidence_cosine=0.1
    )
    assert hit.all()
    assert_allclose(actual, points, atol=2e-14)
    assert_allclose(actual[:, :2], xy, atol=2e-14)
    hit, missing = intersect_affine_terrain(
        origins, -directions, height_m=10, slopes=slopes, minimum_incidence_cosine=0.1
    )
    assert not hit.any() and np.isnan(missing).all()
    for origin, direction in (
        ([0.0, 0.0, 9.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 20.0], [1.0, 0.0, 0.0]),
    ):
        hit, missing = intersect_affine_terrain(
            np.array(origin),
            np.array(direction),
            height_m=10,
            slopes=np.zeros(2),
            minimum_incidence_cosine=0.1,
        )
        assert not hit and np.isnan(missing).all()
    hit, point = intersect_affine_terrain(
        np.array([0.0, 0.0, 10.0]),
        np.array([0.0, 0.0, -1.0]),
        height_m=10,
        slopes=np.zeros(2),
        minimum_incidence_cosine=0.1,
    )
    assert hit
    assert_array_equal(point, [0, 0, 10])


def test_ragged_paths_partial_layers_and_length():
    origins = np.array([[0.0, 0.0, 250.0], [10.0, 20.0, 500.0], [0.0, 0.0, 0.0]])
    directions = np.array([[0.6, 0.0, 0.8], [0.0, 0.8, 0.6], [0.0, 0.0, 1.0]])
    starts, ends, offsets, layers = segment_layer_paths(
        origins, directions, np.array([0.0, 500.0, 1500.0]), minimum_upward_cosine=0.2
    )
    assert_array_equal(offsets, [0, 2, 3, 5])
    assert_array_equal(layers, [0, 1, 1, 0, 1])
    for i in range(3):
        assert_allclose(starts[offsets[i]], origins[i])
        assert_allclose(ends[offsets[i + 1] - 1, 2], 1500)
        assert_allclose(
            np.linalg.norm(
                ends[offsets[i] : offsets[i + 1]] - starts[offsets[i] : offsets[i + 1]], axis=1
            ).sum(),
            (1500 - origins[i, 2]) / directions[i, 2],
        )
    empty = segment_layer_paths(
        np.empty((0, 3)), np.empty((0, 3)), np.array([0.0, 1.0]), minimum_upward_cosine=0.1
    )
    assert empty[0].shape == (0, 3)
    assert_array_equal(empty[2], [0])


@pytest.mark.parametrize(
    "height,slopes,xy",
    [
        (np.nan, np.zeros(2), np.zeros((1, 2))),
        (0.0, np.ones(3), np.zeros((1, 2))),
        (0.0, np.zeros(2), np.zeros((1, 3))),
        (0.0, np.zeros(2), np.array([[np.nan, 0.0]])),
    ],
)
def test_invalid_surface(height, slopes, xy):
    with pytest.raises(ValueError):
        affine_surface_points(xy, height_m=height, slopes=slopes)


@pytest.mark.parametrize(
    "origin,direction,edges,minimum",
    [
        ([[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]], [0.0, 1.0], 0.1),
        ([[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [0.0, 1.0], 0.1),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]], [0.0, 1.0], 0.1),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]], [0.0, 1.0], 0.1),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0.0, 0.0], 0.1),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0.0, 1.0], 0.0),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0.1),
    ],
)
def test_invalid_paths(origin, direction, edges, minimum):
    with pytest.raises(ValueError):
        segment_layer_paths(
            np.array(origin), np.array(direction), np.array(edges), minimum_upward_cosine=minimum
        )


def test_invalid_intersection_threshold():
    with pytest.raises(ValueError):
        intersect_affine_terrain(
            np.zeros(3),
            np.array([0.0, 0.0, -1.0]),
            height_m=0,
            slopes=np.zeros(2),
            minimum_incidence_cosine=0,
        )
