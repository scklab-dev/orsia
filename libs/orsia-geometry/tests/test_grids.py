"""Exact shared supports, coordinate meaning and immutable ownership."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import (
    LayeredCells,
    LocalCartesianFrame,
    PointLattice,
    RectangularCells,
    same_local_frame,
)


def test_synthetic_identity_and_exact_anchored_frame_comparison():
    a = LocalCartesianFrame("test", "synthetic altitude")
    b = LocalCartesianFrame("test", "synthetic altitude")
    assert same_local_frame(a, a)
    assert not same_local_frame(a, b)
    origin, basis = np.array([6e6, 0.0, 0.0]), np.eye(3)
    anchored = LocalCartesianFrame("chart", "ITRF z", origin, basis)
    equal = LocalCartesianFrame("another name", "ITRF z", origin, basis)
    assert same_local_frame(anchored, equal)
    assert not same_local_frame(a, anchored)
    assert not same_local_frame(anchored, replace(equal, vertical_datum="another datum"))
    assert not same_local_frame(anchored, replace(equal, origin_itrf_m=origin + [1e-8, 0.0, 0.0]))
    assert not same_local_frame(anchored, replace(equal, basis_itrf=np.diag([-1.0, -1.0, 1.0])))
    origin.fill(99)
    basis.fill(99)
    assert_array_equal(anchored.origin_itrf_m, [6e6, 0.0, 0.0])
    for values in (anchored.origin_itrf_m, anchored.basis_itrf):
        assert values is not None and not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)


@pytest.mark.parametrize(
    "options",
    [
        {"name": ""},
        {"vertical_datum": ""},
        {"origin_itrf_m": np.zeros(3)},
        {"basis_itrf": np.eye(3)},
        {"origin_itrf_m": np.zeros(2), "basis_itrf": np.eye(3)},
        {"origin_itrf_m": np.zeros(3), "basis_itrf": np.diag([1.0, 1.0, -1.0])},
        {"origin_itrf_m": np.zeros(3), "basis_itrf": np.ones((3, 3))},
        {"origin_itrf_m": np.full(3, np.nan), "basis_itrf": np.eye(3)},
    ],
)
def test_invalid_local_chart(options):
    with pytest.raises(ValueError):
        LocalCartesianFrame(**({"name": "test", "vertical_datum": "synthetic"} | options))


def test_point_lattice_has_y_x_values_but_x_y_coordinates():
    frame = LocalCartesianFrame("test", "synthetic")
    x, y = np.array([2.0, 5.0, 8.0]), np.array([-1.0, 3.0])
    grid = PointLattice(x, y, frame)
    equivalent = PointLattice.regular((2, 3), (3.0, 4.0), (2.0, -1.0), frame)
    assert grid.matches(equivalent) and grid.shape == (2, 3)
    assert_array_equal(grid.xy_m[1, 2], [8.0, 3.0])
    assert not grid.matches(replace(grid, x_m=x + 1e-12))
    assert not grid.matches(replace(grid, y_m=y + 1e-12))
    assert not grid.matches(replace(grid, frame=LocalCartesianFrame("test", "synthetic")))
    x.fill(99)
    y.fill(99)
    assert_array_equal(grid.x_m, [2.0, 5.0, 8.0])
    for axis in (grid.x_m, grid.y_m):
        with pytest.raises(ValueError):
            axis.setflags(write=True)
    assert grid.xy_m.flags.writeable
    assert PointLattice.regular((1, 1), (1.0, 1.0), (0.0, 0.0), frame).shape == (1, 1)


@pytest.mark.parametrize(
    "values",
    [
        np.empty(0),
        np.ones(2),
        np.array([1.0, 0.0]),
        np.ones((1, 1)),
        np.array([0.0, np.nan]),
        np.array([0, 1]),
    ],
)
def test_invalid_point_axes(values):
    with pytest.raises(ValueError):
        PointLattice(cast(Any, values), np.ones(1), LocalCartesianFrame("test", "synthetic"))


@pytest.mark.parametrize(
    "shape,spacing,first",
    [
        ((0, 1), (1.0, 1.0), (0.0, 0.0)),
        ((1, 1), (-1.0, 1.0), (0.0, 0.0)),
        ((1, 1), (np.inf, 1.0), (0.0, 0.0)),
        ((1, 1), (1.0, 1.0), (np.nan, 0.0)),
    ],
)
def test_invalid_regular_lattice(shape, spacing, first):
    with pytest.raises(ValueError):
        PointLattice.regular(shape, spacing, first, LocalCartesianFrame("test", "synthetic"))


def test_cells_retain_area_and_clipped_volume_without_point_value_casts():
    frame = LocalCartesianFrame("test", "synthetic")
    points = PointLattice.regular((2, 3), (3.0, 4.0), (2.0, -1.0), frame)
    cells = RectangularCells.around_points(points, (3.0, 4.0))
    assert cells.shape == (2, 3)
    assert_array_equal(cells.area_m2, np.full((2, 3), 12.0))
    assert_array_equal(cells.centre_xy_m, points.xy_m)
    edges = np.array([0.0, 10.0, 20.0])
    terrain = np.array([[-5.0, 3.0, 10.0], [13.0, 20.0, 30.0]])
    support = LayeredCells(cells, edges, terrain)
    assert support.shape == (2, 3, 2)
    expected = (
        np.array([[[10.0, 10.0], [7.0, 10.0], [0.0, 10.0]], [[0.0, 7.0], [0.0, 0.0], [0.0, 0.0]]])
        * 12
    )
    assert_allclose(support.volume_m3, expected)
    assert support.matches(LayeredCells(cells, edges.copy(), terrain.copy()))
    assert not support.matches(replace(support, height_edges_m=edges + 1.0))
    assert not support.matches(replace(support, surface_altitude_m=terrain + 1.0))
    assert not support.matches(
        replace(support, horizontal=replace(cells, lower_xy_m=cells.lower_xy_m - 1.0))
    )
    assert not cells.matches(replace(cells, upper_xy_m=cells.upper_xy_m + 1.0))
    assert not cells.matches(replace(cells, frame=LocalCartesianFrame("test", "synthetic")))
    edges.fill(99)
    terrain.fill(99)
    assert_array_equal(support.height_edges_m, [0.0, 10.0, 20.0])
    for values in (
        support.height_edges_m,
        support.surface_altitude_m,
        cells.lower_xy_m,
        cells.upper_xy_m,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
    assert support.volume_m3.flags.writeable


def test_invalid_rectangles_and_layer_support():
    frame = LocalCartesianFrame("test", "synthetic")
    for lo, hi in (
        (np.zeros(3), np.ones(3)),
        (np.zeros((1, 2)), np.ones((2, 2))),
        (np.zeros(2), np.zeros(2)),
        (np.array([np.nan, 0.0]), np.ones(2)),
        (np.zeros(2), np.full(2, 1e-200)),
    ):
        with pytest.raises(ValueError):
            RectangularCells(lo, hi, frame)
    cells = RectangularCells(np.zeros((1, 2)), np.ones((1, 2)), frame)
    for edges, terrain in (
        (np.array([0.0]), np.zeros(1)),
        (np.array([1.0, 0.0]), np.zeros(1)),
        (np.array([0.0, 1.0]), np.zeros(2)),
    ):
        with pytest.raises(ValueError):
            LayeredCells(cells, edges, terrain)
    with pytest.raises(ValueError):
        RectangularCells.around_points(PointLattice(np.zeros(1), np.zeros(1), frame), (0.0, 1.0))
    # Empty collections remain useful for empty mass-field selections.
    empty = LayeredCells(
        RectangularCells(np.empty((0, 2)), np.empty((0, 2)), frame),
        np.array([0.0, 1.0]),
        np.empty(0),
    )
    assert empty.volume_m3.shape == (0, 1)
