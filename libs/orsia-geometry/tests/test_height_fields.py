"""Callable terrain, sampled contact and interpolation contracts."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import (
    affine_surface_points,
    intersect_affine_terrain,
    intersect_terrain,
    surface_points,
)
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator


def flat_height(xy):
    return np.zeros(xy.shape[:-1])


def flat_gradient(xy):
    return np.zeros_like(xy)


def controls(**changes: Any) -> dict[str, Any]:
    options: dict[str, Any] = dict(
        height=flat_height,
        gradient=flat_gradient,
        horizontal_bounds_m=(-100.0, 100.0, -100.0, 100.0),
        distance_nodes_m=np.linspace(0.0, 20.0, 11),
        distance_tolerance_m=1e-9,
        height_tolerance_m=1e-10,
        minimum_incidence_cosine=0.1,
    )
    options.update(changes)
    return options


def test_affine_agreement_axes_and_ownership():
    xy = np.array([[[2.0, 3.0], [-5.0, 7.0]]])
    slopes = np.array([0.2, -0.1])

    def height(xy):
        return 10 + xy @ slopes

    def gradient(xy):
        return np.broadcast_to(slopes, xy.shape)

    points, normals = surface_points(xy, height=height, gradient=gradient)
    plane, normal = affine_surface_points(xy, height_m=10, slopes=slopes)
    assert_array_equal(points, plane)
    assert_allclose(normals, np.broadcast_to(normal, points.shape), atol=1e-16)
    origins = points + 9.3 * normals
    directions = -normals
    options = controls(height=height, gradient=gradient)
    hit, actual = intersect_terrain(origins, directions, **options)
    affine_hit, affine_points = intersect_affine_terrain(
        origins, directions, height_m=10, slopes=slopes, minimum_incidence_cosine=0.1
    )
    assert_array_equal(hit, affine_hit)
    assert_allclose(actual, affine_points, atol=1e-10)
    for output in (points, normals, hit, actual):
        assert output.flags.writeable and output.flags.owndata
        for source in (xy, slopes, origins, directions):
            assert not np.shares_memory(output, source)
    for shape in ((2,), (0, 2), (2, 0, 2)):
        chart = np.zeros(shape)
        result, normal = surface_points(chart, height=height, gradient=gradient)
        assert result.shape == normal.shape == (*shape[:-1], 3)
    hit, actual = intersect_terrain(np.empty((0, 3)), np.empty((0, 3)), **options)
    assert hit.shape == (0,) and actual.shape == (0, 3)


@pytest.mark.parametrize("kind", ["regular_grid", "spline"])
def test_interpolated_quadratic_heights_gradients_and_oblique_intersection(kind):
    x = np.linspace(-10.0, 10.0, 9)
    y = np.linspace(-8.0, 8.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    heights = 2.0 + 0.03 * xx**2 + 0.02 * yy**2
    if kind == "regular_grid":
        interpolator = RegularGridInterpolator(
            (x, y),
            heights,
            method="cubic",
            bounds_error=True,
            solver_args={"rtol": 1e-13, "atol": 1e-13},
        )

        def height(xy):
            return interpolator(xy).reshape(xy.shape[:-1])

        def gradient(xy):
            return np.stack(
                [interpolator(xy, nu=nu).reshape(xy.shape[:-1]) for nu in ((1, 0), (0, 1))],
                axis=-1,
            )
    else:
        spline = RectBivariateSpline(x, y, heights)

        def height(xy):
            return spline.ev(xy[..., 0], xy[..., 1])

        def gradient(xy):
            return np.stack(
                [spline.ev(xy[..., 0], xy[..., 1], dx=dx, dy=dy) for dx, dy in ((1, 0), (0, 1))],
                axis=-1,
            )

    for xy in (np.array([2.3, -1.7]), np.array([[[2.3, -1.7], [0.0, 3.1]]])):
        points, normals = surface_points(xy, height=height, gradient=gradient)
        assert_allclose(points[..., 2], 2 + 0.03 * xy[..., 0] ** 2 + 0.02 * xy[..., 1] ** 2)
        expected = np.concatenate((-xy * [0.06, 0.04], np.ones((*xy.shape[:-1], 1))), axis=-1)
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        assert_allclose(normals, expected, atol=1e-12)
    direction = np.array([0.6, 0.0, -0.8])
    target = np.array([2.3, -1.7, float(height(np.array([2.3, -1.7])))])
    hit, actual = intersect_terrain(
        target - 5.3 * direction,
        direction,
        **controls(
            height=height,
            gradient=gradient,
            distance_nodes_m=np.linspace(0, 8, 9),
            horizontal_bounds_m=(-10, 10, -8, 8),
        ),
    )
    assert hit.shape == () and bool(hit)
    assert_allclose(actual, target, atol=2e-10)


def test_origin_misses_distance_limit_and_closed_domain():
    options = controls(horizontal_bounds_m=(0, 1, 0, 1))
    origins = np.array([[0, 0, 0], [1, 1, -1], [0, 0, 21], [0, 0, 20]], dtype=float)
    directions = np.tile([0.0, 0.0, -1.0], (4, 1))
    hit, points = intersect_terrain(origins, directions, **options)
    assert_array_equal(hit, [True, False, False, True])
    assert_array_equal(points[[0, 3]], [[0, 0, 0], [0, 0, 0]])
    assert np.isnan(points[[1, 2]]).all()
    hit, points = intersect_terrain(origins[:1], -directions[:1], **options)
    assert not hit.any() and np.isnan(points).all()
    # Height tolerance recognizes a slightly below-surface origin as a contact.
    hit, _ = intersect_terrain(np.array([0.0, 0.0, -1e-11]), directions[0], **options)
    assert hit


def test_first_detected_crossing_and_rejected_contact_never_skips_to_later_root():
    def height(xy):
        x = xy[..., 0]
        return -(x - 1) * (x - 2) * (3 - x) * (1 + x)

    def gradient(xy):
        x = xy[..., 0]
        return np.stack((4 * x**3 - 15 * x**2 + 10 * x + 5, np.zeros_like(x)), axis=-1)

    options = controls(
        height=height, gradient=gradient, distance_nodes_m=np.array([0.0, 0.5, 1.5, 2.5, 3.5])
    )
    hit, point = intersect_terrain(np.zeros(3), np.array([1.0, 0.0, 0.0]), **options)
    assert hit
    assert_allclose(point, [1, 0, 0], atol=1e-10)
    # First crossing has smaller incidence than the later crossing at x=3.
    options["minimum_incidence_cosine"] = 0.98
    hit, point = intersect_terrain(np.zeros(3), np.array([1.0, 0.0, 0.0]), **options)
    assert not hit and np.isnan(point).all()
    later, _ = intersect_terrain(np.array([2.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), **options)
    assert later
    # An exactly sampled tangent also ends the search, despite a later crossing.
    options.update(
        height=lambda xy: -((xy[..., 0] - 1) ** 2) * (3 - xy[..., 0]),
        gradient=lambda xy: np.stack(
            (3 * xy[..., 0] ** 2 - 10 * xy[..., 0] + 7, np.zeros(xy.shape[:-1])), axis=-1
        ),
        distance_nodes_m=np.array([0.0, 1.0, 2.0, 4.0]),
        minimum_incidence_cosine=0.1,
    )
    hit, point = intersect_terrain(np.zeros(3), np.array([1.0, 0.0, 0.0]), **options)
    assert not hit and np.isnan(point).all()


@pytest.mark.parametrize(
    "height",
    [
        lambda xy: -(xy[..., 0] - 0.25) * (xy[..., 0] - 0.75),
        lambda xy: -((xy[..., 0] - 0.5) ** 2),
    ],
)
def test_hidden_crossings_and_tangencies_are_explicitly_uncertified(height):
    hit, point = intersect_terrain(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        **controls(height=height, distance_nodes_m=np.array([0.0, 1.0])),
    )
    assert not hit and np.isnan(point).all()


@pytest.mark.parametrize(
    "change",
    [
        dict(distance_nodes_m=np.array([1.0, 2.0])),
        dict(distance_nodes_m=np.array([0.0])),
        dict(distance_nodes_m=np.array([0.0, 0.0])),
        dict(distance_nodes_m=np.array([0.0, np.inf])),
        dict(distance_nodes_m=np.zeros((2, 2))),
        dict(distance_tolerance_m=0),
        dict(height_tolerance_m=np.nan),
        dict(minimum_incidence_cosine=0),
        dict(max_iterations=0),
        dict(horizontal_bounds_m=(0, 0, 0, 1)),
        dict(horizontal_bounds_m=(0, 1, 0, np.inf)),
        dict(height=lambda xy: np.zeros((*xy.shape[:-1], 1))),
        dict(height=lambda xy: np.full(xy.shape[:-1], np.nan)),
        dict(gradient=lambda xy: np.zeros(xy.shape[:-1])),
        dict(gradient=lambda xy: np.full(xy.shape, np.inf)),
    ],
)
def test_invalid_intersection_controls_and_callbacks(change):
    with pytest.raises(ValueError):
        intersect_terrain(
            np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), **controls(**change)
        )


def test_domain_precedes_callbacks_and_refinement_failure_is_explicit():
    def unavailable(xy):
        raise RuntimeError("height unavailable")

    options = controls(height=unavailable, horizontal_bounds_m=(-1, 1, -1, 1))
    with pytest.raises(ValueError, match="domain"):
        intersect_terrain(np.zeros(3), np.array([1.0, 0.0, 0.0]), **options)
    with pytest.raises(RuntimeError, match="unavailable"):
        intersect_terrain(np.zeros(3), np.array([0.0, 0.0, -1.0]), **options)
    with pytest.raises(ValueError, match="converge"):
        intersect_terrain(
            np.array([0.0, 0.0, 1.3]),
            np.array([0.0, 0.0, -1.0]),
            **controls(max_iterations=1),
        )
    with pytest.raises(ValueError, match="converge"):
        intersect_terrain(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            **controls(height=lambda xy: np.where(xy[..., 0] < 0.33, -1.0, 1.0)),
        )


@pytest.mark.parametrize("xy", [np.zeros(3), np.zeros(()), np.array([np.nan, 0.0])])
def test_surface_invalid_coordinates(xy):
    with pytest.raises(ValueError):
        surface_points(xy, height=flat_height, gradient=flat_gradient)


def test_terrain_guide_examples():
    guide = Path(__file__).parents[1] / "docs" / "terrain.md"
    blocks = re.findall(r"^```python\n(.*?)^```", guide.read_text(), re.M | re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, str(guide), "exec"), namespace)
