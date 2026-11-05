"""Regression cases for supported near-singular roll and direction-norm roundoff."""

import numpy as np
from orsia_geometry import (
    ellipsoid_intersections,
    radial_nadir,
    sphere_intersections,
    surface_visibility,
    validate_rotations,
)
from orsia_geometry.constants import WGS84_SEMIMAJOR_AXIS_M


def test_nearly_parallel_roll_still_produces_proper_rotation():
    rotation = radial_nadir(np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0 + 1e-8, 0.0]))
    validate_rotations(rotation, ())
    np.testing.assert_allclose(rotation[:, 0], np.array([-1.0, 1.0, 0.0]) / np.sqrt(2), atol=1e-10)


def test_allowed_norm_roundoff_keeps_horizon_discriminant_correct():
    origin = np.array([2.0, 0.0, 0.0])
    direction = np.array([-np.sqrt(0.75 - 1e-11), np.sqrt(0.25 + 1e-11), 0.0]) * (1 + 5e-11)
    sphere_hit, sphere_point = sphere_intersections(origin, direction, radius_m=1)
    # WGS84 equatorial rays share the circular horizon of this independent sphere.
    ellipsoid_hit, ellipsoid_point = ellipsoid_intersections(
        WGS84_SEMIMAJOR_AXIS_M * origin, direction
    )
    assert not sphere_hit and not ellipsoid_hit
    np.testing.assert_array_equal(sphere_point, ellipsoid_point)


def test_scalar_visibility_returns_independent_writable_array():
    mask = surface_visibility(
        np.array([2.0 * WGS84_SEMIMAJOR_AXIS_M, 0.0, 0.0]),
        np.array([WGS84_SEMIMAJOR_AXIS_M, 0.0, 0.0]),
    )
    assert mask.shape == () and mask.flags.owndata and mask.flags.writeable
    mask[...] = False
    assert not mask
