"""Analytic rays, supplied states, roll and moving-target geometry without orbital data."""

import builtins
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from orsia_geometry import (
    axis_rotations,
    compose_rays,
    compute_lines_of_sight,
    ellipsoid_intersections,
    instrument_tracking,
    platform_tracking,
    pointing_rates,
    radial_nadir,
    sphere_intersections,
    surface_visibility,
    validate_rotations,
)
from orsia_geometry.constants import WGS84_SEMIAXES_M, WGS84_SEMIMAJOR_AXIS_M

EPOCH = datetime(2016, 12, 31, 23, 59, 59, tzinfo=UTC)
X, Y, Z = np.eye(3)


def trajectory(times):
    return np.column_stack((2 + times, times * 0, times * 0)), np.tile(X, (len(times), 1))


def identity(times):
    return np.tile(np.eye(3), (len(times), 1, 1))


def observation(times, **kwargs):
    options = dict(
        epoch=EPOCH,
        validity_s=(0, 2),
        trajectory=trajectory,
        platform_attitude=identity,
        instrument_attitude=identity,
        directions_i=Z[None],
        mount_offset_p_m=np.zeros(3),
    )
    options.update(kwargs)
    return compute_lines_of_sight(np.array(times, dtype=float), **options)


def test_compose_independent_rotations_mount_and_ownership():
    times = np.array([0.0, 1.0, 2.0])
    platform = axis_rotations(Z, 0.3 + times * 0.2)
    relative = axis_rotations(X, 0.1 + times * 0.1)
    positions = np.tile(np.array([2.0, 0, 0]), (3, 1))
    sensor = np.array([Z, Y])
    offset = np.array([0.1, 0.2, -0.1])
    origins, rays = compose_rays(positions, platform, relative, sensor, offset)
    for t in range(3):
        for r in range(2):
            np.testing.assert_allclose(origins[t, r], positions[t] + platform[t] @ offset)
            np.testing.assert_allclose(rays[t, r], platform[t] @ relative[t] @ sensor[r])
    assert not np.allclose(rays[:, 0], np.einsum("tij,tjk,k->ti", relative, platform, Z))
    origins[:] = 0
    rays[:] = 0
    np.testing.assert_array_equal(positions[:, 0], 2)
    np.testing.assert_array_equal(sensor[0], Z)
    assert origins.flags.owndata and rays.flags.owndata
    a, b = compose_rays(positions, platform, np.eye(3), np.empty((0, 3)), offset)
    assert a.shape == b.shape == (3, 0, 3)


@pytest.mark.parametrize(
    "change",
    ["positions", "rotation_shape", "rotation_nan", "reflection", "offset", "directions", "unit"],
)
def test_composition_validation(change):
    position, platform, relative, rays, offset = X, np.eye(3), np.eye(3), Z[None], X
    if change == "positions":
        position = np.array([np.nan, 0, 0])
    elif change == "rotation_shape":
        platform = np.eye(2)
    elif change == "rotation_nan":
        platform = np.full((3, 3), np.nan)
    elif change == "reflection":
        relative = np.diag([1.0, 1.0, -1.0])
    elif change == "offset":
        offset = np.zeros(2)
    elif change == "directions":
        rays = np.zeros((3, 2))
    elif change == "unit":
        rays = np.array([[0.0, 0.0, 2.0]])
    with pytest.raises(ValueError):
        compose_rays(position, platform, relative, rays, offset)


def test_surface_root_cases_and_no_mutation():
    origins = np.array(
        [
            [2.0, 0, 0],
            [2.0, 0, 0],
            [0.0, 0, 0],
            [1.0, 0, 0],
            [1.0, -2, 0],
            [2.0, -2, 0],
            [1.0, 0, 0],
        ]
    )
    directions = np.array([-X, X, X, X, Y, Y, -X])
    hit, points = sphere_intersections(origins, directions, radius_m=1)
    np.testing.assert_array_equal(hit, [True, False, True, True, True, False, True])
    np.testing.assert_allclose(points[hit], np.tile(X, (5, 1)), atol=1e-15)
    assert np.isnan(points[~hit]).all()
    assert hit.flags.owndata and points.flags.owndata
    points[:] = 0
    np.testing.assert_array_equal(origins[0], 2 * X)
    # The near root remains accurate at a small height above the surface (metres).
    _, close = sphere_intersections(np.array([1 + 1e-8, 0, 0]), -X, radius_m=1)
    np.testing.assert_allclose(close, X, atol=2e-16)
    # Resolve a near-horizon hit and miss, without changing discriminant signs.
    hit, _ = sphere_intersections(
        np.array([[1 - 1e-8, -2.0, 0], [1 + 1e-8, -2.0, 0]]), np.array([Y, Y]), radius_m=1
    )
    np.testing.assert_array_equal(hit, [True, False])


def test_wgs84_equatorial_polar_intersections_and_visibility():
    axes = np.array(WGS84_SEMIAXES_M)
    origins = np.diag(axes + 700000.0)
    directions = -np.eye(3)
    before = origins.copy(), directions.copy()
    hit, points = ellipsoid_intersections(origins, directions)
    assert hit.all()
    np.testing.assert_allclose(points, np.diag(axes), atol=2e-9, rtol=0)
    np.testing.assert_allclose(np.sum((points / axes) ** 2, axis=-1), 1, atol=1e-15, rtol=0)
    assert surface_visibility(origins, points).all()
    assert not surface_visibility(origins, -points).any()
    assert hit.flags.owndata and hit.flags.writeable
    assert points.flags.owndata and points.flags.writeable
    points[:] = 0
    np.testing.assert_array_equal(origins, before[0])
    np.testing.assert_array_equal(directions, before[1])


def test_wgs84_off_axis_geodetic_target_and_nearest_root():
    # Independent height-zero geodetic construction, 48 deg N, 35 deg E.
    # Defining values are reference fixtures, independent of the imported tuple.
    a, inverse_flattening = 6378137.0, 298.257223563
    e2 = (2 - 1 / inverse_flattening) / inverse_flattening
    latitude, longitude = np.deg2rad([48.0, 35.0])
    normal_radius = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)
    target = normal_radius * np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            (1 - e2) * np.sin(latitude),
        ]
    )
    origin = target + np.array([600000.0, -100000.0, 500000.0])
    direction = target - origin
    direction /= np.linalg.norm(direction)
    hit, point = ellipsoid_intersections(origin, direction)
    assert hit and surface_visibility(origin, target)
    np.testing.assert_allclose(point, target, atol=3e-9, rtol=0)


def test_wgs84_surface_inside_backward_and_horizon_cases():
    a = WGS84_SEMIMAJOR_AXIS_M
    origins = np.array(
        [
            [a + 1e-3, 0, 0],
            [a + 700000, 0, 0],
            [0, 0, 0],
            [a, 0, 0],
            [a, 0, 0],
            [a, -a, 0],
            [a - 1e-3, -700000, 0],
            [a + 1e-3, -700000, 0],
        ]
    )
    directions = np.array([-X, X, X, X, -X, Y, Y, Y])
    hit, points = ellipsoid_intersections(origins, directions)
    np.testing.assert_array_equal(hit, [True, False, True, True, True, True, True, False])
    np.testing.assert_allclose(points[[0, 2, 3, 4, 5]], np.tile(a * X, (5, 1)), atol=2e-9, rtol=0)
    assert np.isnan(points[~hit]).all()
    # A 1 mm inward shift resolves a near-tangent hit; no discriminant snapping.
    assert points[6, 1] < 0
    np.testing.assert_allclose(
        np.sum((points[hit] / WGS84_SEMIAXES_M) ** 2, axis=-1), 1, atol=2e-15, rtol=0
    )
    observer = np.array([[a, 700000, 0], [a - 1e-3, 700000, 0], [a + 1e-3, 700000, 0]])
    np.testing.assert_array_equal(
        surface_visibility(observer, np.tile(a * X, (3, 1))), [True, False, True]
    )


def test_wgs84_empty_batches():
    origins = np.empty((2, 0, 3))
    hit, points = ellipsoid_intersections(origins, origins)
    assert hit.shape == (2, 0) and points.shape == (2, 0, 3)
    assert surface_visibility(origins, origins).shape == (2, 0)


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf])
def test_invalid_sphere_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        sphere_intersections(X, -X, radius_m=radius)


@pytest.mark.parametrize(
    "origin,direction",
    [
        (np.zeros(2), X),
        (X, X[None]),
        (X, 2 * X),
        (np.full(3, np.nan), X),
        (X, np.full(3, np.inf)),
        (np.array(0.0), X),
    ],
)
def test_invalid_rays(origin, direction):
    with pytest.raises(ValueError, match="rays"):
        sphere_intersections(origin, direction, radius_m=1)
    with pytest.raises(ValueError, match="rays"):
        ellipsoid_intersections(origin, direction)


@pytest.mark.parametrize(
    "observer,target",
    [
        (X, X),
        (2 * X, 2 * X),
        (np.full(3, np.nan), X),
        (2 * X, X[None]),
        (np.array(0.0), X),
        (X, np.full(3, np.nan)),
        (X[:2], X[:2]),
        (1.1 * X, (1 + 1e-8) * X),
        (np.zeros(3), X),
    ],
)
def test_invalid_visibility(observer, target):
    with pytest.raises(ValueError, match="visibility"):
        surface_visibility(WGS84_SEMIMAJOR_AXIS_M * observer, WGS84_SEMIMAJOR_AXIS_M * target)


@pytest.mark.parametrize("frame_options", [{}, {"output_frame": "GCRS"}])
def test_lines_of_sight_gcrs_never_imports_or_calls_backend(monkeypatch, frame_options):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "satkit" or name.startswith("satkit."):
            raise AssertionError("GCRS must remain NumPy-only")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("frame adapter must not run")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr("orsia_geometry.los.frame_rotations", forbidden)
    origins, directions = observation([2.0, 0.0, 1.0, 1.0], **frame_options)
    np.testing.assert_array_equal(origins[:, 0, 0], [4.0, 2.0, 3.0, 3.0])
    np.testing.assert_array_equal(directions, np.tile(Z, (4, 1, 1)))
    assert origins.flags.owndata and directions.flags.owndata


@pytest.mark.parametrize("output_frame", ["unknown", "gcrs", "ECEF", ""])
def test_lines_of_sight_invalid_frame_precedes_callbacks_and_backend(monkeypatch, output_frame):
    def forbidden(*args, **kwargs):
        raise AssertionError("callback/backend must not run")

    monkeypatch.setattr("orsia_geometry.los.frame_rotations", forbidden)
    with pytest.raises(ValueError, match="output_frame"):
        observation(
            [0.0],
            output_frame=output_frame,
            trajectory=forbidden,
            platform_attitude=forbidden,
            instrument_attitude=forbidden,
        )


@pytest.mark.parametrize("output_frame", ["GCRS", "ITRF", "TEME"])
def test_lines_of_sight_callback_errors_propagate(monkeypatch, output_frame):
    error = RuntimeError("caller trajectory failure")

    def unavailable(times):
        raise error

    def forbidden(*args, **kwargs):
        raise AssertionError("backend must not run after callback failure")

    monkeypatch.setattr("orsia_geometry.los.frame_rotations", forbidden)
    with pytest.raises(RuntimeError) as caught:
        observation([0.0], trajectory=unavailable, output_frame=output_frame)
    assert caught.value is error


def test_observation_closed_support_elapsed_seconds_and_moving_scan():
    received = []

    def state(times):
        received.append(times.copy())
        return trajectory(times)

    times = np.array([2.0, 0.0, 1.0, 1.0])
    origins, rays = observation(
        times, trajectory=state, instrument_attitude=lambda t: axis_rotations(Y, 0.1 * t)
    )
    np.testing.assert_array_equal(received[0], times)
    np.testing.assert_array_equal(origins[:, 0, 0], 2 + times)
    np.testing.assert_allclose(
        rays[:, 0], np.column_stack((np.sin(0.1 * times), times * 0, np.cos(0.1 * times)))
    )
    assert EPOCH == datetime(2016, 12, 31, 23, 59, 59, tzinfo=UTC)
    # Two seconds straddle the leap-second date purely through caller elapsed time.
    # This test establishes pass-through semantics, not UTC/time-scale conversion.
    a, b = observation([])
    assert a.shape == b.shape == (0, 1, 3)


@pytest.mark.parametrize(
    "times,bounds",
    [
        ([-1.0], (0.0, 2.0)),
        ([np.nan], (0.0, 2.0)),
        ([[0.0]], (0.0, 2.0)),
        ([0.0], (2.0, 0.0)),
        ([0.0], (0.0, np.inf)),
    ],
)
@pytest.mark.parametrize("output_frame", ["GCRS", "ITRF", "TEME"])
def test_support_rejected_before_callbacks(times, bounds, output_frame, monkeypatch):
    def forbidden(_):
        raise AssertionError("callback must not run")

    monkeypatch.setattr("orsia_geometry.los.frame_rotations", forbidden)
    with pytest.raises(ValueError, match="validity"):
        observation(times, trajectory=forbidden, validity_s=bounds, output_frame=output_frame)


@pytest.mark.parametrize(
    "epoch", [datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1)))]
)
@pytest.mark.parametrize("output_frame", ["GCRS", "ITRF", "TEME"])
def test_epoch_validation(epoch, output_frame, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("callback/backend must not run")

    monkeypatch.setattr("orsia_geometry.los.frame_rotations", forbidden)
    with pytest.raises(ValueError, match="UTC"):
        observation([0.0], epoch=epoch, trajectory=forbidden, output_frame=output_frame)


def test_trajectory_shape_and_nonfinite_velocity():
    with pytest.raises(ValueError, match="trajectory"):
        observation([0.0], trajectory=lambda t: (np.ones((1, 2)), np.ones((1, 3))))
    with pytest.raises(ValueError, match="trajectory"):
        observation([0.0], trajectory=lambda t: (np.ones((1, 3)), np.full((1, 3), np.nan)))


def test_nadir_roll_target_tracking_nominal_attitude_and_field_rotation():
    positions = np.array([[2.0, 0, 0], [0, 2.0, 0]])
    velocities = np.array([Y, -X])
    nadir = radial_nadir(positions, velocities)
    np.testing.assert_allclose(nadir[..., 2], -positions / 2)
    np.testing.assert_allclose(nadir[..., 0], velocities)
    validate_rotations(nadir, (2,))
    angles = np.array([0.0, 0.3])
    targets = np.column_stack((np.cos(angles), np.sin(angles), angles * 0))
    ref = np.tile(Z, (2, 1))
    tracking = platform_tracking(positions, targets, ref, mount_offset_p_m=np.zeros(3))
    expected = targets - positions
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(tracking[..., 2], expected, atol=2e-16)
    np.testing.assert_allclose(tracking[..., 0], ref, atol=2e-16)
    assert not surface_visibility(
        WGS84_SEMIMAJOR_AXIS_M * positions, WGS84_SEMIMAJOR_AXIS_M * targets
    )[1]
    relative = instrument_tracking(positions, targets, nadir, ref)
    np.testing.assert_allclose(nadir @ relative, tracking, atol=3e-16)
    # A different roll reference keeps boresight while moving off-axis field rays.
    other = instrument_tracking(positions, targets, nadir, np.tile(Y, (2, 1)))
    np.testing.assert_allclose((nadir @ other)[..., 2], expected, atol=3e-16)
    assert not np.allclose((nadir @ other)[..., 0], tracking[..., 0])


@pytest.mark.parametrize(
    "position,velocity",
    [
        (np.zeros(3), Y),
        (X, X),
        (X, np.zeros(3)),
        (X, X + 1e-14 * Y),
        (X, np.ones(2)),
        (X, np.full(3, np.nan)),
    ],
)
def test_pointing_singularities(position, velocity):
    with pytest.raises(ValueError):
        radial_nadir(position, velocity)


def test_tracking_mount_and_nominal_validation():
    with pytest.raises(ValueError, match="zero mount"):
        platform_tracking(2 * X, X, Y, mount_offset_p_m=X)
    with pytest.raises(ValueError, match="rotation"):
        instrument_tracking(2 * X, X, np.eye(2), Y)
    with pytest.raises(ValueError, match="positions"):
        platform_tracking(X, X[None], Y, mount_offset_p_m=np.zeros(3))


def test_scan_and_pointing_rate_refinement():
    for n in (5, 9, 17):
        times = np.linspace(0, 1, n)
        rotations = axis_rotations(X, 0.2 * times**2)
        rates = pointing_rates(times, rotations)
        # For a fixed axis quadratic angle law the interval mean is exact, rad/s.
        np.testing.assert_allclose(rates, 0.2 * (times[1:] + times[:-1]), atol=2e-15)
    # General moving axes approach an independently differentiated matrix rate.
    errors = []
    for n in (9, 17, 33):
        times = np.linspace(0, 1, n)
        rotations = axis_rotations(Z, times * 0.3) @ axis_rotations(X, times**2 * 0.2)
        rates = pointing_rates(times, rotations)
        mid = (times[1:] + times[:-1]) / 2
        errors.append(np.max(np.abs(rates - np.sqrt(0.3**2 + (0.4 * mid) ** 2))))
    assert errors[2] < errors[1] / 3 and errors[1] < errors[0] / 3


@pytest.mark.parametrize(
    "axis,angle",
    [
        (2 * X, np.array([0.0])),
        ((1 + 2e-10) * X, np.array([np.pi])),
        (X, np.array([np.nan])),
        (np.zeros(2), np.array([0.0])),
    ],
)
def test_invalid_scan(axis, angle):
    with pytest.raises(ValueError, match="axis"):
        axis_rotations(axis, angle)


@pytest.mark.parametrize("error", [-5e-11, 5e-11])
@pytest.mark.parametrize("direction", [X, np.array([1.0, 2.0, 3.0]) / np.sqrt(14)])
def test_scan_normalizes_accepted_axis_for_rotation_composition(error, direction):
    axis = direction * (1 + error)
    angles = np.array([0.0, 0.3, np.pi])
    saved_axis, saved_angles = axis.copy(), angles.copy()
    axis.setflags(write=False)
    rotations = axis_rotations(axis, angles)
    validate_rotations(rotations, (3,))
    expected = axis_rotations(direction, angles)
    np.testing.assert_allclose(rotations, expected, rtol=0, atol=2e-15)
    platform = axis_rotations(Z, np.array([0.1, 0.2, 0.3]))
    _, rays = compose_rays(np.zeros((3, 3)), platform, rotations, Z[None], np.zeros(3))
    np.testing.assert_allclose(rays[:, 0], (platform @ expected)[:, :, 2], rtol=0, atol=2e-15)
    np.testing.assert_array_equal(axis, saved_axis)
    np.testing.assert_array_equal(angles, saved_angles)


@pytest.mark.parametrize(
    "times",
    [np.array([0.0]), np.array([0.0, 0.0]), np.array([0.0, np.nan]), np.array([[0.0, 1.0]])],
)
def test_invalid_rates(times):
    with pytest.raises(ValueError, match="times"):
        pointing_rates(times, np.empty((len(times), 3, 3)))
