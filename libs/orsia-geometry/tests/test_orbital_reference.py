"""Opt-in native checks against ERFA, GMAT and Vallado constants with local auxiliaries."""

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from orsia_geometry import (
    axis_rotations,
    compute_lines_of_sight,
    ellipsoid_intersections,
    frame_rotations,
    instrument_tracking,
    numerical_propagator,
    platform_tracking,
    pointing_rates,
    project_local_east_north,
    radial_nadir,
    sgp4_propagator,
    sphere_intersections,
    surface_visibility,
    transform_rays,
    transform_states,
    wgs84_surface_projection,
    wgs84_surface_projection_from_ray,
)
from orsia_geometry.constants import WGS84_SEMIAXES_M, WGS84_SEMIMAJOR_AXIS_M

pytestmark = pytest.mark.skipif(
    os.environ.get("ORSIA_TEST_ORBITAL") != "1",
    reason="run just --justfile tests/justfile test-orbital with existing local fixtures",
)

# CelesTrak fundamentals-of-astrodynamics 49df0479c950917c0a2c5ac31f2db6477bba57f6:
# datalib/SGP4-VER.TLE, object 5; software/misc/java/SGP4/JAVA_SGP4_v2/java_sgp4_ver.out.
TLE = (
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
)


def native():
    import satkit

    assert not sys._is_gil_enabled()
    assert os.environ.get("PYTHON_GIL") != "0"
    return satkit


def test_native_leap_seconds():
    satkit = native()
    epoch = datetime(2016, 12, 31, 23, 59, 59, tzinfo=UTC)
    times = np.array([2.0, 0.0, 1.0, 1.0])
    origin = satkit.time.from_datetime(epoch)
    instants = [origin + satkit.duration(seconds=float(t)) for t in times]
    assert str(instants[0]).startswith("2017-01-01T00:00:00")
    assert (instants[0] - instants[1]).seconds == pytest.approx(2)
    expected = np.array(
        [
            satkit.frametransform.rotation(
                satkit.frame.GCRF, satkit.frame.ITRF, t
            ).to_rotation_matrix()
            for t in instants
        ]
    )
    actual = frame_rotations(times, epoch=epoch, validity_s=(0, 2))
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    assert actual.flags.owndata and actual.flags.writeable
    # This check's local fixture policy; production retains ordinary backend management.
    assert satkit.utils.is_offline()


def test_native_sgp4_vallado_reference_and_both_element_inputs():
    satkit = native()
    tle = satkit.TLE.from_lines(list(TLE), check_checksum=True)[0]
    epoch = tle.epoch.to_datetime()
    times = np.array([0.0, 21600.0])
    origin = satkit.time.from_datetime(epoch)
    instants = [origin + satkit.duration(seconds=float(t)) for t in times]
    pos, vel = satkit.sgp4(
        tle,
        instants,
        gravconst=satkit.sgp4_gravconst.wgs72,
        opsmode=cast(satkit.sgp4_opsmode, satkit.sgp4_opsmode.afspc),
    )
    truth_p = 1000 * np.array(
        [
            [7022.46529266, -1400.08296755, 0.03995155],
            [-7154.03120202, -3783.17682504, -3536.19412294],
        ]
    )
    truth_v = 1000 * np.array(
        [[1.893841015, 6.405893759, 4.534807250], [4.741887409, -4.151817765, -2.093935425]]
    )
    # Original output rounding and microsecond TLE epoch precision: 0.02 m / 2e-5 m/s.
    np.testing.assert_allclose(pos, truth_p, atol=0.02, rtol=0)
    np.testing.assert_allclose(vel, truth_v, atol=2e-5, rtol=0)
    from orsia_data import parse_orbital_elements

    (elements,) = parse_orbital_elements("\n".join(TLE).encode(), format="tle")
    callback = sgp4_propagator(elements, epoch=epoch, validity_s=(0, 21600))
    gpos, gvel = callback(times)
    expected_p, expected_v = transform_states(
        times, pos, vel, epoch=epoch, validity_s=(0, 21600), from_frame="TEME", to_frame="GCRS"
    )
    np.testing.assert_allclose(gpos, expected_p, atol=1e-9)
    np.testing.assert_allclose(gvel, expected_v, atol=1e-9)
    assert callback(np.array([]))[0].shape == (0, 3)
    assert callback(np.array([0.0]))[0].shape == (1, 3)
    np.testing.assert_array_equal(callback(times[::-1])[0], gpos[::-1])
    # OMM GP mapping of exactly the same TLE mean elements; EPOCH remains an explicit UTC label.
    omm = {
        "NORAD_CAT_ID": 5,
        "EPOCH": epoch.isoformat(),
        "MEAN_MOTION": 10.82419157,
        "ECCENTRICITY": 0.1859667,
        "INCLINATION": 34.2682,
        "RA_OF_ASC_NODE": 348.7242,
        "ARG_OF_PERICENTER": 331.7664,
        "MEAN_ANOMALY": 19.3264,
        "BSTAR": 0.28098e-4,
        "MEAN_MOTION_DOT": 0.00000023,
        "MEAN_MOTION_DDOT": 0.0,
        "TIME_SYSTEM": "UTC",
        "REF_FRAME": "TEME",
        "CENTER_NAME": "EARTH",
        "MEAN_ELEMENT_THEORY": "SGP4",
    }
    (omm_elements,) = parse_orbital_elements(json.dumps(omm).encode(), format="omm")
    other = sgp4_propagator(omm_elements, epoch=epoch, validity_s=(0, 21600))
    np.testing.assert_allclose(other(times)[0], gpos, atol=0.02, rtol=0)


def test_native_numerical_classical_dense_against_gmat_and_refinement():
    native()
    # SatKit upstream 441675131bf9b14a9634d069f7f2599b69c7252c,
    # tests/gmat/cases/leo_iss_j2.json, first 3 samples. GMAT R2026A EarthICRF,
    # EGM96 degree/order 2, Sun/Moon DE440, tides/relativity off; compare with DE421 here.
    epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
    elements = np.array([6778e3, 0.001, *np.deg2rad([51.6, 30.0, 40.0, 50.0])])
    times = np.array([0.0, 3599.999999790452, 7199.999999580905])
    truth = 1000 * np.array(
        [
            [
                -2103.715480048485,
                3643.742096113128,
                5308.456731666616,
                -6.647330004445162,
                -3.833624329580398,
                0.004603821856452621,
            ],
            [
                5961.751559867882,
                489.7145394186537,
                -3201.141734168868,
                2.109884758788286,
                5.588805198146787,
                4.803687372227519,
            ],
            [
                -5070.26773400476,
                -4257.434825558861,
                -1513.045155232562,
                4.13883656159285,
                -2.892266119776178,
                -5.759890269467312,
            ],
        ]
    )
    force_model = dict(gravity_degree=2, gravity_order=2, sun_gravity=True, moon_gravity=True)
    callback = numerical_propagator(
        elements,
        epoch=epoch,
        validity_s=(0, 7200),
        absolute_tolerance=1e-7,
        relative_tolerance=1e-13,
        **force_model,
    )
    positions, velocities = callback(times)
    p_error = np.linalg.norm(positions - truth[:, :3], axis=-1)
    v_error = np.linalg.norm(velocities - truth[:, 3:], axis=-1)
    print("GMAT residuals metres / m/s:", p_error.tolist(), v_error.tolist())
    assert p_error.max() < 0.1 and v_error.max() < 1e-4
    # Tighter force-matched arc vs ordinary default numerical tolerances, dense dates between nodes.
    dense_times = np.linspace(0, 7200, 39)
    coarse = numerical_propagator(elements, epoch=epoch, validity_s=(0, 7200), **force_model)
    error = np.linalg.norm(callback(dense_times)[0] - coarse(dense_times)[0], axis=-1).max()
    print("Dense refinement max metres:", error)
    assert error < 0.1
    assert callback(np.array([]))[0].shape == (0, 3)
    assert callback(np.array([0.0]))[0].shape == (1, 3)
    np.testing.assert_array_equal(callback(times[::-1])[0], positions[::-1])
    np.testing.assert_array_equal(callback(times[[1, 0, 1]])[0], positions[[1, 0, 1]])


@pytest.mark.parametrize(
    "epoch",
    [datetime(2023, 5, 16, 20, tzinfo=UTC), datetime(2016, 12, 31, 23, 59, 59, tzinfo=UTC)],
)
def test_native_two_body_analytic_limit(epoch):
    satkit = native()
    radius = 7e6
    callback = numerical_propagator(
        np.array([radius, 0.0, 0.0, 0.0, 0.0, 0.0]),
        epoch=epoch,
        validity_s=(0, 600),
        gravity_degree=0,
        absolute_tolerance=1e-7,
        relative_tolerance=1e-13,
    )
    times = np.linspace(0, 600, 33)
    rate = np.sqrt(satkit.consts.mu_earth / radius**3)
    expected = radius * np.column_stack((np.cos(rate * times), np.sin(rate * times), times * 0))
    error = np.linalg.norm(callback(times)[0] - expected, axis=-1).max()
    print("Two-body max metres:", error)
    assert error < 0.01


def test_native_numerical_requires_eop_coverage():
    native()
    with pytest.raises(RuntimeError, match="(?i)eop|earth.orientation"):
        numerical_propagator(
            np.array([7e6, 0.0, 0.0, 0.0, 0.0, 0.0]),
            epoch=datetime(2100, 1, 1, tzinfo=UTC),
            validity_s=(0, 1),
        )


def test_native_terrestrial_velocity_consistency():
    native()
    epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
    times = np.array([10.0, 100.0])
    positions = np.tile(np.array([7e6, 1e6, 2e6]), (2, 1))
    velocities = np.zeros_like(positions)
    state = transform_states(times, positions, velocities, epoch=epoch, validity_s=(0, 200))
    h = 0.1
    plus = np.einsum(
        "tij,tj->ti", frame_rotations(times + h, epoch=epoch, validity_s=(0, 200)), positions
    )
    minus = np.einsum(
        "tij,tj->ti", frame_rotations(times - h, epoch=epoch, validity_s=(0, 200)), positions
    )
    reference = (plus - minus) / (2 * h)
    error = np.linalg.norm(state[1] - reference, axis=-1).max()
    print("Frame-velocity refinement max m/s:", error)
    assert error < 0.002
    assert np.linalg.norm(state[1], axis=-1).min() > 400


@pytest.mark.parametrize("output_frame", ["ITRF", "TEME"])
def test_native_lines_of_sight_match_dated_rotations(output_frame):
    native()
    epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
    times = np.array([60.0, 0.0, 30.0, 30.0])
    positions = np.column_stack((7e6 + times, times * 10, times * 0))
    velocities = np.tile([1.0, 10.0, 0.0], (len(times), 1))
    platform = axis_rotations(np.array([0.0, 0.0, 1.0]), times * 0.01)
    instrument = axis_rotations(np.array([1.0, 0.0, 0.0]), times * 0.02)
    sensor = np.eye(3)
    offset = np.array([0.1, -0.2, 0.3])
    actual_origins, actual_directions = compute_lines_of_sight(
        times,
        epoch=epoch,
        validity_s=(0, 60),
        trajectory=lambda t: (positions, velocities),
        platform_attitude=lambda t: platform,
        instrument_attitude=lambda t: instrument,
        directions_i=sensor,
        mount_offset_p_m=offset,
        output_frame=output_frame,
    )
    rotations = frame_rotations(
        times, epoch=epoch, validity_s=(0, 60), from_frame="GCRS", to_frame=output_frame
    )
    for t in range(len(times)):
        for ray in range(len(sensor)):
            expected_origin = rotations[t] @ (positions[t] + platform[t] @ offset)
            expected_direction = rotations[t] @ platform[t] @ instrument[t] @ sensor[ray]
            np.testing.assert_allclose(actual_origins[t, ray], expected_origin, atol=2e-9, rtol=0)
            np.testing.assert_allclose(
                actual_directions[t, ray], expected_direction, atol=1e-15, rtol=0
            )
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("frames", [("GCRS", "ITRF"), ("ITRF", "TEME"), ("TEME", "GCRS")])
def test_native_transform_rays_matches_direct_frame_composition(frames):
    native()
    epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
    times = np.array([[[60.0, 0.0], [30.0, 30.0]], [[0.0, 60.0], [30.0, 0.0]]])
    origins = np.broadcast_to([7e6, 1e6, -2e6], (*times.shape, 3))
    directions = np.broadcast_to([0.6, 0.0, -0.8], origins.shape)
    options = dict(epoch=epoch, validity_s=(0, 60), from_frame=frames[0], to_frame=frames[1])
    actual = transform_rays(times, origins, directions, **options)
    rotations = frame_rotations(times.reshape(-1), **options).reshape((*times.shape, 3, 3))
    for output, value in zip(actual, (origins, directions), strict=True):
        expected = np.einsum("...ij,...j->...i", rotations, value)
        np.testing.assert_allclose(output, expected, atol=2e-9, rtol=0)
    assert not sys._is_gil_enabled()


def test_native_wgs84_projection_matches_primitives_with_missed_rays():
    native()
    epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
    times = np.array([[0.0, 10.0], [10.0, 0.0]])
    chart_origin = np.array([WGS84_SEMIAXES_M[0], 0.0, 0.0])
    positions_itrf = np.broadcast_to(chart_origin + [7e5, 123.0, -89.0], (2, 2, 3))
    directions_itrf = np.array([[[-1.0, 0, 0], [1.0, 0, 0]], [[-1.0, 0, 0], [-1.0, 0, 0]]])
    rotations = frame_rotations(times.ravel(), epoch=epoch, validity_s=(0, 10)).reshape(2, 2, 3, 3)
    origins = np.einsum("...ji,...j->...i", rotations, positions_itrf)
    directions = np.einsum("...ji,...j->...i", rotations, directions_itrf)
    expected_hits, points = ellipsoid_intersections(positions_itrf, directions_itrf)
    expected_xy = project_local_east_north(points, origin_itrf_m=chart_origin)
    surface = wgs84_surface_projection(
        epoch=epoch, validity_s=(0, 10), chart_origin_itrf_m=chart_origin
    )
    xy, hits = surface(origins, directions, times)
    np.testing.assert_array_equal(hits, expected_hits)
    np.testing.assert_allclose(xy, expected_xy, atol=5e-9, rtol=0)
    assert np.isnan(xy[0, 1]).all()
    reference_hit, reference_point = ellipsoid_intersections(
        *transform_rays(
            np.array(times[0, 0]), origins[0, 0], directions[0, 0], epoch=epoch, validity_s=(0, 10)
        )
    )
    assert reference_hit
    expected_surface = wgs84_surface_projection(
        epoch=epoch, validity_s=(0, 10), chart_origin_itrf_m=reference_point
    )
    ray_surface = wgs84_surface_projection_from_ray(
        origin_gcrs_m=origins[0, 0],
        direction_gcrs=directions[0, 0],
        time_s=float(times[0, 0]),
        epoch=epoch,
        validity_s=(0, 10),
    )
    ray_xy, ray_hits = ray_surface(origins, directions, times)
    expected_ray_xy, expected_ray_hits = expected_surface(origins, directions, times)
    np.testing.assert_allclose(ray_xy, expected_ray_xy, atol=5e-9, rtol=0)
    np.testing.assert_array_equal(ray_hits, expected_ray_hits)
    assert not sys._is_gil_enabled()


def test_native_erfa_absolute_frame_fixture_in_fresh_process(tmp_path):
    # ERFA v2.0.1 src/t_erfa_c.c:t_c2t06a. Artificial DUT1 makes TT=UT1=MJD53736.
    # This isolates the numerical reduction; it is deliberately not a physical EOP history.
    rows = []
    for mjd, dut1 in ((53735.0, 64.184), (53736.0, 65.184)):
        row = list(" " * 125)

        def field(start, value, row=row):
            row[start - 1 : start - 1 + len(value)] = value

        field(8, f"{mjd:8.2f}")
        field(17, "I")
        field(19, f"{np.rad2deg(2.55060238e-7) * 3600:9.6f}")
        field(38, f"{np.rad2deg(1.860359247e-6) * 3600:9.6f}")
        field(58, "I")
        field(59, f"{dut1:10.7f}")
        rows.append("".join(row).rstrip())
    (tmp_path / "finals2000A.all").write_text("\n".join(rows) + "\n")
    code = """
from datetime import datetime, UTC
import numpy as np
from orsia_geometry import frame_rotations
expected = np.array([
[-0.1810332128305897282, 0.9834769806938592296, 0.6555550962998436505e-4],
[-0.9834768134136214897, -0.1810332203649130832, 0.5749800844905594110e-3],
[0.5773474024748545878e-3, 0.3961816829632690581e-4, 0.9999998325501747785]])
actual = frame_rotations(np.array([0.]),
    epoch=datetime(2005,12,31,23,58,55,816000,tzinfo=UTC), validity_s=(0,0))[0]
print('ERFA max matrix element error:', np.max(np.abs(actual-expected)))
np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
"""
    env = dict(os.environ, HOME=str(tmp_path), SATKIT_DATA=str(tmp_path), SATKIT_OFFLINE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


def test_readme_leo_ground_trace():
    native()
    path = Path(__file__).parents[1] / "README.md"
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)
    assert blocks, "README must contain a runnable Python example"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    with patch("socket.socket", side_effect=AssertionError("network forbidden in examples")):
        for number, code in enumerate(blocks, 1):
            exec(compile(code, f"<{path}:example-{number}>", "exec"), namespace)
    positions, velocities = namespace["trajectory"](namespace["times_s"])
    assert positions.shape == velocities.shape == (31, 3)
    radius_m = np.linalg.norm(positions, axis=-1)
    speed_m_s = np.linalg.norm(velocities, axis=-1)
    assert np.all((radius_m > 7e6) & (radius_m < 7.1e6))
    assert np.all((speed_m_s > 7e3) & (speed_m_s < 8e3))
    for name in ("ray_origins_itrf_m", "ray_directions_itrf"):
        assert namespace[name].shape == (31, 3, 3)
    hit = namespace["hit"]
    points = namespace["ground_points_itrf_m"]
    assert hit.shape == (31, 3)
    assert hit.all() and points.shape == (31, 3, 3)
    np.testing.assert_allclose(
        np.sum((points / WGS84_SEMIAXES_M) ** 2, axis=-1),
        1,
        atol=1e-12,
        rtol=0,
    )
    displacement_m = np.linalg.norm(points[-1, 0] - points[0, 0])
    assert 300e3 < displacement_m < 500e3
    print("README 60 s nadir ground displacement metres:", displacement_m)
    print("GIL enabled after README native calls:", sys._is_gil_enabled())
    assert not sys._is_gil_enabled()


def test_filter_camera_guide():
    native()
    path = Path(__file__).parents[2] / "orsia-instruments/docs/filter-camera.md"
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)
    assert blocks, "The camera guide must contain the complete inline tutorial"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    with patch("socket.socket", side_effect=AssertionError("network forbidden in examples")):
        for number, code in enumerate(blocks, 1):
            exec(compile(code, f"<{path}:example-{number}>", "exec"), namespace)
    grid, cube = namespace["truth_grid"], namespace["cube"]
    assert grid.image_shape == (3, 2, 2)
    assert cube.values.shape == (6, 6, 2)
    assert cube.valid.any(axis=(0, 1)).all() and (~cube.valid).any()
    assert np.isnan(cube.values[~cube.valid]).all()
    np.testing.assert_allclose(
        np.nanmean(cube.values, axis=(0, 1)), namespace["expected_means"], rtol=1e-12
    )
    print("GIL enabled after camera guide native calls:", sys._is_gil_enabled())
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize(
    "mode", ["nadir", "platform_tracking", "instrument_tracking", "cross_track_scan"]
)
def test_native_wgs84_four_mode_ground_motion(mode):
    native()
    epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
    validity = (0.0, 61.0)
    trajectory = numerical_propagator(
        np.array([WGS84_SEMIMAJOR_AXIS_M + 700000.0, 0.001, np.deg2rad(98), 0.0, 0.0, 0.0]),
        epoch=epoch,
        validity_s=validity,
    )
    # Retain the 60 s arc and resolve the illustrative 0.1 s exposure at 30 s.
    times = np.sort(np.append(np.linspace(0, 60, 31), 30.1))
    rays_i = np.array(
        [[0.0, 0.0, 1.0], [0.001, 0.0, np.sqrt(1 - 1e-6)], [-0.001, 0.0, np.sqrt(1 - 1e-6)]]
    )

    def earth_rotations(t):
        return frame_rotations(t, epoch=epoch, validity_s=validity)

    def nadir(t):
        return radial_nadir(*trajectory(t))

    def identity(t):
        return np.broadcast_to(np.eye(3), (len(t), 3, 3))

    # Fix the ground target at the initial radial intercept, then rotate it with Earth.
    initial_position, _ = trajectory(np.array([0.0]))
    initial_itrf = earth_rotations(np.array([0.0]))[0] @ initial_position[0]
    target_hit, target_itrf = ellipsoid_intersections(
        initial_itrf, -initial_itrf / np.linalg.norm(initial_itrf)
    )
    assert target_hit

    def target(t):
        return np.einsum("tji,j->ti", earth_rotations(t), target_itrf)

    def tracked_platform(t):
        position, velocity = trajectory(t)
        return platform_tracking(position, target(t), velocity, mount_offset_p_m=np.zeros(3))

    def tracked_instrument(t):
        position, velocity = trajectory(t)
        return instrument_tracking(position, target(t), nadir(t), velocity)

    def scan(t):
        return axis_rotations(np.array([1.0, 0.0, 0.0]), 0.03 * np.sin(2 * np.pi * t / 60))

    platform, instrument = {
        "nadir": (nadir, identity),
        "platform_tracking": (tracked_platform, identity),
        "instrument_tracking": (nadir, tracked_instrument),
        "cross_track_scan": (nadir, scan),
    }[mode]
    origins_itrf, directions_itrf = compute_lines_of_sight(
        times,
        epoch=epoch,
        validity_s=validity,
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=instrument,
        directions_i=rays_i,
        mount_offset_p_m=np.zeros(3),
        output_frame="ITRF",
    )
    hits, points_itrf = ellipsoid_intersections(origins_itrf, directions_itrf)
    assert origins_itrf.shape == directions_itrf.shape == points_itrf.shape == (len(times), 3, 3)
    assert hits.shape == (len(times), 3) and hits.all()
    np.testing.assert_allclose(
        np.sum((points_itrf / WGS84_SEMIAXES_M) ** 2, axis=-1), 1, atol=1e-12, rtol=0
    )
    assert surface_visibility(
        origins_itrf[:, 0], np.broadcast_to(target_itrf, (len(times), 3))
    ).all()
    rates = pointing_rates(times, platform(times) @ instrument(times))
    assert np.isfinite(rates).all() and rates.max() > 0
    exposure_points = points_itrf[np.searchsorted(times, [30.0, 30.1]), 0]
    displacement_m = np.linalg.norm(exposure_points[1] - exposure_points[0])
    if "tracking" in mode:
        assert np.linalg.norm(points_itrf[:, 0] - target_itrf, axis=-1).max() < 1e-6
        assert displacement_m < 1e-6
    else:
        assert displacement_m > 100
    assert not sys._is_gil_enabled()


def test_near_nadir_ground_sensitivity_illustration():
    native()
    # Single spherical viewing case; this does not establish a total WGS84 error bound.
    radius, height = WGS84_SEMIMAJOR_AXIS_M, 700000.0
    origin = np.array([[0.0, 0.0, radius + height]])
    angle = 0.1
    direction = np.array([[np.sin(angle), 0.0, -np.cos(angle)]])
    hit, nominal = sphere_intersections(origin, direction, radius_m=radius)
    _, position = sphere_intersections(
        origin + np.array([[1.0, 0.0, 0.0]]), direction, radius_m=radius
    )
    tilted = np.einsum(
        "ij,tj->ti", axis_rotations(np.array([0.0, 1.0, 0.0]), np.array(1e-7)), direction
    )
    _, attitude = sphere_intersections(origin, tilted, radius_m=radius)
    position_response = np.linalg.norm(position - nominal)
    attitude_response = np.linalg.norm(attitude - nominal)
    assert hit.all() and position_response < 1.02 and attitude_response < 0.08
    print("Spherical sensitivity responses metres:", position_response, attitude_response)
