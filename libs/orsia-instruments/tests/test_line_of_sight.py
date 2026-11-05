"""Synthetic sight-line geometry, full-support radiance and detector composition."""

import os
import sys
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from orsia_geometry import axis_rotations, compute_lines_of_sight, sphere_intersections
from orsia_instruments import (
    DetectorElectronics,
    ExpectedElectrons,
    ReadSchedule,
    combine_source_paths,
    gaussian_pixel_fraction,
    line_of_sight_expected_electrons,
    read_schedule,
    sample_lines_of_sight,
    sample_readout,
)
from orsia_radiative import SpectralDomain, SpectralGrid
from scipy.constants import c, h
from scipy.integrate import quad

EPOCH = datetime(2016, 12, 31, 23, 59, 59, tzinfo=UTC)


def trajectory(times):
    # Synthetic straight-line GCRS motion; velocity is its analytic derivative.
    position = np.column_stack((2 + 0.1 * times, np.zeros_like(times), np.zeros_like(times)))
    velocity = np.broadcast_to([0.1, 0.0, 0.0], position.shape).copy()
    return position, velocity


def platform_attitude(times):
    return axis_rotations(np.array([0.0, 0.0, 1.0]), 0.1 * times)


def instrument_attitude(times):
    return np.broadcast_to(MOUNT, (len(times), 3, 3)).copy()


MOUNT = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
FOV = np.array([[-0.03, 0.03, -0.01, 0.01], [0.75, 0.8, -0.01, 0.01]])
PIXELS = np.array([[-0.1, 0.1, -0.1, 0.1], [0.9, 1.5, -0.1, 0.1]])
GRID = SpectralGrid(
    "wavenumber_cm1",
    np.array([4900.0, 5000.0, 5100.0]),
    SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),)),
)


def case(time_order=3, angular_order=3, fov=FOV):
    reads = ReadSchedule(
        ("read-0", "read-1"),
        ("pixel-0", "pixel-1"),
        ("band", "band"),
        ("direct", "direct"),
        np.array([[0.0, 1.0], [0.5, 1.5]]),
    )
    return sample_lines_of_sight(
        reads,
        pixel_bounds_m=PIXELS,
        fov_bounds_uv=fov,
        epoch=EPOCH,
        validity_s=(0.0, 1.5),
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        mount_offset_p_m=np.array([0.2, 0.1, 0.0]),
        focal_length_m=1,
        time_order=time_order,
        angular_order=angular_order,
    )


def uniform_scene(origins, directions, times, wavenumbers):
    assert origins.shape == directions.shape == (*times.shape, 3)
    assert len(wavenumbers) == 3
    return np.full((*times.shape, len(wavenumbers)), 1e-8)


def coverage(rays, grid):
    bounds = rays.acquisition.exposure_bounds_s
    assert bounds.shape == (2, 2) and grid.domain is not None
    assert rays.acquisition.direction_bounds_uv.shape == (2, 4)
    assert rays.support_directions_gcrs.shape[-2:] == (4, 3)
    return (
        (bounds[:, 0] >= 0)
        & (bounds[:, 1] <= 1.5)
        & (grid.points[0] >= 4900)
        & (grid.points[-1] <= 5100)
    )


def acceptance(read, wavelengths, acquisition):
    accepted = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[read], acquisition.image_xy_m[read], 0.02
    )
    return np.broadcast_to(accepted[:, None], (len(accepted), len(wavelengths)))


def test_dated_geometry_and_sphere_hits_are_independent_of_scene():
    rays = case(time_order=1, angular_order=1)
    acq = rays.acquisition
    assert acq.sample_ids == ("read-0", "read-1")
    np.testing.assert_array_equal(acq.exposure_bounds_s, [[0, 1], [0.5, 1.5]])
    np.testing.assert_array_equal(rays.support_times_s[:, [0, -1]], acq.exposure_bounds_s)
    assert rays.support_times_s.shape == (2, 3)
    assert np.linalg.norm(rays.directions_gcrs, axis=-1).min() == pytest.approx(1)
    mid = acq.times_s[0, 0]
    p, velocity = trajectory(np.array([mid]))
    rot = platform_attitude(np.array([mid]))
    np.testing.assert_array_equal(velocity, [[0.1, 0, 0]])
    expected_origin = p[0] + rot[0] @ np.array([0.2, 0.1, 0])
    expected_direction = rot[0] @ MOUNT @ np.array([0, 0, 1])
    np.testing.assert_allclose(rays.origins_gcrs_m[0, 0], expected_origin, atol=1e-15)
    np.testing.assert_allclose(rays.directions_gcrs[0, 0], expected_direction, atol=1e-15)
    np.testing.assert_allclose(acq.image_xy_m[0, 0], [0, 0], atol=1e-15)
    end_position, _ = trajectory(np.array([1.5]))
    end_attitude = platform_attitude(np.array([1.5]))
    np.testing.assert_allclose(
        rays.support_origins_gcrs_m[1, -1, 0],
        end_position[0] + end_attitude[0] @ np.array([0.2, 0.1, 0]),
    )
    u, v = FOV[1, [1, 3]]
    corner = np.array([u, v, np.sqrt(1 - u**2 - v**2)])
    np.testing.assert_allclose(
        rays.support_directions_gcrs[1, -1, 3], end_attitude[0] @ MOUNT @ corner
    )
    hit, points = sphere_intersections(rays.origins_gcrs_m, rays.directions_gcrs, radius_m=1)
    np.testing.assert_array_equal(hit, [[True], [False]])
    np.testing.assert_allclose(np.linalg.norm(points[0, 0]), 1, atol=1e-14)
    assert np.isnan(points[1]).all()
    assert np.isfinite(rays.directions_gcrs[1]).all()  # miss still has a valid scene ray


def test_full_support_coverage_rejects_short_time_and_fov_boundaries():
    rays = case(time_order=1, angular_order=1)
    calls = []

    def short_coverage(current, grid):
        calls.append(True)
        return (current.acquisition.exposure_bounds_s[:, 1] <= 1.4) & (
            np.max(current.support_directions_gcrs[..., 2], axis=(1, 2)) <= 0.7
        )

    with pytest.raises(ValueError, match="entire line-of-sight support"):
        line_of_sight_expected_electrons(
            rays, GRID, uniform_scene, short_coverage, acceptance, collecting_area_m2=1
        )
    assert calls
    # Interior Gauss nodes fit a narrower time domain than the full second read.
    assert rays.acquisition.times_s[1, 0] <= 1.4
    assert rays.acquisition.exposure_bounds_s[1, 1] > 1.4
    with pytest.raises(ValueError, match="coverage"):
        line_of_sight_expected_electrons(
            rays,
            GRID,
            uniform_scene,
            lambda r, g: np.max(r.acquisition.direction_bounds_uv[:, [1, 3]], axis=1) < 0.79,
            acceptance,
            collecting_area_m2=1,
        )
    with pytest.raises(ValueError, match="boolean"):
        line_of_sight_expected_electrons(
            rays,
            GRID,
            uniform_scene,
            lambda *_: np.array([1, 1]),
            acceptance,
            collecting_area_m2=1,
        )


def test_uniform_scene_charge_quadrature_sources_and_seeded_dn():
    low = case(time_order=2, angular_order=2)
    high = case(time_order=4, angular_order=8)
    kwargs = dict(collecting_area_m2=1e-3, throughput=0.7, quantum_efficiency=0.8)
    signal = line_of_sight_expected_electrons(
        high, GRID, uniform_scene, coverage, acceptance, source_name="earth-scene", **kwargs
    )
    coarse = line_of_sight_expected_electrons(
        low, GRID, uniform_scene, coverage, acceptance, **kwargs
    )
    np.testing.assert_allclose(signal.total, coarse.total, rtol=3e-4)
    assert signal.reads is not high.acquisition
    np.testing.assert_array_equal(
        signal.reads.exposure_bounds_s, high.acquisition.exposure_bounds_s
    )
    assert signal.reads.sample_ids == ("read-0", "read-1")
    assert np.all(signal.total > 0)
    spectral = np.trapezoid(1 / GRID.points, GRID.points)
    expected = 1e-8 * 1e-3 * 0.7 * 0.8 * spectral / (h * c * 100)

    # A unity optical law isolates the solid-angle/time and photon-conversion accounting.
    def unity(read, wave, acq):
        return np.ones((acq.times_s.shape[1], len(wave)))

    simple = line_of_sight_expected_electrons(high, GRID, uniform_scene, coverage, unity, **kwargs)
    np.testing.assert_allclose(simple.total, expected * high.acquisition.weights_sr_s.sum(axis=1))
    extra = ExpectedElectrons(signal.reads, {"internal": np.array([2.0, 3.0])})
    joined = combine_source_paths(signal, extra)
    assert set(joined.contributions) == {"earth-scene", "internal"}
    detector = DetectorElectronics(dark_electrons_per_s=0.1, read_noise_electrons=0.5)
    a = sample_readout(joined, detector, np.random.default_rng(71), realisations=3)
    b = sample_readout(joined, detector, np.random.default_rng(71), realisations=3)
    np.testing.assert_array_equal(a.dn, b.dn)
    assert a.dn.shape == (3, 2)
    assert a.reads.sample_ids == signal.reads.sample_ids
    assert a.full_well_saturated.shape == a.adc_clipped.shape == a.dn.shape


def test_direction_cosine_weights_match_independent_solid_angle_integral():
    def angular_slice(u, lower, upper):
        scale = np.sqrt(1 - u * u)
        return np.arcsin(upper / scale) - np.arcsin(lower / scale)

    rays = case(time_order=3, angular_order=8)
    for read, (umin, umax, vmin, vmax) in enumerate(FOV):
        solid_angle, _ = quad(angular_slice, umin, umax, args=(vmin, vmax), epsabs=1e-14)
        duration = np.diff(rays.acquisition.exposure_bounds_s[read])[0]
        assert rays.acquisition.weights_sr_s[read].sum() == pytest.approx(
            duration * solid_angle, rel=1e-12
        )


def test_ray_time_spectral_scene_keeps_node_dependencies():
    rays = case(time_order=2, angular_order=2)
    seen = []

    def varying_scene(origins, directions, times, points):
        seen.append((origins.copy(), directions.copy(), times.copy(), points.copy()))
        return 1e-8 * (1 + times[..., None] + directions[..., 2, None]) * (points / 5000)

    def unity(read, wave, acq):
        return np.ones((acq.times_s.shape[1], len(wave)))

    actual = line_of_sight_expected_electrons(
        rays,
        GRID,
        varying_scene,
        coverage,
        unity,
        collecting_area_m2=1e-3,
    )
    origins, directions, times, points = seen[0]
    np.testing.assert_allclose(origins, rays.origins_gcrs_m)
    np.testing.assert_allclose(directions, rays.directions_gcrs)
    np.testing.assert_allclose(times, rays.acquisition.times_s)
    np.testing.assert_array_equal(points, GRID.points)
    spectral = np.trapezoid((points / 5000) / (h * c * 100 * points), points)
    expected = (
        1e-11
        * spectral
        * np.sum(rays.acquisition.weights_sr_s * (1 + times + directions[..., 2]), axis=1)
    )
    np.testing.assert_allclose(actual.total, expected, rtol=1e-14)
    with pytest.raises(ValueError, match="scene coverage"):
        line_of_sight_expected_electrons(
            rays,
            GRID,
            varying_scene,
            lambda r, g: np.full(2, g.points[-1] <= 5050),
            unity,
            collecting_area_m2=1e-3,
        )


def test_validation_and_owned_rays():
    rays = case(time_order=1, angular_order=1)
    other = replace(rays, origins_gcrs_m=rays.origins_gcrs_m)
    other.origins_gcrs_m[:] = 0
    assert rays.origins_gcrs_m[0, 0, 0] > 2
    with pytest.raises(ValueError, match="FOV"):
        case(fov=np.array([[-0.1, 1.0, 0.0, 0.1], [0.75, 0.8, 0, 0.01]]))
    with pytest.raises(ValueError, match="proper"):
        sample_lines_of_sight(
            read_schedule(rays.acquisition),
            pixel_bounds_m=PIXELS,
            fov_bounds_uv=FOV,
            epoch=EPOCH,
            validity_s=(0.0, 1.5),
            trajectory=trajectory,
            platform_attitude=platform_attitude,
            instrument_attitude=lambda t: np.broadcast_to(
                np.diag([1.0, 1.0, -1.0]), (len(t), 3, 3)
            ),
            mount_offset_p_m=np.zeros(3),
            focal_length_m=1,
        )
    with pytest.raises(ValueError, match="scene radiance"):
        line_of_sight_expected_electrons(
            rays,
            GRID,
            lambda *args: np.full((2, 1, 3), np.nan),
            coverage,
            acceptance,
            collecting_area_m2=1,
        )


def test_independent_noncommuting_scan_and_platform_preserve_support_and_weights():
    reference = case(time_order=2, angular_order=2)
    reads = read_schedule(reference.acquisition)
    seen = []

    def moving_instrument(times):
        seen.append(times.copy())
        return MOUNT @ axis_rotations(np.array([1.0, 0.0, 0.0]), 0.3 * times)

    offset = np.array([0.2, 0.1, -0.4])
    rays = sample_lines_of_sight(
        reads,
        pixel_bounds_m=PIXELS,
        fov_bounds_uv=FOV,
        epoch=EPOCH,
        validity_s=(0.0, 1.5),
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=moving_instrument,
        mount_offset_p_m=offset,
        focal_length_m=1.0,
        time_order=2,
        angular_order=2,
    )
    assert rays.epoch == EPOCH
    assert len(seen) == 2
    np.testing.assert_array_equal(np.array(seen), rays.support_times_s)
    np.testing.assert_array_equal(rays.support_times_s[:, [0, -1]], reads.exposure_bounds_s)
    for name in ("times_s", "directions_uv", "image_xy_m", "weights_sr_s"):
        np.testing.assert_array_equal(
            getattr(rays.acquisition, name), getattr(reference.acquisition, name)
        )
    for i in range(2):
        t = rays.acquisition.times_s[i]
        position, _ = trajectory(t)
        platform = platform_attitude(t)
        relative = moving_instrument(t)
        uv = rays.acquisition.directions_uv[i]
        sensor = np.column_stack((uv, np.sqrt(1 - np.sum(uv**2, axis=-1))))
        expected_origins = position + np.einsum("tij,j->ti", platform, offset)
        expected_directions = np.einsum("tij,tjk,tk->ti", platform, relative, sensor)
        np.testing.assert_allclose(rays.origins_gcrs_m[i], expected_origins, atol=1e-15)
        np.testing.assert_allclose(rays.directions_gcrs[i], expected_directions, atol=1e-15)
        assert not np.allclose(platform @ relative, relative @ platform)
        corners = np.array([[FOV[i, a], FOV[i, b]] for a, b in ((0, 2), (0, 3), (1, 2), (1, 3))])
        corner_i = np.column_stack((corners, np.sqrt(1 - np.sum(corners**2, axis=-1))))
        origins, directions = compute_lines_of_sight(
            rays.support_times_s[i],
            epoch=EPOCH,
            validity_s=(0.0, 1.5),
            trajectory=trajectory,
            platform_attitude=platform_attitude,
            instrument_attitude=moving_instrument,
            directions_i=corner_i,
            mount_offset_p_m=offset,
        )
        np.testing.assert_array_equal(rays.support_origins_gcrs_m[i], origins)
        np.testing.assert_array_equal(rays.support_directions_gcrs[i], directions)


@pytest.mark.parametrize(
    "validity", [(0.1, 1.4), (0.0, 1.4), (0.1, 1.5), (1.0, 0.0), (0.0, np.nan)]
)
def test_complete_exposures_checked_before_any_callback(validity):
    reference = case(time_order=1, angular_order=1)
    calls = []

    def unavailable(times):
        calls.append(times.copy())
        raise AssertionError("unsupported callback must not run")

    # The interior nodes at 0.5 and 1.0 fit (0.1,1.4); endpoints do not.
    with pytest.raises(ValueError, match="complete exposures"):
        sample_lines_of_sight(
            read_schedule(reference.acquisition),
            pixel_bounds_m=PIXELS,
            fov_bounds_uv=FOV,
            epoch=EPOCH,
            validity_s=validity,
            trajectory=unavailable,
            platform_attitude=unavailable,
            instrument_attitude=unavailable,
            mount_offset_p_m=np.zeros(3),
            focal_length_m=1.0,
            time_order=1,
            angular_order=1,
        )
    assert calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"epoch": datetime(2020, 1, 1)},
        {"mount_offset_p_m": np.zeros(2)},
        {"mount_offset_p_m": np.array([np.nan, 0, 0])},
        {"focal_length_m": 0},
        {"time_order": 1.5},
        {"angular_order": 0},
        {"trajectory": lambda t: (np.zeros((len(t), 3)), np.zeros((len(t), 2)))},
    ],
)
def test_sampling_validates_time_geometry_and_trajectory(changes):
    reference = case(time_order=1, angular_order=1)
    options = dict(
        pixel_bounds_m=PIXELS,
        fov_bounds_uv=FOV,
        epoch=EPOCH,
        validity_s=(0.0, 1.5),
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        mount_offset_p_m=np.zeros(3),
        focal_length_m=1.0,
        time_order=1,
        angular_order=1,
    )
    with pytest.raises(ValueError):
        sample_lines_of_sight(read_schedule(reference.acquisition), **(options | changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"epoch": datetime(2020, 1, 1)},
        {"support_times_s": np.zeros(2)},
        {"support_times_s": np.zeros((2, 2))},
        {"support_times_s": np.array([[0.0, 1.0, 0.5, 1.0], [0.5, 1.0, 0.7, 1.5]])},
        {"origins_gcrs_m": np.zeros((2, 1, 2))},
        {"directions_gcrs": np.zeros((2, 1, 3))},
    ],
)
def test_ray_record_validates_shapes_unit_directions_and_order(changes):
    with pytest.raises(ValueError):
        replace(case(time_order=1, angular_order=1), **changes)


@pytest.mark.skipif(
    os.environ.get("ORSIA_TEST_ORBITAL") != "1",
    reason="run just --justfile tests/justfile test-orbital for native frame/time integration",
)
def test_terrestrial_state_and_attitude_are_transformed_before_instrument_sampling():
    from orsia_geometry import frame_rotations, transform_states

    reads = ReadSchedule(("dated",), ("pixel",), ("setting",), ("direct",), np.array([[0.0, 3.0]]))
    offset = np.array([10.0, 5.0, 2.0])
    position_itrf = np.array([7e6, 0.0, 0.0])
    seen = []

    def rotation(times):
        return frame_rotations(
            times, epoch=EPOCH, validity_s=(0.0, 3.0), from_frame="ITRF", to_frame="GCRS"
        )

    def dated_trajectory(times):
        seen.append(times.copy())
        return transform_states(
            times,
            np.broadcast_to(position_itrf, (len(times), 3)).copy(),
            np.zeros((len(times), 3)),
            epoch=EPOCH,
            validity_s=(0.0, 3.0),
            from_frame="ITRF",
            to_frame="GCRS",
        )

    rays = sample_lines_of_sight(
        reads,
        pixel_bounds_m=PIXELS[:1],
        fov_bounds_uv=FOV[:1],
        epoch=EPOCH,
        validity_s=(0.0, 3.0),
        trajectory=dated_trajectory,
        platform_attitude=rotation,
        instrument_attitude=instrument_attitude,
        mount_offset_p_m=offset,
        focal_length_m=1.0,
        time_order=3,
        angular_order=2,
    )
    assert not sys._is_gil_enabled()
    # Three physical SI seconds cross the leap second at the end of 2016.
    expected_end_rotation = frame_rotations(
        np.array([0.0]),
        epoch=datetime(2017, 1, 1, 0, 0, 1, tzinfo=UTC),
        validity_s=(0.0, 0.0),
        from_frame="ITRF",
        to_frame="GCRS",
    )
    np.testing.assert_array_equal(rotation(np.array([3.0])), expected_end_rotation)
    np.testing.assert_array_equal(seen[0], rays.support_times_s[0])
    assert seen[0][0] == 0.0 and seen[0][-1] == 3.0
    times = rays.acquisition.times_s[0]
    rotations = rotation(times)
    position, velocity = dated_trajectory(times)
    assert np.linalg.norm(velocity, axis=-1).min() > 400.0
    assert not np.allclose(position, position_itrf)
    expected_origins = np.einsum("tij,j->ti", rotations, position_itrf + offset)
    uv = rays.acquisition.directions_uv[0]
    sensor = np.column_stack((uv, np.sqrt(1 - np.sum(uv**2, axis=-1))))
    expected_directions = np.einsum("tij,jk,tk->ti", rotations, MOUNT, sensor)
    np.testing.assert_allclose(rays.origins_gcrs_m[0], expected_origins, atol=2e-9, rtol=0)
    np.testing.assert_allclose(rays.directions_gcrs[0], expected_directions, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(
        read_schedule(rays.acquisition).exposure_bounds_s, reads.exposure_bounds_s
    )
