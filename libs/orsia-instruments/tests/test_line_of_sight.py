"""Synthetic sight-line geometry, full-support radiance and detector composition."""

from dataclasses import replace

import numpy as np
import pytest
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
    sphere_intersections,
)
from orsia_radiative import SpectralDomain, SpectralGrid
from scipy.constants import c, h
from scipy.integrate import quad


def state(times):
    angle = 0.1 * times
    rotation = np.zeros((len(times), 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angle)
    rotation[:, 1, 0] = np.sin(angle)
    rotation[:, 0, 1] = -np.sin(angle)
    rotation[:, 2, 2] = 1
    position = np.column_stack((2 + 0.1 * times, np.zeros_like(times), np.zeros_like(times)))
    return position, rotation


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
        platform_state=state,
        mount_offset_p_m=np.array([0.2, 0.1, 0.0]),
        mount_rotation_p_i=MOUNT,
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
    assert rays.support_directions_e.shape[-2:] == (4, 3)
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
    assert np.linalg.norm(rays.directions_e, axis=-1).min() == pytest.approx(1)
    mid = acq.times_s[0, 0]
    p, rot = state(np.array([mid]))
    expected_origin = p[0] + rot[0] @ np.array([0.2, 0.1, 0])
    expected_direction = rot[0] @ MOUNT @ np.array([0, 0, 1])
    np.testing.assert_allclose(rays.origins_e_m[0, 0], expected_origin, atol=1e-15)
    np.testing.assert_allclose(rays.directions_e[0, 0], expected_direction, atol=1e-15)
    np.testing.assert_allclose(acq.image_xy_m[0, 0], [0, 0], atol=1e-15)
    end_position, end_attitude = state(np.array([1.5]))
    np.testing.assert_allclose(
        rays.support_origins_e_m[1, -1, 0],
        end_position[0] + end_attitude[0] @ np.array([0.2, 0.1, 0]),
    )
    u, v = FOV[1, [1, 3]]
    corner = np.array([u, v, np.sqrt(1 - u**2 - v**2)])
    np.testing.assert_allclose(
        rays.support_directions_e[1, -1, 3], end_attitude[0] @ MOUNT @ corner
    )
    hit, points = sphere_intersections(rays, radius_m=1)
    np.testing.assert_array_equal(hit, [[True], [False]])
    np.testing.assert_allclose(np.linalg.norm(points[0, 0]), 1, atol=1e-14)
    assert np.isnan(points[1]).all()
    assert np.isfinite(rays.directions_e[1]).all()  # miss still has a valid scene ray


def test_full_support_coverage_rejects_short_time_and_fov_boundaries():
    rays = case(time_order=1, angular_order=1)
    calls = []

    def short_coverage(current, grid):
        calls.append(True)
        return (current.acquisition.exposure_bounds_s[:, 1] <= 1.4) & (
            np.max(current.support_directions_e[..., 2], axis=(1, 2)) <= 0.7
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
    assert signal.acquisition is high.acquisition
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
    np.testing.assert_allclose(origins, rays.origins_e_m)
    np.testing.assert_allclose(directions, rays.directions_e)
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
    other = replace(rays, origins_e_m=rays.origins_e_m)
    other.origins_e_m[:] = 0
    assert rays.origins_e_m[0, 0, 0] > 2
    with pytest.raises(ValueError, match="FOV"):
        case(fov=np.array([[-0.1, 1.0, 0.0, 0.1], [0.75, 0.8, 0, 0.01]]))
    with pytest.raises(ValueError, match="proper"):
        sample_lines_of_sight(
            read_schedule(rays.acquisition),
            pixel_bounds_m=PIXELS,
            fov_bounds_uv=FOV,
            platform_state=state,
            mount_offset_p_m=np.zeros(3),
            mount_rotation_p_i=np.diag([1, 1, -1]),
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
