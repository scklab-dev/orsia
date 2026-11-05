"""End-to-end local spectral imagers, actual joint footprints and supplied calibration."""

import sys
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_instruments import (
    DetectorProperties,
    GaussianPSF,
    calibrate_frame_electrons,
    calibrate_frames,
    camera_acquisition_grid,
    cavity_imager,
    filter_camera,
    filter_strip,
    focal_plane_imager,
    frame_response,
    gaussian_filter,
    group_native_bands,
    interferometric_imager,
    lvf_imager,
    raster_read_schedule,
    reconstruct_radiance_cube,
    regular_pixel_bounds,
    select_raster_reads,
    simulate_frames,
    spatial_spectral_response,
    stack_frame_electrons,
)
from orsia_inverse.linear import solve_linear
from orsia_radiative import SpectralDomain, wavenumber_grid
from scipy.integrate import quad


def detector(**kwargs):
    parameters = dict(
        pixel_bounds_m=regular_pixel_bounds((1, 2), pitch_m=20e-6),
        pitch_m=None,
        quantum_efficiency=0.8,
        dark_electrons_per_s=2.0,
        read_noise_electrons=1.0,
        full_well_electrons=1e8,
        electrons_per_dn=1.0,
        bias_dn=100.0,
        nonlinearity=0,
        adc_bits=24,
    )
    parameters.update(kwargs)
    return DetectorProperties(**parameters)


def spectral(count=81):
    return wavenumber_grid(
        np.linspace(4500.0, 5500.0, count),
        domain=SpectralDomain("wavenumber_cm1", ((4500.0, 5500.0),)),
    )


def identity(times):
    return np.broadcast_to(np.eye(3), (len(times), 3, 3))


def platform(times):
    return np.broadcast_to(np.diag([-1.0, 1.0, -1.0]), (len(times), 3, 3))


def trajectory(times):
    position = np.column_stack((0.2 * times, times * 0, times * 0 + 300))
    return position, np.broadcast_to([0.2, 0.0, 0.0], position.shape)


def surface(origins, directions, times):
    distance = -origins[..., 2] / directions[..., 2]
    return (origins + directions * distance[..., None])[..., :2], distance > 0


def grid(camera, frames=8, order=8, schedule=None, **kwargs):
    if schedule is None:
        start = np.arange(frames) * 0.01
        schedule = raster_read_schedule(
            np.column_stack((start, start + 0.005)), camera.detector.shape, port=camera.optics.port
        )
    parameters = dict(
        epoch=datetime(2023, 5, 16, tzinfo=UTC),
        validity_s=(0, 1),
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301.0,
        time_order=2,
        angular_order=order,
    )
    parameters.update(kwargs)
    return camera_acquisition_grid(camera, schedule, **parameters)


def optical():
    return dict(
        focal_length_m=0.3, collecting_area_m2=np.pi * (0.3 / 4) ** 2 / 4, spatial=GaussianPSF(4e-6)
    )


def lvf(gradient=0.01, **kwargs):
    return focal_plane_imager(
        detector(),
        lvf_imager(**optical(), reference_m=2e-6, gradient=gradient, sigma_m=50e-9, **kwargs),
        support_radius_m=4e-6 * np.sqrt(-2 * np.log(1e-6)),
    )


def port(sign, **kwargs):
    return focal_plane_imager(
        detector(),
        interferometric_imager(**optical(), opd_m=np.linspace(0, 30e-6, 8), port=sign, **kwargs),
        support_radius_m=25e-6,
    )


def coverage(acquisition, spectral_grid):
    return np.ones(len(acquisition.read_bands), dtype=bool)


def uniform(xy, directions, times, points):
    return np.full((*times.shape, len(points)), 2e-5)


def heterogeneous(xy, directions, times, points):
    return np.where(
        xy[..., 0, None] < 0,
        np.where(points < 5000, 1e-5, 3e-5),
        np.where(points < 5000, 4e-5, 2e-5),
    )


def joint(instruments, grids, **kwargs):
    parameters = dict(
        x_edges_m=np.array([-1.0, 0, 1]),
        y_edges_m=np.array([-1.0, 1]),
        spectral_edges_cm1=np.array([4500.0, 5000.0, 5500.0]),
        max_bytes=100_000_000,
    )
    parameters.update(kwargs)
    return spatial_spectral_response(instruments, grids, spectral(), **parameters)


def test_lvf_zero_gradient_matches_fixed_filter_and_native_cube():
    camera = lvf(0, pixel_order=14)
    fixed = filter_camera(
        detector(),
        [
            filter_strip(
                0, 1, band="fixed", transmission=lambda wave: gaussian_filter(wave, 2e-6, 50e-9)
            )
        ],
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(4e-6),
    )
    acquisition = grid(camera, frames=3, order=16)
    truth = simulate_frames(camera, acquisition, spectral(), uniform, coverage)
    reference = simulate_frames(
        fixed, grid(fixed, frames=3, order=16), spectral(), uniform, coverage
    )
    assert_allclose(truth.images.photoelectrons, reference.images.photoelectrons, rtol=3e-7)
    estimated = frame_response(camera, acquisition, spectral())
    grouped, bands = group_native_bands(estimated)
    assert len(bands) == 1
    calibrated = calibrate_frames(truth.images, grouped, detector=camera.detector)
    cube = reconstruct_radiance_cube(
        calibrated,
        acquisition,
        x_edges_m=np.linspace(-0.1, 0.1, 11),
        y_edges_m=np.array([-0.1, 0.1]),
        bands=bands,
    )
    assert_allclose(cube.values[cube.valid], 2e-5)
    assert np.any(~cube.valid)
    assert not sys._is_gil_enabled()


def test_fixed_lvf_scan_keeps_achieved_bands_and_estimated_perturbations():
    camera = lvf()
    acquisition = grid(camera)
    truth = simulate_frames(camera, acquisition, spectral(), heterogeneous, coverage)
    estimated = frame_response(camera, acquisition, spectral())
    grouped, bands = group_native_bands(estimated, response_tolerance=1e-7)
    assert len(bands) == 2
    assert np.sum(np.abs(bands[0].weights - bands[1].weights)) > 0.5
    calibrated = calibrate_frames(truth.images, grouped, detector=camera.detector)
    kwargs = dict(
        x_edges_m=np.linspace(-0.1, 0.1, 41), y_edges_m=np.array([-0.1, 0.1]), bands=bands
    )
    cube = reconstruct_radiance_cube(calibrated, acquisition, **kwargs)
    assert cube.values.shape == (1, 40, 2)
    assert np.any(cube.valid[..., 0] & cube.valid[..., 1])
    shifted = replace(acquisition, xy_m=acquisition.xy_m + np.array([0.02, 0]))
    displaced = reconstruct_radiance_cube(calibrated, shifted, **kwargs)
    assert not np.array_equal(cube.coverage, displaced.coverage)
    low_qe = replace(camera, detector=replace(camera.detector, quantum_efficiency=0.4))
    wrong = frame_response(low_qe, acquisition, spectral())
    biased = calibrate_frames(truth.images, wrong, detector=low_qe.detector)
    assert_allclose(biased.values, calibrated.values * 2)
    assert_allclose(
        truth.images.photoelectrons,
        simulate_frames(
            camera, acquisition, spectral(), heterogeneous, coverage
        ).images.photoelectrons,
    )


def test_joint_two_port_heterogeneous_spatial_opd_closes_direct_and_inverse():
    instruments = [port(sign, opd_offset_m=lambda x, y: x * 0.05) for sign in ("+", "-")]
    grids = [grid(camera, order=10) for camera in instruments]
    response = joint(instruments, grids)
    state = np.array([1e-5, 3e-5, 4e-5, 2e-5])
    frames = [
        simulate_frames(camera, acquisition, spectral(), heterogeneous, coverage)
        for camera, acquisition in zip(instruments, grids, strict=True)
    ]
    charges = [
        calibrate_frame_electrons(f.images, detector=c.detector)
        for f, c in zip(frames, instruments, strict=True)
    ]
    observations, valid = stack_frame_electrons(response, charges)
    assert valid.all()
    with pytest.raises(ValueError, match="identities"):
        stack_frame_electrons(response, charges[::-1])
    with pytest.raises(ValueError, match="schedule"):
        stack_frame_electrons(response, charges[:1])
    with pytest.raises(ValueError, match="realization"):
        stack_frame_electrons(response, charges, realisation=1)
    assert_allclose(response.matrix @ state, observations, rtol=1e-14)
    assert response.schedules[0].reads.sample_ids == response.schedules[1].reads.sample_ids
    assert set(response.schedules[0].reads.port_ids) == {"+"}
    assert set(response.schedules[1].reads.port_ids) == {"-"}
    assert response.state_shape == (1, 2, 2)
    assert np.all(response.coverage > 0)
    covariance = np.diag(observations + 4) + np.full((len(observations), len(observations)), 2.0)
    solved = solve_linear(
        response.matrix, observations, covariance=covariance, max_bytes=10_000_000
    )
    assert solved.rank == 4
    assert solved.identifiable.all()
    assert_allclose(solved.state, state, rtol=2e-13)
    assert_allclose(
        solved.covariance, solved.gain @ covariance @ solved.gain.T, rtol=1e-12, atol=1e-25
    )
    selected = np.array([15, 2, 8, 0], dtype=np.int64)
    partial = [
        grid(camera, schedule=select_raster_reads(g.schedule, selected), order=10)
        for camera, g in zip(instruments, grids, strict=True)
    ]
    subset = joint(instruments, partial)
    assert_allclose(subset.matrix, response.matrix[np.r_[selected, selected + 16]])
    with pytest.raises(ValueError, match="identities"):
        stack_frame_electrons(subset, charges)


def test_port_conservation_and_unequal_gain_covariance_correction():
    cameras = [port(sign) for sign in ("+", "-")]
    grids = [grid(camera) for camera in cameras]
    pair = [
        frame_response(camera, acquisition, spectral())
        for camera, acquisition in zip(cameras, grids, strict=True)
    ]
    zero = port("+", visibility=0)
    reference = frame_response(zero, grid(zero), spectral()).spectral_coefficients
    assert_allclose(
        pair[0].spectral_coefficients + pair[1].spectral_coefficients, 2 * reference, rtol=3e-15
    )
    for camera, acquisition, gain in zip(cameras, grids, (2.0, 4.0), strict=True):
        physical = replace(camera, detector=replace(camera.detector, electrons_per_dn=gain))
        frames = simulate_frames(
            physical,
            acquisition,
            spectral(),
            uniform,
            coverage,
            rng=np.random.default_rng(3),
            realisations=2,
        )
        corrected = calibrate_frame_electrons(
            frames.images, detector=physical.detector, covariance=np.ones(16)
        )
        assert corrected.covariance is not None
        assert_allclose(corrected.covariance, gain**2)
        assert corrected.valid.all()
        assert frames.images.full_well_saturated is not None
        flagged = frames.images.full_well_saturated.copy()
        flagged.reshape(2, -1)[0, 0] = True
        bad = replace(frames.images, full_well_saturated=flagged)
        with pytest.raises(ValueError, match="clipped"):
            calibrate_frame_electrons(bad, detector=physical.detector)
        masked = calibrate_frame_electrons(bad, detector=physical.detector, mask_flagged=True)
        assert np.isnan(masked.values[:, 0]).all()
        assert not masked.valid[:, 0].any()


def test_cavity_native_multi_peak_spacing_scan_and_resonance_reference():
    spacings = np.array([4e-6, 4.2e-6, 4.4e-6])
    camera = focal_plane_imager(
        detector(),
        cavity_imager(
            **optical(),
            spacing_m=spacings,
            reflectivity=0.7,
            order_transmission=1,
            internal_angle_rad=0,
        ),
        support_radius_m=25e-6,
    )
    acquisition = grid(camera, frames=3, order=14)
    wave_grid = wavenumber_grid(
        np.linspace(2000.0, 9000.0, 1401),
        domain=SpectralDomain("wavenumber_cm1", ((2000.0, 9000.0),)),
    )
    response = frame_response(camera, acquisition, wave_grid)
    grouped, bands = group_native_bands(response)
    assert len(bands) == 3
    values = response.spectral_coefficients[0]
    peaks = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    assert len(peaks) >= 5
    # Remove photon/quadrature weighting: local Airy law at normal incidence.
    expected = 1 / (
        1 + 4 * 0.7 / (1 - 0.7) ** 2 * np.sin(2 * np.pi * 4e-6 * 100 * wave_grid.points) ** 2
    )
    actual = values[1:-1] * wave_grid.points[1:-1]
    assert_allclose(actual / actual.max(), expected[1:-1] / expected[1:-1].max(), rtol=1e-13)
    truth = simulate_frames(camera, acquisition, wave_grid, uniform, coverage)
    calibrated = calibrate_frames(truth.images, grouped, detector=camera.detector)
    cube = reconstruct_radiance_cube(
        calibrated,
        acquisition,
        x_edges_m=np.array([-1.0, 1.0]),
        y_edges_m=np.array([-1.0, 1.0]),
        bands=bands,
    )
    assert_allclose(cube.values, 2e-5)


@pytest.mark.parametrize(
    "kwargs,match,error",
    [
        ({"max_bytes": 1}, "budget", MemoryError),
        ({"x_edges_m": np.array([-0.001, 0.001])}, "continuous", ValueError),
        ({"spectral_edges_cm1": np.array([4600.0, 5500.0])}, "spectral", ValueError),
        ({"x_edges_m": np.array([1.0, 0.0])}, "increasing", ValueError),
        ({"read_batch_size": 0}, "positive", ValueError),
    ],
)
def test_joint_resource_and_support_boundaries(kwargs, match, error):
    camera = lvf()
    with pytest.raises(error, match=match):
        joint([camera], [grid(camera)], **kwargs)


def test_lvf_pixel_and_incident_quadrature_refinement():
    outputs = []
    for order in (10, 22, 30):
        camera = lvf(pixel_order=order)
        outputs.append(
            simulate_frames(
                camera, grid(camera, frames=1, order=order), spectral(161), uniform, coverage
            ).images.photoelectrons
        )
    assert np.max(abs(outputs[2] - outputs[1])) < np.max(abs(outputs[1] - outputs[0]))
    assert_allclose(outputs[2], outputs[1], rtol=2e-6)


def test_zero_visibility_has_coverage_without_spectral_identifiability():
    cameras = [port(sign, visibility=0) for sign in ("+", "-")]
    response = joint(cameras, [grid(camera) for camera in cameras])
    values = response.matrix @ np.array([1e-5, 3e-5, 4e-5, 2e-5])
    fitted = solve_linear(
        response.matrix, values, covariance=np.ones(len(values)), max_bytes=10_000_000
    )
    assert np.all(response.coverage > 0)
    assert fitted.rank == 2
    assert not fitted.identifiable.any()
    assert np.isinf(fitted.condition)


def test_grouping_dark_response_and_tolerance_chain():
    camera = lvf()
    response = frame_response(camera, grid(camera, frames=1), spectral())
    with pytest.raises(ValueError, match="tolerance"):
        group_native_bands(response, response_tolerance=-1)
    dark = replace(
        response,
        spectral_coefficients=np.zeros_like(response.spectral_coefficients),
        spatial_weights=np.zeros_like(response.spatial_weights),
    )
    grouped, bands = group_native_bands(dark)
    assert not bands
    assert grouped.read_bands == ("", "")
    # Equal nominal labels never force merging of incompatible achieved responses.
    same_labels = replace(response, read_bands=("same", "same"))
    grouped, bands = group_native_bands(same_labels)
    assert len(bands) == 2
    assert grouped.reads is response.reads


def test_calibrated_charge_dense_covariance_flags_and_missing_reads():
    camera = lvf()
    acquisition = grid(camera, frames=1)
    frames = simulate_frames(
        camera, acquisition, spectral(), uniform, coverage, rng=np.random.default_rng(9)
    )
    estimate = replace(camera.detector, electrons_per_dn=np.array([[2.0, 3.0]]))
    covariance = np.array([[2.0, 0.4], [0.4, 3.0]])
    corrected = calibrate_frame_electrons(frames.images, detector=estimate, covariance=covariance)
    assert corrected.covariance is not None
    assert_allclose(
        corrected.covariance, covariance * np.array([2.0, 3.0])[:, None] * np.array([2.0, 3.0])
    )
    missing = calibrate_frame_electrons(
        frames.images, detector=estimate, read_valid=np.array([False, False])
    )
    assert np.isnan(missing.values).all()
    expected = calibrate_frame_electrons(
        frames.images, detector=estimate, covariance=covariance, use_expectation=True
    )
    assert expected.covariance is not None
    assert_allclose(expected.covariance, covariance)
    invalid: tuple[dict[str, Any], ...] = (dict(read_valid=np.ones(2)), dict(covariance=np.ones(3)))
    for kwargs in invalid:
        with pytest.raises(ValueError):
            calibrate_frame_electrons(frames.images, detector=estimate, **kwargs)


def test_physical_port_setting_and_support_contracts():
    camera = port("+")
    acquisition = grid(camera)
    with pytest.raises(ValueError, match="port"):
        frame_response(port("-"), acquisition, spectral())
    with pytest.raises(ValueError, match="exposure"):
        frame_response(camera, grid(camera, frames=1), spectral())
    with pytest.raises(ValueError, match="radius"):
        focal_plane_imager(camera.detector, camera.optics, support_radius_m=-1)
    with pytest.raises(ValueError, match="nonempty"):
        joint([], [])


def test_cavity_spatial_offset_and_estimated_spectral_perturbation():
    def instrument(spacing, order):
        return focal_plane_imager(
            detector(),
            cavity_imager(
                **optical(),
                spacing_m=spacing,
                order_transmission=0.9,
                internal_angle_rad=0.1,
                refractive_index=1.2,
                coating_phase_rad=0.2,
                spacing_offset_m=lambda x, y: 0.001 * x,
                pixel_order=order,
            ),
            support_radius_m=25e-6,
        )

    fine = instrument(4e-6, 24)
    acquisition = grid(fine, frames=1, order=20)
    truth = simulate_frames(fine, acquisition, spectral(), heterogeneous, coverage)
    coarse = frame_response(instrument(4e-6, 16), acquisition, spectral())
    assert_allclose(coarse.spectral_coefficients, truth.response.spectral_coefficients, rtol=2e-6)
    wrong = frame_response(instrument(4.05e-6, 24), acquisition, spectral())
    assert not np.allclose(
        wrong.spectral_coefficients, truth.response.spectral_coefficients, rtol=1e-2
    )


def test_cavity_spectral_quadrature_converges_to_independent_integral():
    def instrument(reflectivity):
        return focal_plane_imager(
            detector(),
            cavity_imager(
                **optical(), spacing_m=4e-6, reflectivity=reflectivity, order_transmission=1
            ),
            support_radius_m=25e-6,
        )

    camera, relay = instrument(0.7), instrument(0)
    acquisition = grid(camera, frames=1)
    independent = quad(
        lambda nu: 1 / (nu * (1 + 4 * 0.7 / 0.3**2 * np.sin(2 * np.pi * 4e-6 * 100 * nu) ** 2)),
        4500,
        5500,
        epsabs=1e-13,
    )[0] / np.log(5500 / 4500)
    errors = []
    for count in (41, 81, 161):
        actual = frame_response(camera, acquisition, spectral(count)).spectral_coefficients.sum(
            axis=1
        )
        unfiltered = frame_response(relay, acquisition, spectral(count)).spectral_coefficients.sum(
            axis=1
        )
        errors.append(float(np.max(abs(actual / unfiltered - independent))))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-6
