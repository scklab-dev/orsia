"""Independent finite-slit dispersion, photon integration and scan reconstruction checks."""

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
    calibrate_frames,
    dispersive_spectrometer,
    frame_exposure_bounds,
    frame_response,
    ground_integration_bounds,
    raster_read_schedule,
    read_schedule,
    reconstruct_radiance_cube,
    regular_pixel_bounds,
    simulate_frames,
    spectrometer_acquisition_grid,
)
from orsia_instruments.acquisition.los import focal_plane_direction_bounds, sample_lines_of_sight
from orsia_radiative import SpectralBand, SpectralDomain, spectral_band, wavenumber_grid
from scipy.constants import c, h
from scipy.integrate import quad
from scipy.special import ndtr

EPOCH = datetime(2023, 5, 16, tzinfo=UTC)


def ideal_detector(pixels):
    """Attach explicit ideal photoelectric properties to a test focal plane."""
    return DetectorProperties(
        pixel_bounds_m=pixels,
        pitch_m=None,
        quantum_efficiency=1,
        dark_electrons_per_s=0,
        read_noise_electrons=0,
        full_well_electrons=1e6,
        electrons_per_dn=1,
        bias_dn=0,
        nonlinearity=0,
        adc_bits=16,
    )


def spectral_grid(count=65):
    return wavenumber_grid(
        np.linspace(4500.0, 5500.0, count),
        domain=SpectralDomain("wavenumber_cm1", ((4500.0, 5500.0),)),
    )


def identity(times):
    return np.broadcast_to(np.eye(3), (len(times), 3, 3))


def platform(times):
    return np.broadcast_to(np.diag([-1.0, 1.0, -1.0]), (len(times), 3, 3))


def trajectory(times):
    positions = np.column_stack((times * 0.2, times * 0, times * 0 + 300))
    return positions, np.broadcast_to([0.2, 0.0, 0.0], positions.shape)


def surface(origins, directions, times):
    distance = -origins[..., 2] / directions[..., 2]
    return (origins + distance[..., None] * directions)[..., :2], -directions, distance > 0


def coverage(grid, spectral):
    return np.ones(len(grid.read_bands), dtype=bool)


def uniform(xy, directions, times, query_spectral):
    points = query_spectral.points
    return np.full((*times.shape, len(points)), 2e-5)


def instrument(**kwargs):
    parameters = dict(
        slit_bounds_m=np.array([-5e-6, 5e-6, -20e-6, 20e-6]),
        focal_length_m=0.3,
        collecting_area_m2=0.004,
        spatial=GaussianPSF(4e-6),
        reference_m=2e-6,
        dispersion=200.0,
    )
    parameters.update(kwargs)
    return dispersive_spectrometer(
        ideal_detector(regular_pixel_bounds((2, 5), pitch_m=20e-6)), **parameters
    )


def acquisition(spec, *, frames=3, slit_order=20, time_order=2, **kwargs):
    parameters = dict(
        epoch=EPOCH,
        validity_s=(0, 1),
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301.0,
        slit_order=slit_order,
        time_order=time_order,
    )
    parameters.update(kwargs)
    return spectrometer_acquisition_grid(
        spec,
        raster_read_schedule(
            frame_exposure_bounds(start_s=0, frame_count=frames, frame_rate_hz=20, exposure_s=0.01),
            (spec).pixel_bounds_m.shape[:2],
            port=(spec).optics.port,
        ),
        **parameters,
    )


def bands(response):
    result = []
    for name in dict.fromkeys(response.read_bands):
        coefficients = response.spectral_coefficients[response.read_bands.index(name)]
        result.append(SpectralBand(name, response.grid, coefficients / coefficients.sum()))
    return result


def test_slit_rays_are_entrance_coordinates_and_support_includes_scan():
    spec = instrument()
    grid = acquisition(spec)
    rays = grid.rays.acquisition
    assert grid.image_shape == (3, 2, 5)
    assert grid.read_bands[:10] == spec.pixel_bands
    assert grid.rays.acquisition.sample_ids[-1] == "s2-p9"
    assert grid.rays.acquisition.pixel_ids[0] == grid.rays.acquisition.pixel_ids[10]
    # Detector pixels have different x, yet they receive identical entrance rays.
    assert_allclose(rays.image_xy_m[0], rays.image_xy_m[4], atol=0, rtol=0)
    assert np.all(rays.image_xy_m >= spec.slit_bounds_m[[0, 2]])
    assert np.all(rays.image_xy_m <= spec.slit_bounds_m[[1, 3]])
    expected_xy = np.stack(
        (0.2 * rays.times_s - 1000 * rays.image_xy_m[..., 0], 1000 * rays.image_xy_m[..., 1]),
        axis=-1,
    )
    assert_allclose(grid.xy_m, expected_xy, atol=1e-17)
    for frame in range(3):
        start, end = rays.exposure_bounds_s[frame * 10]
        # Exact planar full-slit extrema over the complete moving exposure.
        lower, upper = np.array([0.2 * start - 0.005, -0.02]), np.array([0.2 * end + 0.005, 0.02])
        assert np.all(grid.bounds_m[frame * 10, [0, 2]] <= lower)
        assert np.all(grid.bounds_m[frame * 10, [1, 3]] >= upper)
    enclosing = ground_integration_bounds(grid)
    assert enclosing[0] < -0.005 and enclosing[1] > 0.027
    assert grid.psf_support_radius_m == 0
    # A post-slit PSF changes detector acceptance, never the scene-ray domain.
    broad = acquisition(instrument(spatial=GaussianPSF(15e-6)))
    assert_allclose(grid.xy_m, broad.xy_m, atol=0, rtol=0)
    assert_allclose(grid.bounds_m, broad.bounds_m, atol=0, rtol=0)
    spec.slit_bounds_m[:] = 0
    assert grid.entrance_bounds_m is not None
    assert_allclose(grid.entrance_bounds_m, [-5e-6, 5e-6, -20e-6, 20e-6])


def test_monochromatic_line_placement_and_finite_slit_width():
    spec = instrument()
    grid = acquisition(spec, frames=1, slit_order=28)
    spectral = spectral_grid(401)
    # One-node triangular line integrates to the same unit photon count at either line.
    counts = []
    for wavenumber in (5000.0, 0.01 / 2.1e-6):
        index = int(np.argmin(abs(spectral.points - wavenumber)))

        def line(xy, directions, times, query_spectral, index=index):
            points = query_spectral.points
            values = np.zeros(len(points))
            values[index] = points[index] * 1e-8
            return np.broadcast_to(values, (*times.shape, len(points)))

        result = simulate_frames(
            replace(
                spec,
                detector=DetectorProperties(
                    pixel_bounds_m=(spec).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            grid,
            spectral,
            line,
            coverage,
        )
        counts.append(result.images.photoelectrons[0].sum(axis=0))
        # Independent paraxial achieved-band reference: finite slit CDF integral,
        # trapezoidal spectral measure and photon energy. Field-cosine variation
        # across this slit is below 1e-8 and bounds the reference approximation.
        accepted = []
        for point in spectral.points:
            shift = 200 * (0.01 / point - 2e-6)
            accepted.append(
                quad(
                    lambda x, shift=shift: (
                        ndtr((10e-6 - x - shift) / 4e-6) - ndtr((-10e-6 - x - shift) / 4e-6)
                    ),
                    -5e-6,
                    5e-6,
                    epsabs=1e-17,
                )[0]
            )
        intervals = np.diff(spectral.points)
        spectral_measure = np.r_[
            intervals[0] / 2, (intervals[:-1] + intervals[1:]) / 2, intervals[-1] / 2
        ]
        independent = np.array(accepted) * spectral_measure / spectral.points
        expected_mean = independent[index] * spectral.points[index] * 1e-8 / independent.sum()
        calibrated = calibrate_frames(
            result.images,
            result.response,
            detector=DetectorProperties(
                pixel_bounds_m=spec.pixel_bounds_m,
                pitch_m=None,
                quantum_efficiency=1,
                dark_electrons_per_s=0,
                read_noise_electrons=0,
                full_well_electrons=1e6,
                electrons_per_dn=1,
                bias_dn=0,
                nonlinearity=0,
                adc_bits=16,
            ),
        )
        cube = reconstruct_radiance_cube(
            calibrated,
            acquisition(spec, frames=1, slit_order=28),
            x_edges_m=np.array([-0.04, 0.04]),
            y_edges_m=np.array([-0.04, 0.04]),
            bands=[bands(result.response)[2]],
        )
        assert cube.values.item() == pytest.approx(expected_mean, rel=2e-8)
    assert np.argmax(counts[0]) == 2
    assert np.argmax(counts[1]) == 3
    # Independently integrate the finite slit and detector CDF at zero shift.
    fraction = (
        quad(lambda x: ndtr((10e-6 - x) / 4e-6) - ndtr((-10e-6 - x) / 4e-6), -5e-6, 5e-6)[0] / 10e-6
    )
    measured_fraction = counts[0][2] / counts[0].sum()
    assert measured_fraction == pytest.approx(fraction, rel=2e-8)
    narrow = instrument(slit_bounds_m=np.array([-0.1e-6, 0.1e-6, -20e-6, 20e-6]))
    narrow_grid = acquisition(narrow, frames=1)
    narrow_result = simulate_frames(
        replace(
            narrow,
            detector=DetectorProperties(
                pixel_bounds_m=(narrow).pixel_bounds_m,
                pitch_m=None,
                quantum_efficiency=1,
                dark_electrons_per_s=0,
                read_noise_electrons=0,
                full_well_electrons=1e6,
                electrons_per_dn=1,
                bias_dn=0,
                nonlinearity=0,
                adc_bits=16,
            ),
        ),
        narrow_grid,
        spectral,
        line,
        coverage,
    )
    assert narrow_result.images.photoelectrons.sum() < np.sum(counts[1]) / 40


def test_independent_photon_accounting_with_spatial_spectral_coupling():
    smile, keystone = 0.4, 25.0
    spec = instrument(smile=smile, keystone=keystone, transmission=0.8, throughput=0.7)
    grid = acquisition(spec, frames=1, slit_order=28)
    spectral = spectral_grid(17)
    qe = 0.6

    def scene(xy, directions, times, query_spectral):
        points = query_spectral.points
        return (1 + 5 * xy[..., 0, None] + 2 * xy[..., 1, None]) * points * 1e-8

    result = simulate_frames(
        replace(
            spec,
            detector=DetectorProperties(
                pixel_bounds_m=(spec).pixel_bounds_m,
                pitch_m=None,
                quantum_efficiency=qe,
                dark_electrons_per_s=0,
                read_noise_electrons=0,
                full_well_electrons=1e6,
                electrons_per_dn=1,
                bias_dn=0,
                nonlinearity=0,
                adc_bits=16,
            ),
        ),
        grid,
        spectral,
        scene,
        coverage,
    )
    # Scalar CDFs and nested adaptive integration supply an independent forward reference.
    # L(sigma)=sigma*1e-8 cancels the photon 1/sigma factor exactly.
    pixel = spec.pixel_bounds_m[0, 2]
    sigma = 4e-6
    integrals = []
    for point in spectral.points:
        delta = 0.01 / point - 2e-6

        def integrand_y(y, delta=delta):
            fy = ndtr((pixel[3] - y - keystone * delta) / sigma) - ndtr(
                (pixel[2] - y - keystone * delta) / sigma
            )

            def integrand_x(x):
                cx = x + 200 * delta + smile * y
                fx = ndtr((pixel[1] - cx) / sigma) - ndtr((pixel[0] - cx) / sigma)
                pupil_jacobian = 0.3**2 / (0.3**2 + x * x + y * y) ** 2
                # Time-mean X = speed*mid-exposure - height*x/focal_length.
                return fx * fy * pupil_jacobian * (1 + 5 * (0.001 - 1000 * x) + 2 * 1000 * y)

            return quad(integrand_x, -5e-6, 5e-6, epsabs=1e-18)[0]

        integrals.append(quad(integrand_y, -20e-6, 20e-6, epsabs=1e-18)[0])
    expected = (
        np.trapezoid(integrals, spectral.points)
        * 1e-8
        / (h * c * 100)
        * 0.004
        * 0.01
        * 0.8
        * 0.7
        * qe
    )
    assert result.images.photoelectrons[0, 0, 2] == pytest.approx(expected, rel=3e-11)
    response = result.response
    # Smile/keystone change achieved spectral responses across spatial detector rows.
    normalized = response.spectral_coefficients / response.spectral_coefficients.sum(
        axis=1, keepdims=True
    )
    assert np.sum(abs(normalized[2] - normalized[7])) > 0.05
    # The complete forward kernel couples the spectral line to its surface footprint.
    centroids = []
    for index in (4, 12):

        def line(xy, directions, times, query_spectral, index=index):
            points = query_spectral.points
            values = np.zeros(len(points))
            values[index] = 1e-4
            return (1 + 5 * xy[..., 0, None]) * values

        patterned = simulate_frames(
            replace(
                spec,
                detector=DetectorProperties(
                    pixel_bounds_m=(spec).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            grid,
            spectral,
            line,
            coverage,
        )
        flat = simulate_frames(
            replace(
                spec,
                detector=DetectorProperties(
                    pixel_bounds_m=(spec).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            grid,
            spectral,
            lambda x, d, t, p, index=index: np.broadcast_to(
                np.eye(1, len(p.points), index)[0] * 1e-4, (*t.shape, len(p.points))
            ),
            coverage,
        )
        centroids.append(
            (patterned.images.photoelectrons[0, 0, 2] / flat.images.photoelectrons[0, 0, 2] - 1) / 5
        )
    assert abs(centroids[0] - centroids[1]) > 0.001
    assert not sys._is_gil_enabled()


def test_complete_supplied_calibration_scan_cube_noise_and_missing_support():
    spec = instrument(smile=0.4, keystone=25)
    grid = acquisition(spec)
    spectral = spectral_grid()

    def spectrum(xy, directions, times, query_spectral):
        points = query_spectral.points
        return np.broadcast_to(points * 1e-8, (*times.shape, len(points)))

    detector = DetectorProperties(
        pixel_bounds_m=spec.pixel_bounds_m,
        pitch_m=None,
        quantum_efficiency=0.7,
        electrons_per_dn=0.01,
        bias_dn=2000,
        dark_electrons_per_s=2,
        read_noise_electrons=0.1,
        adc_bits=32,
        full_well_electrons=1e6,
        nonlinearity=0,
    )
    measured = simulate_frames(
        replace(spec, detector=detector),
        grid,
        spectral,
        spectrum,
        coverage,
        rng=np.random.default_rng(42),
        realisations=2,
    )
    repeated = simulate_frames(
        replace(spec, detector=detector),
        grid,
        spectral,
        spectrum,
        coverage,
        rng=np.random.default_rng(42),
        realisations=2,
        read_batch_size=1,
    )
    assert measured.images.dn is not None and measured.images.dn.shape == (2, 3, 2, 5)
    assert repeated.images.dn is not None
    assert np.array_equal(measured.images.dn, repeated.images.dn)
    supplied = frame_response(replace(spec, detector=detector), grid, spectral)
    assert_allclose(measured.response.spectral_coefficients, supplied.spectral_coefficients)
    requested = bands(supplied)
    expected = np.array([np.dot(band.weights, spectral.points * 1e-8) for band in requested])
    kwargs: dict[str, Any] = dict(
        x_edges_m=np.array([-0.04, 0.04]), y_edges_m=np.array([-0.04, 0.04]), bands=requested
    )
    calibrated = calibrate_frames(
        measured.images, supplied, detector=detector, use_expectation=True
    )
    # Estimated coordinates are independently recomputed; truth stays outside calibrated data.
    estimated = acquisition(spec)
    cube = reconstruct_radiance_cube(calibrated, estimated, **kwargs)
    assert cube.valid.all()
    # Summing thousands of positive quadrature contributions permits float64 roundoff.
    assert_allclose(cube.values[0, 0], expected, rtol=2e-14)
    assert_allclose([band.weights for band in cube.bands], [band.weights for band in requested])
    noisy = calibrate_frames(measured.images, supplied, detector=detector)
    noisy_cube = reconstruct_radiance_cube(noisy, estimated, **kwargs)
    assert np.max(abs(noisy_cube.values[0, 0] / expected - 1)) < 0.15
    # Excluding every scan read for one detector channel gives an invalid band plane.
    good = np.array([name != "r0-c2" for name in supplied.read_bands])
    missing = reconstruct_radiance_cube(calibrated, estimated, read_valid=good, **kwargs)
    assert not missing.valid[0, 0, 2] and np.isnan(missing.values[0, 0, 2])
    with pytest.raises(ValueError, match="incompatible"):
        reconstruct_radiance_cube(
            calibrated,
            estimated,
            x_edges_m=kwargs["x_edges_m"],
            y_edges_m=kwargs["y_edges_m"],
            bands=[spectral_band("r0-c2", spectral, np.ones(len(spectral.points)))],
        )
    # Supplied wavelength mapping and radiometric scale affect processing only.
    saved = measured.images.photoelectrons.copy()
    shifted = instrument(smile=0.4, keystone=25, reference_m=2.01e-6)
    shifted_response = frame_response(replace(shifted, detector=detector), estimated, spectral)
    shifted_calibrated = calibrate_frames(
        measured.images, shifted_response, use_expectation=True, detector=detector
    )
    assert np.max(abs(shifted_calibrated.values / calibrated.values - 1)) > 0.01
    with pytest.raises(ValueError, match="incompatible"):
        reconstruct_radiance_cube(shifted_calibrated, estimated, **kwargs)
    scaled = replace(supplied, spectral_coefficients=supplied.spectral_coefficients * 2)
    assert_allclose(
        calibrate_frames(measured.images, scaled, use_expectation=True, detector=detector).values,
        calibrated.values / 2,
    )
    assert_allclose(measured.images.photoelectrons, saved, atol=0, rtol=0)
    assert measured.images.full_well_saturated is not None
    flags = measured.images.full_well_saturated.copy()
    flags[:, :, 0, 2] = True
    flagged_images = replace(measured.images, full_well_saturated=flags)
    with pytest.raises(ValueError, match="clipped"):
        calibrate_frames(flagged_images, supplied, detector=detector)
    masked = calibrate_frames(flagged_images, supplied, detector=detector, mask_flagged=True)
    flagged_cube = reconstruct_radiance_cube(masked, estimated, **kwargs)
    assert not flagged_cube.valid[0, 0, 2]
    assert np.isnan(flagged_cube.values[0, 0, 2])


def test_moving_pattern_and_independent_estimated_geometry_shift():
    spec = instrument()
    grid = acquisition(spec)
    spectral = spectral_grid(33)

    def pattern(xy, directions, times, query_spectral):
        points = query_spectral.points
        return np.broadcast_to(
            (2e-5 * (1 + 5 * xy[..., 0]))[..., None], (*times.shape, len(points))
        )

    result = simulate_frames(
        replace(
            spec,
            detector=DetectorProperties(
                pixel_bounds_m=(spec).pixel_bounds_m,
                pitch_m=None,
                quantum_efficiency=1,
                dark_electrons_per_s=0,
                read_noise_electrons=0,
                full_well_electrons=1e6,
                electrons_per_dn=1,
                bias_dn=0,
                nonlinearity=0,
                adc_bits=16,
            ),
        ),
        grid,
        spectral,
        pattern,
        coverage,
    )
    calibrated = calibrate_frames(
        result.images,
        result.response,
        detector=DetectorProperties(
            pixel_bounds_m=spec.pixel_bounds_m,
            pitch_m=None,
            quantum_efficiency=1,
            dark_electrons_per_s=0,
            read_noise_electrons=0,
            full_well_electrons=1e6,
            electrons_per_dn=1,
            bias_dn=0,
            nonlinearity=0,
            adc_bits=16,
        ),
    )
    # For this spectrally flat gradient the spatial marginal is exact.
    centroid = np.sum(grid.xy_m[..., 0] * result.response.spatial_weights, axis=1)
    assert_allclose(calibrated.values[0], 2e-5 * (1 + 5 * centroid), rtol=2e-14)
    # Differencing two integrated means amplifies their float64 summation error.
    assert_allclose(calibrated.values[0, 10:20] - calibrated.values[0, :10], 1e-6, rtol=1e-13)
    edges = np.arange(-0.05, 0.101, 0.01)
    kwargs = dict(x_edges_m=edges, y_edges_m=np.array([-0.1, 0.1]), bands=bands(result.response))
    cube = reconstruct_radiance_cube(calibrated, grid, **kwargs)

    def biased(times):
        positions, velocities = trajectory(times)
        return positions + [0.03, 0, 0], velocities

    estimated = acquisition(spec, trajectory=biased)
    shifted = reconstruct_radiance_cube(calibrated, estimated, **kwargs)
    assert_allclose(shifted.values[:, 3:], cube.values[:, :-3], equal_nan=True, rtol=1e-14)
    assert (~cube.valid).any() and np.isnan(cube.values[~cube.valid]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dispersion=0),
        dict(slit_bounds_m=np.zeros(4)),
        dict(slit_bounds_m=np.ones(3)),
        dict(focal_length_m=0),
    ],
)
def test_invalid_physical_spectrometer(kwargs):
    with pytest.raises(ValueError):
        instrument(**kwargs)


def test_rejected_support_and_grid_mismatches():
    spec = instrument()
    grid = acquisition(spec, frames=1)
    spectral = spectral_grid(5)
    with pytest.raises(ValueError, match="certify"):
        simulate_frames(
            replace(
                spec,
                detector=DetectorProperties(
                    pixel_bounds_m=(spec).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            grid,
            spectral,
            uniform,
            lambda g, s: np.zeros(10, dtype=bool),
        )
    with pytest.raises(ValueError, match="radiance"):
        simulate_frames(
            replace(
                spec,
                detector=DetectorProperties(
                    pixel_bounds_m=(spec).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            grid,
            spectral,
            lambda x, d, t, p: np.zeros(1),
            coverage,
        )
    with pytest.raises(ValueError, match="slit"):
        frame_response(
            replace(
                instrument(slit_bounds_m=spec.slit_bounds_m * 2),
                detector=DetectorProperties(
                    pixel_bounds_m=(
                        instrument(slit_bounds_m=spec.slit_bounds_m * 2)
                    ).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            grid,
            spectral,
        )
    with pytest.raises(ValueError, match="nonempty"):
        spectrometer_acquisition_grid(
            spec,
            raster_read_schedule(
                np.empty((0, 2)), (spec).pixel_bounds_m.shape[:2], port=(spec).optics.port
            ),
            epoch=EPOCH,
            validity_s=(0, 1),
            trajectory=trajectory,
            platform_attitude=platform,
            instrument_attitude=identity,
            surface=surface,
            motion_bound_m_s=0.2,
            angular_bound_m=301,
        )
    with pytest.raises(ValueError, match="angular"):
        acquisition(spec, angular_bound_m=0)
    with pytest.raises(ValueError, match="surface"):
        acquisition(
            spec,
            surface=lambda o, d, t: (np.zeros((*t.shape, 2)), -d, np.zeros(t.shape, dtype=bool)),
        )
    with pytest.raises(ValueError, match="entrance"):
        sample_lines_of_sight(
            read_schedule(grid.rays.acquisition),
            pixel_bounds_m=grid.rays.acquisition.pixel_bounds_m,
            fov_bounds_uv=grid.rays.acquisition.direction_bounds_uv,
            epoch=EPOCH,
            validity_s=(0, 1),
            trajectory=trajectory,
            platform_attitude=platform,
            instrument_attitude=identity,
            mount_offset_p_m=np.zeros(3),
            focal_length_m=0.3,
            entrance_bounds_m=np.zeros((10, 4)),
        )
    with pytest.raises(ValueError, match="enclose"):
        sample_lines_of_sight(
            read_schedule(grid.rays.acquisition),
            pixel_bounds_m=grid.rays.acquisition.pixel_bounds_m,
            fov_bounds_uv=grid.rays.acquisition.direction_bounds_uv / 2,
            epoch=EPOCH,
            validity_s=(0, 1),
            trajectory=trajectory,
            platform_attitude=platform,
            instrument_attitude=identity,
            mount_offset_p_m=np.zeros(3),
            focal_length_m=0.3,
            entrance_bounds_m=np.broadcast_to(spec.slit_bounds_m, (10, 4)),
        )
    with pytest.raises(ValueError):
        focal_plane_direction_bounds(np.zeros((1, 4)), 0.3)


def test_slit_time_and_spectral_refinement():
    spec = instrument(smile=0.4, keystone=25)

    def scene(xy, directions, times, query_spectral):
        points = query_spectral.points
        return (
            2e-5
            * (1 + 1e9 * times[..., None] ** 4)
            * np.exp((points - 5000) / 1000)
            * (1 + (xy[..., 1, None] * 30) ** 2)
        )

    def run(t, a, k):
        return simulate_frames(
            replace(
                spec,
                detector=DetectorProperties(
                    pixel_bounds_m=(spec).pixel_bounds_m,
                    pitch_m=None,
                    quantum_efficiency=1,
                    dark_electrons_per_s=0,
                    read_noise_electrons=0,
                    full_well_electrons=1e6,
                    electrons_per_dn=1,
                    bias_dn=0,
                    nonlinearity=0,
                    adc_bits=16,
                ),
            ),
            acquisition(spec, frames=1, time_order=t, slit_order=a),
            spectral_grid(k),
            scene,
            coverage,
        ).images.photoelectrons[0, 0, 2]

    reference = run(4, 32, 257)
    for low, high in (
        (run(1, 32, 257), run(3, 32, 257)),
        (run(4, 4, 257), run(4, 20, 257)),
        (run(4, 32, 9), run(4, 32, 129)),
    ):
        assert abs(high / reference - 1) < abs(low / reference - 1)
        assert abs(high / reference - 1) < 1e-4
