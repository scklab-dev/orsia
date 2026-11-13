"""Independent fixed-filter photon, projection, calibration and registration checks."""

import sys
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_geometry import WGS84SurfaceProjection
from orsia_geometry.constants import WGS84_SEMIAXES_M
from orsia_instruments import (
    CalibratedFrames,
    DetectorProperties,
    FocalPlaneImager,
    GaussianPSF,
    ImagerSizing,
    ImagingOptics,
    airy_psf,
    calibrate_frames,
    camera_acquisition_grid,
    filter_camera,
    filter_strip,
    frame_exposure_bounds,
    frame_response,
    gaussian_filter,
    ground_integration_bounds,
    raster_read_schedule,
    reconstruct_radiance_cube,
    regular_pixel_bounds,
    select_raster_reads,
    simulate_frames,
    sized_focal_length,
    wgs84_surface_projection_from_camera,
)
from orsia_radiative import SpectralBand, SpectralDomain, spectral_band, wavenumber_grid
from scipy.constants import c, h
from scipy.integrate import quad
from scipy.special import j0, j1

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


def connected_grid(points):
    return wavenumber_grid(
        points, domain=SpectralDomain("wavenumber_cm1", ((float(points[0]), float(points[-1])),))
    )


def identity(times):
    return np.broadcast_to(np.eye(3), (len(times), 3, 3))


def platform(times):
    return np.broadcast_to(np.diag([-1.0, 1.0, -1.0]), (len(times), 3, 3))


def trajectory(times):
    positions = np.column_stack((times * 0, times * 0.2, times * 0 + 300))
    return positions, np.broadcast_to([0.0, 0.2, 0.0], positions.shape)


def surface(origins, directions, times):
    distance = -origins[..., 2] / directions[..., 2]
    points = origins + distance[..., None] * directions
    return points[..., :2], -directions, distance > 0


def coverage(grid, spectral):
    return np.ones(len(grid.read_bands), dtype=bool)


def uniform(xy, directions, times, query_spectral):
    points = query_spectral.points
    return np.full((*times.shape, len(points)), 2e-5)


@pytest.mark.parametrize("batch_size", [1, 5, 100])
def test_flat_local_query_order_and_response_contraction(batch_size):
    from orsia_instruments.photodetection.frames import frame_response_coefficients

    cam = camera(shape=(2, 2))
    grid = acquisition(cam, angular_order=2, frames=2)
    spectral = connected_grid(np.linspace(4000, 5000, 5))
    observed = []

    def radiance(xy, view, times, query_spectral):
        points = query_spectral.points
        assert xy.shape == (len(times), 2) and view.shape == (len(times), 3)
        assert times.ndim == points.ndim == 1
        observed.append((xy.copy(), view.copy(), times.copy()))
        return (2 + xy[:, 0, None] + times[:, None] + view[:, 2, None]) * points * 1e-9

    expected_values = (
        (
            2
            + grid.xy_m[..., 0, None]
            + grid.rays.acquisition.times_s[..., None]
            + grid.view_local[..., 2, None]
        )
        * spectral.points
        * 1e-9
    )
    expected = np.zeros(len(grid.read_bands))
    for selected, spectral_slice, coefficients in frame_response_coefficients(
        cam, grid, spectral, batch_size=3
    ):
        expected[selected] += np.einsum(
            "nqk,nqk->n", coefficients, expected_values[selected, :, spectral_slice]
        )
    result = simulate_frames(cam, grid, spectral, radiance, coverage, read_batch_size=batch_size)
    assert_allclose(result.images.photoelectrons.ravel(), expected, rtol=2e-15)
    for index, source in enumerate((grid.xy_m, grid.view_local, grid.rays.acquisition.times_s)):
        actual = np.concatenate([call[index] for call in observed])
        np.testing.assert_array_equal(actual, source.reshape(actual.shape))
    np.testing.assert_array_equal(grid.view_local, -grid.rays.directions_gcrs)


@pytest.mark.parametrize("kind", ["shape", "nan", "norm"])
def test_surface_and_grid_reject_invalid_local_views(kind):
    cam = camera(shape=(1, 1))

    def malformed(origins, directions, times):
        xy, view, hits = surface(origins, directions, times)
        if kind == "shape":
            view = view[..., :2]
        elif kind == "nan":
            view[:] = np.nan
        else:
            view *= 2
        return xy, view, hits

    with pytest.raises(ValueError):
        acquisition(cam, surface_callback=malformed)
    grid = acquisition(cam, angular_order=2, frames=1)
    bad = grid.view_local.copy()
    if kind == "shape":
        bad = bad[..., :2]
    elif kind == "nan":
        bad[:] = np.nan
    else:
        bad *= 2
    with pytest.raises(ValueError):
        replace(grid, view_local=bad)
    with pytest.raises(ValueError, match="coordinates"):
        replace(grid, xy_m=np.full_like(grid.xy_m, np.nan))


@pytest.mark.parametrize("kind", ["old_axes", "nan", "negative"])
def test_flat_radiance_output_validation(kind):
    cam = camera(shape=(1, 1))
    grid = acquisition(cam, frames=1, angular_order=2)
    spectral = connected_grid(np.linspace(4000, 5000, 5))

    def invalid(xy, view, times, query_spectral):
        points = query_spectral.points
        if kind == "old_axes":
            return np.ones((1, len(times), len(points)))
        return np.full((len(times), len(points)), np.nan if kind == "nan" else -1.0)

    with pytest.raises(ValueError, match="radiance"):
        simulate_frames(cam, grid, spectral, invalid, coverage)


def camera(shape=(4, 2), *, sigma=4e-6, f_number=4, **kwargs):
    pixels = regular_pixel_bounds(shape, pitch_m=20e-6)
    strips = [
        filter_strip(
            i, i + 1, band=f"b{i}", transmission=lambda w, i=i: np.full_like(w, 0.4 + i * 0.1)
        )
        for i in range(shape[0])
    ]
    return filter_camera(
        ideal_detector(pixels),
        strips,
        focal_length_m=0.3,
        f_number=f_number,
        spatial=GaussianPSF(sigma),
        **kwargs,
    )


def acquisition(
    cam,
    *,
    time_order=2,
    angular_order=10,
    frames=3,
    surface_callback=surface,
    trajectory_callback=trajectory,
    angular_bound=301,
    **kwargs,
):
    bounds = frame_exposure_bounds(start_s=0, frame_count=frames, frame_rate_hz=20, exposure_s=0.01)
    return camera_acquisition_grid(
        cam,
        raster_read_schedule(bounds, (cam).pixel_bounds_m.shape[:2], port=(cam).optics.port),
        epoch=EPOCH,
        validity_s=(0, 1),
        trajectory=trajectory_callback,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface_callback,
        motion_bound_m_s=0.2,
        angular_bound_m=angular_bound,
        time_order=time_order,
        angular_order=angular_order,
        **kwargs,
    )


def achieved_bands(response):
    bands = []
    for name in dict.fromkeys(response.read_bands):
        if name:
            i = response.read_bands.index(name)
            weights = response.spectral_coefficients[i]
            bands.append(SpectralBand(name, response.grid, weights / weights.sum()))
    return tuple(bands)


def test_sizing_layout_and_schedule():
    assert sized_focal_length(ImagerSizing(600000, 20, 10e-6)) == pytest.approx(0.30)
    bounds = frame_exposure_bounds(start_s=0, frame_count=41, frame_rate_hz=250, exposure_s=0.001)
    assert bounds.shape == (41, 2)
    assert bounds[-1, 1] == pytest.approx(0.161)
    cam = camera()
    assert cam.row_bands == ("b0", "b1", "b2", "b3")
    assert cam.optics.collecting_area_m2 == pytest.approx(np.pi * (0.3 / 4) ** 2 / 4)
    assert cam.support_radius_m == pytest.approx(4e-6 * np.sqrt(-2 * np.log(1e-6)))
    pixels = regular_pixel_bounds((3, 1), pitch_m=20e-6)
    strip = filter_strip(0, 2, band="a", transmission=np.ones_like)
    with pytest.raises(ValueError, match="covered"):
        filter_camera(
            ideal_detector(pixels),
            [strip],
            focal_length_m=0.3,
            f_number=4,
            spatial=GaussianPSF(4e-6),
        )
    dark = filter_camera(
        ideal_detector(pixels),
        [strip],
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(4e-6),
        allow_uncovered_rows=True,
    )
    assert dark.row_bands == ("a", "a", "")
    spectral = connected_grid(np.linspace(4000, 5000, 7))
    grid = acquisition(dark, frames=1)
    result = simulate_frames(
        replace(
            dark,
            detector=DetectorProperties(
                pixel_bounds_m=(dark).pixel_bounds_m,
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
        coverage,
    )
    assert np.all(result.images.photoelectrons[:, 2] == 0)
    calibrated = calibrate_frames(
        result.images,
        result.response,
        detector=DetectorProperties(
            pixel_bounds_m=dark.pixel_bounds_m,
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
    assert not calibrated.valid[0, -1]
    assert np.isnan(calibrated.values[0, -1])
    for strips in ([strip, strip], [filter_strip(0, 4, band="a", transmission=np.ones_like)]):
        with pytest.raises(ValueError, match="overlap|exceed"):
            filter_camera(
                ideal_detector(pixels),
                strips,
                focal_length_m=0.3,
                f_number=4,
                spatial=GaussianPSF(4e-6),
            )


@pytest.mark.parametrize("sizing", [(0, 1, 1), (1, -1, 1), (1, 1, np.inf)])
def test_bad_reference_sizing(sizing):
    with pytest.raises(ValueError):
        ImagerSizing(*sizing)


@pytest.mark.parametrize(
    "kwargs", [dict(frame_count=0), dict(frame_rate_hz=0), dict(exposure_s=1), dict(start_s=np.nan)]
)
def test_bad_schedule(kwargs):
    parameters = dict(start_s=0, frame_count=2, frame_rate_hz=10, exposure_s=0.01)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        frame_exposure_bounds(**parameters)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(f_number=0),
        dict(collecting_area_m2=1),
        dict(omitted_psf_mass=0),
        dict(pixel_order=0),
        dict(support_radius_m=1),
        dict(spatial=np.ones_like),
    ],
)
def test_bad_camera_settings(kwargs):
    pixels = regular_pixel_bounds((1, 1), pitch_m=20e-6)
    parameters = dict(focal_length_m=0.3, f_number=4, spatial=GaussianPSF(4e-6))
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        filter_camera(
            ideal_detector(pixels),
            [filter_strip(0, 1, band="u", transmission=np.ones_like)],
            **parameters,
        )


@pytest.mark.parametrize("reference_pixel", [None, (0, 1)])
@pytest.mark.parametrize("irregular", [False, True])
def test_camera_chart_reference_geometry_and_snapshot(monkeypatch, reference_pixel, irregular):
    # Only the dated frame boundary is substituted. The camera/ray/WGS84/chart
    # composition runs unchanged, against independent scalar intersection algebra.
    import orsia_geometry.frames as frames

    real_frame_rotations = frames.frame_rotations

    def dated_rotation(times):
        angle = 0.1 + 0.002 * times
        cosine, sine = np.cos(angle), np.sin(angle)
        rotations = np.zeros((*times.shape, 3, 3))
        rotations[..., 0, 0] = rotations[..., 1, 1] = cosine
        rotations[..., 0, 1], rotations[..., 1, 0] = -sine, sine
        rotations[..., 2, 2] = 1
        return rotations

    monkeypatch.setattr(frames, "frame_rotations", lambda times, **kwargs: dated_rotation(times))
    if irregular:
        pixels = np.array(
            [
                [[0.001, 0.002, -0.004, -0.002], [0.0024, 0.004, -0.003, -0.001]],
                [[0.0011, 0.0022, 0.0005, 0.0015], [0.0024, 0.0044, 0.001, 0.003]],
            ]
        )
        detector_centre = np.array([0.0027, -0.0005])
    else:
        pixels = regular_pixel_bounds((2, 2), pitch_m=20e-6) + [0.003, 0.003, -0.002, -0.002]
        detector_centre = np.array([0.003, -0.002])
    cam = filter_camera(
        ideal_detector(pixels),
        [filter_strip(0, 2, band="a", transmission=np.ones_like)],
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(4e-6),
    )
    centre = (
        detector_centre
        if reference_pixel is None
        else pixels[reference_pixel].reshape(2, 2).mean(axis=1)
    )
    sensor = np.array([*centre, 0.3])
    sensor /= np.linalg.norm(sensor)
    angle = 0.04
    platform_rotation = np.array(
        [[-np.sin(angle), 0, -np.cos(angle)], [np.cos(angle), 0, -np.sin(angle)], [0, -1, 0]]
    )
    angle = 0.025
    instrument_rotation = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    base_position = np.array([WGS84_SEMIAXES_M[0] + 7e5, 1500.0, 1e4])
    velocity = np.array([2.0, 7500.0, 20.0])
    offset = np.array([12.0, -34.0, 56.0])
    calls = []

    def orbit(times):
        calls.append(times.copy())
        return base_position + times[:, None] * velocity, np.broadcast_to(velocity, (len(times), 3))

    def attitude(times):
        calls.append(times.copy())
        return np.broadcast_to(platform_rotation, (len(times), 3, 3))

    def instrument(times):
        calls.append(times.copy())
        return np.broadcast_to(instrument_rotation, (len(times), 3, 3))

    reference_time = 0.6
    reference_origin = base_position + reference_time * velocity + platform_rotation @ offset
    reference_direction = platform_rotation @ instrument_rotation @ sensor
    dated = dated_rotation(np.array(reference_time))
    origin_itrf, direction_itrf = dated @ reference_origin, dated @ reference_direction
    axes = np.array(WGS84_SEMIAXES_M)
    a = np.sum((direction_itrf / axes) ** 2)
    b = np.sum(origin_itrf * direction_itrf / axes**2)
    c0 = np.sum((origin_itrf / axes) ** 2) - 1
    distance = (-b - np.sqrt(b**2 - a * c0)) / a
    intercept = origin_itrf + distance * direction_itrf
    up = intercept / axes**2
    up /= np.linalg.norm(up)
    east = np.array([-intercept[1], intercept[0], 0])
    east /= np.linalg.norm(east)
    chart_axes = np.array([east, np.cross(up, east)])
    projection = wgs84_surface_projection_from_camera(
        cam,
        trajectory=orbit,
        platform_attitude=attitude,
        instrument_attitude=instrument,
        epoch=EPOCH,
        validity_s=(0, 1),
        reference_time_s=reference_time,
        mount_offset_p_m=offset,
        reference_pixel=reference_pixel,
    )
    assert type(projection) is WGS84SurfaceProjection
    assert_allclose(projection.chart_origin_itrf_m, intercept, atol=5e-9, rtol=0)
    assert len(calls) == 3
    for times in calls:
        np.testing.assert_array_equal(times, [reference_time])
    np.testing.assert_array_equal(cam.pixel_bounds_m, pixels)
    np.testing.assert_array_equal(offset, [12, -34, 56])
    # Include repeated dates, different ray axes, a miss and a non-reference time.
    times = np.array([[reference_time, reference_time], [0.8, 0.8]])
    origins = np.broadcast_to(reference_origin, (2, 2, 3)).copy()
    directions = np.array([[reference_direction, -reference_direction]] * 2)
    snapshot_origins, snapshot_directions, snapshot_times = (
        origins.copy(),
        directions.copy(),
        times.copy(),
    )
    xy, view, hits = projection(origins, directions, times)
    np.testing.assert_array_equal(hits, [[True, False], [True, False]])
    assert np.isnan(xy[:, 1]).all()
    assert_allclose(xy[0, 0], 0, atol=5e-9, rtol=0)
    # Construct a known second ellipsoid point and its dated ray independently.
    second = intercept + 90 * east - 65 * chart_axes[1]
    second /= np.sqrt(np.sum((second / axes) ** 2))
    direction = second - dated_rotation(np.array(0.8)) @ reference_origin
    direction /= np.linalg.norm(direction)
    directions[1, 0] = dated_rotation(np.array(0.8)).T @ direction
    expected = chart_axes @ (second - intercept)
    xy, view, hits = projection(origins, directions, times)
    assert_allclose(xy[1, 0], expected, atol=5e-9, rtol=0)
    expected_view = -np.einsum(
        "...ij,...j->...i", dated_rotation(times), directions
    ) @ np.column_stack((chart_axes.T, up))
    assert_allclose(view, expected_view, atol=2e-15, rtol=0)
    assert xy.dtype == np.float64 and xy.flags.owndata and xy.flags.writeable
    assert hits.dtype == np.bool_ and hits.flags.owndata and hits.flags.writeable
    np.testing.assert_array_equal(origins, snapshot_origins)
    np.testing.assert_array_equal(times, snapshot_times)
    np.testing.assert_array_equal(directions[0], snapshot_directions[0])
    # Detector snapshots reject writes; mutable source changes preserve the chart.
    with pytest.raises(ValueError, match="read-only"):
        cam.pixel_bounds_m[:] += 0.01
    pixels[:] += 0.01
    offset[:] = 0
    base_position[:] = 0
    platform_rotation[:] = instrument_rotation[:] = 0
    after_xy, after_view, after_hits = projection(origins, directions, times)
    assert len(calls) == 3
    assert_allclose(after_xy, xy, atol=0, rtol=0)
    assert_allclose(after_view, view, atol=0, rtol=0)
    np.testing.assert_array_equal(after_hits, hits)
    # Restore the real time-validation boundary before checking support errors.
    monkeypatch.setattr(frames, "frame_rotations", real_frame_rotations)
    with pytest.raises(ValueError, match="validity"):
        projection(origins, directions, times + 1)


def test_camera_chart_default_centre_time_and_mount(monkeypatch):
    import orsia_geometry.frames as frames

    monkeypatch.setattr(
        frames,
        "frame_rotations",
        lambda times, **kwargs: np.broadcast_to(np.eye(3), (len(times), 3, 3)),
    )
    position = np.array([WGS84_SEMIAXES_M[0] + 7e5, 0, 0])
    rotation = np.array([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]])
    projection = wgs84_surface_projection_from_camera(
        camera(shape=(2, 2)),
        trajectory=lambda t: (np.broadcast_to(position, (len(t), 3)), np.zeros((len(t), 3))),
        platform_attitude=lambda t: np.broadcast_to(rotation, (len(t), 3, 3)),
        instrument_attitude=identity,
        epoch=EPOCH,
        validity_s=(0, 0),
    )
    xy, view, hit = projection(position, np.array([-1.0, 0, 0]), np.array(0.0))
    assert hit and xy.shape == (2,)
    assert_allclose(xy, [0, 0], atol=1e-10, rtol=0)


@pytest.mark.parametrize("pixel", [(-1, 0), (0, -1), (4, 0), (0, 2)])
def test_camera_chart_rejects_out_of_range_native_pixel_before_callbacks(pixel):
    def unused(times):
        raise AssertionError("invalid pixel must fail before callbacks or backend access")

    with pytest.raises(ValueError, match="reference_pixel"):
        wgs84_surface_projection_from_camera(
            camera(),
            trajectory=unused,
            platform_attitude=unused,
            instrument_attitude=unused,
            epoch=EPOCH,
            validity_s=(0, 1),
            reference_pixel=pixel,
        )


@pytest.mark.parametrize("time", [np.nan, np.inf, -0.1, 1.1])
def test_camera_chart_rejects_invalid_reference_time_before_callbacks(time):
    def unused(times):
        raise AssertionError("invalid time must fail before callbacks or backend access")

    with pytest.raises(ValueError, match="validity"):
        wgs84_surface_projection_from_camera(
            camera(),
            trajectory=unused,
            platform_attitude=unused,
            instrument_attitude=unused,
            epoch=EPOCH,
            validity_s=(0, 1),
            reference_time_s=time,
        )


def test_airy_normalization_encircled_energy_and_support():
    wave = np.array([1e-6, 2e-6])
    peak = airy_psf(np.zeros((1, 2, 2)), wave, np.zeros((1, 2)), 0, f_number=4)
    assert_allclose(peak[0], np.pi / (4 * (wave * 4) ** 2))
    radius = 20e-6
    integral = quad(
        lambda r: (
            float(airy_psf(np.array([[[r, 0.0]]]), wave[:1], np.zeros((1, 2)), 0, f_number=4)[0, 0])
            * 2
            * np.pi
            * r
        ),
        0,
        radius,
        epsabs=1e-12,
    )[0]
    z = np.pi * radius / (wave[0] * 4)
    assert integral == pytest.approx(1 - j0(z) ** 2 - j1(z) ** 2, abs=2e-13)
    pixels = regular_pixel_bounds((1, 1), pitch_m=20e-6)
    strips = [filter_strip(0, 1, band="airy", transmission=np.ones_like)]
    cam = filter_camera(
        ideal_detector(pixels),
        strips,
        focal_length_m=0.3,
        f_number=4,
        airy_wavelength_max_m=2e-6,
        omitted_psf_mass=0.1,
        pixel_order=12,
    )
    z = np.pi * cam.support_radius_m / (2e-6 * 4)
    assert j0(z) ** 2 + j1(z) ** 2 == pytest.approx(0.1, abs=1e-13)
    grid = acquisition(cam, frames=1, angular_order=20)
    result = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
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
        connected_grid(np.linspace(5000, 6000, 7)),
        uniform,
        coverage,
    )
    assert result.images.photoelectrons[0, 0, 0] > 0
    with pytest.raises(ValueError, match="maximum"):
        airy_psf(np.zeros((1, 2, 2)), wave, np.zeros((1, 2)), 0, f_number=4, wavelength_max_m=1e-6)
    with pytest.raises(ValueError):
        airy_psf(np.zeros((1, 2, 2)), wave, np.zeros((1, 2)), 0, f_number=0)
    with pytest.raises(ValueError):
        filter_camera(
            ideal_detector(pixels),
            strips,
            focal_length_m=0.3,
            f_number=4,
            spatial=lambda o, w, i, e: np.ones(o.shape[:2]),
        )
    with pytest.raises(ValueError):
        filter_camera(
            ideal_detector(pixels),
            strips,
            focal_length_m=0.3,
            f_number=4,
            airy_wavelength_max_m=2e-6,
            spatial=GaussianPSF(1e-6),
        )
    custom = filter_camera(
        ideal_detector(pixels),
        strips,
        focal_length_m=0.3,
        f_number=4,
        support_radius_m=3e-6,
        spatial=lambda o, w, i, e: np.broadcast_to(
            (np.linalg.norm(o, axis=-1) < 3e-6) / (np.pi * (3e-6) ** 2), o.shape[:2]
        ),
    )
    assert custom.omitted_psf_mass is None
    assert (
        simulate_frames(
            replace(
                custom,
                detector=DetectorProperties(
                    pixel_bounds_m=(custom).pixel_bounds_m,
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
            acquisition(custom, frames=1),
            connected_grid(np.linspace(5000, 6000, 5)),
            uniform,
            coverage,
        ).images.photoelectrons.min()
        > 0
    )


def test_surface_projection_native_mapping_and_continuous_bounds():
    cam = camera()
    grid = acquisition(cam)
    rays = grid.rays
    expected = (
        rays.origins_gcrs_m[..., :2]
        - rays.origins_gcrs_m[..., 2, None]
        * rays.directions_gcrs[..., :2]
        / rays.directions_gcrs[..., 2, None]
    )
    assert_allclose(grid.xy_m, expected, atol=1e-15)
    assert grid.image_shape == (3, 4, 2)
    assert np.array_equal(grid.native_indices[-1], [2, 3, 1])
    assert grid.rays.acquisition.pixel_ids[0] == grid.rays.acquisition.pixel_ids[8]
    bounds = ground_integration_bounds(grid)
    assert np.all(grid.xy_m >= bounds[[0, 2]])
    assert np.all(grid.xy_m <= bounds[[1, 3]])
    # Dense analytic sweep checks the independently bounded continuous footprint.
    for i in (0, 7, 23):
        uv_bounds = rays.acquisition.direction_bounds_uv[i]
        u = np.linspace(uv_bounds[0], uv_bounds[1], 17)
        v = np.linspace(uv_bounds[2], uv_bounds[3], 19)
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
        for t in np.linspace(
            rays.acquisition.exposure_bounds_s[i, 0], rays.acquisition.exposure_bounds_s[i, 1], 13
        ):
            sensor = np.column_stack((uv, np.sqrt(1 - (uv**2).sum(axis=1))))
            direction = sensor @ platform(np.array([t]))[0].T
            origin = np.broadcast_to(trajectory(np.array([t]))[0][0], direction.shape)
            xy, _, _ = surface(origin, direction, np.full(len(uv), t))
            assert np.all(xy >= grid.bounds_m[i, [0, 2]])
            assert np.all(xy <= grid.bounds_m[i, [1, 3]])
    with pytest.raises(ValueError):
        acquisition(
            cam,
            surface_callback=lambda o, d, t: (
                np.zeros((*t.shape, 2)),
                -d,
                np.zeros(t.shape, dtype=bool),
            ),
        )
    with pytest.raises(ValueError):
        acquisition(
            cam,
            surface_callback=lambda o, d, t: (
                np.full((*t.shape, 2), np.nan),
                -d,
                np.ones(t.shape, dtype=bool),
            ),
        )


def test_independent_photon_counting_and_qe_area_time_scaling():
    cam = camera(shape=(1, 1), omitted_psf_mass=1e-9)
    grid = acquisition(cam, frames=1, angular_order=26)
    points = np.array([4000.0, 4090, 4300, 4750, 5000])
    spectral = connected_grid(points)

    def linear_spectrum(xy, direction, times, query_spectral):
        wavenumbers = query_spectral.points
        return np.broadcast_to(wavenumbers * 1e-8, (*times.shape, len(wavenumbers)))

    detector = DetectorProperties(
        pixel_bounds_m=cam.pixel_bounds_m,
        pitch_m=None,
        quantum_efficiency=0.7,
        dark_electrons_per_s=0,
        read_noise_electrons=0,
        full_well_electrons=1e6,
        electrons_per_dn=1,
        bias_dn=0,
        nonlinearity=0,
        adc_bits=16,
    )
    result = simulate_frames(
        replace(cam, detector=detector),
        grid,
        spectral,
        linear_spectrum,
        coverage,
        read_batch_size=1,
    )
    # L(sigma)=sigma*1e-8 makes the photon integrand exactly constant.
    expected = (
        1e-8
        * (points[-1] - points[0])
        / (h * c * 100)
        * cam.optics.collecting_area_m2
        * 0.4
        * 0.7
        * 0.01
        * (20e-6 / 0.3) ** 2
    )
    print("Independent photon relative error:", result.images.photoelectrons.item() / expected - 1)
    assert result.images.photoelectrons.item() == pytest.approx(expected, rel=2e-7)
    calibrated = calibrate_frames(result.images, result.response, detector=detector)
    assert calibrated.values.item() == pytest.approx(
        np.dot(result.response.spectral_coefficients[0], points * 1e-8)
        / result.response.spectral_coefficients.sum(),
        rel=1e-14,
    )
    energy = spectral_band("b0", spectral, np.ones_like(points))
    print(
        "Photon mean / energy mean:",
        calibrated.values.item(),
        np.dot(points * 1e-8, energy.weights),
    )
    assert abs(calibrated.values.item() / np.dot(points * 1e-8, energy.weights) - 1) > 0.003
    half_qe = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
                pitch_m=None,
                quantum_efficiency=0.35,
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
        linear_spectrum,
        coverage,
    )
    assert_allclose(half_qe.images.photoelectrons, result.images.photoelectrons / 2, rtol=1e-14)
    small_area = camera(shape=(1, 1), f_number=4 * np.sqrt(2), omitted_psf_mass=1e-9)
    smaller = simulate_frames(
        replace(small_area, detector=detector), grid, spectral, linear_spectrum, coverage
    )
    assert_allclose(smaller.images.photoelectrons, result.images.photoelectrons / 2, rtol=1e-14)
    shorter_grid = replace(
        grid,
        schedule=replace(
            grid.schedule,
            reads=replace(
                grid.schedule.reads, exposure_bounds_s=grid.schedule.reads.exposure_bounds_s / 2
            ),
        ),
        rays=replace(
            grid.rays,
            acquisition=replace(
                grid.rays.acquisition,
                reads=replace(
                    grid.rays.acquisition.reads,
                    exposure_bounds_s=grid.rays.acquisition.exposure_bounds_s / 2,
                ),
                times_s=grid.rays.acquisition.times_s / 2,
                weights_sr_s=grid.rays.acquisition.weights_sr_s / 2,
            ),
            support_times_s=grid.rays.support_times_s / 2,
        ),
    )
    shorter = simulate_frames(
        replace(cam, detector=detector), shorter_grid, spectral, linear_spectrum, coverage
    )
    assert_allclose(shorter.images.photoelectrons, result.images.photoelectrons / 2, rtol=1e-14)


def test_off_axis_pupil_projection_recovers_cosine_fourth_irradiance():
    on_axis = camera(shape=(1, 1), omitted_psf_mass=1e-9)
    offset = 0.03
    off_axis = filter_camera(
        ideal_detector(regular_pixel_bounds((1, 1), pitch_m=20e-6, centre_xy_m=(offset, 0))),
        on_axis.strips,
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(4e-6),
        omitted_psf_mass=1e-9,
    )
    spectral = connected_grid(np.linspace(4000, 5000, 5))
    counts = []
    for cam in (on_axis, off_axis):
        grid = acquisition(cam, frames=1, angular_order=26, angular_bound=1000)
        counts.append(
            simulate_frames(
                replace(
                    cam,
                    detector=DetectorProperties(
                        pixel_bounds_m=(cam).pixel_bounds_m,
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
                coverage,
            ).images.photoelectrons.item()
        )
    expected_ratio = (1 + (offset / 0.3) ** 2) ** -2
    error = counts[1] / counts[0] / expected_ratio - 1
    print("Off-axis cosine-fourth relative error:", error)
    assert abs(error) < 1e-7


def test_four_strip_calibration_cube_noise_and_missing_coverage():
    cam = camera()
    grid = acquisition(cam)
    spectral = connected_grid(np.linspace(4000, 5000, 9))
    detector = DetectorProperties(
        pixel_bounds_m=cam.pixel_bounds_m,
        pitch_m=None,
        quantum_efficiency=0.7,
        bias_dn=2000,
        electrons_per_dn=0.01,
        dark_electrons_per_s=4,
        read_noise_electrons=0.1,
        adc_bits=32,
        full_well_electrons=1e6,
        nonlinearity=0,
    )
    result = simulate_frames(
        replace(cam, detector=detector),
        grid,
        spectral,
        uniform,
        coverage,
        rng=np.random.default_rng(42),
        realisations=3,
    )
    assert result.images.dn is not None
    assert result.images.dn.shape == (3, 3, 4, 2)
    repeated = simulate_frames(
        replace(cam, detector=detector),
        grid,
        spectral,
        uniform,
        coverage,
        rng=np.random.default_rng(42),
        realisations=3,
        read_batch_size=1,
    )
    assert repeated.images.dn is not None
    assert np.array_equal(result.images.dn, repeated.images.dn)
    assert_allclose(
        result.response.spectral_coefficients,
        frame_response(replace(cam, detector=detector), grid, spectral).spectral_coefficients,
    )
    expected = calibrate_frames(
        result.images, result.response, use_expectation=True, detector=detector
    )
    assert_allclose(expected.values, 2e-5, rtol=2e-15)
    calibrated = calibrate_frames(result.images, result.response, detector=detector)
    assert calibrated.values.shape == (3, 24)
    cube = reconstruct_radiance_cube(
        expected,
        grid,
        x_edges_m=np.array([-1, 1.0]),
        y_edges_m=np.linspace(-0.08, 0.12, 21),
        bands=achieved_bands(result.response),
    )
    assert cube.values.shape == (20, 1, 4)
    assert_allclose(cube.values[cube.valid], 2e-5, rtol=5e-15)
    assert np.isnan(cube.values[~cube.valid]).all()
    assert np.any(cube.valid.any(axis=2) & ~cube.valid.all(axis=2))
    assert not cube.valid[0].any()
    print("Four-strip cube valid/invalid:", int(cube.valid.sum()), int((~cube.valid).sum()))
    noisy_cube = reconstruct_radiance_cube(
        calibrated,
        grid,
        x_edges_m=np.array([-1, 1.0]),
        y_edges_m=np.array([-1, 1.0]),
        bands=achieved_bands(result.response),
    )
    assert np.max(np.abs(noisy_cube.values / 2e-5 - 1)) < 0.1
    changed_response = replace(
        result.response, spectral_coefficients=result.response.spectral_coefficients * 2
    )
    assert_allclose(
        calibrate_frames(
            result.images, changed_response, use_expectation=True, detector=detector
        ).values,
        1e-5,
    )
    # A scalar gain cannot create a different requested energy response.
    incompatible = [spectral_band("b0", spectral, np.ones(9))]
    with pytest.raises(ValueError, match="incompatible"):
        reconstruct_radiance_cube(
            expected,
            grid,
            x_edges_m=np.array([-1, 1.0]),
            y_edges_m=np.array([-1, 1.0]),
            bands=incompatible,
        )


def test_estimated_geometry_changes_registration_without_changing_truth():
    cam = camera(shape=(1, 6))
    grid = acquisition(cam, frames=1, angular_order=14)
    spectral = connected_grid(np.linspace(4000, 5000, 9))

    def pattern(xy, direction, times, query_spectral):
        points = query_spectral.points
        return np.broadcast_to(
            (2e-5 * (1 + xy[..., 0] * 5))[..., None], (*times.shape, len(points))
        )

    measurements = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
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
    original = measurements.images.photoelectrons.copy()
    calibrated = calibrate_frames(
        measurements.images,
        measurements.response,
        detector=DetectorProperties(
            pixel_bounds_m=cam.pixel_bounds_m,
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
    edges = np.arange(-0.2, 0.21, 0.01)
    kwargs: dict[str, Any] = dict(
        x_edges_m=edges, y_edges_m=np.array([-1, 1.0]), bands=achieved_bands(measurements.response)
    )
    cube = reconstruct_radiance_cube(calibrated, grid, **kwargs)

    def biased_trajectory(times):
        position, velocity = trajectory(times)
        return position + np.array([0.03, 0, 0]), velocity

    shifted = acquisition(cam, frames=1, angular_order=14, trajectory_callback=biased_trajectory)
    shifted_cube = reconstruct_radiance_cube(calibrated, shifted, **kwargs)
    assert_allclose(shifted_cube.values[:, 3:], cube.values[:, :-3], equal_nan=True, rtol=1e-14)
    print("Estimated-geometry registration shift: 0.03 m, exactly 3 output cells")
    assert np.array_equal(measurements.images.photoelectrons, original)
    # Calibrated native linear-gradient means equal weighted projected coordinates.
    centroid = np.sum(grid.xy_m[..., 0] * measurements.response.spatial_weights, axis=1)
    assert_allclose(calibrated.values[0], 2e-5 * (1 + centroid * 5), rtol=2e-15)
    pixel_x = cam.pixel_bounds_m[..., :2].mean(axis=-1).ravel()
    analytic_centroid = -300 * pixel_x / 0.3
    assert_allclose(calibrated.values[0], 2e-5 * (1 + analytic_centroid * 5), rtol=2e-8)


def test_direction_dependent_scene_support_and_validation():
    cam = camera(shape=(1, 1))
    grid = acquisition(cam, frames=1)
    spectral = connected_grid(np.linspace(4000, 5000, 5))
    calls = []

    def scene(xy, direction, times, query_spectral):
        points = query_spectral.points
        calls.append(xy.copy())
        return np.broadcast_to((2e-5 * direction[..., 2])[..., None], (*times.shape, len(points)))

    with pytest.raises(ValueError, match="certify"):
        simulate_frames(
            replace(
                cam,
                detector=DetectorProperties(
                    pixel_bounds_m=(cam).pixel_bounds_m,
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
            scene,
            lambda g, s: np.zeros(1, dtype=bool),
        )
    assert not calls
    result = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
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
        scene,
        coverage,
    )
    assert_allclose(calls[0], grid.xy_m.reshape(-1, 2))
    plain = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
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
        coverage,
    )
    assert result.images.photoelectrons.item() < plain.images.photoelectrons.item()
    for bad in (
        lambda x, d, t, p: np.zeros(1),
        lambda x, d, t, p: -np.ones((*t.shape, len(p.points))),
    ):
        with pytest.raises(ValueError, match="radiance"):
            simulate_frames(
                replace(
                    cam,
                    detector=DetectorProperties(
                        pixel_bounds_m=(cam).pixel_bounds_m,
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
                bad,
                coverage,
            )
    with pytest.raises(ValueError):
        frame_response(
            replace(
                cam,
                detector=DetectorProperties(
                    pixel_bounds_m=(cam).pixel_bounds_m,
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
            read_batch_size=0,
        )
    with pytest.raises(ValueError):
        simulate_frames(
            replace(
                replace(cam, detector=replace(cam.detector, pixel_bounds_m=cam.pixel_bounds_m * 2)),
                detector=DetectorProperties(
                    pixel_bounds_m=(
                        replace(
                            cam,
                            detector=replace(cam.detector, pixel_bounds_m=cam.pixel_bounds_m * 2),
                        )
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
            uniform,
            coverage,
        )


def test_detector_flags_nonlinearity_and_quality_masks():
    cam = camera(shape=(1, 2))
    grid = acquisition(cam, frames=1)
    spectral = connected_grid(np.linspace(4000, 5000, 5))
    result = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
                pitch_m=None,
                full_well_electrons=0.1,
                quantum_efficiency=1,
                dark_electrons_per_s=0,
                read_noise_electrons=0,
                electrons_per_dn=1,
                bias_dn=0,
                nonlinearity=0,
                adc_bits=16,
            ),
        ),
        grid,
        spectral,
        uniform,
        coverage,
        rng=np.random.default_rng(2),
    )
    assert result.images.full_well_saturated is not None
    assert result.images.full_well_saturated.any()
    with pytest.raises(ValueError, match="clipped"):
        calibrate_frames(
            result.images,
            result.response,
            detector=DetectorProperties(
                pixel_bounds_m=cam.pixel_bounds_m,
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
    masked = calibrate_frames(
        result.images,
        result.response,
        mask_flagged=True,
        detector=DetectorProperties(
            pixel_bounds_m=cam.pixel_bounds_m,
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
    assert not masked.valid.any()
    ideal = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
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
        coverage,
        rng=np.random.default_rng(3),
    )
    with pytest.raises(ValueError, match="nonlinear"):
        calibrate_frames(
            ideal.images,
            ideal.response,
            detector=DetectorProperties(
                pixel_bounds_m=cam.pixel_bounds_m,
                pitch_m=None,
                nonlinearity=0.1,
                quantum_efficiency=1,
                dark_electrons_per_s=0,
                read_noise_electrons=0,
                full_well_electrons=1e6,
                electrons_per_dn=1,
                bias_dn=0,
                adc_bits=16,
            ),
        )
    with pytest.raises(ValueError, match="flag"):
        calibrate_frames(
            replace(ideal.images, adc_clipped=None),
            ideal.response,
            detector=DetectorProperties(
                pixel_bounds_m=cam.pixel_bounds_m,
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
    expected = calibrate_frames(
        ideal.images,
        ideal.response,
        use_expectation=True,
        detector=DetectorProperties(
            pixel_bounds_m=cam.pixel_bounds_m,
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
    kwargs: dict[str, Any] = dict(
        x_edges_m=np.array([-1, 1.0]),
        y_edges_m=np.array([-1, 1.0]),
        bands=achieved_bands(ideal.response),
    )
    empty = reconstruct_radiance_cube(expected, grid, read_valid=np.zeros(2, dtype=bool), **kwargs)
    assert not empty.valid.any() and np.isnan(empty.values).all()
    changes: tuple[dict[str, Any], ...] = (
        dict(read_valid=np.ones(3, dtype=bool)),
        dict(realisation=1),
        dict(response_tolerance=-1),
        dict(minimum_coverage=-1),
        dict(bands=[]),
        dict(x_edges_m=np.array([1.0, 0])),
    )
    for bad_kwargs in changes:
        with pytest.raises(ValueError):
            reconstruct_radiance_cube(expected, grid, **(kwargs | bad_kwargs))
    # Spectral grouping labels are independent of geometric read correspondence.
    relabelled = reconstruct_radiance_cube(
        expected, replace(grid, read_bands=("geometry-label",) * 2), **kwargs
    )
    assert_allclose(relabelled.values, reconstruct_radiance_cube(expected, grid, **kwargs).values)
    with pytest.raises(ValueError, match="timing"):
        calibrate_frames(
            replace(
                ideal.images,
                schedule=replace(
                    ideal.images.schedule,
                    reads=replace(
                        ideal.images.schedule.reads,
                        exposure_bounds_s=ideal.images.schedule.reads.exposure_bounds_s + 1,
                    ),
                ),
            ),
            ideal.response,
            detector=DetectorProperties(
                pixel_bounds_m=cam.pixel_bounds_m,
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
    with pytest.raises(ValueError):
        CalibratedFrames(ideal.response, np.zeros((1, 2)), np.zeros((1, 2), dtype=bool))


def test_time_angular_and_spectral_convergence():
    cam = camera(shape=(1, 1))

    def scene(xy, direction, times, query_spectral):
        points = query_spectral.points
        return (
            (1 + 2e7 * times[..., None] ** 4)
            * (1 + xy[..., 0, None] ** 2 * 100)
            * np.exp((points[None, :] - 4000) / 1000)
            * 1e-5
        )

    def run(t, a, k):
        grid = acquisition(cam, frames=1, time_order=t, angular_order=a)
        return simulate_frames(
            replace(
                cam,
                detector=DetectorProperties(
                    pixel_bounds_m=(cam).pixel_bounds_m,
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
            connected_grid(np.linspace(4000, 5000, k)),
            scene,
            coverage,
        ).images.photoelectrons.item()

    reference = run(5, 30, 513)
    low, medium, high = run(1, 4, 9), run(2, 10, 33), run(3, 22, 257)
    assert abs(high / reference - 1) < 4e-6
    assert abs(high - reference) < abs(medium - reference) < abs(low - reference)

    print(
        "Joint refinement relative errors:",
        [value / reference - 1 for value in (low, medium, high)],
    )
    temporal = [run(t, 30, 513) for t in (1, 2, 3)]
    spatial = [run(5, a, 513) for a in (4, 10, 22)]
    spectral = [run(5, 30, k) for k in (9, 33, 257)]
    for name, values in (("time", temporal), ("angular", spatial), ("spectral", spectral)):
        errors = np.abs(np.array(values) / reference - 1)
        print(name, "refinement relative errors:", errors.tolist())
        assert errors[-1] < errors[0]


def test_psf_tail_refinement_and_selected_response_contracts():
    spectral = connected_grid(np.linspace(4000, 5000, 5))
    counts = []
    for budget in (0.1, 0.01, 1e-8):
        cam = camera(shape=(1, 1), omitted_psf_mass=budget)
        grid = acquisition(cam, frames=1, angular_order=32)
        counts.append(
            simulate_frames(
                replace(
                    cam,
                    detector=DetectorProperties(
                        pixel_bounds_m=(cam).pixel_bounds_m,
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
                coverage,
            ).images.photoelectrons.item()
        )
    errors = np.abs(np.array(counts[:-1]) / counts[-1] - 1)
    print("PSF tail-budget refinement relative errors:", errors.tolist())
    assert errors[1] < errors[0] < 0.1
    cam = camera(shape=(1, 2))
    grid = acquisition(cam, frames=1)
    result = simulate_frames(
        replace(
            cam,
            detector=DetectorProperties(
                pixel_bounds_m=(cam).pixel_bounds_m,
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
        coverage,
        rng=np.random.default_rng(8),
    )
    flags = np.array([[[[True, False]]]])
    images = replace(result.images, full_well_saturated=flags)
    calibrated = calibrate_frames(
        images,
        result.response,
        mask_flagged=True,
        detector=DetectorProperties(
            pixel_bounds_m=cam.pixel_bounds_m,
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
    assert np.array_equal(calibrated.valid, [[False, True]])
    assert np.isnan(calibrated.values[0, 0])
    with pytest.raises(ValueError):
        replace(result.response, spatial_weights=result.response.spatial_weights * 2)
    with pytest.raises(ValueError):
        replace(result.response, spectral_coefficients=-result.response.spectral_coefficients)
    with pytest.raises(ValueError):
        replace(result.response, node_times_s=result.response.node_times_s + 1)
    with pytest.raises(ValueError, match="identities"):
        wrong_reads = replace(result.response.reads, pixel_ids=("wrong", "wrong"))
        calibrate_frames(
            result.images,
            replace(result.response, schedule=replace(result.response.schedule, reads=wrong_reads)),
            detector=DetectorProperties(
                pixel_bounds_m=cam.pixel_bounds_m,
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
    with pytest.raises(ValueError, match="geometry"):
        frame_response(
            replace(
                cam,
                detector=DetectorProperties(
                    pixel_bounds_m=(cam).pixel_bounds_m,
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
            replace(grid, focal_length_m=0.4),
            spectral,
        )


def test_spectral_strip_frame_sequence_preserves_band_means_offline(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr("socket.socket", forbid_network)
    strips = [
        filter_strip(
            index,
            index + 1,
            band=f"b{index}",
            transmission=partial(
                gaussian_filter, centre_m=0.01 / centre, width_m=0.0001 / centre, peak=0.8
            ),
        )
        for index, centre in enumerate((4100, 4350, 4650, 4900))
    ]
    cam = filter_camera(
        ideal_detector(regular_pixel_bounds((4, 4), pitch_m=20e-6)),
        strips,
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(4e-6),
    )
    exposures = frame_exposure_bounds(
        start_s=0, frame_count=41, frame_rate_hz=250, exposure_s=0.001
    )
    grid = camera_acquisition_grid(
        cam,
        raster_read_schedule(exposures, (cam).pixel_bounds_m.shape[:2], port=(cam).optics.port),
        epoch=EPOCH,
        validity_s=(0, 1),
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301,
        time_order=2,
        angular_order=8,
    )
    spectral = connected_grid(np.linspace(3900, 5100, 65))

    def linear_spectrum(xy, directions, times, query_spectral):
        points = query_spectral.points
        return np.broadcast_to(points * 1e-8, (*times.shape, len(points)))

    detector = DetectorProperties(
        pixel_bounds_m=cam.pixel_bounds_m,
        pitch_m=None,
        quantum_efficiency=0.7,
        bias_dn=1000,
        electrons_per_dn=0.2,
        read_noise_electrons=2,
        dark_electrons_per_s=1,
        adc_bits=24,
        full_well_electrons=1e6,
        nonlinearity=0,
    )
    measurements = simulate_frames(
        replace(cam, detector=detector),
        grid,
        spectral,
        linear_spectrum,
        coverage,
        rng=np.random.default_rng(20261102),
    )
    assert measurements.images.photoelectrons.shape == (41, 4, 4)
    assert measurements.images.dn is not None
    assert measurements.images.dn.shape == (1, 41, 4, 4)
    supplied_response = frame_response(replace(cam, detector=detector), grid, spectral)
    bands = achieved_bands(supplied_response)
    expected = np.array([np.dot(spectral.points * 1e-8, band.weights) for band in bands])
    assert np.all(np.diff(expected) > 0)
    kwargs: dict[str, Any] = dict(
        x_edges_m=np.linspace(-0.06, 0.06, 7),
        y_edges_m=np.linspace(-0.06, 0.12, 10),
        bands=bands,
    )
    for use_expectation in (True, False):
        calibrated = calibrate_frames(
            measurements.images,
            supplied_response,
            detector=detector,
            use_expectation=use_expectation,
        )
        cube = reconstruct_radiance_cube(calibrated, grid, **kwargs)
        assert cube.valid.any(axis=(0, 1)).all()
        assert (~cube.valid).any()
        assert np.isnan(cube.values[~cube.valid]).all()
        assert np.isfinite(cube.values[cube.valid]).all()
        if use_expectation:
            relative_error = cube.values / expected - 1
            assert np.max(np.abs(relative_error[cube.valid])) < 1e-12
    assert not sys._is_gil_enabled()


def test_read_batches_preserve_map_common_noise_and_subsequent_rng():
    cam = camera(shape=(2, 2))
    maps = np.arange(4.0).reshape(2, 2)
    detector = replace(
        cam.detector,
        quantum_efficiency=0.6 + maps * 0.05,
        dark_electrons_per_s=1.0 + maps,
        read_noise_electrons=2.0 + maps,
        electrons_per_dn=1.0 + maps * 0.2,
        bias_dn=100.0 + maps,
    )
    original = cam.optics

    def response(exposure, read, wave, current):
        return original.response(exposure, read, wave, current) * (0.5 + 0.05 * read)

    optics = ImagingOptics(
        original.focal_length_m,
        original.collecting_area_m2,
        response,
        original.settings,
        original.throughput,
        original.port,
    )
    cam = FocalPlaneImager(detector, optics, cam.support_radius_m)
    schedule = raster_read_schedule(
        np.array([[0.0, 0.01], [0.05, 0.06]]), (2, 2), row_offsets_s=np.array([0.0, 0.002])
    )
    selected = camera_acquisition_grid(
        cam,
        schedule,
        epoch=EPOCH,
        validity_s=(0.0, 1.0),
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301.0,
        time_order=2,
        angular_order=4,
    )
    spectral = connected_grid(np.array([4900.0, 5000.0, 5100.0]))
    results = []
    tails = []
    for batch in (1, 3, 64):
        rng = np.random.default_rng(561)
        result = simulate_frames(
            cam,
            selected,
            spectral,
            uniform,
            coverage,
            rng=rng,
            realisations=3,
            read_batch_size=batch,
            row_noise_electrons=2.0,
            column_noise_electrons=3.0,
            exposure_noise_electrons=4.0,
        )
        results.append(result)
        tails.append(rng.standard_normal(10))
    for result, tail in zip(results[1:], tails[1:], strict=True):
        assert_allclose(result.images.photoelectrons, results[0].images.photoelectrons, rtol=1e-14)
        np.testing.assert_array_equal(result.images.dn, results[0].images.dn)
        np.testing.assert_array_equal(tail, tails[0])


def test_camera_selection_retains_original_native_indices_through_calibration():
    cam = camera(shape=(2, 2))
    full = raster_read_schedule(
        np.array([[0.0, 0.01], [0.05, 0.06]]), (2, 2), row_offsets_s=np.array([0.0, 0.002])
    )
    schedule = select_raster_reads(full, np.array([1, 4, 7]))
    selected = camera_acquisition_grid(
        cam,
        schedule,
        epoch=EPOCH,
        validity_s=(0.0, 1.0),
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301.0,
        time_order=2,
        angular_order=4,
    )
    spectral = connected_grid(np.array([4900.0, 5000.0, 5100.0]))
    result = simulate_frames(cam, selected, spectral, uniform, coverage, read_batch_size=2)
    assert result.images.photoelectrons.shape == (3,)
    assert result.images.schedule is schedule and result.response.schedule is schedule
    calibrated = calibrate_frames(result.images, result.response, detector=cam.detector)
    assert_allclose(calibrated.values, 2e-5, rtol=1e-14)
    np.testing.assert_array_equal(selected.native_indices, full.native_indices[[1, 4, 7]])


def test_camera_optical_description_cannot_drift_from_response():
    cam = camera(shape=(1, 1))
    opaque = filter_strip(0, 1, band=cam.strips[0].band, transmission=np.zeros_like)
    invalid_changes: tuple[dict[str, Any], ...] = (
        {"strips": (opaque,)},
        {"optics": cam.optics},
        {"support_radius_m": cam.support_radius_m * 2},
        {"omitted_psf_mass": 0.1},
        {"row_bands": ("opaque",)},
    )
    for changes in invalid_changes:
        with pytest.raises(TypeError):
            replace(cam, **changes)
    grid = acquisition(cam, frames=1, angular_order=2, time_order=1)
    spectral = connected_grid(np.linspace(4500, 5000, 3))
    signal = simulate_frames(cam, grid, spectral, uniform, coverage).images.photoelectrons
    rebuilt = filter_camera(
        cam.detector,
        [opaque],
        focal_length_m=cam.optics.focal_length_m,
        f_number=4,
        spatial=GaussianPSF(4e-6),
    )
    assert rebuilt.row_bands == cam.row_bands
    dark = simulate_frames(rebuilt, grid, spectral, uniform, coverage).images.photoelectrons
    assert np.any(signal > 0)
    assert np.count_nonzero(dark) == 0
    lower_qe = replace(cam, detector=replace(cam.detector, quantum_efficiency=0.5))
    actual = simulate_frames(lower_qe, grid, spectral, uniform, coverage).images.photoelectrons
    assert_allclose(actual, signal * 0.5, rtol=2e-15)
    changed_geometry = replace(cam.detector, pixel_bounds_m=cam.pixel_bounds_m + 1e-6)
    with pytest.raises(ValueError, match="pixel geometry"):
        replace(cam, detector=changed_geometry)
