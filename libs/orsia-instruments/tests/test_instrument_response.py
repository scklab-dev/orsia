"""Independent radiometry, local scan support and positive instrument response contracts."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from orsia_instruments import (
    DetectorElectronics,
    DetectorOperatingPoints,
    DetectorResponse,
    ExpectedElectrons,
    GroundGrid,
    InstrumentResponse,
    LinearTrajectory,
    ReadSchedule,
    SceneRadiance,
    along_track_acquisition,
    along_track_exposure_bounds,
    along_track_raster_reads,
    along_track_read_schedule,
    along_track_strip_templates,
    correct_digital_readings,
    detector_response,
    digitize,
    electron_covariance,
    electron_moments,
    expected_electrons,
    filter_imager,
    gaussian_along_track_response,
    gaussian_cross_track_response,
    gaussian_filter,
    ground_fixed_pointing,
    ground_scroll_pointing,
    instrument_response,
    photoelectron_response,
    raster_exposure_bounds,
    rectangular_acquisition,
    response_electrons,
    sample_readout,
    uniform_radiance_field,
)
from orsia_radiative import ObservationGeometry, Radiance, SpectralDomain, SpectralGrid
from scipy.constants import c, h


def zero_pointing(read_index, times_s):
    return np.zeros((len(times_s), 2))


def scan(**changes):
    geometry = dict(
        sample_ids=("read-0", "read-1"),
        pixel_ids=("pixel+", "pixel-"),
        setting_ids=("setting", "setting"),
        port_ids=("+", "-"),
        target_xy_m=np.zeros((2, 2)),
        target_offsets_at_exposure_centre_m=np.array([[20.0, 0.0], [20.0, 0.0]]),
        pixel_bounds_m=np.tile([-1e-5, 1e-5, -1e-5, 1e-5], (2, 1)),
        exposure_s=0.2,
    )
    trajectory = dict(height_m=1000.0, speed_m_s=100.0, time_origin_s=0.0)
    support = dict(
        along_edges_m=np.array([-20.0, 0.0, 20.0]),
        cross_edges_m=np.array([-20.0, 0.0, 20.0]),
    )
    orders = dict(time_order=3, angular_order=3, cross_order=None)
    for group in (geometry, trajectory, support, orders):
        group.update({name: changes.pop(name) for name in tuple(changes) if name in group})
    support_frame = changes.pop("support_frame", "ground")
    changes.setdefault("pointing_law", zero_pointing)
    motion = LinearTrajectory(**trajectory)
    reads = along_track_read_schedule(
        sample_ids=geometry["sample_ids"],
        pixel_ids=geometry["pixel_ids"],
        setting_ids=geometry["setting_ids"],
        port_ids=geometry["port_ids"],
        target_along_m=geometry["target_xy_m"][:, 0],
        ground_along_offsets_m=geometry["target_offsets_at_exposure_centre_m"][:, 0],
        exposure_s=geometry["exposure_s"],
        trajectory=motion,
    )
    return along_track_acquisition(
        reads,
        motion,
        target_xy_m=geometry["target_xy_m"],
        pixel_bounds_m=geometry["pixel_bounds_m"],
        cross_offset_m=geometry["target_offsets_at_exposure_centre_m"][:, 1],
        **support,
        **orders,
        support_frame=support_frame,
        focal_length_m=1.0,
        **changes,
    )


def test_removed_records_and_direct_scan_contract():
    import orsia_instruments as instruments
    from orsia_instruments import acquisition

    for name in (
        "AlongTrackIntegrationOrders",
        "AlongTrackReadSpecification",
        "GroundSpatialSupport",
    ):
        assert not hasattr(instruments, name)
        assert not hasattr(acquisition, name)
    assert instruments.linear_scan_acquisition is along_track_acquisition


def test_strip_templates_and_raster_window_keep_explicit_geometry_and_read_order():
    x = np.array([-4e-6, 4e-6, -4e-6, 4e-6])
    offsets = np.array([-30.0, 30.0, -30.0, 30.0])
    durations = np.array([0.1, 0.2, 0.1, 0.2])
    trajectory = LinearTrajectory(height_m=800, speed_m_s=200, time_origin_s=5)
    templates, pixels = along_track_strip_templates(
        sample_ids=("plus-0", "plus-1", "minus-0", "minus-1"),
        pixel_ids=("a+", "b+", "a-", "b-"),
        setting_ids=("0", "1", "0", "1"),
        port_ids=("+", "+", "-", "-"),
        detector_x_m=x,
        ground_along_offsets_m=offsets,
        active_width_m=6e-6,
        exposure_s=durations,
        trajectory=trajectory,
    )
    x[:] = 100
    offsets[:] = 100
    durations[:] = 100
    assert pixels[0].tolist() == pytest.approx([-7e-6, -1e-6, -3e-6, 3e-6])
    np.testing.assert_allclose(templates.exposure_bounds_s.mean(axis=1), [5.15, 4.85, 5.15, 4.85])
    assert templates.port_ids[2] == "-"
    shifted_read = along_track_read_schedule(
        sample_ids=("shifted",),
        pixel_ids=("a",),
        setting_ids=("0",),
        port_ids=("+",),
        target_along_m=np.array([25.0]),
        ground_along_offsets_m=np.array([10.0]),
        exposure_s=0.1,
        trajectory=trajectory,
    )
    np.testing.assert_allclose(shifted_read.exposure_bounds_s.mean(axis=1), [5.075])
    rows = np.array([-10.0, 5.0, 35.0])
    ground_offsets = np.array([-30.0, 30.0, -30.0, 30.0])
    bounds = raster_exposure_bounds(templates, rows, ground_offsets, trajectory)
    assert bounds.shape == (3, 4, 2)
    np.testing.assert_allclose(
        np.diff(bounds, axis=-1)[..., 0], np.tile([0.1, 0.2, 0.1, 0.2], (3, 1))
    )
    expected_centres = 5 + (rows[:, None] - ground_offsets[None, :]) / 200
    np.testing.assert_allclose(bounds.mean(axis=-1), expected_centres)
    reads = along_track_raster_reads(
        templates,
        rows,
        ground_offsets,
        3,
        trajectory,
        sample_id=lambda r, i, j: f"{r}:{i}:{j}",
        pixel_id=lambda r, i, j: f"{r}:{j}",
    )
    np.testing.assert_allclose(
        bounds.transpose(1, 0, 2), reads.exposure_bounds_s.reshape(4, 3, 3, 2)[:, :, 0]
    )
    assert reads.port_ids[0] == "+" and reads.port_ids[-1] == "-"
    bounds[:] = 0
    assert reads.exposure_bounds_s[0, 0] != 0
    with pytest.raises(ValueError, match="aligned identities"):
        along_track_strip_templates(
            sample_ids=("a", "b"),
            pixel_ids=("a",),
            setting_ids=("0", "1"),
            port_ids=("+", "-"),
            detector_x_m=np.zeros(2),
            ground_along_offsets_m=np.zeros(2),
            active_width_m=1e-5,
            exposure_s=0.2,
            trajectory=trajectory,
        )
    with pytest.raises(ValueError, match="exposure_s"):
        along_track_strip_templates(
            sample_ids=("a",),
            pixel_ids=("a",),
            setting_ids=("0",),
            port_ids=("+",),
            detector_x_m=np.zeros(1),
            ground_along_offsets_m=np.zeros(1),
            active_width_m=1e-5,
            exposure_s=0,
            trajectory=trajectory,
        )
    with pytest.raises(ValueError, match="detector geometry"):
        along_track_strip_templates(
            sample_ids=("a",),
            pixel_ids=("a",),
            setting_ids=("0",),
            port_ids=("+",),
            detector_x_m=np.zeros(1),
            ground_along_offsets_m=np.zeros(1),
            active_width_m=0,
            exposure_s=0.2,
            trajectory=trajectory,
        )
    with pytest.raises(ValueError, match="offset per template"):
        raster_exposure_bounds(templates, rows, ground_offsets[:2], trajectory)


def grid():
    points = np.array([4000.0, 4700.0, 6000.0])
    return SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4000.0, 6000.0),))
    )


def acceptance(read, wave, acquisition):
    return np.full((acquisition.times_s.shape[1], len(wave)), 0.3 if read == 0 else 0.7)


def projected(acquisition=None):
    acquisition = scan() if acquisition is None else acquisition
    xy = acquisition.directions_uv * 1000
    xy[..., 0] += 100 * acquisition.times_s
    return detector_response(
        spectral_grid=grid(),
        acquisition=acquisition,
        optical_response=acceptance,
        ground_xy_m=xy,
        along_m=np.array([-20.0, 0.0, 20.0]),
        cross_m=np.array([-20.0, 0.0, 20.0]),
        support_bounds_m=np.tile([-20.0, 20.0, -20.0, 20.0], (2, 1)),
        collecting_area_m2=0.02,
        throughput=0.5,
        quantum_efficiency=0.8,
    )


def test_gaussian_scan_lift_matches_full_optical_law_and_checks_node_pairing():
    trajectory = LinearTrajectory(height_m=1000, speed_m_s=100)
    reads = along_track_read_schedule(
        sample_ids=("a",),
        pixel_ids=("strip",),
        setting_ids=("0",),
        port_ids=("direct",),
        target_along_m=np.zeros(1),
        ground_along_offsets_m=np.zeros(1),
        exposure_s=0.2,
        trajectory=trajectory,
    )
    params = dict(
        target_xy_m=np.zeros((1, 2)),
        pixel_bounds_m=np.array([[-10e-6, 10e-6, -10e-6, 10e-6]]),
        along_edges_m=np.array([-20.0, 0.0, 20.0]),
        focal_length_m=1,
        time_order=2,
        angular_order=3,
    )
    line = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=zero_pointing,
        cross_edges_m=np.array([-0.5, 0.5]),
        cross_order=1,
    )
    full = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=zero_pointing,
        cross_edges_m=np.array([-20.0, 0.0, 20.0]),
    )
    optics = filter_imager(
        focal_length_m=1,
        collecting_area_m2=0.1,
        psf_sigma_m=4e-6,
        filter=lambda wave: gaussian_filter(wave, centre_m=2e-6, width_m=1e-7),
    )
    lifted = gaussian_cross_track_response(line, optics, sigma_m=4e-6)(full)
    wavelengths = np.array([1.9e-6, 2e-6])
    np.testing.assert_allclose(
        lifted(0, wavelengths, full), optics.response(0, 0, wavelengths, full), rtol=3e-15
    )

    def shifted_pointing(read_index, times_s):
        return np.column_stack((np.zeros(len(times_s)), np.full(len(times_s), 5e-6)))

    shifted_line = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=shifted_pointing,
        cross_edges_m=np.array([-0.5, 0.5]),
        cross_order=1,
    )
    shifted_full = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=shifted_pointing,
        cross_edges_m=np.array([-20.0, 0.0, 20.0]),
    )
    shifted_law = gaussian_cross_track_response(shifted_line, optics, sigma_m=4e-6)(shifted_full)
    np.testing.assert_allclose(
        shifted_law(0, wavelengths, shifted_full),
        optics.response(0, 0, wavelengths, shifted_full),
        rtol=3e-15,
    )

    def slewing_cross_pointing(read_index, times_s):
        return np.column_stack((np.zeros(len(times_s)), 5e-6 + times_s * 2e-6))

    slewing_line = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=slewing_cross_pointing,
        cross_edges_m=np.array([-0.5, 0.5]),
        cross_order=1,
    )
    slewing_full = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=slewing_cross_pointing,
        cross_edges_m=np.array([-20.0, 0.0, 20.0]),
    )
    slewing_law = gaussian_cross_track_response(slewing_line, optics, sigma_m=4e-6)(slewing_full)
    np.testing.assert_allclose(
        slewing_law(0, wavelengths, slewing_full),
        optics.response(0, 0, wavelengths, slewing_full),
        rtol=3e-15,
    )

    def distant_pointing(read_index, times_s):
        return np.column_stack((np.zeros(len(times_s)), np.full(len(times_s), 100.0)))

    distant_line = along_track_acquisition(
        reads,
        trajectory,
        **params,
        pointing_law=distant_pointing,
        cross_edges_m=np.array([-0.5, 0.5]),
        cross_order=1,
    )
    with pytest.raises(ValueError, match="reference acceptance"):
        gaussian_cross_track_response(distant_line, optics, sigma_m=4e-6)(shifted_full)
    with pytest.raises(ValueError, match="repeat the optical line"):
        gaussian_cross_track_response(line, optics, sigma_m=4e-6)(scan())
    with pytest.raises(ValueError, match="sigma"):
        gaussian_cross_track_response(line, optics, sigma_m=0)


def test_gaussian_along_track_response_matches_independent_read_composition():
    trajectory = LinearTrajectory(height_m=1000, speed_m_s=100, time_origin_s=2)
    reads = along_track_read_schedule(
        sample_ids=("read-0", "read-1"),
        pixel_ids=("pixel-0", "pixel-1"),
        setting_ids=("setting-0", "setting-1"),
        port_ids=("+", "+"),
        target_along_m=np.zeros(2),
        ground_along_offsets_m=np.array([10.0, -10.0]),
        exposure_s=np.array([0.2, 0.3]),
        trajectory=trajectory,
    )
    pixels = np.tile([-10e-6, 10e-6, -10e-6, 10e-6], (2, 1))
    edges = np.array([-20.0, 0.0, 20.0])
    optics = tuple(
        filter_imager(
            focal_length_m=1,
            collecting_area_m2=0.1 + read * 0.03,
            psf_sigma_m=4e-6,
            throughput=0.8 - read * 0.2,
            filter=lambda wave, read=read: gaussian_filter(
                wave,
                centre_m=2e-6 + read * 2e-8,
                width_m=1e-7,
            ),
        )
        for read in range(2)
    )
    knots = np.array([-20.0, 0.0, 20.0])
    result = gaussian_along_track_response(
        grid(),
        reads,
        trajectory,
        optics.__getitem__,
        pixel_bounds_m=pixels,
        along_edges_m=edges,
        cross_edges_m=edges,
        time_order=2,
        angular_order=3,
        ground_grid=GroundGrid(knots, knots),
        focal_length_m=1,
        sigma_m=4e-6,
        pointing_law=zero_pointing,
        quantum_efficiency=0.7,
    )
    assert result.coefficients.shape == (2, 3, 3, 3)
    assert np.all(result.coefficients >= 0)
    assert result.reads.sample_ids == ("read-0", "read-1")
    np.testing.assert_allclose(result.support_bounds_m, [[-20, 20, -20, 20]] * 2)
    for read in range(2):
        local_reads = ReadSchedule(
            (reads.sample_ids[read],),
            (reads.pixel_ids[read],),
            (reads.setting_ids[read],),
            (reads.port_ids[read],),
            reads.exposure_bounds_s[read : read + 1],
        )
        line = along_track_acquisition(
            local_reads,
            trajectory,
            target_xy_m=np.zeros((1, 2)),
            pixel_bounds_m=pixels[read : read + 1],
            along_edges_m=edges,
            cross_edges_m=np.array([-0.5, 0.5]),
            cross_order=1,
            time_order=2,
            angular_order=3,
            focal_length_m=1,
            pointing_law=zero_pointing,
        )
        reference = instrument_response(
            grid(),
            local_reads,
            trajectory,
            gaussian_cross_track_response(line, optics[read], sigma_m=4e-6),
            target_xy_m=np.zeros((1, 2)),
            pixel_bounds_m=pixels[read : read + 1],
            along_edges_m=edges,
            cross_edges_m=edges,
            time_order=2,
            angular_order=3,
            ground_grid=GroundGrid(knots, knots),
            focal_length_m=1,
            pointing_law=zero_pointing,
            collecting_area_m2=optics[read].collecting_area_m2,
            throughput=optics[read].throughput,
            quantum_efficiency=0.7,
        )
        np.testing.assert_array_equal(result.coefficients[read], reference.coefficients[0])
        np.testing.assert_array_equal(
            result.reads.exposure_bounds_s[read], reference.reads.exposure_bounds_s[0]
        )
    with pytest.raises(ValueError, match="pixel bounds"):
        gaussian_along_track_response(
            grid(),
            reads,
            trajectory,
            optics.__getitem__,
            pixel_bounds_m=pixels[:1],
            along_edges_m=edges,
            cross_edges_m=edges,
            pointing_law=zero_pointing,
            ground_grid=GroundGrid(knots, knots),
            focal_length_m=1,
            sigma_m=4e-6,
        )


def test_along_track_raster_reads_matches_explicit_timing_and_identities():
    trajectory = LinearTrajectory(height_m=1000, speed_m_s=100, time_origin_s=3)
    templates = ReadSchedule(
        ("first", "second"),
        ("pixel-a", "pixel-b"),
        ("a", "b"),
        ("+", "-"),
        np.array([[0.0, 0.2], [1.0, 1.4]]),
    )
    rows = np.array([-20.0, 0.0, 20.0])
    offsets = np.array([10.0, -10.0])

    def sample_id(r, i, j):
        return f"read-{r}-{i}-{j}"

    def pixel_id(r, i, j):
        return f"pixel-{r}-{j}"

    actual = along_track_raster_reads(
        templates,
        rows,
        offsets,
        2,
        trajectory,
        sample_id=sample_id,
        pixel_id=pixel_id,
    )
    indices = tuple(np.ndindex((2, 3, 2)))
    reference = along_track_read_schedule(
        sample_ids=tuple(sample_id(*index) for index in indices),
        pixel_ids=tuple(pixel_id(*index) for index in indices),
        setting_ids=tuple(templates.setting_ids[r] for r, i, j in indices),
        port_ids=tuple(templates.port_ids[r] for r, i, j in indices),
        target_along_m=np.array([rows[i] for r, i, j in indices]),
        ground_along_offsets_m=np.array([offsets[r] for r, i, j in indices]),
        exposure_s=np.array([np.diff(templates.exposure_bounds_s[r])[0] for r, i, j in indices]),
        trajectory=trajectory,
    )
    assert actual.sample_ids == reference.sample_ids
    assert actual.pixel_ids == reference.pixel_ids
    assert actual.setting_ids == reference.setting_ids
    assert actual.port_ids == reference.port_ids
    np.testing.assert_array_equal(actual.exposure_bounds_s, reference.exposure_bounds_s)
    before = actual.exposure_bounds_s.copy()
    templates.exposure_bounds_s[:] = [[2, 3], [3, 4]]
    np.testing.assert_array_equal(actual.exposure_bounds_s, before)
    with pytest.raises(ValueError, match="target rows"):
        along_track_raster_reads(
            templates,
            rows,
            offsets,
            0,
            trajectory,
            sample_id=sample_id,
            pixel_id=pixel_id,
        )


def test_digital_read_correction_preserves_identity_and_refuses_clipping():
    reads = ReadSchedule(
        ("a", "b"),
        ("p", "q"),
        ("s", "s"),
        ("direct", "direct"),
        np.array([[0.0, 0.5], [0.0, 1.0]]),
    )
    electronics = DetectorElectronics(
        dark_electrons_per_s=2,
        electrons_per_dn=4,
        bias_dn=100,
        full_well_electrons=1000,
        adc_bits=12,
    )
    reading = digitize(reads, np.array([[9.0, 14.0]]), np.zeros((1, 2)), electronics)
    corrected = correct_digital_readings(reading, electronics)
    np.testing.assert_array_equal(reading.dn, [[102, 104]])
    np.testing.assert_array_equal(corrected, [[7.0, 14.0]])
    assert corrected.flags.writeable and not np.shares_memory(corrected, reading.dn)
    clipped = digitize(reads, np.array([[1000.0, 14.0]]), np.zeros((1, 2)), electronics)
    with pytest.raises(ValueError, match="clipped"):
        correct_digital_readings(clipped, electronics)
    points = DetectorOperatingPoints(
        dark_electrons_per_s=np.array([2.0, 0.0]),
        read_noise_electrons=np.zeros(2),
        full_well_electrons=np.full(2, 1000.0),
        electrons_per_dn=np.array([4.0, 2.0]),
        bias_dn=np.array([100.0, 7.0]),
        nonlinearity=np.zeros(2),
        adc_bits=12,
    )
    mixed = digitize(reads, np.array([[9.0, 14.0]]), np.zeros((1, 2)), points)
    np.testing.assert_array_equal(correct_digital_readings(mixed, points), [[7.0, 14.0]])
    with pytest.raises(ValueError, match="nonlinear"):
        correct_digital_readings(mixed, replace(points, nonlinearity=np.array([0.1, 0.0])))


def test_scan_exact_exposure_full_support_and_paraxial_measure():
    reads = scan(time_origin_s=10.0)
    np.testing.assert_allclose(reads.exposure_bounds_s, np.tile([9.7, 9.9], (2, 1)))
    np.testing.assert_allclose(
        reads.direction_bounds_uv, np.tile([-0.01, 0.05, -0.02, 0.02], (2, 1))
    )
    np.testing.assert_allclose(reads.weights_sr_s.sum(axis=1), 40 * 40 * 0.2 / 1000**2, rtol=2e-14)
    np.testing.assert_allclose(reads.image_xy_m, reads.directions_uv)
    assert reads.sample_ids == ("read-0", "read-1") and reads.port_ids == ("+", "-")
    assert reads.times_s.min() > 9.7 and reads.times_s.max() < 9.9
    # The support endpoints extend beyond Gauss nodes in both time and direction.
    assert reads.direction_bounds_uv[0, 0] < reads.directions_uv[0, :, 0].min()
    shifted = scan(target_xy_m=np.array([[-2475.0, -2475.0], [2475.0, 2475.0]]))
    np.testing.assert_allclose(shifted.exposure_bounds_s.mean(axis=1) * 100 + 20, [-2475, 2475])
    # A target-column coordinate translates the ground patch; detector cross slopes stay local.
    np.testing.assert_allclose(shifted.direction_bounds_uv, scan().direction_bounds_uv, atol=1e-15)


def test_ground_pointing_fixed_scroll_and_zero_parity():
    trajectory = LinearTrajectory(height_m=1000.0, speed_m_s=100.0)
    targets = np.array([[0.0, 0.0], [30.0, 15.0]])
    centres = np.array([0.0, 0.3])
    fixed = ground_fixed_pointing(trajectory, targets, centres)
    slow = ground_scroll_pointing(trajectory, targets, centres, ground_rate_m_s=25.0)
    platform_fixed = ground_scroll_pointing(trajectory, targets, centres, ground_rate_m_s=100.0)
    t = np.array([-0.1, 0.0, 0.1])
    np.testing.assert_allclose(fixed(0, t)[:, 0], [0.01, 0, -0.01])
    np.testing.assert_allclose(slow(0, t)[:, 0], [0.0075, 0, -0.0075])
    np.testing.assert_allclose(platform_fixed(0, t)[:, 0], 0)
    np.testing.assert_allclose(fixed(1, centres[1:]), [[0.0, 0.0]])
    original = targets.copy()
    targets[:] = 100
    np.testing.assert_allclose(slow(0, t)[:, 0], [0.0075, 0, -0.0075])
    targets[:] = original
    zero = scan()
    explicit_zero = scan(pointing_law=lambda read, t: np.zeros((len(t), 2)))
    np.testing.assert_array_equal(zero.image_xy_m, explicit_zero.image_xy_m)
    np.testing.assert_array_equal(zero.weights_sr_s, explicit_zero.weights_sr_s)
    centres = zero.exposure_bounds_s.mean(axis=1)
    fixed_law = ground_fixed_pointing(trajectory, np.zeros((2, 2)), centres)
    slow_law = ground_scroll_pointing(trajectory, np.zeros((2, 2)), centres, ground_rate_m_s=25.0)
    fixed_scan = scan(pointing_law=fixed_law)
    slow_scan = scan(pointing_law=slow_law)
    for actual in (fixed_scan, slow_scan):
        np.testing.assert_array_equal(actual.directions_uv, zero.directions_uv)
        np.testing.assert_array_equal(actual.weights_sr_s, zero.weights_sr_s)
    # In a fixed ground patch the stationary gaze removes exposure-time smear.
    fixed_image = fixed_scan.image_xy_m.reshape(2, 3, 6, 6, 2)
    np.testing.assert_allclose(fixed_image, np.broadcast_to(fixed_image[:, :1], fixed_image.shape))
    np.testing.assert_allclose(
        slow_scan.image_xy_m[..., 0] - fixed_scan.image_xy_m[..., 0],
        -25.0 * (slow_scan.times_s - centres[:, None]) / trajectory.height_m,
        atol=2e-17,
    )
    slit = scan(support_frame="platform", pointing_law=fixed_law)
    np.testing.assert_allclose(
        slit.image_xy_m[..., 0] - fixed_scan.image_xy_m[..., 0],
        100.0 * (slit.times_s - centres[:, None]) / trajectory.height_m,
        atol=2e-17,
    )
    with_cross = ground_fixed_pointing(
        trajectory, np.zeros((2, 2)), centres, cross_offset_m=np.array([5.0, -5.0])
    )
    np.testing.assert_allclose(with_cross(0, np.array([centres[0]]))[0, 1], 0.005)
    for bad in (
        lambda read, t: np.zeros((len(t), 3)),
        lambda read, t: np.full((len(t), 2), np.nan),
    ):
        with pytest.raises(ValueError, match="pointing_law"):
            scan(pointing_law=bad)


def test_platform_slit_support_and_swept_ground():
    acquisition = scan(support_frame="platform", along_edges_m=np.array([-5.0, 0.0, 5.0]))
    ground_fixed = scan(along_edges_m=np.array([-5.0, 0.0, 5.0]))
    np.testing.assert_array_equal(acquisition.times_s, ground_fixed.times_s)
    np.testing.assert_array_equal(
        acquisition.directions_uv[..., 1], ground_fixed.directions_uv[..., 1]
    )
    # A fixed patch has constant along-ground nodes through exposure time; a slit sweeps.
    ground_x = (ground_fixed.directions_uv[..., 0] * 1000 + 100 * ground_fixed.times_s).reshape(
        2, 3, 6, 6
    )
    np.testing.assert_allclose(
        ground_x, np.broadcast_to(ground_x[:, :1], ground_x.shape), atol=1e-13
    )
    assert not np.allclose(acquisition.directions_uv, ground_fixed.directions_uv)
    np.testing.assert_allclose(
        acquisition.direction_bounds_uv[:, :2], np.tile([0.015, 0.025], (2, 1))
    )
    ground = acquisition.directions_uv[..., 0] * 1000 + 100 * acquisition.times_s
    assert ground.min() > -15 and ground.max() < 15
    np.testing.assert_allclose(acquisition.weights_sr_s.sum(axis=1), 10 * 40 * 0.2 / 1e6)


def test_support_frame_requires_explicit_ground_or_platform():
    with pytest.raises(ValueError, match="support_frame"):
        scan(support_frame="boresight")


def test_grouped_scan_defaults_per_read_timing_and_borrowed_inputs():
    targets = np.array([[30.0, 70.0], [80.0, 90.0]])
    offsets = np.array([[10.0, 3.0], [20.0, -4.0]])
    pixels = np.tile([-1e-5, 1e-5, -1e-5, 1e-5], (2, 1))
    durations = np.array([0.2, 0.4])
    edges = np.array([-20.0, 0.0, 20.0])
    identities = dict(
        sample_ids=("first", "second"),
        pixel_ids=("pixel-0", "pixel-1"),
        setting_ids=("setting-0", "setting-1"),
        port_ids=("+", "-"),
    )
    trajectory = LinearTrajectory(height_m=1000.0, speed_m_s=100.0, time_origin_s=10.0)
    reads = along_track_read_schedule(
        **identities,
        target_along_m=targets[:, 0],
        ground_along_offsets_m=offsets[:, 0],
        exposure_s=durations,
        trajectory=trajectory,
    )
    originals = [a.copy() for a in (targets, offsets, pixels, durations, edges)]
    acquisition = along_track_acquisition(
        reads,
        trajectory,
        target_xy_m=targets,
        pixel_bounds_m=pixels,
        along_edges_m=edges,
        cross_edges_m=edges,
        cross_offset_m=offsets[:, 1],
        focal_length_m=2.0,
        pointing_law=zero_pointing,
    )
    assert acquisition.times_s.shape == (2, 3 * 10 * 10)
    np.testing.assert_allclose(acquisition.exposure_bounds_s, [[10.1, 10.3], [10.4, 10.8]])
    np.testing.assert_allclose(
        acquisition.direction_bounds_uv,
        [[-0.02, 0.04, -0.017, 0.023], [-0.02, 0.06, -0.024, 0.016]],
        atol=1e-15,
    )
    np.testing.assert_allclose(acquisition.weights_sr_s.sum(axis=1), durations * 40 * 40 / 1e6)
    np.testing.assert_array_equal(acquisition.image_xy_m, 2 * acquisition.directions_uv)
    for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids"):
        assert getattr(acquisition, name) == getattr(reads, name)
    for original, current in zip(
        originals, (targets, offsets, pixels, durations, edges), strict=True
    ):
        np.testing.assert_array_equal(current, original)
        for output in (acquisition.times_s, acquisition.pixel_bounds_m, acquisition.weights_sr_s):
            assert not np.shares_memory(current, output)
    acquisition.pixel_bounds_m[:] = 0
    np.testing.assert_array_equal(pixels, originals[2])
    # Reuse the physical read, trajectory and orders with a narrow cross-track support.
    line = along_track_acquisition(
        reads,
        trajectory,
        target_xy_m=targets,
        pixel_bounds_m=pixels,
        along_edges_m=edges,
        cross_edges_m=np.array([-0.5, 0.5]),
        cross_offset_m=offsets[:, 1],
        cross_order=1,
        focal_length_m=2.0,
        pointing_law=zero_pointing,
    )
    assert line.times_s.shape == (2, 3 * 10)
    np.testing.assert_array_equal(line.exposure_bounds_s, acquisition.exposure_bounds_s)


def test_along_track_read_schedule_matches_full_scan_without_optical_nodes():
    target = np.array([30.0, 80.0])
    offsets = np.array([10.0, 20.0])
    durations = np.array([0.2, 0.4])
    trajectory = LinearTrajectory(height_m=1000.0, speed_m_s=100.0, time_origin_s=10.0)
    kwargs = dict(
        sample_ids=("first", "second"),
        pixel_ids=("a", "b"),
        setting_ids=("one", "two"),
        port_ids=("+", "-"),
    )
    reads = along_track_read_schedule(
        **kwargs,
        target_along_m=target,
        ground_along_offsets_m=offsets,
        exposure_s=durations,
        trajectory=trajectory,
    )
    full = along_track_acquisition(
        reads,
        trajectory,
        target_xy_m=np.column_stack((target, [70.0, 90.0])),
        cross_offset_m=np.array([3.0, -4.0]),
        pixel_bounds_m=np.tile([-1e-5, 1e-5, -1e-5, 1e-5], (2, 1)),
        along_edges_m=np.array([-20.0, 0.0, 20.0]),
        cross_edges_m=np.array([-20.0, 0.0, 20.0]),
        focal_length_m=1.0,
        pointing_law=zero_pointing,
    )
    assert isinstance(reads, ReadSchedule)
    for name in kwargs:
        assert getattr(reads, name) == getattr(full, name)
    np.testing.assert_array_equal(reads.exposure_bounds_s, full.exposure_bounds_s)
    np.testing.assert_allclose(reads.exposure_bounds_s, [[10.1, 10.3], [10.4, 10.8]])
    assert not np.shares_memory(reads.exposure_bounds_s, target)
    target[0] = -100.0
    durations[0] = 4.0
    np.testing.assert_array_equal(reads.exposure_bounds_s, full.exposure_bounds_s)
    reads.exposure_bounds_s[0, 0] = 2.0
    assert full.exposure_bounds_s[0, 0] != 2.0


def test_along_track_read_schedule_validates_positions_timing_and_identities():
    kwargs: dict[str, Any] = dict(
        sample_ids=("a", "b"),
        pixel_ids=("p", "p"),
        setting_ids=("s", "s"),
        port_ids=("+", "-"),
        target_along_m=np.array([0.0, 20.0]),
        ground_along_offsets_m=np.array([0.0, 10.0]),
        exposure_s=0.2,
        trajectory=LinearTrajectory(height_m=1000.0, speed_m_s=100.0),
    )
    np.testing.assert_array_equal(
        along_track_read_schedule(**kwargs).exposure_bounds_s, [[-0.1, 0.1], [0.0, 0.2]]
    )
    for change, message in (
        ({"target_along_m": np.zeros((2, 2))}, "(reads,)"),
        ({"ground_along_offsets_m": np.zeros(1)}, "(reads,)"),
        ({"target_along_m": np.array([0.0, np.nan])}, "finite"),
        ({"exposure_s": 0.0}, "positive"),
        ({"sample_ids": ("a", "a")}, "unique"),
        ({"trajectory": LinearTrajectory(height_m=1000.0, speed_m_s=-1.0)}, "positive"),
    ):
        with pytest.raises(ValueError, match=message):
            along_track_read_schedule(**cast("dict[str, Any]", kwargs | change))


def test_scan_and_rectangular_angular_measures_remain_distinct():
    local = scan(support_frame="platform")
    sky = rectangular_acquisition(
        sample_ids=local.sample_ids,
        pixel_ids=local.pixel_ids,
        setting_ids=local.setting_ids,
        port_ids=local.port_ids,
        exposure_bounds_s=local.exposure_bounds_s,
        pixel_bounds_m=local.pixel_bounds_m,
        direction_bounds_uv=(0.0, 0.04, -0.02, 0.02),
        pointing=lambda times: np.zeros((len(times), 2)),
        focal_length_m=1.0,
        time_order=3,
        angular_order=8,
    )
    np.testing.assert_allclose(local.weights_sr_s.sum(axis=1), 0.2 * 0.04 * 0.04, rtol=2e-15)
    assert np.all(sky.weights_sr_s.sum(axis=1) > local.weights_sr_s.sum(axis=1))


def test_analytic_photon_area_exposure_qe_spectral_scaling():
    acquisition = scan()
    # L(sigma)=h*c*100*sigma gives one photon per unit radiance integral,
    # so the trapezoid is exact on a deliberately nonuniform spectral grid.
    g = grid()
    photons = h * c * 100 * g.points
    field = uniform_radiance_field(
        Radiance(g, photons, ObservationGeometry(0.0)),
        time_bounds_s=(-1.0, 1.0),
        u_bounds=(-0.1, 0.1),
        v_bounds=(-0.1, 0.1),
        spectral_bounds_cm1=(4000.0, 6000.0),
    )
    expected = (6000 - 4000) * (40 * 40 * 0.2 / 1000**2) * 0.02 * 0.5 * 0.8 * np.array([0.3, 0.7])
    signal = expected_electrons(
        field,
        acquisition,
        acceptance,
        collecting_area_m2=0.02,
        throughput=0.5,
        quantum_efficiency=0.8,
    )
    np.testing.assert_allclose(signal.total, expected, rtol=2e-14)
    dense = photoelectron_response(
        g,
        acquisition,
        acceptance,
        collecting_area_m2=0.02,
        throughput=0.5,
        quantum_efficiency=0.8,
    )
    callback = photoelectron_response(
        g,
        acquisition,
        acceptance,
        collecting_area_m2=0.02,
        throughput=0.5,
        quantum_efficiency=lambda r, w: np.full((len(r), len(w)), 0.8),
    )
    assert dense.flags.writeable and dense.shape == (*acquisition.times_s.shape, len(g.points))
    np.testing.assert_array_equal(callback, dense)
    np.testing.assert_allclose(np.sum(dense * photons, axis=(1, 2)), expected, rtol=2e-14)
    np.testing.assert_allclose(
        response_electrons(projected(), np.broadcast_to(photons, (3, 3, 3))).total,
        expected,
        rtol=2e-14,
    )
    for keyword, value in (
        ("collecting_area_m2", 0.04),
        ("throughput", 1.0),
        ("quantum_efficiency", 0.4),
    ):
        params = dict(collecting_area_m2=0.02, throughput=0.5, quantum_efficiency=0.8)
        params[keyword] = value
        result = expected_electrons(field, acquisition, acceptance, **params)
        factor = 0.5 if keyword == "quantum_efficiency" else 2
        np.testing.assert_allclose(result.total, factor * expected, rtol=2e-14)
    twice = scan(exposure_s=0.4)
    np.testing.assert_allclose(
        expected_electrons(
            field,
            twice,
            acceptance,
            collecting_area_m2=0.02,
            throughput=0.5,
            quantum_efficiency=0.8,
        ).total,
        2 * expected,
        rtol=2e-14,
    )


def test_projection_against_independent_bilinear_polynomial_and_direct_field():
    acquisition = scan()
    response = projected(acquisition)
    np.testing.assert_allclose(
        response.coefficients, projected(acquisition).coefficients, rtol=1e-15
    )
    x, y = np.meshgrid(response.along_m, response.cross_m, indexing="ij")
    spatial = 1 + 0.01 * x + 0.02 * y + 0.0001 * x * y
    spectrum = h * c * 100 * grid().points
    values = spatial[..., None] * spectrum
    result = response_electrons(response, values)
    # Odd x,y and xy integrate to zero over the symmetric ground support.
    analytic = np.array([0.3, 0.7]) * 0.02 * 0.5 * 0.8 * 0.2 * 40 * 40 / 1e6 * 2000
    np.testing.assert_allclose(result.total, analytic, rtol=2e-14)
    # Independently sample the actual moving field at each time/direction node.
    t = np.unique(np.r_[acquisition.exposure_bounds_s.ravel(), acquisition.times_s.ravel()])
    u = np.unique(
        np.r_[
            acquisition.directions_uv[..., 0].ravel(),
            acquisition.direction_bounds_uv[:, :2].ravel(),
        ]
    )
    v = np.unique(
        np.r_[
            acquisition.directions_uv[..., 1].ravel(),
            acquisition.direction_bounds_uv[:, 2:].ravel(),
        ]
    )
    xx = 1000 * u[None, :, None] + 100 * t[:, None, None]
    yy = 1000 * v[None, None, :]
    from orsia_instruments import RadianceField

    field = RadianceField(
        grid(), t, u, v, (1 + 0.01 * xx + 0.02 * yy + 0.0001 * xx * yy)[..., None] * spectrum
    )
    direct = expected_electrons(
        field,
        acquisition,
        acceptance,
        collecting_area_m2=0.02,
        throughput=0.5,
        quantum_efficiency=0.8,
    )
    np.testing.assert_allclose(result.total, direct.total, rtol=2e-14)
    assert result.acquisition.sample_ids == acquisition.sample_ids
    assert response.coefficients.min() >= 0
    np.testing.assert_allclose(
        response.coefficients.sum(axis=(1, 2))[1] / response.coefficients.sum(axis=(1, 2))[0], 7 / 3
    )


def test_halo_correlation_block_direct_and_native_identities():
    response = projected()
    axis = np.arange(-60.0, 61.0, 20.0)
    rows, cols = np.array([1, 5]), np.array([1, 5])
    templates = response.reads
    reads = ReadSchedule(
        tuple(f"sample-{i}" for i in range(8)),
        tuple(f"pixel-{i}" for i in range(8)),
        ("setting",) * 8,
        ("+",) * 4 + ("-",) * 4,
        np.tile([2.0, 2.2], (8, 1)),
    )
    maps = np.arange(1.0, 99.0).reshape(2, 7, 7)
    spectra = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    common: dict = dict(
        spectra=spectra,
        scene_along_m=axis,
        scene_cross_m=axis,
        target_indices=(rows, cols),
        reads=reads,
    )
    direct = response_electrons(response, maps, method="direct", **common)
    actual = response_electrons(response, maps, method="correlation", **common)
    np.testing.assert_allclose(actual.total, direct.total, rtol=2e-15)
    dense = np.einsum("cij,ck->ijk", maps, spectra)
    dense_result = response_electrons(
        response,
        dense,
        scene_along_m=axis,
        scene_cross_m=axis,
        target_indices=(rows, cols),
        reads=reads,
        method="correlation",
    )
    np.testing.assert_allclose(dense_result.total, direct.total, rtol=2e-15)
    assert direct.acquisition is reads
    for row, column, offset in ((1, 1, 0), (1, 5, 1), (5, 1, 2), (5, 5, 3)):
        patch = dense[row - 1 : row + 2, column - 1 : column + 2]
        np.testing.assert_allclose(
            direct.total[[offset, 4 + offset]],
            response_electrons(response, patch).total,
            rtol=2e-15,
        )
    invalid: dict = dict(common)
    invalid["target_indices"] = (np.array([0]), cols)
    with pytest.raises(ValueError, match="halo"):
        response_electrons(response, maps, **invalid)
    invalid = dict(common)
    invalid["reads"] = replace(reads, port_ids=("wrong",) * 8)
    with pytest.raises(ValueError, match="settings/ports"):
        response_electrons(response, maps, **invalid)
    invalid = dict(common)
    invalid["reads"] = replace(reads, exposure_bounds_s=np.tile([0.0, 1.0], (8, 1)))
    with pytest.raises(ValueError, match="durations"):
        response_electrons(response, maps, **invalid)
    assert templates.port_ids == ("+", "-")


def test_physical_detector_then_signed_reconstruction_and_covariance():
    reads = scan()
    signal = ExpectedElectrons(reads, {"scene": np.array([20.0, 5.0])})
    detector = DetectorElectronics(full_well_electrons=10, electrons_per_dn=1, adc_bits=8)
    mean, _ = electron_moments(signal, detector)
    reading = digitize(reads, mean[None], np.zeros((1, 2)), detector)
    W = np.array([[1.0, -1.0]])
    np.testing.assert_array_equal(reading.dn, [[10, 5]])
    np.testing.assert_array_equal(reading.full_well_saturated, [[True, False]])
    np.testing.assert_array_equal(reading.dn @ W.T, [[5]])
    # Reconstruction before clipping would give 15, physically inconsistent with 5.
    assert (W @ mean)[0] == 15
    covariance = electron_covariance(
        signal, detector, exposure_ids=("same", "same"), exposure_noise_electrons=3
    )
    np.testing.assert_array_equal(covariance, [[29, 9], [9, 14]])
    np.testing.assert_array_equal(W @ covariance @ W.T, [[25]])
    rng = np.random.default_rng(123)
    draws = sample_readout(
        signal,
        detector,
        rng,
        realisations=4,
        exposure_ids=("same", "same"),
        exposure_noise_electrons=3,
    )
    assert draws.dn.shape == (4, 2) and draws.acquisition is reads
    assert draws.full_well_saturated.shape == (4, 2)


def test_ownership_and_whole_support():
    response = projected()
    assert isinstance(response, DetectorResponse)
    assert not isinstance(response, InstrumentResponse)
    copied = replace(response)
    assert not np.shares_memory(copied.coefficients, response.coefficients)
    with pytest.raises(ValueError, match="entire ground support"):
        replace(response, support_bounds_m=np.tile([-20.0, 21.0, -20.0, 20.0], (2, 1)))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(response, coefficients=-response.coefficients)
    with pytest.raises(ValueError, match="ground axes"):
        replace(response, along_m=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="coefficients must"):
        replace(response, coefficients=np.zeros((1, 1, 1, 1)))
    with pytest.raises(ValueError, match="finite ordered"):
        replace(response, support_bounds_m=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="nonnegative"):
        response_electrons(response, -np.ones((3, 3, 3)))
    with pytest.raises(ValueError, match="lattice"):
        response_electrons(response, np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="positive"):
        scan(speed_m_s=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        scan(along_edges_m=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="support_frame"):
        scan(support_frame="curved")
    with pytest.raises(ValueError, match="strictly increasing"):
        scan(cross_edges_m=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="positive integers"):
        scan(time_order=0)
    with pytest.raises(ValueError, match="positive integers"):
        scan(angular_order=2.5)
    with pytest.raises(ValueError, match="positive integers"):
        scan(cross_order=0)
    with pytest.raises(ValueError, match="unit disk"):
        scan(height_m=1)


def test_primitive_rejects_missing_spectral_support():
    with pytest.raises(ValueError, match="connected"):
        photoelectron_response(
            SpectralGrid("wavenumber_cm1", grid().points),
            scan(),
            acceptance,
            collecting_area_m2=1,
        )
    with pytest.raises(ValueError, match="QE callback"):
        photoelectron_response(
            grid(),
            scan(),
            acceptance,
            collecting_area_m2=1,
            quantum_efficiency=lambda r, w: np.ones(len(w)),
        )
    with pytest.raises(ValueError, match="shape"):
        photoelectron_response(
            grid(), scan(), lambda r, w, a: np.ones((1, 1)), collecting_area_m2=1
        )


def test_record_contract_is_positive_physical_response():
    response = projected()
    assert isinstance(response, DetectorResponse)
    assert response.coefficients.shape == (2, 3, 3, 3)
    assert response.grid == grid()


def test_scan_response_derives_full_ground_support_and_preserves_direct_projection():
    identities = dict(
        sample_ids=("one",),
        pixel_ids=("pixel",),
        setting_ids=("setting",),
        port_ids=("+",),
    )
    trajectory = LinearTrajectory(height_m=1000.0, speed_m_s=100.0, time_origin_s=10.0)
    reads = along_track_read_schedule(
        **identities,
        target_along_m=np.array([30.0]),
        ground_along_offsets_m=np.array([10.0]),
        exposure_s=0.2,
        trajectory=trajectory,
    )
    params = dict(
        target_xy_m=np.array([[30.0, 70.0]]),
        cross_offset_m=np.array([3.0]),
        pixel_bounds_m=np.array([[-1e-5, 1e-5, -1e-5, 1e-5]]),
        along_edges_m=np.array([-20.0, 0.0, 20.0]),
        cross_edges_m=np.array([-20.0, 0.0, 20.0]),
        time_order=2,
        angular_order=2,
        pointing_law=zero_pointing,
    )
    ground_grid = GroundGrid(
        np.array([-30.0, 0.0, 30.0, 60.0, 90.0]), np.array([30.0, 60.0, 90.0, 120.0])
    )
    for support_frame, expected_support in (
        ("ground", np.array([[10.0, 50.0, 50.0, 90.0]])),
        ("platform", np.array([[0.0, 60.0, 50.0, 90.0]])),
    ):
        acquisition = along_track_acquisition(
            reads,
            trajectory,
            **params,
            support_frame=support_frame,
            focal_length_m=1.0,
        )
        ground_xy = acquisition.directions_uv * trajectory.height_m
        ground_xy[..., 0] += trajectory.speed_m_s * (acquisition.times_s - trajectory.time_origin_s)
        ground_xy[..., 1] += 67.0
        kwargs = dict(collecting_area_m2=0.02, throughput=0.5, quantum_efficiency=0.8)
        explicit = detector_response(
            grid(),
            acquisition,
            acceptance,
            ground_xy_m=ground_xy,
            along_m=ground_grid.along_m,
            cross_m=ground_grid.cross_m,
            support_bounds_m=expected_support,
            **kwargs,
        )
        actual = instrument_response(
            spectral_grid=grid(),
            reads=reads,
            trajectory=trajectory,
            optical_factory=lambda current: acceptance,
            **params,
            support_frame=support_frame,
            ground_grid=ground_grid,
            focal_length_m=1.0,
            **kwargs,
        )
        assert type(actual) is InstrumentResponse
        assert type(explicit) is DetectorResponse
        assert not np.shares_memory(actual.coefficients, explicit.coefficients)
        assert not np.shares_memory(actual.along_m, ground_grid.along_m)
        assert not np.shares_memory(actual.coefficients, replace(actual).coefficients)
        np.testing.assert_allclose(actual.coefficients, explicit.coefficients, rtol=0, atol=0)
        np.testing.assert_allclose(actual.support_bounds_m, expected_support, rtol=0, atol=2e-13)
        np.testing.assert_array_equal(actual.reads.exposure_bounds_s, acquisition.exposure_bounds_s)


def test_arline_of_sight_radiance_direct_and_translated_application_and_timing():
    response = projected()
    lattice = GroundGrid(response.along_m, response.cross_m)
    values = np.ones((3, 3, 3))
    scene = SceneRadiance(lattice, response.grid, values)
    values[:] = 2.0
    np.testing.assert_array_equal(
        response_electrons(response, scene.values).total,
        response_electrons(response, np.ones((3, 3, 3))).total,
    )
    # Array application uses response.grid; callers select and align source spectra.
    axis = np.arange(-60.0, 61.0, 20.0)
    maps = np.ones((2, 7, 7))
    spectra = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dense_scene = SceneRadiance(
        GroundGrid(axis, axis), response.grid, np.einsum("cij,ck->ijk", maps, spectra)
    )
    with pytest.raises(ValueError, match="nonnegative"):
        response_electrons(response, -maps, spectra=spectra)
    reads = ReadSchedule(
        tuple(f"sample-{i}" for i in range(8)),
        tuple(f"pixel-{i}" for i in range(8)),
        ("setting",) * 8,
        ("+",) * 4 + ("-",) * 4,
        np.tile([2.0, 2.2], (8, 1)),
    )
    indices = (np.array([1, 5]), np.array([1, 5]))
    for method in ("direct", "correlation"):
        actual = response_electrons(
            response,
            maps,
            spectra=spectra,
            scene_along_m=axis,
            scene_cross_m=axis,
            target_indices=indices,
            reads=reads,
            method=method,
        )
        dense_counts = response_electrons(
            response,
            dense_scene.values,
            scene_along_m=axis,
            scene_cross_m=axis,
            target_indices=indices,
            reads=reads,
            method=method,
        )
        np.testing.assert_allclose(actual.total, dense_counts.total, rtol=1e-14)
    with pytest.raises(ValueError, match="halo"):
        response_electrons(
            response,
            maps,
            spectra=spectra,
            scene_along_m=axis,
            scene_cross_m=axis,
            target_indices=(np.array([0]), indices[1]),
            reads=reads,
        )
    targets = np.array([[30.0, 70.0], [80.0, 90.0]])
    offsets = np.array([[10.0, 3.0], [20.0, -4.0]])
    bounds = along_track_exposure_bounds(
        targets,
        offsets,
        np.array([0.2, 0.4]),
        LinearTrajectory(height_m=1000.0, speed_m_s=100.0, time_origin_s=10.0),
    )
    np.testing.assert_allclose(bounds, [[10.1, 10.3], [10.4, 10.8]])
    with pytest.raises(ValueError, match="positive"):
        along_track_exposure_bounds(
            targets, offsets, 0.0, LinearTrajectory(height_m=1000.0, speed_m_s=100.0)
        )
