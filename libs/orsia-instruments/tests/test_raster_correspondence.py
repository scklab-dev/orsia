"""Retained raster identity, read-axis selections and estimated electronic correction."""

from dataclasses import replace

import numpy as np
import pytest
from orsia_instruments import (
    DetectorProperties,
    ImageMeasurements,
    NativeBandResponse,
    calibrate_frames,
    detector_operating_points,
    image_charge,
    image_readout,
    raster_read_schedule,
    regular_pixel_bounds,
    select_raster_reads,
)
from orsia_radiative import SpectralDomain, SpectralGrid


def detector():
    return DetectorProperties(
        regular_pixel_bounds((2, 3), pitch_m=1.0),
        None,
        quantum_efficiency=1.0,
        dark_electrons_per_s=0.0,
        read_noise_electrons=0.0,
        full_well_electrons=1e6,
        electrons_per_dn=1.0,
        bias_dn=0.0,
        nonlinearity=0.0,
        adc_bits=16,
    )


def test_rolling_snapshot_selection_and_dense_boundary():
    bounds = np.array([[0.0, 1.0], [2.0, 4.0]])
    offsets = np.array([0.0, 0.25])
    schedule = raster_read_schedule(
        bounds, (2, 3), row_offsets_s=offsets, setting_ids=("a", "b"), port="+"
    )
    bounds.fill(99)
    offsets.fill(99)
    assert schedule.complete and schedule.native_shape == (2, 2, 3)
    for f, r, c in schedule.native_indices:
        index = f * 6 + r * 3 + c
        np.testing.assert_array_equal(
            schedule.reads.exposure_bounds_s[index],
            np.array([[0.0, 1.0], [2.0, 4.0]])[f] + r * 0.25,
        )
    for values in (schedule.native_indices, schedule.reads.exposure_bounds_s):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    for indices in (np.array([0, 4, 7, 11]), np.roll(np.arange(12), 3)):
        selected = select_raster_reads(schedule, indices)
        assert not selected.complete and selected.native_shape == schedule.native_shape
        assert selected.reads.sample_ids == tuple(schedule.reads.sample_ids[i] for i in indices)
        np.testing.assert_array_equal(selected.native_indices, schedule.native_indices[indices])
        with pytest.raises(ValueError, match="shape"):
            image_charge({"dense": np.ones(schedule.native_shape)}, schedule=selected)
        charge = image_charge(
            {"selected": np.arange(len(indices), dtype=np.float64)}, schedule=selected
        )
        electronics = detector()
        images = image_readout(charge, detector=electronics)
        assert images.photoelectrons.shape == (len(indices),)
        assert images.schedule is selected


def test_selected_estimated_detector_equation_and_measurement_identity():
    schedule = select_raster_reads(
        raster_read_schedule(
            np.array([[0.0, 1.0], [2.0, 4.0]]), (2, 3), row_offsets_s=np.array([0.0, 0.25])
        ),
        np.array([0, 4, 7, 11]),
    )
    true = replace(detector(), bias_dn=100.0, read_noise_electrons=1.0)
    images = image_readout(
        image_charge({"scene": np.array([30.0, 40.0, 50.0, 60.0])}, schedule=schedule),
        detector=true,
        rng=np.random.default_rng(187),
        realisations=3,
    )
    assert images.dn is not None
    dn = images.dn.copy()
    estimated = replace(
        true,
        pixel_bounds_m=true.pixel_bounds_m + [2.0, 2.0, 3.0, 3.0],
        electrons_per_dn=np.arange(6.0).reshape(2, 3) + 1.0,
        bias_dn=np.arange(6.0).reshape(2, 3) + 10.0,
        dark_electrons_per_s=np.arange(6.0).reshape(2, 3) + 0.5,
        quantum_efficiency=0.01,
    )
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.array([4000.0, 5000.0]),
        SpectralDomain("wavenumber_cm1", ((4000.0, 5000.0),)),
    )
    response = NativeBandResponse(
        schedule,
        grid,
        ("band",) * 4,
        np.full((4, 2), 0.5),
        np.ones((4, 1)),
        schedule.reads.exposure_bounds_s.mean(axis=1)[:, None],
        np.zeros((4, 1, 2)),
    )
    calibrated = calibrate_frames(images, response, detector=estimated)
    _, rows, columns = schedule.native_indices.T
    points = detector_operating_points(estimated, rows=rows, columns=columns)
    duration = np.diff(schedule.reads.exposure_bounds_s, axis=1).ravel()
    expected = (
        points.electrons_per_dn * (dn - points.bias_dn) - points.dark_electrons_per_s * duration
    )
    np.testing.assert_array_equal(calibrated.values, expected)
    np.testing.assert_array_equal(images.dn, dn)
    assert calibrated.valid.all()
    bad_schedule = replace(
        schedule,
        reads=replace(
            schedule.reads, sample_ids=tuple("other-" + s for s in schedule.reads.sample_ids)
        ),
    )
    with pytest.raises(ValueError, match="identities"):
        calibrate_frames(images, replace(response, schedule=bad_schedule), detector=estimated)
    bad_schedule = replace(
        schedule,
        reads=replace(schedule.reads, exposure_bounds_s=schedule.reads.exposure_bounds_s + 0.1),
    )
    with pytest.raises(ValueError, match="timing"):
        calibrate_frames(images, replace(response, schedule=bad_schedule), detector=estimated)


def test_measurement_direct_construction_snapshots_all_arrays_and_mappings():
    schedule = raster_read_schedule(np.array([[0.0, 1.0]]), (1, 2))
    source = np.array([[[2.0, 3.0]]])
    setting = np.array([0.5])
    settings = {"position_m": setting}
    contributions = {"scene": source}
    measurements = ImageMeasurements(source, schedule, None, None, None, settings, contributions)
    source.fill(99)
    setting.fill(99)
    settings.clear()
    contributions.clear()
    np.testing.assert_array_equal(measurements.photoelectrons, [[[2.0, 3.0]]])
    np.testing.assert_array_equal(measurements.contributions["scene"], [[[2.0, 3.0]]])
    np.testing.assert_array_equal(measurements.settings["position_m"], [0.5])
    assert not np.shares_memory(measurements.photoelectrons, measurements.contributions["scene"])
    assert not np.shares_memory(measurements.contributions["scene"], source)
    with pytest.raises(TypeError):
        measurements.settings["other"] = setting  # pyrefly: ignore[unsupported-operation]


def test_nonlinearity_is_checked_only_on_reads_selected_for_correction():
    schedule = select_raster_reads(
        raster_read_schedule(np.array([[0.0, 1.0]]), (1, 2)), np.array([0])
    )
    electronics = DetectorProperties(
        regular_pixel_bounds((1, 2), pitch_m=1.0),
        None,
        quantum_efficiency=1.0,
        dark_electrons_per_s=0.0,
        read_noise_electrons=0.0,
        full_well_electrons=1e6,
        electrons_per_dn=1.0,
        bias_dn=0.0,
        nonlinearity=np.array([[0.0, 0.1]]),
        adc_bits=16,
    )
    measurements = ImageMeasurements(
        np.array([10.0]),
        schedule,
        np.array([[10]]),
        np.array([[False]]),
        np.array([[False]]),
        {},
        {},
    )
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.array([4000.0, 5000.0]),
        SpectralDomain("wavenumber_cm1", ((4000.0, 5000.0),)),
    )
    response = NativeBandResponse(
        schedule,
        grid,
        ("band",),
        np.full((1, 2), 0.5),
        np.ones((1, 1)),
        np.array([[0.5]]),
        np.zeros((1, 1, 2)),
    )
    corrected = calibrate_frames(measurements, response, detector=electronics)
    np.testing.assert_array_equal(corrected.values, [[10.0]])
    with pytest.raises(ValueError, match="nonlinear"):
        calibrate_frames(measurements, response, detector=replace(electronics, nonlinearity=0.1))
