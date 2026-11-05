"""Detector-owned active geometry, stable maps and explicit factory contracts."""

import inspect
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from orsia_instruments import (
    DetectorProperties,
    GaussianPSF,
    dispersive_spectrometer,
    filter_camera,
    filter_strip,
    frame_response,
    regular_detector,
    regular_pixel_bounds,
    simulate_frames,
)
from orsia_instruments.imaging.acquisition import regular_pixel_bounds as legacy_bounds


@pytest.fixture
def properties():
    return dict(
        quantum_efficiency=0.7,
        dark_electrons_per_s=2,
        read_noise_electrons=3,
        full_well_electrons=1e6,
        electrons_per_dn=2,
        bias_dn=100,
        nonlinearity=0.01,
        adc_bits=16,
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (2, 1), (2, 3)])
def test_regular_detector_retains_explicit_pitch_shape_and_active_gaps(shape, properties):
    detector = regular_detector(
        shape,
        pitch_m=(4.0, 6.0),
        active_size_m=(2.0, 3.0),
        centre_xy_m=(10.0, 20.0),
        **properties,
    )
    assert detector.shape == shape
    assert detector.pitch_m == (4.0, 6.0)
    expected = regular_pixel_bounds(
        shape, pitch_m=(4.0, 6.0), active_size_m=(2.0, 3.0), centre_xy_m=(10.0, 20.0)
    )
    np.testing.assert_array_equal(detector.pixel_bounds_m, expected)
    assert not detector.pixel_bounds_m.flags.writeable
    scalar = regular_detector(
        shape, pitch_m=4.0, active_size_m=2.0, centre_xy_m=(0, 0), **properties
    )
    assert scalar.pitch_m == (4.0, 4.0)


def test_irregular_detector_snapshots_geometry_and_property_maps(properties):
    pixels = np.array(
        [
            [[-5.0, -4.0, -3.0, -2.0], [0.0, 2.0, -3.0, -1.0]],
            [[-4.0, -3.0, 0.0, 1.0], [1.0, 4.0, 3.0, 5.0]],
        ]
    )
    values = np.full((2, 2), 0.4, dtype=np.float32)
    detector = DetectorProperties(
        pixel_bounds_m=pixels, pitch_m=None, **(properties | {"quantum_efficiency": values})
    )
    expected = pixels.copy()
    pixels[:] = 0
    values[:] = 1
    assert detector.shape == (2, 2) and detector.pitch_m is None
    np.testing.assert_array_equal(detector.pixel_bounds_m, expected)
    assert not np.shares_memory(detector.pixel_bounds_m, pixels)
    assert np.asarray(detector.quantum_efficiency).dtype == np.float32
    assert not np.shares_memory(np.asarray(detector.quantum_efficiency), values)
    for name in properties:
        if name != "adc_bits":
            value = np.asarray(getattr(detector, name))
            assert not value.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                value[...] = 0
    with pytest.raises(ValueError, match="read-only"):
        detector.pixel_bounds_m[:] = 0


@pytest.mark.parametrize(
    "pixels",
    [
        np.empty((0, 1, 4)),
        np.empty((1, 0, 4)),
        np.ones((1, 4)),
        np.ones((1, 1, 3)),
        np.zeros((1, 1, 4)),
        np.array([[[1.0, 0.0, 0.0, 1.0]]]),
        np.array([[[0.0, 1.0, 1.0, 0.0]]]),
        np.array([[[0.0, np.inf, 0.0, 1.0]]]),
        np.full((1, 1, 4), np.nan),
    ],
)
def test_invalid_active_rectangles_fail_at_detector_construction(pixels, properties):
    with pytest.raises(ValueError, match="finite ordered"):
        DetectorProperties(pixel_bounds_m=pixels, pitch_m=None, **properties)


@pytest.mark.parametrize(
    "pitch", [(0.0, 1.0), (1.0, -1.0), (np.nan, 1.0), (1.0, np.inf), (1.0,), 1.0]
)
def test_detector_requires_explicit_positive_xy_pitch_or_none(pitch, properties):
    with pytest.raises(ValueError, match="pitch"):
        DetectorProperties(
            pixel_bounds_m=regular_pixel_bounds((2, 3), pitch_m=1),
            pitch_m=pitch,
            **properties,
        )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("coordinate", [0, 1])
def test_regular_detector_rejects_incoherent_centre_spacing(axis, coordinate, properties):
    pixels = regular_pixel_bounds((2, 3), pitch_m=(2.0, 3.0), active_size_m=1.0)
    selected = (1, slice(None)) if axis == 0 else (slice(None), 1)
    for edge in (2 * coordinate, 2 * coordinate + 1):
        pixels[(*selected, edge)] += 0.1
    with pytest.raises(ValueError, match="spacing"):
        DetectorProperties(pixel_bounds_m=pixels, pitch_m=(2.0, 3.0), **properties)
    assert DetectorProperties(pixel_bounds_m=pixels, pitch_m=None, **properties).shape == (2, 3)


def test_detector_rejects_active_sizes_larger_than_pitch(properties):
    with pytest.raises(ValueError, match="active dimensions"):
        DetectorProperties(
            pixel_bounds_m=regular_pixel_bounds((1, 1), pitch_m=2),
            pitch_m=(1.0, 2.0),
            **properties,
        )


@pytest.mark.parametrize(
    "name",
    [
        "quantum_efficiency",
        "dark_electrons_per_s",
        "read_noise_electrons",
        "full_well_electrons",
        "electrons_per_dn",
        "bias_dn",
        "nonlinearity",
    ],
)
@pytest.mark.parametrize("shape", [(1, 3), (2, 1), (3, 2)])
def test_property_maps_must_match_detector_shape_exactly(name, shape, properties):
    with pytest.raises(ValueError, match="map must match"):
        DetectorProperties(
            pixel_bounds_m=regular_pixel_bounds((2, 3), pitch_m=1),
            pitch_m=None,
            **(properties | {name: np.full(shape, 0.5)}),
        )


def test_regular_factory_requires_every_argument(properties):
    arguments: dict[str, Any] = dict(
        shape=(1, 1), pitch_m=1.0, active_size_m=0.8, centre_xy_m=(0.0, 0.0), **properties
    )
    parameters = inspect.signature(regular_detector).parameters
    assert set(parameters) == set(arguments)
    for name, parameter in parameters.items():
        assert parameter.default is inspect.Parameter.empty
        with pytest.raises(TypeError, match=name):
            regular_detector(**{key: value for key, value in arguments.items() if key != name})


def test_instrument_factories_share_attached_detector_and_legacy_bounds_path(properties):
    detector = regular_detector(
        (2, 3), pitch_m=20e-6, active_size_m=16e-6, centre_xy_m=(0.0, 0.0), **properties
    )
    camera = filter_camera(
        detector,
        [filter_strip(0, 2, band="a", transmission=np.ones_like)],
        focal_length_m=0.3,
        f_number=4.0,
        spatial=GaussianPSF(4e-6),
    )
    spectrometer = dispersive_spectrometer(
        detector,
        slit_bounds_m=np.array([-5e-6, 5e-6, -20e-6, 20e-6]),
        focal_length_m=0.3,
        collecting_area_m2=0.004,
        spatial=GaussianPSF(4e-6),
        reference_m=2e-6,
        dispersion=200.0,
    )
    for instrument in (camera, spectrometer):
        assert instrument.detector is detector
        assert instrument.pixel_bounds_m is detector.pixel_bounds_m
        calibrated = replace(instrument, detector=replace(detector, quantum_efficiency=0.5))
        assert calibrated.detector is not detector
        assert float(np.asarray(detector.quantum_efficiency)) == 0.7
    assert legacy_bounds is regular_pixel_bounds
    for function in (filter_camera, dispersive_spectrometer):
        parameter = inspect.signature(function).parameters["detector"]
        assert parameter.default is inspect.Parameter.empty
    for function in (frame_response, frame_response, simulate_frames):
        assert "detector" not in inspect.signature(function).parameters
