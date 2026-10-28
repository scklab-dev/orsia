"""Consumer ownership and finite physical support after removing the copy wrapper."""

from dataclasses import replace

import numpy as np
import pytest
from orsia_instruments import (
    DetectorMaps,
    DetectorOperatingPoints,
    ExpectedElectrons,
    ReadSchedule,
    exposure_sequence,
    image_acquisition,
    image_charge,
    image_readout,
    regular_pixel_bounds,
)
from orsia_instruments.optics.dispersion import polynomial_image_map
from orsia_instruments.optics.response import exposure_setting, spatial_offset, spectral_value


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["strided", "fortran", "readonly"])
def test_physical_read_copies_preserve_dtype_and_ownership(dtype, layout):
    source = np.arange(8, dtype=dtype).reshape(2, 4)[:, ::2]
    if layout == "fortran":
        source = np.asfortranarray(source)
    elif layout == "readonly":
        source.setflags(write=False)
    expected = source.copy()
    reads = ReadSchedule(("a", "b"), ("p", "p"), ("s", "s"), ("+", "+"), source)
    retained = reads.exposure_bounds_s
    np.testing.assert_array_equal(retained, expected)
    assert retained.dtype == source.dtype
    assert retained.flags.writeable
    assert not np.shares_memory(retained, source)
    retained[0, 0] = -1
    np.testing.assert_array_equal(source, expected)
    if source.flags.writeable:
        source[:] = 100
        assert retained[1, 0] == expected[1, 0]


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_charge_and_settings_use_numpy_copies(dtype):
    source = np.arange(12, dtype=dtype).reshape(1, 2, 6)[:, :, ::2]
    expected = source.copy()
    charge = image_charge({"source": source}, exposure_bounds_s=np.array([[0, 1]]))
    source[:] = 100
    retained = charge.signal.contributions["source"]
    assert retained.dtype == dtype
    assert retained.flags.writeable
    np.testing.assert_array_equal(retained, expected.ravel())
    assert not np.shares_memory(retained, source)
    settings = np.array([2], dtype=dtype)
    described = replace(charge, settings={"opd_m": settings})
    settings[:] = 4
    np.testing.assert_array_equal(described.settings["opd_m"], [2])
    assert np.asarray(described.settings["opd_m"]).dtype == dtype
    measured = image_readout(described)
    assert measured.photoelectrons.dtype == charge.signal.total.dtype
    assert measured.photoelectrons.flags.writeable
    with pytest.raises(ValueError, match="finite"):
        replace(charge, settings={"opd_m": np.nan})


@pytest.mark.parametrize("value", [2, [2, 3], np.array([2, 3], dtype=np.float32)])
def test_scalar_vector_settings_preserve_dtype(value):
    retained = exposure_setting(value, "opd")
    np.testing.assert_array_equal(retained, value)
    assert retained.dtype == np.asarray(value).dtype
    assert retained.flags.writeable
    assert not np.shares_memory(retained, np.asarray(value))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_physical_validation_remains_at_consumers(bad):
    reads = ReadSchedule(("a",), ("p",), ("s",), ("+",), np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        replace(reads, exposure_bounds_s=np.array([[0.0, bad]]))
    with pytest.raises(ValueError, match="finite"):
        ExpectedElectrons(reads, {"source": np.array([bad])})
    with pytest.raises(ValueError, match="finite"):
        exposure_setting(bad, "opd")
    with pytest.raises(ValueError, match="finite"):
        regular_pixel_bounds((1, 1), pitch_m=1, centre_xy_m=[bad, 0])
    with pytest.raises(ValueError, match="finite"):
        DetectorMaps(bias_dn=bad)
    zeros, ones = np.zeros(1), np.ones(1)
    with pytest.raises(ValueError, match="finite"):
        DetectorOperatingPoints(zeros, zeros, ones, ones, np.array([bad]), zeros)


def test_detector_maps_and_constant_pointing_preserve_dtype():
    source = np.array([[1, 2]], dtype=np.float32)
    detector = DetectorMaps(bias_dn=source)
    source[:] = 0
    retained = np.asarray(detector.bias_dn)
    assert retained.dtype == np.float32 and retained.flags.writeable
    np.testing.assert_array_equal(retained, [[1, 2]])
    assert not np.shares_memory(retained, source)
    pointing = np.zeros(2, dtype=np.float32)
    sequence = exposure_sequence(
        start_s=0,
        duration_s=1,
        pointing_uv=pointing,
        direction_bounds_uv=(-0.1, 0.1, -0.1, 0.1),
    )
    pointing[:] = 1
    retained_pointing = sequence.pointing(np.array([0.5]))
    assert retained_pointing.dtype == np.float32
    np.testing.assert_array_equal(retained_pointing, [[0, 0]])


def test_image_acquisition_snapshots_strided_inputs():
    exposures = np.arange(8.0).reshape(2, 4)[:, ::2]
    pixels = regular_pixel_bounds((2, 1), pitch_m=0.01)
    offsets = np.array([0.0, 99.0, 0.25, 99.0])[::2]
    expected_exposures, expected_pixels = exposures.copy(), pixels.copy()
    acquisition = image_acquisition(
        exposure_bounds_s=exposures,
        setting_ids=("a", "b"),
        pixel_bounds_m=pixels,
        row_offsets_s=offsets,
        direction_bounds_uv=(-0.1, 0.1, -0.1, 0.1),
        pointing=lambda t: np.zeros((len(t), 2)),
        focal_length_m=1.0,
        time_order=1,
        angular_order=1,
    )
    exposures.fill(99)
    pixels.fill(99)
    offsets.fill(99)
    for i in range(4):
        setting, row = divmod(i, 2)
        np.testing.assert_array_equal(
            acquisition.exposure_bounds_s[i], expected_exposures[setting] + row * 0.25
        )
        np.testing.assert_array_equal(acquisition.pixel_bounds_m[i], expected_pixels[row, 0])
    assert acquisition.exposure_bounds_s.flags.writeable
    assert not np.shares_memory(acquisition.exposure_bounds_s, exposures)
    assert not np.shares_memory(acquisition.pixel_bounds_m, pixels)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_polynomial_mapping_owns_floating_output_for_dtype_preserving_geometry(dtype):
    image = np.array([[1, 99, 2, 99]], dtype=dtype)[:, ::2]
    image.setflags(write=False)
    centres = polynomial_image_map(
        image, np.array([1.0, 1.5]), reference_m=1.0, dispersion=0.5, keystone=0.25
    )
    np.testing.assert_array_equal(centres, [[[1.0, 2.0], [1.25, 2.125]]])
    assert centres.dtype == np.float64 and centres.flags.writeable
    assert not np.shares_memory(centres, image)
    np.testing.assert_array_equal(image, [[1, 2]])


def test_optical_callbacks_borrow_arrays_and_keep_shape_and_finiteness_checks():
    wave = np.array([1.0, 2.0])
    values = np.array([0.2, 99.0, 0.4, 99.0])[::2]
    values.setflags(write=False)
    assert spectral_value(lambda w: values, wave, "transmission") is values
    assert spectral_value(0.5, wave, "transmission").shape == ()
    assert spatial_offset(lambda x, y: x + y, 0.25, 0.5, "offset") == 0.75
    with pytest.raises(ValueError, match="shape"):
        spectral_value(lambda w: values[:, None], wave, "transmission")
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            spatial_offset(lambda x, y, value=bad: value, 0.0, 0.0, "offset")
