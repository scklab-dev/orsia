"""Static dtype relationships and unchanged numerical ownership contracts."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray
from orsia_instruments.acquisition import ReadSchedule, read_schedule
from orsia_instruments.imaging.acquisition import ExposureSequence
from orsia_instruments.optics.filters import gaussian_filter
from orsia_instruments.optics.response import exposure_setting


def test_snapshot_dtype_inference_and_ownership() -> None:
    single = np.array([1, 2], dtype=np.float32)
    double = np.array([1, 2], dtype=np.float64)
    integers = np.array([1, 2], dtype=np.int32)
    single_copy = assert_type(exposure_setting(single, "single"), NDArray[np.float32])
    double_copy = assert_type(exposure_setting(double, "double"), NDArray[np.float64])
    integer_copy = assert_type(exposure_setting(integers, "integers"), NDArray[np.int32])
    scalar_copy = assert_type(exposure_setting(np.float32(1), "scalar"), NDArray[np.float32])
    assert_type(exposure_setting(1, "integer"), NDArray[np.int_])
    assert_type(exposure_setting(1.0, "float"), NDArray[np.float64])
    assert_type(exposure_setting([1, 2], "integers"), NDArray[np.int_])
    assert_type(exposure_setting([1.0, 2.0], "floats"), NDArray[np.float64])
    for source, snapshot in (
        (single, single_copy),
        (double, double_copy),
        (integers, integer_copy),
    ):
        assert snapshot.dtype == source.dtype
        assert snapshot.flags.writeable
        assert not np.shares_memory(source, snapshot)
        np.testing.assert_array_equal(snapshot, source)
    assert scalar_copy.dtype == np.float32
    assert scalar_copy.shape == ()


def test_physical_read_and_exposure_dtype_inference() -> None:
    bounds = np.array([[0, 1]], dtype=np.float32)
    reads = ReadSchedule(("read",), ("pixel",), ("setting",), ("port",), bounds)
    assert_type(reads, ReadSchedule[np.float32])
    assert_type(reads.exposure_bounds_s, NDArray[np.float32])
    assert_type(read_schedule(reads), ReadSchedule[np.float32])
    assert read_schedule(reads) is reads
    assert reads.exposure_bounds_s.dtype == np.float32
    assert not np.shares_memory(bounds, reads.exposure_bounds_s)

    def pointing(times: NDArray[np.float64]) -> NDArray[np.float32]:
        return np.zeros((len(times), 2), dtype=np.float32)

    sequence = ExposureSequence(bounds, (-0.1, 0.1, -0.1, 0.1), pointing)
    assert_type(sequence, ExposureSequence[np.float32])
    assert_type(sequence.bounds_s, NDArray[np.float32])
    assert sequence.bounds_s.dtype == np.float32
    assert not np.shares_memory(bounds, sequence.bounds_s)


def test_real_scalar_and_array_inputs_keep_existing_conversion() -> None:
    scalar = gaussian_filter(np.float32(2), np.int32(2), np.float64(0.1))
    assert_type(scalar, NDArray[np.float64])
    assert np.asarray(scalar).dtype == np.float64
    vector = gaussian_filter(np.array([1, 2], dtype=np.float32), 2, 0.1)
    assert vector.dtype == np.float64
