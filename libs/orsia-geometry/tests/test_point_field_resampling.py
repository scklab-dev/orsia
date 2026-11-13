"""Explicit coordinate interpolation, conservative endpoint masks and continuation."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import LocalCartesianFrame, PointLattice, resample_point_field

FRAME = LocalCartesianFrame("resampling test", "synthetic altitude")


def test_nonuniform_coordinates_and_signed_trailing_fields():
    source = PointLattice(np.array([-2.0, 0.0, 3.0]), np.array([-1.0, 4.0]), FRAME)
    target = PointLattice(np.array([-1.0, 1.5]), np.array([0.0, 1.0]), FRAME)
    x, y = np.meshgrid(source.x_m, source.y_m)
    values = np.stack((1 + x + 2 * y + 0.1 * x * y, -3 + x - y), axis=-1)
    actual, valid = resample_point_field(values, source, target, boundary="reject")
    x, y = np.meshgrid(target.x_m, target.y_m)
    expected = np.stack((1 + x + 2 * y + 0.1 * x * y, -3 + x - y), axis=-1)
    assert_allclose(actual, expected, rtol=2e-15, atol=4 * np.finfo(float).eps)
    assert valid.all() and actual.flags.writeable and valid.flags.writeable
    assert not np.shares_memory(actual, values)
    assert (actual[..., 1] < 0).all()


def test_nearest_is_explicit_and_masks_include_zero_weight_endpoints():
    source = PointLattice(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0]), FRAME)
    target = PointLattice(np.array([-1.0, 1.0, 3.0]), np.array([-1.0, 0.0, 3.0]), FRAME)
    values = source.xy_m[..., 0] + source.xy_m[..., 1]
    with pytest.raises(ValueError, match="outside"):
        resample_point_field(values, source, target, boundary="reject")
    actual, valid = resample_point_field(values, source, target, boundary="nearest")
    expected = np.clip(target.x_m, 0, 2)[None, :] + np.clip(target.y_m, 0, 2)[:, None]
    assert_array_equal(actual, expected)
    assert valid.all()
    mask = np.ones(source.shape, dtype=bool)
    mask[1, 2] = False
    exact = PointLattice(np.array([1.0]), np.array([0.0]), FRAME)
    # Conservative bounded bracket includes the zero-weight upper endpoints.
    actual, valid = resample_point_field(
        values, source, exact, boundary="reject", source_valid=mask
    )
    assert not valid.any()
    assert_array_equal(actual, 0)


def test_singleton_axes_and_frame_rejection():
    source = PointLattice(np.array([1.0]), np.array([2.0]), FRAME)
    target = PointLattice(np.array([-1.0, 1.0, 4.0]), np.array([0.0, 2.0]), FRAME)
    values = np.array([[[3.0, -1.0]]])
    actual, valid = resample_point_field(values, source, target, boundary="nearest")
    assert_array_equal(actual, np.broadcast_to(values, (*target.shape, 2)))
    assert valid.all()
    with pytest.raises(ValueError, match="outside"):
        resample_point_field(values, source, target, boundary="reject")
    with pytest.raises(ValueError, match="frame"):
        resample_point_field(
            values,
            source,
            replace(target, frame=LocalCartesianFrame(FRAME.name, FRAME.vertical_datum)),
            boundary="nearest",
        )


@pytest.mark.parametrize("kind", ["shape", "nan", "mask_shape", "mask_dtype", "boundary"])
def test_invalid_inputs(kind):
    source = PointLattice(np.arange(3.0), np.arange(2.0), FRAME)
    values = np.ones(source.shape)
    mask = np.ones(source.shape, dtype=bool)
    boundary = "reject"
    if kind == "shape":
        values = values[:1]
    if kind == "nan":
        values[0, 0] = np.nan
    if kind == "mask_shape":
        mask = mask[:1]
    if kind == "mask_dtype":
        mask = np.ones(source.shape)
    if kind == "boundary":
        boundary = "implicit"
    with pytest.raises(ValueError):
        resample_point_field(
            values, source, source, boundary=cast(Any, boundary), source_valid=cast(Any, mask)
        )
