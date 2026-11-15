"""Finite Fourier values, explicit physical supports and reconstruction policies."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import LocalCartesianFrame, PointLattice, resample_point_field
from orsia_instruments.response.spatial import (
    form_detector_images,
    form_detector_images_with_tangents,
    gaussian_pixel_transfer,
    gaussian_sigma_from_mtf,
)

FRAME = LocalCartesianFrame("Fourier test", "synthetic surface")


def grids():
    source = PointLattice.regular((32, 34), (1.0, 1.0), (-16.5, -15.5), FRAME)
    target = PointLattice.regular((6, 7), (1.0, 1.0), (-3.5, -2.5), FRAME)
    return source, target


def sample(values, source, target, displacement=None, **changes):
    offsets = [np.zeros((target.shape[0], 2, 2))] if displacement is None else displacement
    options: dict[str, Any] = dict(
        detector_pitch_xy_m=(1.0, 1.0), pixel_size_xy_m=(1.0, 1.0), negative_policy="preserve"
    )
    return form_detector_images(
        values, np.ones(source.shape, dtype=bool), source, target, offsets, **(options | changes)
    )


def test_integer_samples_match_independent_inverse_fft():
    source, target = grids()
    values = np.ones((*source.shape, 1))
    signal, valid = sample(values, source, target)
    assert valid[0].all()
    sigma = np.sqrt(-2 * np.log(0.30)) / np.pi
    padding = int(np.ceil(np.sqrt(-2 * np.log(1e-6)) * sigma + 0.5)) + 2
    padded = np.pad(values[..., 0], padding)
    fr = np.fft.fftfreq(padded.shape[0])[:, None]
    fc = np.fft.fftfreq(padded.shape[1])[None, :]
    transfer = np.exp(-2 * np.pi**2 * sigma**2 * (fr**2 + fc**2)) * np.sinc(fr) * np.sinc(fc)
    oracle = np.fft.ifft2(np.fft.fft2(padded) * transfer).real
    selected = oracle[padding + 13 : padding + 19, padding + 13 : padding + 20]
    assert_allclose(signal[0], np.repeat(selected[..., None], 2, axis=2), atol=2e-14)


def test_signed_ringing_explicit_clip_and_physical_rejection():
    source = PointLattice.regular((32, 32), (1.0, 1.0), (-16.0, -16.0), FRAME)
    target = PointLattice(np.array([1.5]), np.array([0.0]), FRAME)
    values = np.zeros((*source.shape, 1))
    values[16, 16, 0] = 1
    options = dict(optical_mtf=1.0, pixel_size_xy_m=None)
    signed, _ = sample(values, source, target, **options)
    clipped, _ = sample(values, source, target, **options, negative_policy="clip")
    assert signed[0][0, 0, 0] < -0.1
    assert_array_equal(clipped[0], 0)
    with pytest.raises(ValueError, match="materially negative"):
        sample(values, source, target, **options, negative_policy="reject")
    with pytest.raises(ValueError, match="nonnegative supported"):
        sample(-values, source, target, **options, negative_policy="reject")
    opposite, _ = sample(-values, source, target, **options)
    assert_allclose(opposite[0], -signed[0], rtol=1e-15)
    opposite_clip, _ = sample(-values, source, target, **options, negative_policy="clip")
    assert np.any(opposite_clip[0] + clipped[0] > 0)  # Clipping has no linear/transpose claim.
    integer = PointLattice(np.array([1.0]), np.array([0.0]), FRAME)
    roundoff, _ = sample(values, source, integer, **options, negative_policy="reject")
    assert np.all(roundoff[0] >= 0) and np.max(roundoff[0]) < 1e-14


def test_coordinate_order_nonuniform_output_and_row_displacements():
    source = PointLattice.regular((32, 32), (1.0, 1.0), (-16.0, -16.0), FRAME)
    target = PointLattice(np.array([-1.0, 0.0, 3.0]), np.array([0.0, 2.0]), FRAME)
    x, y = np.meshgrid(source.x_m, source.y_m)
    values = (100 + x + 10 * y)[..., None]
    offsets = [np.array([[[1.0, 2.0]], [[-2.0, 1.0]]])]
    result, valid = sample(values, source, target, offsets, optical_mtf=1.0, pixel_size_xy_m=None)
    expected = (
        100
        + target.x_m[None, :]
        + offsets[0][:, 0, 0, None]
        + 10 * (target.y_m[:, None] + offsets[0][:, 0, 1, None])
    )
    assert valid[0].all()
    assert_allclose(result[0][..., 0], expected, rtol=3e-15, atol=1e-12)
    with pytest.raises(ValueError, match="frame"):
        sample(
            values,
            source,
            replace(target, frame=LocalCartesianFrame(FRAME.name, FRAME.vertical_datum)),
        )


def test_mask_and_finite_boundary_rule():
    source, target = grids()
    values = np.ones((*source.shape, 1))
    valid = np.ones(source.shape, dtype=bool)
    valid[15, 16] = False
    _, mask = form_detector_images(
        values,
        valid,
        source,
        target,
        [np.zeros((6, 1, 2))],
        detector_pitch_xy_m=(1.0, 1.0),
        pixel_size_xy_m=(1.0, 1.0),
        negative_policy="preserve",
    )
    assert not mask[0].all()
    outside = PointLattice(target.x_m + 1000, target.y_m, FRAME)
    result, mask = sample(values, source, outside)
    assert not mask[0].any()
    assert_array_equal(result[0], 0)


def test_historical_anchor_combination_is_explicit_grid_math():
    source = PointLattice.regular((6, 7), (1.0, 1.0), (-3.0, -2.5), FRAME)
    target = PointLattice.regular((8, 9), (1.0, 1.0), (-4.0, -3.5), FRAME)
    coefficients, support = resample_point_field(
        np.ones((6, 7, 1, 2)),
        source,
        target,
        boundary="nearest",
        source_valid=np.ones((6, 7), dtype=bool),
    )
    anchors = np.broadcast_to([0.2, 0.3], (*target.shape, 2))
    summed = np.einsum("yxba,yxa->yxb", coefficients, anchors)
    # Only this historical reference assumes two identical reads in supplied sums.
    one_read = summed / 2
    assert support.all()
    assert_array_equal(one_read, 0.25)


@pytest.mark.parametrize(
    "change",
    [
        {"optical_mtf": 0.0},
        {"tail_tolerance": 1.0},
        {"fourier_backend": "invalid"},
        {"negative_policy": "implicit"},
        {"detector_pitch_xy_m": (0.0, 1.0)},
        {"pixel_size_xy_m": (1.0, np.inf)},
    ],
)
def test_invalid_controls(change):
    source, target = grids()
    with pytest.raises(ValueError):
        sample(np.ones((*source.shape, 1)), source, target, **change)


def test_invalid_fields_grids_and_displacements():
    source, target = grids()
    values = np.ones((*source.shape, 1))
    irregular = replace(source, x_m=source.x_m + np.linspace(0, 1, source.x_m.size) ** 2)
    singleton = PointLattice(np.zeros(1), source.y_m, FRAME)
    for current, data in ((irregular, values), (singleton, values[:, :1])):
        with pytest.raises(ValueError, match="source axes"):
            sample(data, current, target)
    for offsets in ([np.zeros((5, 1, 2))], [np.full((6, 1, 2), np.nan)]):
        with pytest.raises(ValueError, match="displacements"):
            sample(values, source, target, offsets)
    with pytest.raises(ValueError, match="source/support"):
        sample(np.full(values.shape, np.nan), source, target)
    with pytest.raises(ValueError, match="too small"):
        sample(values, source, target, detector_pitch_xy_m=(100.0, 100.0))
    with pytest.raises(ValueError, match="source/support"):
        form_detector_images(
            values,
            cast(Any, np.ones(source.shape)),
            source,
            target,
            [np.zeros((6, 1, 2))],
            detector_pitch_xy_m=(1.0, 1.0),
            pixel_size_xy_m=None,
            negative_policy="preserve",
        )


def test_optional_backend_plumbing_without_claiming_native_qualification(monkeypatch):
    calls = []

    class Plan:
        def __init__(self, kind, shape, **options):
            assert kind == 2 and options == dict(
                eps=1e-12, isign=1, nthreads=1, modeord=1, dtype="complex128"
            )
            self.shape = shape
            calls.append(self)

        def setpts(self, y, x):
            self.y, self.x = y, x

        def execute(self, coefficients):
            fy = np.fft.fftfreq(self.shape[0]) * self.shape[0]
            fx = np.fft.fftfreq(self.shape[1]) * self.shape[1]
            return np.sum(
                (np.exp(1j * self.y[:, None] * fy) @ coefficients)
                * np.exp(1j * self.x[:, None] * fx),
                axis=1,
            )

    source, target = grids()
    values = np.random.default_rng(123).normal(size=(*source.shape, 1))
    expected, mask = sample(values, source, target)
    monkeypatch.setitem(sys.modules, "finufft", SimpleNamespace(Plan=Plan))
    actual, actual_mask = sample(values, source, target, fourier_backend="nufft")
    assert len(calls) == 1
    assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-12)
    assert_array_equal(actual_mask[0], mask[0])
    tangents = np.stack((-values, 2 * values), axis=-1)
    actual, derivative, tangent_mask = form_detector_images_with_tangents(
        values,
        tangents,
        np.ones(source.shape, dtype=bool),
        source,
        target,
        [np.zeros((target.shape[0], 2, 2))],
        detector_pitch_xy_m=(1.0, 1.0),
        pixel_size_xy_m=(1.0, 1.0),
        fourier_backend="nufft",
    )
    assert len(calls) == 2  # A fresh plan per call, shared by the mean and both directions.
    assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-12)
    assert_allclose(derivative[0][..., 0], -expected[0], rtol=1e-11, atol=1e-12)
    assert_allclose(derivative[0][..., 1], 2 * expected[0], rtol=1e-11, atol=1e-12)
    assert_array_equal(tangent_mask[0], mask[0])
    monkeypatch.setitem(sys.modules, "finufft", None)
    with pytest.raises(ImportError):
        sample(values, source, target, fourier_backend="nufft")


def test_shared_transfer_domains():
    assert gaussian_sigma_from_mtf(1.0) == 0
    assert_allclose(
        gaussian_pixel_transfer(
            np.array([0.0, 0.5]),
            np.array([0.0, 0.5]),
            optical_mtf=0.3,
            pitch_ratio_yx=(1.0, 1.0),
            pixel_width_yx_samples=None,
        ),
        [[1.0, 0.3], [0.3, 0.09]],
        rtol=2e-15,
    )
    with pytest.raises(ValueError):
        gaussian_sigma_from_mtf(np.nan)
    with pytest.raises(ValueError):
        gaussian_pixel_transfer(
            np.zeros((1, 1)),
            np.zeros(1),
            optical_mtf=0.3,
            pitch_ratio_yx=(1.0, 1.0),
            pixel_width_yx_samples=None,
        )
    with pytest.raises(ValueError):
        gaussian_pixel_transfer(
            np.zeros(1),
            np.zeros(1),
            optical_mtf=0.3,
            pitch_ratio_yx=(1.0, 1.0),
            pixel_width_yx_samples=(0.0, 1.0),
        )


@pytest.mark.parametrize("directions", [0, 1, 3])
def test_image_tangents_preserve_multichannel_support_and_signed_fields(directions):
    source, target = grids()
    rng = np.random.default_rng(814)
    values = rng.normal(size=(*source.shape, 2))
    tangents = rng.normal(size=(*values.shape, directions))
    support = np.ones(source.shape, dtype=bool)
    support[10, 11] = False
    offsets = [rng.uniform(-1, 1, size=(target.shape[0], count, 2)) for count in (1, 3)]
    options = dict(detector_pitch_xy_m=(1.0, 1.0), pixel_size_xy_m=None)
    actual, derivative, masks = form_detector_images_with_tangents(
        values, tangents, support, source, target, offsets, **options
    )
    expected, expected_masks = form_detector_images(
        values, support, source, target, offsets, negative_policy="preserve", **options
    )
    for channel in range(2):
        assert_array_equal(actual[channel], expected[channel])
        assert_array_equal(masks[channel], expected_masks[channel])
        assert derivative[channel].shape == (*actual[channel].shape, directions)
    for direction in range(directions):
        reference, _ = form_detector_images(
            tangents[..., direction],
            support,
            source,
            target,
            offsets,
            negative_policy="preserve",
            **options,
        )
        for channel in range(2):
            assert_array_equal(derivative[channel][..., direction], reference[channel])
