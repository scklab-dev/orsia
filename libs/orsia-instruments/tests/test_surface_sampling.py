"""Surface photon contractions and fixed spatial tangents on synthetic fields."""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from orsia_geometry import LocalCartesianFrame, PointLattice
from orsia_instruments.acquisition.schedule import ReadSchedule
from orsia_instruments.photodetection.surface import (
    sample_fourier_surface_electrons,
    sample_surface_electron_tangents,
    sample_surface_electrons,
)
from orsia_instruments.response.spatial import (
    form_detector_images,
    form_detector_images_with_tangents,
)
from orsia_instruments.response.spectral import SpectralElectronResponse
from orsia_radiative import SpectralGrid

FRAME = LocalCartesianFrame("surface test", "synthetic")
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def inputs():
    reads = ReadSchedule(
        ("a", "b", "c"),
        ("a", "b", "c"),
        ("band",) * 3,
        ("direct",) * 3,
        np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 4.0]]),
    )
    response = SpectralElectronResponse(
        reads,
        SpectralGrid("wavelength_nm", np.array([500.0, 510.0, 530.0])),
        np.arange(1.0, 10.0).reshape(3, 3),
    )
    points = PointLattice.regular((2, 2), (1.0, 1.0), (0.0, 0.0), FRAME)
    view = np.tile([0.0, 0.0, 1.0], (3, 1))
    support = np.array([[True, True], [True, False]])
    return response, points, view, support


def light(xy, view, times, spectral):
    return 2 + xy[:, :1] + times[:, None] + spectral.points[None, :] / 1000


@pytest.mark.parametrize("read_block,spectral_block", [(1, 1), (2, 2), (8, 8)])
def test_surface_blocks_support_and_total_tangent(read_block, spectral_block):
    response, points, view, support = inputs()
    options = dict(
        frame=FRAME,
        epoch=EPOCH,
        support=support,
        read_block_size=read_block,
        spectral_block_size=spectral_block,
    )

    def tangent(xy, view, times, spectral):
        values = light(xy, view, times, spectral)
        return values, -values[..., None] * 0.2

    def seed(reads, spectral):
        return response.coefficients[reads, spectral, None] * 0.3

    values, derivative, mask = sample_surface_electron_tangents(
        tangent, response, points, view, 1, response_tangent=seed, **options
    )
    direct, direct_mask = sample_surface_electrons(light, response, points, view, **options)
    np.testing.assert_allclose(values, direct)
    np.testing.assert_array_equal(mask, direct_mask)
    np.testing.assert_allclose(derivative[..., 0], 0.1 * direct, rtol=2e-15)
    times = response.reads.exposure_bounds_s.mean(axis=1)
    for pixel in np.ndindex(points.shape):
        if support[pixel]:
            expected = np.sum(
                response.coefficients
                * light(np.broadcast_to(points.xy_m[pixel], (3, 2)), view, times, response.grid),
                axis=1,
            )
            np.testing.assert_allclose(values[pixel], expected)
        else:
            np.testing.assert_array_equal(values[pixel], 0)
    epsilon = 1e-5
    plus = SpectralElectronResponse(
        response.reads, response.grid, response.coefficients * (1 + 0.3 * epsilon)
    )
    minus = SpectralElectronResponse(
        response.reads, response.grid, response.coefficients * (1 - 0.3 * epsilon)
    )
    pv, _ = sample_surface_electrons(
        lambda *args: light(*args) * (1 - 0.2 * epsilon), plus, points, view, **options
    )
    mv, _ = sample_surface_electrons(
        lambda *args: light(*args) * (1 + 0.2 * epsilon), minus, points, view, **options
    )
    np.testing.assert_allclose(derivative[..., 0], (pv - mv) / (2 * epsilon), rtol=3e-10)


def test_surface_errors_and_no_callback_on_missing_support():
    response, points, view, support = inputs()

    def fail(*args):
        raise AssertionError("unsupported callback")

    values, mask = sample_surface_electrons(
        fail, response, points, view, frame=FRAME, epoch=EPOCH, support=np.zeros_like(support)
    )
    assert not values.any() and not mask.any()
    for updates in (
        dict(read_block_size=0),
        dict(view_local=-view),
        dict(epoch=datetime(2026, 1, 1)),
        dict(support=support[:, :1]),
    ):
        options: dict[str, Any] = (
            dict(view_local=view, frame=FRAME, epoch=EPOCH, support=support) | updates
        )
        with pytest.raises(ValueError):
            sample_surface_electrons(light, response, points, **options)
    with pytest.raises(ValueError):
        sample_surface_electron_tangents(
            lambda *args: (light(*args), np.ones((3, 3, 2))),
            response,
            points,
            view,
            1,
            frame=FRAME,
            epoch=EPOCH,
            support=support,
        )


def test_fourier_surface_and_fixed_response_tangents():
    response, _, view, _ = inputs()
    source = PointLattice.regular((16, 16), (1.0, 1.0), (0.0, 0.0), FRAME)
    target = PointLattice.regular((2, 2), (1.0, 1.0), (7.0, 7.0), FRAME)
    support = np.ones(source.shape, bool)
    offsets = np.zeros((2, 3, 2))
    options = dict(detector_pitch_xy_m=(1.0, 1.0), pixel_size_xy_m=None, optical_mtf=1.0)
    values, valid = sample_fourier_surface_electrons(
        light,
        response,
        view,
        source,
        support,
        target,
        offsets,
        frame=FRAME,
        epoch=EPOCH,
        negative_policy="preserve",
        read_block_size=2,
        **options,
    )
    fields, _ = sample_surface_electrons(
        light, response, source, view, frame=FRAME, epoch=EPOCH, support=support
    )
    displacements = [offsets[:, i : i + 1] for i in range(3)]
    expected, expected_mask = form_detector_images(
        fields, support, source, target, displacements, negative_policy="preserve", **options
    )
    np.testing.assert_allclose(values, np.concatenate(expected, axis=2))
    np.testing.assert_array_equal(valid, np.concatenate(expected_mask, axis=2))
    _, derivatives, masks = form_detector_images_with_tangents(
        fields, -fields[..., None], support, source, target, displacements, **options
    )
    for expected_value, derivative, mask in zip(expected, derivatives, masks, strict=True):
        np.testing.assert_allclose(derivative[..., 0], -expected_value, atol=1e-12)
        assert mask.all()


def test_fourier_rejects_extra_view_rows():
    response, _, view, _ = inputs()
    source = PointLattice.regular((16, 16), (1.0, 1.0), (0.0, 0.0), FRAME)
    target = PointLattice.regular((2, 2), (1.0, 1.0), (7.0, 7.0), FRAME)
    with pytest.raises(ValueError, match="axes"):
        sample_fourier_surface_electrons(
            light,
            response,
            np.vstack((view, view[:1])),
            source,
            np.ones(source.shape, bool),
            target,
            np.zeros((2, 3, 2)),
            frame=FRAME,
            epoch=EPOCH,
            detector_pitch_xy_m=(1.0, 1.0),
            pixel_size_xy_m=None,
            optical_mtf=1.0,
            negative_policy="preserve",
        )
