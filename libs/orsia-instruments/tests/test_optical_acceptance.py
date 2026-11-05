"""Explicit optical snapshots, exact spectral reuse and prepared table mass."""

from unittest.mock import patch

import numpy as np
import pytest
from orsia_instruments import (
    GaussianPSF,
    ReadSchedule,
    TabulatedPSF,
    photoelectron_response,
    pixel_acceptance,
    rectangular_acquisition,
    sample_optical_acceptance,
)
from orsia_instruments.optics import psfs
from orsia_radiative import SpectralDomain, SpectralGrid


def grid(points):
    points = np.asarray(points, dtype=np.float64)
    return SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((points[0], points[-1]),))
    )


def test_acceptance_scratch_snapshot_exact_support_and_physical_reuse():
    acquisition = rectangular_acquisition(
        reads=ReadSchedule(
            ("a", "b"), ("p", "q"), ("s", "s"), ("+", "+"), np.array([[0.0, 1.0], [2.0, 3.0]])
        ),
        pixel_bounds_m=np.tile([-1.0, 1.0, -1.0, 1.0], (2, 1)),
        direction_bounds_uv=(-0.1, 0.1, -0.1, 0.1),
        pointing=lambda times: np.zeros((len(times), 2)),
        focal_length_m=1.0,
        time_order=1,
        angular_order=2,
    )
    spectral = grid([4000.0, 4500.0, 5000.0])
    wave = 0.01 / spectral.points
    scratch = np.empty((4, 3))
    calls = []

    def response(read, wavelengths, current):
        calls.append(read)
        local = scratch[:, : len(wavelengths)]
        local[:] = (read + 1) * np.arange(1.0, 5.0)[:, None] / 10 * wavelengths / wave.max()
        return local

    acceptance = sample_optical_acceptance(acquisition, response, wave[::-1])
    assert calls == [0, 1]
    retained = acceptance.fractions.copy()
    scratch.fill(99)
    np.testing.assert_array_equal(acceptance.fractions, retained)
    for values in (acceptance.fractions, acceptance.wavelengths_m):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    base = photoelectron_response(spectral, acceptance, collecting_area_m2=2.0)
    changed = photoelectron_response(
        spectral, acceptance, collecting_area_m2=4.0, throughput=0.5, quantum_efficiency=0.25
    )
    np.testing.assert_array_equal(changed, base * 0.25)
    subset = grid([4000.0, 5000.0])
    subset_acceptance = sample_optical_acceptance(acquisition, response, 0.01 / subset.points)
    np.testing.assert_array_equal(
        photoelectron_response(subset, acceptance, collecting_area_m2=2.0),
        photoelectron_response(subset, subset_acceptance, collecting_area_m2=2.0),
    )
    assert calls == [0, 1, 0, 1]
    changed_support = sample_optical_acceptance(acquisition, response, np.nextafter(wave, np.inf))
    with pytest.raises(ValueError, match="exact optical acceptance support"):
        photoelectron_response(spectral, changed_support, collecting_area_m2=2.0)


def test_prepared_table_mass_interpolator_reuse_and_uniform_gaussian_accuracy():
    x, y = np.array([-1.0, 1.0]), np.array([-1.0, 1.0])
    density = np.full((2, 2), 0.1)
    with patch.object(
        psfs, "RegularGridInterpolator", wraps=psfs.RegularGridInterpolator
    ) as prepare:
        model = TabulatedPSF(x, y, density)
        for order in (2, 4, 8):
            fractions = pixel_acceptance(
                np.array([1.0, 2.0]),
                [-1.0, 1.0, -1.0, 1.0],
                np.zeros((3, 2)),
                spatial=model,
                transmission=0.5,
                pixel_order=order,
            )
            np.testing.assert_allclose(fractions, 0.2, atol=1e-15)
        assert prepare.call_count == 1
    assert model.mass == pytest.approx(0.4)
    density.fill(99)
    x.fill(99)
    assert model.mass == pytest.approx(0.4)
    analytic = pixel_acceptance(
        np.array([1.0, 2.0]),
        [-1.0, 1.0, -1.0, 1.0],
        np.zeros((3, 2)),
        spatial=GaussianPSF(0.1),
        pixel_order=1,
    )
    np.testing.assert_allclose(analytic, 1.0, atol=1e-15)
