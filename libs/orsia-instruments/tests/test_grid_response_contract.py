"""Explicit point/frame support, y/x array meaning and retained scan-native ordering."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import LocalCartesianFrame, PointLattice
from orsia_instruments import (
    GridElectronResponse,
    ReadSchedule,
    SurfaceRadianceSamples,
    response_electrons,
)
from orsia_radiative import SpectralDomain, SpectralGrid

FRAME = LocalCartesianFrame("grid response", "synthetic surface")


def response():
    lattice = PointLattice(np.arange(-2.0, 3.0), np.array([-10.0, 0.0, 10.0]), FRAME)
    spectral = SpectralGrid(
        "wavenumber_cm1",
        np.array([1000.0, 1001.0, 1002.0]),
        SpectralDomain("wavenumber_cm1", ((1000.0, 1002.0),)),
    )
    reads = ReadSchedule(
        ("a", "b"), ("p", "p"), ("s0", "s1"), ("+", "-"), np.array([[0.0, 1.0], [2.0, 3.0]])
    )
    coefficients = np.arange(1.0, 91.0).reshape(2, 3, 5, 3)
    bounds = np.tile([-2.0, 2.0, -10.0, 10.0], (2, 1))
    return GridElectronResponse(reads, spectral, lattice, bounds, coefficients)


def test_nonsquare_lattice_meaning_and_snapshot_ownership():
    kernel = response()
    assert kernel.coefficients.shape == (2, 3, 5, 3)
    x, y = np.meshgrid(kernel.lattice.x_m, kernel.lattice.y_m)
    values = (10 + 0.1 * x + 0.02 * y)[..., None] * np.array([1.0, 2.0, 4.0])
    samples = SurfaceRadianceSamples(kernel.lattice, kernel.grid, values)
    expected = np.array([np.sum(row * values) for row in kernel.coefficients])
    assert_allclose(response_electrons(kernel, samples).total, expected, rtol=2e-15)
    assert_allclose(response_electrons(kernel, values).total, expected, rtol=2e-15)
    values[:] = 0
    assert np.all(samples.values > 0) and samples.values.flags.writeable
    assert samples.lattice is kernel.lattice
    copied = replace(kernel)
    assert copied.lattice is kernel.lattice and copied.reads is kernel.reads
    assert copied.coefficients.flags.writeable and not np.shares_memory(
        copied.coefficients, kernel.coefficients
    )
    with pytest.raises(ValueError):
        kernel.lattice.x_m.setflags(write=True)


@pytest.mark.parametrize("change", ["x", "y", "frame", "spectrum", "coordinate"])
def test_equal_shape_does_not_establish_common_support(change):
    kernel = response()
    lattice, spectral = kernel.lattice, kernel.grid
    if change == "x":
        lattice = replace(lattice, x_m=lattice.x_m + 1)
    if change == "y":
        lattice = replace(lattice, y_m=lattice.y_m + 1)
    if change == "frame":
        lattice = replace(lattice, frame=LocalCartesianFrame(FRAME.name, FRAME.vertical_datum))
    if change == "spectrum":
        spectral = SpectralGrid("wavenumber_cm1", spectral.points + 1)
    if change == "coordinate":
        spectral = SpectralGrid("wavelength_nm", spectral.points)
    samples = SurfaceRadianceSamples(lattice, spectral, np.ones((3, 5, 3)))
    with pytest.raises(ValueError):
        response_electrons(kernel, samples)
    with pytest.raises(ValueError):
        response_electrons(kernel, samples, spectra=np.ones((1, 3)))


def test_explicit_record_lattice_conflict_and_sample_domains():
    kernel = response()
    samples = SurfaceRadianceSamples(kernel.lattice, kernel.grid, np.ones((3, 5, 3)))
    with pytest.raises(ValueError, match="explicit scene lattice"):
        response_electrons(
            kernel, samples, scene_lattice=replace(kernel.lattice, x_m=kernel.lattice.x_m + 1)
        )
    for values in (np.ones((5, 3, 3)), -np.ones((3, 5, 3)), np.full((3, 5, 3), np.nan)):
        with pytest.raises(ValueError):
            SurfaceRadianceSamples(kernel.lattice, kernel.grid, values)
    single = PointLattice(np.zeros(1), np.zeros(1), FRAME)
    assert SurfaceRadianceSamples(single, kernel.grid, np.ones((1, 1, 3))).values.shape == (1, 1, 3)
    with pytest.raises(ValueError, match="ground axes"):
        replace(kernel, lattice=single)


def test_translation_yx_fields_preserve_x_then_y_scan_native_reads():
    kernel = response()
    lattice = PointLattice(np.arange(-4.0, 5.0), np.arange(-30.0, 31.0, 10.0), FRAME)
    x, y = np.meshgrid(lattice.x_m, lattice.y_m)
    values = (20 + x + 0.1 * y)[..., None] * np.array([1.0, 2.0, 3.0])
    samples = SurfaceRadianceSamples(lattice, kernel.grid, values)
    ys, xs = np.array([1, 4]), np.array([2, 5, 6])
    reads = ReadSchedule(
        tuple(f"{r}-{ix}-{iy}" for r in range(2) for ix in xs for iy in ys),
        ("p",) * 12,
        ("s0",) * 6 + ("s1",) * 6,
        ("+",) * 6 + ("-",) * 6,
        np.tile([10.0, 11.0], (12, 1)),
    )
    expected = np.array(
        [
            np.sum(kernel.coefficients[r] * values[iy - 1 : iy + 2, ix - 2 : ix + 3])
            for r in range(2)
            for ix in xs
            for iy in ys
        ]
    )
    for method in ("direct", "correlation"):
        actual = response_electrons(
            kernel, samples, target_indices=(ys, xs), reads=reads, method=method
        )
        assert actual.reads is reads
        assert_allclose(actual.total, expected, rtol=3e-15)
        factors = response_electrons(
            kernel,
            values[..., 0][None],
            spectra=np.array([[1.0, 2.0, 3.0]]),
            scene_lattice=lattice,
            target_indices=(ys, xs),
            reads=reads,
            method=method,
        )
        assert_allclose(factors.total, expected, rtol=3e-15)
    with pytest.raises(ValueError, match="frame"):
        response_electrons(
            kernel,
            replace(
                samples,
                lattice=replace(lattice, frame=LocalCartesianFrame("other", "synthetic surface")),
            ),
            target_indices=(ys, xs),
            reads=reads,
        )
    with pytest.raises(ValueError, match="halo"):
        response_electrons(kernel, samples, target_indices=(np.array([0]), xs), reads=reads)
    with pytest.raises(ValueError, match="lattice"):
        response_electrons(kernel, samples)


def test_wavelength_coordinate_and_required_spectral_support():
    kernel = response()
    nm = SpectralGrid(
        "wavelength_nm",
        np.array([500.0, 600.0, 700.0]),
        SpectralDomain("wavelength_nm", ((500.0, 700.0),)),
    )
    converted = replace(kernel, grid=nm)
    samples = SurfaceRadianceSamples(converted.lattice, nm, np.ones((3, 5, 3)))
    assert_array_equal(
        response_electrons(converted, samples).total, converted.coefficients.sum(axis=(1, 2, 3))
    )
    with pytest.raises(ValueError, match="coverage"):
        replace(kernel, grid=SpectralGrid("wavenumber_cm1", kernel.grid.points))


def test_nonsquare_projection_has_independent_bilinear_photon_oracle():
    from orsia_instruments import Acquisition, detector_response
    from scipy.constants import c, h

    lattice = PointLattice(np.array([-2.0, 0.0, 3.0]), np.array([-10.0, 10.0]), FRAME)
    spectral = SpectralGrid(
        "wavenumber_cm1",
        np.array([1000.0, 2000.0]),
        SpectralDomain("wavenumber_cm1", ((1000.0, 2000.0),)),
    )
    reads = ReadSchedule(("r",), ("p",), ("s",), ("direct",), np.array([[0.0, 1.0]]))
    acquisition = Acquisition(
        reads,
        np.array([[0.2, 0.8]]),
        np.zeros((1, 2, 2)),
        np.zeros((1, 2, 2)),
        np.array([[0.1, 0.3]]),
        np.array([[-0.1, 0.1, -0.1, 0.1]]),
        np.array([[-0.1, 0.1, -0.1, 0.1]]),
    )
    xy = np.array([[[-1.0, -5.0], [1.0, 8.0]]])
    projected = detector_response(
        spectral,
        acquisition,
        lambda read, wave, current: np.full((2, len(wave)), 0.25),
        ground_xy_m=xy,
        lattice=lattice,
        support_bounds_m=np.array([[-2.0, 3.0, -10.0, 10.0]]),
        collecting_area_m2=2.0,
        throughput=0.4,
        quantum_efficiency=0.8,
    )
    x, y = np.meshgrid(lattice.x_m, lattice.y_m)
    spatial = 3 + 0.2 * x + 0.03 * y + 0.001 * x * y
    samples = SurfaceRadianceSamples(
        lattice, spectral, spatial[..., None] * (h * c * 100 * spectral.points)
    )
    actual = response_electrons(projected, samples)
    node_value = 3 + 0.2 * xy[0, :, 0] + 0.03 * xy[0, :, 1] + 0.001 * xy[0, :, 0] * xy[0, :, 1]
    expected = 1000 * 0.25 * 2 * 0.4 * 0.8 * np.dot([0.1, 0.3], node_value)
    assert projected.coefficients.shape == (1, 2, 3, 2)
    assert_allclose(actual.total, [expected], rtol=3e-15)


def test_bilinear_basis_rejects_overflowing_coordinate_intervals():
    kernel = response()
    with np.errstate(over="ignore"):
        lattice = PointLattice(np.array([-1e308, 1e308]), kernel.lattice.y_m, FRAME)
    with pytest.raises(ValueError, match="finite positive intervals"):
        replace(kernel, lattice=lattice, coefficients=np.zeros((2, 3, 2, 3)))
