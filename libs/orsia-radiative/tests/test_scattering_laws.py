"""Analytic spectral laws, coordinate order, numerical limits and ownership."""

import numpy as np
import pytest
from orsia_radiative import (
    SpectralDomain,
    SpectralGrid,
    angstrom_optical_depth,
    convert_spectral_grid,
    rayleigh_from_air_columns,
    wavenumber_grid,
)
from orsia_radiative.optics import scattering


def test_public_exports_are_owned_by_scattering():
    assert angstrom_optical_depth is scattering.angstrom_optical_depth
    assert rayleigh_from_air_columns is scattering.rayleigh_from_air_columns


@pytest.mark.parametrize("domain", [None, SpectralDomain("wavenumber_cm1", ((4000, 21000),))])
def test_rayleigh_reference_scaling_order_and_ownership(domain):
    grid = wavenumber_grid(np.array([5000.0, 10000.0, 20000.0]), domain=domain)
    columns = np.array([0.0, 1e28, 3e28])
    before = columns.copy()
    result = rayleigh_from_air_columns(
        grid=grid,
        dry_air_columns_molecules_m2=columns,
        reference_cross_section_m2=2e-30,
        reference_wavelength_um=1.0,
    )
    # Wavelengths are 2, 1, 0.5 um in the original wavenumber order.
    expected = np.array([[0.0, 0.0, 0.0], [0.00125, 0.02, 0.32], [0.00375, 0.06, 0.96]])
    np.testing.assert_allclose(result.optical_depth, expected, rtol=2e-15)
    assert result.grid is grid and result.grid.domain is domain
    assert result.optical_depth.dtype == np.float64
    assert result.optical_depth.flags.writeable
    assert not np.shares_memory(result.optical_depth, columns)
    assert not np.shares_memory(result.optical_depth, grid.points)
    np.testing.assert_array_equal(columns, before)
    columns[:] = 5
    np.testing.assert_allclose(result.optical_depth, expected, rtol=2e-15)
    result.optical_depth[:] = 0
    np.testing.assert_array_equal(columns, np.full(3, 5))
    np.testing.assert_array_equal(grid.points, [5000.0, 10000.0, 20000.0])


@pytest.mark.parametrize("columns", [np.array([0.0]), np.array([1e28])[::-1]])
def test_rayleigh_single_point_and_zero_cross_section(columns):
    grid = wavenumber_grid(np.array([10000.0]))
    result = rayleigh_from_air_columns(grid, columns, 0.0, 1.0)
    np.testing.assert_array_equal(result.optical_depth, np.zeros((1, 1)))


@pytest.mark.parametrize("columns", [[], [[1.0]], -1.0, [-1.0], [np.nan], [np.inf]])
def test_rayleigh_invalid_columns(columns):
    with pytest.raises(ValueError, match="columns"):
        rayleigh_from_air_columns(
            wavenumber_grid(np.array([10000.0])), np.array(columns), 1e-30, 1.0
        )


@pytest.mark.parametrize("cross_section", [-1.0, np.nan, np.inf])
def test_rayleigh_invalid_cross_section(cross_section):
    with pytest.raises(ValueError, match="cross section"):
        rayleigh_from_air_columns(
            wavenumber_grid(np.array([10000.0])), np.array([1e28]), cross_section, 1.0
        )


@pytest.mark.parametrize("reference", [0.0, -1.0, np.nan, np.inf])
def test_rayleigh_invalid_reference_wavelength(reference):
    with pytest.raises(ValueError, match="reference wavelength"):
        rayleigh_from_air_columns(
            wavenumber_grid(np.array([10000.0])), np.array([1e28]), 1e-30, reference
        )


def test_rayleigh_rejects_wavelength_grid():
    with pytest.raises(ValueError, match="wavenumber_cm1"):
        rayleigh_from_air_columns(
            SpectralGrid("wavelength_nm", np.array([500.0])), np.array([1e28]), 1e-30, 1.0
        )


@pytest.mark.parametrize("columns,cross_section,reference", [(1.0, 1.0, 1e100), (1e308, 10.0, 1.0)])
def test_rayleigh_rejects_nonfinite_evaluation(columns, cross_section, reference):
    with pytest.raises(ValueError, match="finite"):
        rayleigh_from_air_columns(
            wavenumber_grid(np.array([10000.0])), np.array([columns]), cross_section, reference
        )


@pytest.mark.parametrize(
    "alpha,expected", [(2.0, [0.8, 0.2, 0.05]), (0.0, [0.2] * 3), (-1.0, [0.1, 0.2, 0.4])]
)
@pytest.mark.parametrize("coordinate", ["wavelength_nm", "wavenumber_cm1"])
def test_angstrom_reference_scaling_coordinates_and_ownership(alpha, expected, coordinate):
    wavelength_grid = SpectralGrid("wavelength_nm", np.array([250.0, 500.0, 1000.0]))
    grid = (
        wavelength_grid if coordinate == "wavelength_nm" else convert_spectral_grid(wavelength_grid)
    )
    result = angstrom_optical_depth(
        grid=grid,
        reference_optical_depth=0.2,
        reference_wavelength_nm=500.0,
        angstrom_exponent=alpha,
    )
    np.testing.assert_allclose(
        result, expected if coordinate == "wavelength_nm" else expected[::-1]
    )
    assert result.shape == (3,) and result.dtype == np.float64
    assert result.flags.writeable and result.flags.owndata
    assert not np.shares_memory(result, grid.points)
    result[:] = 0
    np.testing.assert_array_equal(wavelength_grid.points, [250.0, 500.0, 1000.0])


def test_angstrom_zero_depth_single_point_and_domain_independence():
    points = np.array([500.0])
    grid = SpectralGrid("wavelength_nm", points, SpectralDomain("wavelength_nm", ((400, 600),)))
    np.testing.assert_array_equal(angstrom_optical_depth(grid, 0.0, 550.0, 1.3), [0.0])
    np.testing.assert_array_equal(
        angstrom_optical_depth(grid, 0.2, 550.0, 1.3),
        angstrom_optical_depth(SpectralGrid("wavelength_nm", points), 0.2, 550.0, 1.3),
    )


@pytest.mark.parametrize("depth", [-1.0, np.nan, np.inf])
def test_angstrom_invalid_reference_depth(depth):
    with pytest.raises(ValueError, match="reference optical depth"):
        angstrom_optical_depth(SpectralGrid("wavelength_nm", np.array([500.0])), depth, 550.0, 1.3)


@pytest.mark.parametrize("reference", [0.0, -1.0, np.nan, np.inf])
def test_angstrom_invalid_reference_wavelength(reference):
    with pytest.raises(ValueError, match="reference wavelength"):
        angstrom_optical_depth(
            SpectralGrid("wavelength_nm", np.array([500.0])), 0.1, reference, 1.3
        )


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -np.inf])
def test_angstrom_invalid_exponent(alpha):
    with pytest.raises(ValueError, match="exponent"):
        angstrom_optical_depth(SpectralGrid("wavelength_nm", np.array([500.0])), 0.1, 550.0, alpha)


def test_angstrom_rejects_nonfinite_evaluation():
    with pytest.raises(ValueError, match="finite"):
        angstrom_optical_depth(SpectralGrid("wavelength_nm", np.array([1.0])), 1.0, 1000.0, 1000.0)


@pytest.mark.parametrize("law", ["rayleigh", "angstrom"])
def test_unrepresentable_reciprocal_wavelength(law):
    grid = wavenumber_grid(np.array([1e-308]))
    with pytest.raises(ValueError, match="finite"):
        if law == "rayleigh":
            rayleigh_from_air_columns(grid, np.array([1e28]), 1e-30, 1.0)
        else:
            angstrom_optical_depth(grid, 0.1, 550.0, 1.3)
