"""Bounded pure solar adaptation and analytic coordinate-change checks."""

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    SolarIrradianceSpectrum,
    SpectralDomain,
    SpectralGrid,
    SpectralIrradiance,
    convert_spectral_grid,
    density_per_nm_to_per_cm1,
    interpolate_point_density,
    scale_spectral_irradiance,
    solar_irradiance,
)


def grid(points: list[float]) -> SpectralGrid:
    return SpectralGrid("wavenumber_cm1", np.array(points, dtype=np.float64))


def test_exact_native_identity_and_singleton() -> None:
    source = grid([1.0, 2.0, 4.0])
    values = np.array([0.0, 1.0, 4.0])
    result = interpolate_point_density(source, values, source)
    np.testing.assert_array_equal(result, values)
    assert not np.shares_memory(result, values) and result.flags.writeable
    np.testing.assert_array_equal(
        interpolate_point_density(grid([2.0]), np.array([3.0]), grid([2.0])), [3.0]
    )
    np.testing.assert_array_equal(interpolate_point_density(source, values, grid([3.0])), [2.5])
    with pytest.raises(ValueError):
        interpolate_point_density(grid([2.0]), np.array([3.0]), grid([3.0]))


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0]),
        np.array([1.0, np.inf]),
        np.array([-1.0, 2.0]),
        np.array([[1.0, 2.0]]),
    ],
)
def test_invalid_values(values: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        interpolate_point_density(grid([1.0, 2.0]), values, grid([1.5]))


def test_coordinate_rejection_and_support_gaps() -> None:
    wavelength = SpectralGrid("wavelength_nm", np.array([1.0, 2.0]))
    for source, target in ((wavelength, grid([1.0])), (grid([1.0, 2.0]), wavelength)):
        with pytest.raises(ValueError):
            interpolate_point_density(source, np.ones(2), target)
    with pytest.raises(ValueError):
        solar_irradiance(SolarIrradianceSpectrum(wavelength.points, np.ones(2)), wavelength)
    source = SpectralGrid(
        "wavenumber_cm1",
        np.array([1.0, 2.0, 4.0, 5.0]),
        SpectralDomain("wavenumber_cm1", ((1.0, 2.0), (3.0, 6.0))),
    )
    values = np.array([1.0, 2.0, 4.0, 5.0])
    np.testing.assert_array_equal(
        interpolate_point_density(source, values, grid([2.0, 4.0, 4.5])), [2.0, 4.0, 4.5]
    )
    for target in ([2.5], [3.5]):
        with pytest.raises(ValueError, match="gap"):
            interpolate_point_density(source, values, grid(target))
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_point_density(source, values, grid([6.0]))


def test_converted_domain_and_ancillary_jacobian() -> None:
    source = SpectralGrid(
        "wavelength_nm",
        np.array([500.0, 600.0, 900.0, 1000.0]),
        SpectralDomain("wavelength_nm", ((500.0, 600.0), (900.0, 1000.0))),
    )
    converted = convert_spectral_grid(source)
    density = np.array([1.0, 2.0, 3.0, 4.0])
    spectrum = SolarIrradianceSpectrum(source.points, density)
    result = solar_irradiance(spectrum, converted, wavelength_domain=source.domain)
    assert result.grid is converted
    np.testing.assert_array_equal(result.values, density[::-1] * source.points[::-1] ** 2 / 1e7)
    # Pure Jacobian may also transform verified absolute-density uncertainty,
    # but does not assign that meaning to the native field or propagate covariance.
    uncertainty = density / 10
    np.testing.assert_allclose(
        density_per_nm_to_per_cm1(source.points, uncertainty), result.values / 10
    )
    with pytest.raises(ValueError, match="gap"):
        solar_irradiance(spectrum, grid([15000.0]), wavelength_domain=source.domain)


def test_analytic_coordinate_change_converges_not_exact_trapezoids() -> None:
    errors = []
    for count in (33, 65, 129):
        wavelength = np.linspace(500.0, 1000.0, count)
        # E_lambda=1: exact integral across this interval is 500 W m-2.
        source = SpectralGrid("wavelength_nm", wavelength)
        target = convert_spectral_grid(source)
        result = solar_irradiance(SolarIrradianceSpectrum(wavelength, np.ones(count)), target)
        errors.append(abs(np.trapezoid(result.values, target.points) - 500.0))
    assert errors[2] < errors[1] / 3 < errors[0] / 9
    assert errors[2] < 0.01


@pytest.mark.parametrize(
    "distance,reference",
    [(0.0, 1.0), (-1.0, 1.0), (np.inf, 1.0), (1.0, np.nan), (1.0, 0.0), (1e-300, 1e300)],
)
def test_invalid_scaling(distance: float, reference: float) -> None:
    with pytest.raises(ValueError):
        scale_spectral_irradiance(
            SpectralIrradiance(grid([1.0]), np.ones(1)),
            distance_au=distance,
            reference_distance_au=reference,
        )


def test_nonunit_reference_and_nonmutation() -> None:
    irradiance = SpectralIrradiance(grid([1.0, 2.0]), np.array([2.0, 4.0]))
    result = scale_spectral_irradiance(irradiance, distance_au=3.0, reference_distance_au=2.0)
    np.testing.assert_allclose(result.values, irradiance.values * 4 / 9)
    assert not np.shares_memory(result.values, irradiance.values)
    same = scale_spectral_irradiance(irradiance, distance_au=2.0, reference_distance_au=2.0)
    np.testing.assert_array_equal(same.values, irradiance.values)
