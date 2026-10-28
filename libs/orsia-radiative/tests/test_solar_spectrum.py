import numpy as np
import pytest
from orsia_radiative import (
    SolarIrradianceSpectrum,
    SpectralGrid,
    SpectralIrradiance,
    interpolate_point_density,
    scale_spectral_irradiance,
    solar_irradiance,
)


def test_point_density_interpolation_preserves_knots_and_linear_reference() -> None:
    source = SpectralGrid("wavenumber_cm1", np.array([1.0, 2.0, 4.0]))
    target = SpectralGrid("wavenumber_cm1", np.array([1.0, 1.5, 3.0, 4.0]))
    values = np.array([2.0, 4.0, 8.0])

    result = interpolate_point_density(source, values, target)

    assert np.array_equal(result, np.array([2.0, 3.0, 6.0, 8.0]))
    assert np.array_equal(values, np.array([2.0, 4.0, 8.0]))


def test_point_density_rejects_extrapolation() -> None:
    source = SpectralGrid("wavenumber_cm1", np.array([1.0, 2.0]))
    target = SpectralGrid("wavenumber_cm1", np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_point_density(source, np.array([1.0, 1.0]), target)


def test_solar_irradiance_applies_jacobian_and_retains_grid() -> None:
    wavelength = SpectralGrid("wavelength_nm", np.array([500.0, 1000.0]))
    target = SpectralGrid("wavenumber_cm1", np.array([10000.0, 20000.0]))

    result = solar_irradiance(
        SolarIrradianceSpectrum(wavelength.points, np.array([2.0, 1.0])), target
    )

    assert result.grid is target
    assert np.array_equal(result.values, np.array([0.1, 0.05]))


def test_inverse_square_distance_scaling() -> None:
    grid = SpectralGrid("wavenumber_cm1", np.array([10000.0]))
    irradiance = SpectralIrradiance(grid, np.array([4.0]))

    result = scale_spectral_irradiance(irradiance, distance_au=2.0, reference_distance_au=1.0)

    assert result.grid is grid
    assert result.values[0] == 1.0
    assert irradiance.values[0] == 4.0
