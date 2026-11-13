"""Analytic optical-mixture weighting with a caller-supplied spherical kernel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_radiative.optics.aerosol_mixture import (
    LognormalAerosolPopulation,
    lognormal_aerosol_optics,
)


def constant_sphere(
    index: complex, diameter: NDArray[np.float64], wavelength: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    assert index.imag <= 0 and wavelength > 0
    one = np.ones_like(diameter)
    return one * 2, one, one * 0, one * 0.4


def test_lognormal_volume_and_optical_weighting() -> None:
    population: LognormalAerosolPopulation = dict(
        volume_fraction=0.4,
        median_radius_um=0.11,
        geometric_width=1.4,
        wavelength_um=[0.5, 2.5],
        real_index=[1.5, 1.5],
        absorption_index=[0.01, 0.01],
    )
    second = population.copy()
    second["volume_fraction"] = 0.6
    second["median_radius_um"] = 0.5
    second["geometric_width"] = 2.0
    wave = np.array([550.0, 1000.0, 1800.0, 2200.0])
    relative, albedo, asymmetry = lognormal_aerosol_optics(
        wave,
        [population, second],
        constant_sphere,
        reference_wavelength_nm=550.0,
        quadrature_order=32,
        calculation_step_nm=100.0,
    )
    assert_allclose(relative, 1, atol=1e-15)
    assert_allclose(albedo, 0.5, atol=1e-15)
    assert_allclose(asymmetry, 0.4, atol=1e-15)
    with pytest.raises(ValueError):
        lognormal_aerosol_optics(
            wave[::-1], [population], constant_sphere, reference_wavelength_nm=550.0
        )
    invalid_population = population.copy()
    invalid_population["geometric_width"] = 1.0
    with pytest.raises(ValueError):
        lognormal_aerosol_optics(
            wave,
            [invalid_population],
            constant_sphere,
            reference_wavelength_nm=550.0,
        )

    def invalid(
        index: complex, diameter: NDArray[np.float64], wavelength: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return diameter * 0, diameter * 0, diameter * 0, diameter * 0

    with pytest.raises(ValueError, match="positive finite"):
        lognormal_aerosol_optics(wave, [population], invalid, reference_wavelength_nm=550.0)


def test_explicit_humidity_mass_ratio_density() -> None:
    from orsia_radiative.atmosphere.columns import dry_air_molar_density

    p, t, q = np.array([100000.0, 80000.0]), np.array([290.0, 280.0]), np.array([0.01, 0.005])
    ratio = q / (1 - q)
    expected = (p - p * ratio / (0.621945 + ratio)) / (8.31446261815324 * t)
    assert_allclose(
        dry_air_molar_density(p, t, q, water_dry_mass_ratio=0.621945), expected, rtol=0, atol=0
    )
    with pytest.raises(ValueError):
        dry_air_molar_density(p, t, q + 1, water_dry_mass_ratio=0.621945)
