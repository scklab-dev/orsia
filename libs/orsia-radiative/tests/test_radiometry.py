"""Planck units, independent SI expressions, limits and array ownership."""

from math import expm1, pi

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_radiative.radiometry import planck_wavenumber
from scipy.integrate import quad


@pytest.mark.parametrize("temperature", [180.0, 300.0, 5800.0])
def test_planck_matches_independent_frequency_density(temperature: float) -> None:
    grid = np.array([1.0, 500.0, 1000.0, 2000.0, 10000.0])
    frequency = 299792458.0 * 100 * grid
    # Frequency density times d(frequency)/d(cm-1), not coordinate conversion alone.
    expected = np.array(
        [
            2
            * 6.62607015e-34
            * nu**3
            / 299792458.0**2
            / expm1(6.62607015e-34 * nu / (1.380649e-23 * temperature))
            * (100 * 299792458.0)
            for nu in frequency
        ]
    )
    np.testing.assert_allclose(planck_wavenumber(grid, temperature), expected, rtol=3e-14)


def test_planck_shape_ownership_monotonic_temperature_and_stefan_boltzmann() -> None:
    grid = np.array([500.0, 1000.0])
    temperatures = np.array([200.0, 300.0, 400.0])
    saved = grid.copy(), temperatures.copy()
    values = planck_wavenumber(grid, temperatures)
    assert values.shape == (3, 2) and values.dtype == np.float64 and values.flags.writeable
    assert np.all(np.diff(values, axis=0) > 0)
    assert not np.shares_memory(values, grid) and not np.shares_memory(values, temperatures)
    np.testing.assert_array_equal(grid, saved[0])
    np.testing.assert_array_equal(temperatures, saved[1])
    np.testing.assert_array_equal(values[1], planck_wavenumber(grid, 300.0))
    integral, _ = quad(
        lambda nu: float(planck_wavenumber(np.array([nu]), 300.0)[0]),
        1e-6,
        20000.0,
        epsabs=1e-8,
    )
    # Integral of radiance is sigma*T^4/pi; hemispheric irradiance is pi times this.
    sigma = 2 * pi**5 * 1.380649e-23**4 / (15 * 6.62607015e-34**3 * 299792458.0**2)
    assert integral == pytest.approx(sigma * 300**4 / pi, rel=2e-12)


def test_planck_rayleigh_jeans_wien_and_extreme_intermediates() -> None:
    grid = np.array([1e-150, 1e-6, 1e200, 1e300])
    values = planck_wavenumber(grid, 300.0)
    expected_rj = 2 * 1.380649e-23 * 299792458.0 * 1e6 * 300 * grid[:2] ** 2
    np.testing.assert_allclose(values[:2], expected_rj, rtol=3e-9, atol=0)
    np.testing.assert_array_equal(values[2:], [0.0, 0.0])
    # Nu^3 would overflow, but the resulting blackbody radiance is representable.
    value = planck_wavenumber(np.array([1e105]), 1e105)
    expected = np.exp(
        np.log(2 * 6.62607015e-34 * 299792458.0**2 * 1e8)
        + 315 * np.log(10.0)
        - np.log(expm1(100 * 6.62607015e-34 * 299792458.0 / 1.380649e-23))
    )
    np.testing.assert_allclose(value, [expected], rtol=3e-13)
    assert planck_wavenumber(np.array([1.0]), 1e-300)[0] == 0
    with pytest.raises(ValueError, match="exceeds"):
        planck_wavenumber(np.array([1e200]), 1e200)


@pytest.mark.parametrize(
    "temperature",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        np.array([]),
        np.array([[300.0]]),
        np.array([300.0, 0.0]),
        np.array([300.0, np.nan]),
    ],
)
def test_invalid_temperature(temperature: float | NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="temperature"):
        planck_wavenumber(np.array([1000.0]), temperature)


@pytest.mark.parametrize(
    "grid",
    [
        np.array([]),
        np.array([0.0]),
        np.array([-1.0]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1000.0, 999.0]),
        np.array([1000.0, 1000.0]),
        np.array([[1000.0]]),
        np.array([1000], dtype=np.int64),
    ],
)
def test_invalid_sampling(grid: NDArray[np.float64 | np.int64]) -> None:
    with pytest.raises(ValueError, match="wavenumbers"):
        # The integer fixture intentionally exercises an ill-typed dynamic call.
        planck_wavenumber(grid, 300.0)  # pyrefly: ignore [bad-argument-type]
