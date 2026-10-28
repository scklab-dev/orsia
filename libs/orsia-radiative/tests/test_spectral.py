from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    density_per_cm1_to_per_nm,
    density_per_nm_to_per_cm1,
    reverse_spectral_points,
    wavelength_to_wavenumber,
    wavenumber_to_wavelength,
)


def test_nm_cm1_point_conversion_reverses_order_and_roundtrips() -> None:
    wavelengths = np.array([1000.0, 2000.0], dtype=np.float64)
    converted = wavelength_to_wavenumber(wavelengths)
    np.testing.assert_array_equal(converted, [5000.0, 10000.0])
    np.testing.assert_array_equal(wavenumber_to_wavelength(converted), wavelengths)
    np.testing.assert_array_equal(
        reverse_spectral_points(np.array([3.0, 4.0]), wavelengths), [4, 3]
    )


def test_spectral_density_jacobian_and_dimensionless_depth_are_distinct() -> None:
    wavelengths = np.array([1000.0, 2000.0])
    converted = density_per_nm_to_per_cm1(wavelengths, np.array([1.0, 1.0]))
    np.testing.assert_allclose(converted, (0.4, 0.1))
    np.testing.assert_allclose(
        reverse_spectral_points(np.array([0.2, 0.4]), wavelengths), (0.4, 0.2)
    )
    np.testing.assert_allclose(
        density_per_cm1_to_per_nm(wavelength_to_wavenumber(wavelengths), converted),
        (1.0, 1.0),
    )


def test_declared_quadrature_preserves_integrated_spectral_irradiance() -> None:
    wavelengths = np.array([1000.0, 1200.0, 2000.0])
    source = np.array([3.0, 4.0, 8.0])
    target = density_per_nm_to_per_cm1(wavelengths, source)
    weights_nm = np.array([100.0, 500.0, 300.0])
    weights_cm1 = weights_nm[::-1] * 1e7 / wavelengths[::-1] ** 2
    assert sum(target * weights_cm1) == pytest.approx(sum(source * weights_nm))


@pytest.mark.parametrize(
    "points",
    [np.array([1.0, 1.0]), np.array([0.0, 2.0]), np.array([np.nan, 2.0])],
)
def test_spectral_conversion_rejects_invalid_coordinates(points: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="wavenumbers_cm1"):
        wavelength_to_wavenumber(points)


def test_spectral_conversion_rejects_invalid_density_support() -> None:
    with pytest.raises(ValueError, match="spectral values"):
        density_per_nm_to_per_cm1(np.array([1000.0, 2000.0]), np.array([1.0]))


@pytest.mark.parametrize("values", [np.array([1.0]), np.array([np.nan, 1.0])])
def test_inverse_density_and_point_reversal_reject_invalid_support(
    values: NDArray[np.float64],
) -> None:
    for operation in (density_per_cm1_to_per_nm, reverse_spectral_points):
        with pytest.raises(ValueError, match="spectral values"):
            if operation is reverse_spectral_points:
                operation(values, np.array([1000.0, 2000.0]))
            else:
                operation(np.array([1000.0, 2000.0]), values)


def test_density_jacobians_reject_finite_inputs_whose_conversion_overflows() -> None:
    for operation in (density_per_nm_to_per_cm1, density_per_cm1_to_per_nm):
        with np.errstate(over="ignore"):
            with pytest.raises(ValueError, match="converted spectral density"):
                operation(np.array([10000.0, 20000.0]), np.full(2, 1e308))


def test_public_readme_example() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    namespace: dict[str, object] = {}
    for block in readme.read_text(encoding="utf-8").split("```python\n")[1:]:
        exec(block.split("```", 1)[0], namespace)
