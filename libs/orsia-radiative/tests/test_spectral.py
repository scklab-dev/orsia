from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    ObservationGeometry,
    Radiance,
    SpectralDomain,
    SpectralGrid,
    SpectralIrradiance,
    convert_spectral_grid,
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


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_density_conversion_preserves_other_axes_and_discrete_pairing(axis):
    points = np.array([500.0, 700.0, 1300.0])
    source = np.arange(24.0).reshape(2, 3, 4) - 10
    source = np.moveaxis(source, 1, axis)
    converted = density_per_nm_to_per_cm1(points, source, axis=axis)
    expected = np.moveaxis(
        np.moveaxis(source, axis, -1)[..., ::-1] * [0.169, 0.049, 0.025], -1, axis
    )
    np.testing.assert_allclose(converted, expected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        density_per_cm1_to_per_nm(wavelength_to_wavenumber(points), converted, axis=axis),
        source,
        rtol=2e-15,
        atol=0,
    )
    assert converted.shape == source.shape and converted.flags.writeable
    assert not np.shares_memory(converted, source)
    np.testing.assert_array_equal(
        reverse_spectral_points(source, points, axis=axis), np.flip(source, axis)
    )
    # Already quadrature-inclusive coefficients transform reciprocally, without regridding.
    response = np.arange(24.0).reshape(2, 3, 4) + 1
    response = np.moveaxis(response, 1, axis)
    paired = np.moveaxis(
        np.moveaxis(response, axis, -1)[..., ::-1] / [0.169, 0.049, 0.025], -1, axis
    )
    np.testing.assert_allclose(
        np.sum(paired * converted, axis=axis), np.sum(response * source, axis=axis), rtol=3e-15
    )


@pytest.mark.parametrize("axis", [2, -3])
def test_spectral_conversion_rejects_outside_axes(axis):
    points = np.array([500.0, 700.0, 1300.0])
    with pytest.raises(ValueError, match="spectral values"):
        density_per_nm_to_per_cm1(points, np.ones((2, 3)), axis=axis)


def test_scalar_and_empty_nonspectral_axes():
    points = np.array([500.0])
    with pytest.raises(ValueError, match="spectral values"):
        reverse_spectral_points(np.array(1.0), points)
    assert density_per_nm_to_per_cm1(points, np.empty((0, 1, 2)), axis=1).shape == (0, 1, 2)


@pytest.mark.parametrize("kind", ["radiance", "irradiance"])
def test_physical_spectra_accept_either_density_coordinate_with_explicit_conversion(kind):
    domain = SpectralDomain("wavelength_nm", ((500.0, 700.0), (1200.0, 1400.0)))
    grid = SpectralGrid("wavelength_nm", np.array([500.0, 700.0, 1300.0]), domain)
    values = np.array([1.0, 2.0, 3.0])
    geometry = ObservationGeometry(20.0)

    def construct(spectral, samples):
        if kind == "radiance":
            return Radiance(spectral, samples, geometry)
        return SpectralIrradiance(spectral, samples)

    original = construct(grid, values)
    values[:] = 99
    np.testing.assert_array_equal(original.values, [1.0, 2.0, 3.0])
    assert original.grid is grid and original.values.flags.writeable
    with pytest.raises(ValueError, match="convert explicitly"):
        _ = original.wavenumbers_cm1
    converted = construct(
        convert_spectral_grid(grid), density_per_nm_to_per_cm1(grid.points, original.values)
    )
    np.testing.assert_allclose(converted.values, [0.507, 0.098, 0.025], rtol=2e-15)
    assert converted.wavenumbers_cm1 is converted.grid.points
    restored = construct(
        convert_spectral_grid(converted.grid),
        density_per_cm1_to_per_nm(converted.grid.points, converted.values),
    )
    assert restored.grid == original.grid
    np.testing.assert_allclose(restored.values, original.values, rtol=2e-15)
    if kind == "radiance":
        assert isinstance(converted, Radiance)
        assert converted.geometry is geometry
    for invalid in (np.ones(2), np.array([-1.0, 0.0, 1.0]), np.full(3, np.nan)):
        with pytest.raises(ValueError):
            construct(grid, invalid)


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


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "docs/state-and-spectra.md",
        "docs/columns.md",
        "docs/spectroscopy.md",
        "docs/transfer.md",
    ],
)
def test_public_documented_examples(relative_path: str) -> None:
    document = Path(__file__).resolve().parents[1] / relative_path
    namespace: dict[str, object] = {}
    for block in document.read_text(encoding="utf-8").split("```python\n")[1:]:
        exec(block.split("```", 1)[0], namespace)
