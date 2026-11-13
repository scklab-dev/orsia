"""Explicit coordinate and point-density conversion, without quadrature."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.spectral.grid import _point_snapshot, _validate_points

_NM_CM = 1e7


def wavelength_to_wavenumber(wavelengths_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a sampled wavelength axis for wavenumber-domain radiative inputs.

    Args:
        wavelengths_nm: Positive increasing float64 wavelength points in nm.

    Returns:
        Owned increasing wavenumber points in cm-1, in reversed sample order.

    Raises:
        ValueError: The wavelength axis is empty, nonfinite or not increasing.
    """
    _validate_points(wavelengths_nm)
    converted = _NM_CM / wavelengths_nm[::-1]
    return _point_snapshot(converted)


def wavenumber_to_wavelength(wavenumbers_cm1: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a sampled wavenumber axis for wavelength-domain source inputs.

    Args:
        wavenumbers_cm1: Positive increasing float64 wavenumber points in cm-1.

    Returns:
        Owned increasing wavelength points in nm, in reversed sample order.

    Raises:
        ValueError: The wavenumber axis is empty, nonfinite or not increasing.
    """
    _validate_points(wavenumbers_cm1)
    return _point_snapshot(_NM_CM / wavenumbers_cm1[::-1])


def reverse_spectral_points(
    values: NDArray[np.float64], source_points: NDArray[np.float64], *, axis: int = -1
) -> NDArray[np.float64]:
    """Match point values to a reciprocal spectral axis by reversing their order.

    Args:
        values: Finite point values in source order, with any nonspectral axes.
        source_points: Positive increasing source spectral coordinates.
        axis: Spectral array axis, negative indices allowed; defaults to the last axis.

    Returns:
        Fresh writable samples with the same shape and reversed spectral order,
        without density conversion. Other axes and signed values are preserved.

    Raises:
        ValueError: Point axis or sample shape/finite values fail validation.
    """
    _validate_points(source_points)
    if (
        not -values.ndim <= axis < values.ndim
        or values.shape[axis] != source_points.size
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("spectral values must be finite and match the point grid")
    return np.flip(values, axis=axis).copy()


def density_per_nm_to_per_cm1(
    wavelengths_nm: NDArray[np.float64], values_per_nm: NDArray[np.float64], *, axis: int = -1
) -> NDArray[np.float64]:
    """Convert wavelength point densities into per-wavenumber densities.

    Args:
        wavelengths_nm: Positive increasing wavelength points in nm.
        values_per_nm: Finite point densities per nm in wavelength order, possibly signed.
        axis: Spectral array axis, negative indices allowed; defaults to the last axis.

    Returns:
        Fresh writable densities per cm-1, with the same shape, reversed spectral
        order and point Jacobian. All nonspectral axes retain their order.

    Raises:
        ValueError: Samples or converted densities are invalid or nonfinite.
    """
    reversed_values = reverse_spectral_points(values_per_nm, wavelengths_nm, axis=axis)
    result = np.moveaxis(reversed_values, axis, -1) * (wavelengths_nm[::-1] ** 2 / _NM_CM)
    if not np.all(np.isfinite(result)):
        raise ValueError("converted spectral density must be finite")
    return np.moveaxis(result, -1, axis)


def density_per_cm1_to_per_nm(
    wavenumbers_cm1: NDArray[np.float64], values_per_cm1: NDArray[np.float64], *, axis: int = -1
) -> NDArray[np.float64]:
    """Convert wavenumber point densities into per-wavelength densities.

    Args:
        wavenumbers_cm1: Positive increasing wavenumber points in cm-1.
        values_per_cm1: Finite point densities per cm-1 in wavenumber order, possibly signed.
        axis: Spectral array axis, negative indices allowed; defaults to the last axis.

    Returns:
        Fresh writable densities per nm, with the same shape, reversed spectral
        order and point Jacobian. All nonspectral axes retain their order.

    Raises:
        ValueError: Samples or converted densities are invalid or nonfinite.
    """
    reversed_values = reverse_spectral_points(values_per_cm1, wavenumbers_cm1, axis=axis)
    result = np.moveaxis(reversed_values, axis, -1) * (wavenumbers_cm1[::-1] ** 2 / _NM_CM)
    if not np.all(np.isfinite(result)):
        raise ValueError("converted spectral density must be finite")
    return np.moveaxis(result, -1, axis)
