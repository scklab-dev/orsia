"""Evaluate spatial two-anchor Lambertian reflectance on explicit spectra."""

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid


def affine_reflectance_weights(
    spectral: SpectralGrid, anchor_wavelengths_nm: tuple[float, float]
) -> NDArray[np.float64]:
    """Return the two affine-in-wavelength basis weights on supplied samples.

    Args:
        spectral: Positive increasing nm or cm-1 samples; output keeps their order.
        anchor_wavelengths_nm: Two finite positive increasing wavelengths, nm.
            Requested samples must lie within their closed interval.

    Returns:
        Fresh nonnegative (spectral,anchor=2) weights summing to one. These are
        dimensionless surface basis weights, with no spectral-density Jacobian.

    Raises:
        ValueError: Anchor order or requested wavelength support is invalid.
    """
    lower, upper = anchor_wavelengths_nm
    if not np.isfinite(anchor_wavelengths_nm).all() or not 0 < lower < upper:
        raise ValueError("two positive increasing surface anchor wavelengths required")
    wave = spectral.points if spectral.coordinate == "wavelength_nm" else 1e7 / spectral.points
    if np.any((wave < lower) | (wave > upper)):
        raise ValueError("spectral samples lie outside the surface anchor interval")
    fraction = (wave - lower) / (upper - lower)
    return np.stack((1 - fraction, fraction), axis=-1)


def affine_surface_reflectance(
    anchor_reflectance: NDArray[np.float64],
    spectral: SpectralGrid,
    anchor_wavelengths_nm: tuple[float, float],
) -> NDArray[np.float64]:
    """Evaluate the declared affine surface law without spatial resampling.

    Args:
        anchor_reflectance: Finite fractions (*samples,anchor=2) in [0,1].
            Each leading sample retains its supplied spatial interpretation.
        spectral: Increasing nm or cm-1 points inside the anchor interval.
        anchor_wavelengths_nm: Two ordered positive physical anchor wavelengths.

    Returns:
        Fresh fractions (*samples,spectral), with the original leading axes.

    Raises:
        ValueError: Reflectance dimensions, physical values or spectral support fail.
    """
    if (
        anchor_reflectance.ndim < 1
        or anchor_reflectance.shape[-1] != 2
        or not np.isfinite(anchor_reflectance).all()
        or np.any((anchor_reflectance < 0) | (anchor_reflectance > 1))
    ):
        raise ValueError("surface anchors require finite (...,2) fractions in [0,1]")
    return anchor_reflectance @ affine_reflectance_weights(spectral, anchor_wavelengths_nm).T
