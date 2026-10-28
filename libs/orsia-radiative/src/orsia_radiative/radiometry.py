"""Pure blackbody radiometry and explicit inverse-square irradiance scaling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import constants

from orsia_radiative.radiance import SpectralIrradiance
from orsia_radiative.spectral.grid import _validate_points

# Convert the second radiation constant from SI metres to centimetres.
_SECOND_RADIATION_CM_K = 100 * constants.h * constants.c / constants.k


def planck_wavenumber(
    wavenumbers_cm1: NDArray[np.float64], temperature_k: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Planck radiance in W m-2 sr-1 per (cm-1), not per m-1 or wavelength.

    Positive increasing float64 coordinates have shape (points,). Temperature
    is a positive finite scalar in K or a float64 (layers,) array. Return a fresh
    writable float64 (points,) or (layers, points) array respectively; no input
    is mutated or retained. B = 2*h*c**2*1e8*nu_cm**3 / expm1(c2*nu_cm/T).
    Logarithmic evaluation avoids intermediate overflow and handles the
    Rayleigh-Jeans and Wien limits. Underflow returns zero; an unrepresentable
    radiance raises ValueError rather than returning infinity.

    Args:
        wavenumbers_cm1: Positive increasing spectral points in cm-1.
        temperature_k: Positive scalar or (layers,) temperatures in kelvin.

    Returns:
        Owned spectral radiance (points,) or (layers,points) in W m-2 sr-1 per cm-1.

    Raises:
        ValueError: Spectral or temperature inputs are invalid or output overflows.
    """
    _validate_points(wavenumbers_cm1)
    temperatures = np.asarray(temperature_k)
    if (
        temperatures.ndim > 1
        or temperatures.size == 0
        or not np.all(np.isfinite(temperatures))
        or np.any(temperatures <= 0)
    ):
        raise ValueError("temperature must be positive finite scalar or layer values")
    log_temperature = np.log(temperatures)[..., None]
    log_wavenumber = np.log(wavenumbers_cm1)
    log_x = np.log(_SECOND_RADIATION_CM_K) + log_wavenumber - log_temperature
    with np.errstate(over="ignore", under="ignore"):
        # Form the dimensionless ratio directly to avoid amplifying log-roundoff
        # in the Wien exponent; log_x remains finite for extreme input ratios.
        x = (wavenumbers_cm1 / temperatures[..., None]) * _SECOND_RADIATION_CM_K
        correction = np.empty_like(x)
        small = log_x < -10
        correction[small] = -x[small] / 2 - x[small] ** 2 / 24
        correction[~small] = log_x[~small] - x[~small] - np.log(-np.expm1(-x[~small]))
        # B_RJ = 2*k*c*1e6*T*nu_cm**2; x/expm1(x) is its quantum correction.
        values = np.exp(
            np.log(2 * constants.k * constants.c * 1e6)
            + log_temperature
            + 2 * log_wavenumber
            + correction
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Planck radiance exceeds finite float64 support")
    return values


def scale_spectral_irradiance(
    irradiance: SpectralIrradiance,
    *,
    distance_au: float,
    reference_distance_au: float,
) -> SpectralIrradiance:
    """Scale beam-normal irradiance by (reference_distance_au/distance_au)**2.

    Both distances must be finite and positive, in AU. No source reference
    distance is assumed; the caller must establish it from the source. Date,
    ephemeris and geometry remain consumer policy. No cosine is applied.
    Retain the grid and return fresh writable values; do not mutate inputs.

    Args:
        irradiance: Materialised beam-normal irradiance at reference distance.
        distance_au: Destination distance from the source in astronomical units.
        reference_distance_au: Supplied source reference distance in astronomical units.

    Returns:
        Fresh scaled irradiance samples on the original spectral grid.

    Raises:
        ValueError: Distances are invalid or scaled values overflow float64 support.
    """
    if not all(np.isfinite(value) and value > 0 for value in (distance_au, reference_distance_au)):
        raise ValueError("distance and reference distance must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        factor = np.square(np.float64(reference_distance_au) / distance_au)
        values = irradiance.values * factor
    if not np.isfinite(factor) or not np.all(np.isfinite(values)):
        raise ValueError("scaled irradiance must be finite")
    return SpectralIrradiance(irradiance.grid, values)


__all__ = ["planck_wavenumber", "scale_spectral_irradiance"]
