"""Normalised scalar Rayleigh and bounded Henyey-Greenstein phase laws."""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def henyey_greenstein_phase(asymmetry: float, cosine: float) -> float: ...


@overload
def henyey_greenstein_phase(
    asymmetry: NDArray[np.float64], cosine: float | NDArray[np.float64]
) -> NDArray[np.float64]: ...


@overload
def henyey_greenstein_phase(
    asymmetry: float, cosine: NDArray[np.float64]
) -> NDArray[np.float64]: ...


def henyey_greenstein_phase(
    asymmetry: float | NDArray[np.float64], cosine: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """HG phase P=(1-g²)/(1+g²-2g*cosine)^(3/2), with 4π mean one.

    Dimensionless float64 inputs broadcast using NumPy rules. Asymmetry must
    be finite in [-0.8, 0.8]; cosine must be finite in [-1, 1], without clipping.
    Scalar inputs return a Python float; if either input is an array, return a
    fresh writable array, including for zero-dimensional inputs. Inputs are
    never mutated or retained.

    Args:
        asymmetry: Scalar or array HG anisotropy coefficient in [-0.8,0.8].
        cosine: Scalar or array scattering-angle cosine in [-1,1].

    Returns:
        Broadcast scalar or float64 array of dimensionless phase values.

    Raises:
        ValueError: The anisotropy or scattering angle leaves its physical support.
    """
    scalar_result = np.isscalar(asymmetry) and np.isscalar(cosine)
    asymmetry = np.asarray(asymmetry, dtype=np.float64)
    if not np.all(np.isfinite(asymmetry)) or np.any(np.abs(asymmetry) > 0.8):
        raise ValueError("HG asymmetry must be finite and in [-0.8, 0.8]")
    cosine = _phase_cosine(cosine)
    phase = np.asarray(
        (1 - asymmetry**2) / (1 + asymmetry**2 - 2 * asymmetry * cosine) ** 1.5,
        dtype=np.float64,
    )
    return float(phase) if scalar_result else phase


@overload
def rayleigh_phase(cosine: float) -> float: ...


@overload
def rayleigh_phase(cosine: NDArray[np.float64]) -> NDArray[np.float64]: ...


def rayleigh_phase(cosine: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Scalar (unpolarised) Rayleigh phase P=3/4*(1+cosine²), 4π mean one.

    Dimensionless float64 cosine must be finite in [-1, 1], without clipping.
    Scalar input returns a Python float; array input returns a fresh writable
    array of the input shape, including for zero-dimensional input. Inputs are
    never mutated or retained.

    Args:
        cosine: Scalar or array scattering-angle cosine in [-1,1].

    Returns:
        Scalar or float64 array of dimensionless Rayleigh phase values.

    Raises:
        ValueError: The scattering angle is nonfinite or outside [-1,1].
    """
    scalar_result = np.isscalar(cosine)
    cosine = _phase_cosine(cosine)
    phase = np.asarray(0.75 * (1 + cosine**2), dtype=np.float64)
    return float(phase) if scalar_result else phase


def _phase_cosine(cosine: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a float64 scattering-angle cosine array within physical support.

    The returned array may share storage with an array input.

    Returns:
        Dimensionless cosine values within [-1, 1].

    Raises:
        ValueError: Cosine is nonfinite or outside [-1, 1].
    """
    values = np.asarray(cosine, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any((values < -1) | (values > 1)):
        raise ValueError("scattering-angle cosine must be finite and in [-1, 1]")
    return values
