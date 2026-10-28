"""Array-only fixed Gaussian weighting shared by estimation and diagnostics."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular


def covariance_factor(covariance: NDArray[np.float64], size: int, name: str) -> NDArray[np.float64]:
    """Validate variances or a dense covariance and return an owned square root.

    A vector contains variances, not standard deviations; its factor is a vector.
    A matrix factor is lower triangular L with C=L L.T. Dense symmetry is checked
    to 1e-12 of the largest absolute entry, then roundoff asymmetry is averaged.
    Reject nonpositive or ill-conditioned spectra (min/max <=1e-12), without
    regularisation. Normalisation avoids overflow in symmetry/eigenvalue checks.
    Inputs are not modified. The factor has the base units of the covariance.

    Args:
        covariance: Finite float64 (size,) variances or (size,size) covariance.
        size: Expected number of observation or prior coordinates.
        name: Input label inserted into validation errors.

    Returns:
        Owned diagonal scale vector or lower-triangular covariance factor.

    Raises:
        ValueError: Covariance shape, entries, symmetry or condition is invalid.
    """
    if covariance.shape not in ((size,), (size, size)) or size == 0:
        raise ValueError(f"{name} must have shape ({size},) or ({size}, {size})")
    if covariance.dtype != np.float64 or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain finite float64 values")
    scale = float(np.max(np.abs(covariance)))
    if scale == 0:
        raise ValueError(f"{name} must be positive definite and well-conditioned")
    normalised = covariance / scale
    if covariance.ndim == 1:
        if np.min(normalised) <= 1e-12:
            raise ValueError(f"{name} must be positive definite and well-conditioned")
        return np.sqrt(covariance)
    if not np.allclose(normalised, normalised.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    normalised = (normalised + normalised.T) / 2
    try:
        eigenvalues = np.linalg.eigvalsh(normalised)
        if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
            raise ValueError(f"{name} must be positive definite and well-conditioned")
        return np.linalg.cholesky(normalised) * np.sqrt(scale)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} factorisation failed") from exc


def whiten(
    factor: NDArray[np.float64], values: NDArray[np.float64], *, transpose: bool = False
) -> NDArray[np.float64]:
    """Apply L^-1 (or L^-T) to a vector or matrix of column right-hand sides.

    Accept a validated factor from covariance_factor and compatible finite
    values. No input mutation or shared work buffers; return a writable array.
    Callers check outputs where overflow can affect the numerical calculation.

    Args:
        factor: Prepared diagonal scales or lower-triangular matrix.
        values: Vector or matrix with covariance coordinates on the first axis.
        transpose: Apply the transposed inverse for sensitivity propagation.

    Returns:
        Writable whitened vector or column-wise matrix in factor coordinates.
    """
    if factor.ndim == 1:
        return values / (factor if values.ndim == 1 else factor[:, None])
    return solve_triangular(
        factor, values, lower=True, trans="T" if transpose else "N", check_finite=False
    )
