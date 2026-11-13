"""Estimate one robust offset shared by normalized spectra with declared errors.

Selection, repeated-observation dependence and two-pass variance bookkeeping
remain caller responsibilities. The conditional covariance stays a common source.
"""

from collections.abc import Sequence
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from orsia_inverse.contrast import helmert_basis, normalized_contrast

type FloatArray = NDArray[np.float64]
type ArrayInput = NDArray[np.floating | np.integer] | Sequence[float] | Sequence[Sequence[float]]


class SharedContrastOffset(TypedDict):
    """One scene-shared normalized-radiance correction and conditional covariance.

    All returned arrays own writable storage. The covariance is common to every
    corrected spectrum; repeated per-pixel independent noise would change its meaning.

    Attributes:
        channel_relative_correction: (band,) zero-mean dimensionless correction.
        contrast_correction: (band-1,) normalized Helmert-coordinate correction.
        contrast_posterior_covariance: (band-1,band-1) common dimensionless covariance;
            NaN when the selected population is too small.
        converged: Whether a robust correction update met the stopping tolerance.
        iterations: Number of attempted robust updates, at most four.
        selected_pixel_count: Number of supplied preselected spectra.
    """

    channel_relative_correction: FloatArray
    contrast_correction: FloatArray
    contrast_posterior_covariance: FloatArray
    converged: bool
    iterations: int
    selected_pixel_count: int


def estimate_shared_contrast_offset(
    measurements: ArrayInput,
    predictions: ArrayInput,
    covariances: ArrayInput,
    *,
    minimum_pixels: int = 128,
    huber_threshold: float = 3.0,
) -> SharedContrastOffset:
    """Estimate the source four-iteration robust scene-shared contrast correction.

    Inputs are preselected (pixel,band), (pixel,band) and (pixel,band,band).
    Selection belongs to the caller. Returns correction and its common
    covariance, never an independently repeated per-pixel noise term. Too few
    pixels returns zero correction and converged=False with NaN covariance.

    Args:
        measurements: Borrowed finite (pixel,band) spectra with positive per-spectrum mean.
        predictions: Matching finite predicted spectra with positive mean, same radiance unit.
        covariances: (pixel,band,band) conditional covariance in squared radiance units.
        minimum_pixels: Minimum preselected pixel count to attempt the fit.
        huber_threshold: Huber multiplier of covariance-normalized residual RMS.

    Returns:
        Owned relative channel/contrast correction, common contrast covariance and
        iteration/selection diagnostics. Insufficient pixels return zero correction,
        NaN covariance and converged=False; no independent-pixel duplication is made.

    Raises:
        ValueError: Spectral/covariance axes or normalized-contrast domains fail.
        numpy.linalg.LinAlgError: Conditional covariance or information cannot factor.
    """
    y, predicted, cov = (np.asarray(v, float) for v in (measurements, predictions, covariances))
    if y.ndim != 2 or predicted.shape != y.shape or cov.shape != (*y.shape, y.shape[1]):
        raise ValueError("shared contrast-offset array axes differ")
    h = helmert_basis(y.shape[1])
    correction = np.zeros(h.shape[1])
    posterior = np.full((h.shape[1], h.shape[1]), np.nan)
    converged, iterations = False, 0
    if len(y) >= minimum_pixels:
        residuals, precisions = [], []
        for observed, prediction, covariance in zip(y, predicted, cov, strict=True):
            z, _, cz = normalized_contrast(observed, covariance)
            residuals.append(z - normalized_contrast(prediction)[0])
            assert cz is not None
            precisions.append(cho_solve(cho_factor(cz), np.eye(len(z))))
        residuals, precisions = np.array(residuals), np.array(precisions)
        for iteration in range(1, 5):
            offset = residuals - correction
            norm = np.sqrt(
                np.maximum(
                    np.einsum("ni,nij,nj->n", offset, precisions, offset) / len(correction), 0
                )
            )
            weights = np.minimum(1.0, huber_threshold / np.maximum(norm, np.finfo(float).eps))
            information = np.einsum("n,nij->ij", weights, precisions)
            rhs = np.einsum("n,nij,nj->i", weights, precisions, residuals)
            factor = cho_factor(information)
            updated = cast(FloatArray, cho_solve(factor, rhs))
            posterior = cast(FloatArray, cho_solve(factor, np.eye(len(correction))))
            iterations = iteration
            converged = np.linalg.norm(updated - correction) <= 1e-10
            correction = updated
            if converged:
                break
    return dict(
        channel_relative_correction=h @ correction,
        contrast_correction=correction,
        contrast_posterior_covariance=(posterior + posterior.T) / 2,
        converged=bool(converged),
        iterations=iterations,
        selected_pixel_count=len(y),
    )
