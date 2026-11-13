"""Evaluate compensated spatial contrasts with correlated independent/shared errors."""

from collections.abc import Callable, Mapping
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]


class ApertureContrast(TypedDict):
    """Spatial contrast, uncertainty and strongest supported aperture.

    Arrays own writable storage. Field-unit quantities retain the signal's unit;
    score quantities are dimensionless. Unsupported floating entries are NaN.

    Attributes:
        score_sigma: (row,column) contrast divided by its standard uncertainty.
        sigma: (row,column) contrast standard uncertainty in field units.
        valid_support_mask: Boolean (row,column) supported-aperture centres.
        best_score_sigma: Largest supported score, using row-major C-order ties.
        best_row: Zero-based best aperture row.
        best_column: Zero-based best aperture column.
        best_contrast: Signed contrast at that centre, in field units.
        best_sigma: Standard uncertainty at that centre, in field units.
        variance_contributions: Optional signed contributions in squared field units.
    """

    score_sigma: FloatArray
    sigma: FloatArray
    valid_support_mask: BoolArray
    best_score_sigma: float
    best_row: int
    best_column: int
    best_contrast: float
    best_sigma: float
    variance_contributions: dict[str, float]


def scan_compensated_apertures(
    signal: FloatArray,
    valid: BoolArray,
    covariance: Callable[[NDArray[np.int64], FloatArray], float],
    *,
    variance_terms: Mapping[str, Callable[[NDArray[np.int64], FloatArray], float]] | None = None,
    core_radius: int = 1,
    background_inner: int = 3,
    background_outer: int = 6,
    minimum_valid_fraction: float = 1.0,
) -> ApertureContrast:
    """Scan signed core-minus-annulus contrasts with caller-supplied field covariance.

    Each covariance call receives selected flat C-order field indices and signed
    aperture weights. It returns their covariance quadratic form in squared
    field units. The caller composes all correlations, including cross-source
    terms, before this boundary. Invalid field samples never enter a contraction.

    Args:
        signal: Borrowed float64 scalar field (row,column), finite on valid samples.
        valid: Boolean sample support with the same shape.
        covariance: Pure selected-index covariance quadratic-form callable.
        variance_terms: Optional named quadratic-form contributions at the best
            aperture. Signed cross terms are allowed. These supplied terms are
            reported independently; total variance always comes from covariance.
        core_radius: Positive circular core radius, in pixels.
        background_inner: Annulus inner radius greater than core_radius.
        background_outer: Annulus outer radius greater than background_inner.
        minimum_valid_fraction: Required valid fraction of each aperture component.

    Returns:
        Writable score/uncertainty/support maps, signed best contrast and optional
        signed variance contributions. Best indices are zero-based; ties use C order.

    Raises:
        ValueError: Shapes, valid values, controls, variance or available centers fail.
    """
    if (
        signal.ndim != 2
        or signal.dtype != np.float64
        or valid.shape != signal.shape
        or valid.dtype != np.bool_
        or not np.isfinite(signal[valid]).all()
    ):
        raise ValueError("signal and valid must share finite supported two-dimensional axes")
    if (
        not 0 < core_radius < background_inner < background_outer
        or not 0 < minimum_valid_fraction <= 1
    ):
        raise ValueError("invalid compensated aperture")
    radius = background_outer
    rr, cc = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance = rr**2 + cc**2
    core = distance <= core_radius**2
    background = (distance >= background_inner**2) & (distance <= radius**2)
    score = np.full(signal.shape, np.nan)
    sigma, contrast = score.copy(), score.copy()
    support = np.zeros(signal.shape, bool)
    best_indices = np.empty(0, dtype=np.int64)
    best_weights = np.empty(0, dtype=np.float64)
    best_score = -np.inf
    for r in range(radius, signal.shape[0] - radius):
        for c in range(radius, signal.shape[1] - radius):
            region = (slice(r - radius, r + radius + 1), slice(c - radius, c + radius + 1))
            core_valid = core & valid[region]
            background_valid = background & valid[region]
            if core_valid.sum() < np.ceil(
                core.sum() * minimum_valid_fraction
            ) or background_valid.sum() < np.ceil(background.sum() * minimum_valid_fraction):
                continue
            aperture = core_valid / core_valid.sum() - background_valid / background_valid.sum()
            selected = aperture != 0
            indices = ((r + rr[selected]) * signal.shape[1] + c + cc[selected]).astype(np.int64)
            weights = aperture[selected]
            total = covariance(indices, weights)
            if not np.isfinite(total) or total <= 0:
                raise ValueError("aperture variance must be finite and positive")
            contrast[r, c] = weights @ signal.ravel()[indices]
            sigma[r, c] = np.sqrt(total)
            score[r, c] = contrast[r, c] / sigma[r, c]
            support[r, c] = True
            if score[r, c] > best_score:
                best_score = float(score[r, c])
                best_indices, best_weights = indices, weights
    if not support.any():
        raise ValueError("no supported aperture")
    best = np.unravel_index(np.nanargmax(score), score.shape)
    terms = (
        {}
        if variance_terms is None
        else {
            name: float(term(best_indices, best_weights)) for name, term in variance_terms.items()
        }
    )
    if not all(np.isfinite(value) for value in terms.values()):
        raise ValueError("reported variance contributions must be finite")
    return dict(
        score_sigma=score,
        sigma=sigma,
        valid_support_mask=support,
        best_score_sigma=float(score[best]),
        best_row=int(best[0]),
        best_column=int(best[1]),
        best_contrast=float(contrast[best]),
        best_sigma=float(sigma[best]),
        variance_contributions=terms,
    )


def aperture_scorer(
    valid: BoolArray,
    radii: tuple[int, int, int],
    support: BoolArray,
    sigma: FloatArray,
) -> Callable[[FloatArray], FloatArray]:
    """Prepare a fixed core-minus-annulus normalization for independent score maps.

    Args:
        valid: Borrowed Boolean (row,column) data support.
        radii: Core, inner and outer annulus radii in pixels.
        support: Borrowed Boolean supported aperture centres on valid's grid.
        sigma: Borrowed positive contrast RMS on supported centres, in field units.

    Returns:
        Callable from a field in sigma's units to a fresh dimensionless score map;
        unsupported centres contain NaN. Keep inputs stable while using the callable.
    """
    core_radius, inner, outer = radii
    rr, cc = np.mgrid[-outer : outer + 1, -outer : outer + 1]
    radius = rr**2 + cc**2
    core = (radius <= core_radius**2).astype(float)
    background = ((radius >= inner**2) & (radius <= outer**2)).astype(float)
    core_count = convolve(valid.astype(float), core, mode="constant", cval=0)
    background_count = convolve(valid.astype(float), background, mode="constant", cval=0)

    def score(field: FloatArray) -> FloatArray:
        masked = np.where(valid, field, 0)
        result = np.full(valid.shape, np.nan)
        result[support] = (
            convolve(masked, core, mode="constant", cval=0)[support] / core_count[support]
            - convolve(masked, background, mode="constant", cval=0)[support]
            / background_count[support]
        ) / sigma[support]
        return result

    return score


def calibrate_null_threshold(
    samples: FloatArray, false_alarm_probability: float
) -> tuple[float, int]:
    """Calibrate a strict exceedance threshold on an independent null population.

    Args:
        samples: Finite one-dimensional calibration maxima, excluding validation draws.
        false_alarm_probability: Target upper-tail probability strictly between zero and one.

    Returns:
        Threshold and one-based order-statistic rank ceil((N+1)*(1-probability)).
        Validation and detection must compare with strict greater-than separately.

    Raises:
        ValueError: Probability, population or requested finite-sample rank is invalid.
    """
    rank = int(np.ceil((len(samples) + 1) * (1 - false_alarm_probability)))
    if (
        samples.ndim != 1
        or not np.isfinite(samples).all()
        or not 0 < false_alarm_probability < 1
        or rank > len(samples)
        or rank < 1
    ):
        raise ValueError("finite null population and attainable false-alarm target required")
    return float(np.sort(samples)[rank - 1]), rank
