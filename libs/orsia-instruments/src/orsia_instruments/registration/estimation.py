"""Estimate image translations and row-knot corrections with signed interpolation.

Coordinates are zero-based pixels; correction samples the moving image at
output coordinate plus correction. Numerical inputs are borrowed unchanged."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, TypedDict, Unpack

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, minimum_filter
from scipy.sparse import csr_matrix

from orsia_instruments.registration.resampling import (
    STENCIL_BLOCK_CELLS,
    RegistrationMap,
    interpolation_stencil,
    interpolation_weights,
)

type FloatArray = NDArray[np.float64]


type BoolArray = NDArray[np.bool_]


type IntArray = NDArray[np.int64]


type ArrayInput = (
    float
    | np.floating
    | NDArray[np.floating | np.integer | np.bool_]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
)


class TranslationEstimate(TypedDict):
    """Constant image displacement and conditional fit diagnostics.

    Returned arrays own writable storage. Pixel coordinates follow row then column.

    Attributes:
        correction_pixel: (2,) correction added to output coordinates when sampling moving.
        covariance_pixel2: (2,2) conditional row/column displacement covariance, pixel2.
        correlation: Normalized reference/prediction correlation on fixed support.
        sample_count: Number of fixed supported image samples in the fit.
        band: Optional caller-supplied band index for product diagnostics.
    """

    correction_pixel: FloatArray
    covariance_pixel2: FloatArray
    correlation: float
    sample_count: int
    band: NotRequired[int]


class RowRegistrationEstimate(TranslationEstimate):
    """Row-knot displacement field, conditional covariance and photometric fit.

    Inherited fields describe the mean correction and fixed-support diagnostics.
    Returned numerical arrays own writable storage.

    Attributes:
        row_correction_pixel: (row,2) interpolated row/column correction in pixels.
        row_knot_rows: (knot,) zero-based integer knot-row coordinates.
        row_joint_covariance_pixel2: (2*knot,2*knot) conditional displacement covariance,
            with row/column coordinates interleaved per knot, in pixel2.
        gain: Positive dimensionless fitted standardized-image gain.
        offset: Dimensionless fitted standardized-image offset.
    """

    row_correction_pixel: FloatArray
    row_knot_rows: IntArray
    row_joint_covariance_pixel2: FloatArray
    gain: float
    offset: float


class TranslationOptions(TypedDict, total=False):
    """Optional numerical controls for constant-translation estimation.

    Attributes:
        maximum_shift: Absolute correction bound in pixels, in (0.15,1).
        maximum_iterations: Positive maximum robust-refinement iterations.
        tolerance: Final maximum correction-update magnitude, pixels.
        minimum_samples: Minimum fixed supported observations.
        minimum_correlation: Minimum normalized-image correlation.
        robust_threshold: Huber multiplier of robust residual scale.
        maximum_disagreement: Maximum coarse/refined correction difference, pixels.
        kernel: keys_cubic, lanczos3 or lanczos5 signed interpolation.
        smoothing_sigma: Log-image Gaussian smoothing sigma, pixels.
    """

    maximum_shift: float
    maximum_iterations: int
    tolerance: float
    minimum_samples: int
    minimum_correlation: float
    robust_threshold: float
    maximum_disagreement: float
    kernel: str
    smoothing_sigma: float


def sample_image(
    image: ArrayInput,
    row: ArrayInput,
    column: ArrayInput,
    support: ArrayInput | None = None,
    kernel: str = "lanczos5",
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """Sample values and two derivatives using one complete signed source stencil.

    Image/support are (row,column). Coordinate arrays share any shape. Returns
    owned values, row/column derivatives and Boolean support; invalid values NaN.
    Input arrays remain unchanged. Derivatives are image units per pixel.

    Args:
        image: Borrowed two-dimensional image in caller units.
        row: Finite zero-based row coordinates, arbitrary shape.
        column: Matching column coordinates, pixels.
        support: Optional Boolean image support; nonfinite samples are always excluded.
        kernel: keys_cubic, lanczos3 or lanczos5 signed kernel.

    Returns:
        Owned values, row/column derivatives and Boolean support with coordinate axes.
        Invalid values/derivatives are NaN; derivatives use image units per pixel.

    Raises:
        ValueError: Image/coordinate axes, kernel or coordinate finiteness fail.
    """
    value, dr, dc, valid = _sample_image(image, row, column, support, kernel, derivatives=True)
    assert dr is not None and dc is not None
    return value, dr, dc, valid


def registration_operator(
    shape: tuple[int, int, int],
    corrections: ArrayInput,
    support: BoolArray,
    *,
    reference_band: int = 0,
    kernel: str = "lanczos5",
) -> tuple[RegistrationMap, BoolArray]:
    """Return compact signed resampling and its read-only complete-stencil support.

    Shape is (row,column,band); corrections are (band,2) or (row,band,2) pixels.
    The map owns read-only copies of corrections and source support. Zero-shift
    rows use exact identity including borders. Use apply/rows/propagate_any for
    bounded operations; a global registration CSR is deliberately unavailable.
    Invalid shapes, shifts, reference band or kernel raise ValueError.

    Args:
        shape: Positive (row,column,band) integer dimensions in C-order cell identity.
        corrections: (band,2) or (row,band,2) row/column pixel corrections.
        support: Boolean source cube support.
        reference_band: Zero-based band with exactly zero correction.
        kernel: keys_cubic, lanczos3 or lanczos5 signed kernel.

    Returns:
        Owned compact map and a read-only view of its complete-stencil support.

    Raises:
        ValueError: Shape, correction, reference band or interpolation domain fails.
    """
    mapping = RegistrationMap(
        shape, corrections, support, reference_band=reference_band, kernel=kernel
    )
    return mapping, mapping.valid


def highpass_log_radiance(
    radiance: ArrayInput, support: ArrayInput, smoothing_sigma: float = 0.7
) -> tuple[FloatArray, BoolArray]:
    """Prepare source log-radiance smoothing/highpass with complete FIR support.

    Radiance/support have (row,column) axes; smoothing sigma is pixels. Returns
    high-pass values and eroded support. Masks exclude nonpositive/nonfinite values.

    Args:
        radiance: Borrowed (row,column) radiance in any consistent positive unit.
        support: Boolean image support, intersected with positive finite radiance.
        smoothing_sigma: FIR Gaussian smoothing width in pixels; nonpositive skips it.

    Returns:
        Owned high-pass log-radiance and complete-FIR Boolean support.
    """
    y = np.asarray(radiance, float)
    mask = np.asarray(support, bool) & np.isfinite(y) & (y > 0)
    image = np.log(y, where=mask, out=np.zeros_like(y))
    if smoothing_sigma > 0:
        radius = int(np.ceil(3 * smoothing_sigma))
        kernel = _gaussian(smoothing_sigma, radius)
        weight = convolve(mask.astype(float), kernel, mode="constant", cval=0)
        image = convolve(image * mask, kernel, mode="constant", cval=0)
        mask &= np.abs(weight - kernel.sum()) <= 1e-12
    kernel = _gaussian(2.5, 4)
    blurred = convolve(image * mask, kernel, mode="constant", cval=0)
    weight = convolve(mask.astype(float), kernel, mode="constant", cval=0)
    mask &= np.abs(weight - kernel.sum()) <= 1e-12 * max(kernel.sum(), 1)
    return np.where(mask, image - blurred, 0), mask


def estimate_translation(
    reference: ArrayInput,
    moving: ArrayInput,
    reference_support: ArrayInput,
    moving_support: ArrayInput,
    *,
    maximum_shift: float = 0.5,
    maximum_iterations: int = 8,
    tolerance: float = 1e-4,
    minimum_samples: int = 256,
    minimum_correlation: float = 0.5,
    robust_threshold: float = 3.0,
    maximum_disagreement: float = 0.05,
    kernel: str = "lanczos5",
    smoothing_sigma: float = 0.7,
) -> TranslationEstimate:
    """Estimate source constant translation from positive measured radiance images.

    Returns correction pixels, IRLS covariance, correlation and fixed sample count.
    Search and normalized-derivative robust refinement follow source translation.jl.
    Raises ValueError for support, texture, bounds, disagreement or convergence failure.

    Args:
        reference: Positive reference radiance image (row,column), any consistent unit.
        moving: Corresponding moving radiance image.
        reference_support: Boolean reference support.
        moving_support: Boolean moving support.
        maximum_shift: Absolute correction bound in pixels, in (0.15,1).
        maximum_iterations: Positive maximum robust-refinement iterations.
        tolerance: Maximum final correction update, pixels.
        minimum_samples: Minimum fixed supported sample count.
        minimum_correlation: Minimum normalized-image correlation.
        robust_threshold: Positive Huber threshold multiplier of robust residual scale.
        maximum_disagreement: Maximum coarse/refined correction difference, pixels.
        kernel: keys_cubic, lanczos3 or lanczos5 interpolation.
        smoothing_sigma: Log-image Gaussian smoothing sigma, pixels.

    Returns:
        Owned row/column correction, conditional pixel-squared covariance, correlation
        and fixed sample count. Corrections sample moving at output plus correction.

    Raises:
        ValueError: Controls, fixed support, texture, correlation or convergence fail.
        numpy.linalg.LinAlgError: Robust normal equations are singular.
    """
    if not 0.15 < maximum_shift < 1 or maximum_iterations < 1:
        raise ValueError("residual shift bound must lie in (.15,1)")
    ref, rs = highpass_log_radiance(reference, reference_support, smoothing_sigma)
    mov, ms = highpass_log_radiance(moving, moving_support, smoothing_sigma)
    radius = {"keys_cubic": 2, "lanczos3": 3, "lanczos5": 5}[kernel] + int(np.floor(maximum_shift))
    fixed = rs & minimum_filter(ms, size=2 * radius + 1, mode="constant", cval=0)
    rr, cc = np.nonzero(fixed)
    if len(rr) < minimum_samples:
        raise ValueError("insufficient fixed registration support")
    target = ref[rr, cc]
    target = _standardize(target)

    def evaluate(
        shift: FloatArray, *, derivatives: bool = True
    ) -> tuple[FloatArray, FloatArray | None]:
        value, dr, dc, valid = _sample_image(
            mov, rr + shift[0], cc + shift[1], ms, kernel, derivatives=derivatives
        )
        if not valid.all():
            raise ValueError("registration changed fixed support")
        normalized = _standardize(value)
        if not derivatives:
            return normalized, None
        assert dr is not None and dc is not None
        scale = value.std()
        jac = np.stack((dr, dc), axis=1)
        jac = (
            jac - jac.mean(axis=0) - normalized[:, None] * (normalized[:, None] * jac).mean(axis=0)
        ) / scale
        return normalized, jac

    limit, best, score = maximum_shift - 0.15, np.zeros(2), -np.inf
    for step, bounds in ((0.1, None), (0.02, 0.1)):
        low = np.full(2, -limit) if bounds is None else np.maximum(best - bounds, -limit)
        high = np.full(2, limit) if bounds is None else np.minimum(best + bounds, limit)
        for r in np.arange(low[0], high[0] + 1e-12, step):
            for c in np.arange(low[1], high[1] + 1e-12, step):
                candidate = np.array([r, c])
                value, _ = evaluate(candidate, derivatives=False)
                candidate_score = np.mean(target * value)
                if candidate_score > score:
                    best, score = candidate, candidate_score
    initial, correction = best.copy(), best.copy()

    def equations(
        shift: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float, FloatArray, FloatArray, FloatArray]:
        value, jac = evaluate(shift)
        assert jac is not None
        residual = target - value
        scale = 1.4826 * np.median(abs(residual - np.median(residual)))
        threshold = max(robust_threshold * scale, np.sqrt(np.finfo(float).eps))
        weights = np.minimum(1, threshold / np.maximum(abs(residual), np.finfo(float).eps))
        info = jac.T @ (weights[:, None] * jac)
        if np.linalg.det(info) <= 1e-10 * np.max(np.diag(info)) ** 2:
            raise ValueError("scene texture does not constrain both axes")
        return (
            value,
            residual,
            float(threshold),
            weights,
            info,
            np.linalg.solve(info, jac.T @ (weights * residual)),
        )

    for _ in range(maximum_iterations):
        _, residual, threshold, _, _, update = equations(correction)
        if np.max(abs(update)) <= tolerance:
            break
        loss = _huber(residual, threshold)
        accepted = False
        for backtrack in range(17):
            trial = correction + 0.5**backtrack * update
            if np.max(abs(trial)) >= maximum_shift:
                continue
            trial_loss = _huber(target - evaluate(trial, derivatives=False)[0], threshold)
            if trial_loss <= loss + 32 * np.finfo(float).eps * max(loss, np.finfo(float).eps):
                correction, accepted = trial, True
                break
        if not accepted:
            raise ValueError("registration refinement failed to decrease robust objective")
    value, residual, _, weights, info, update = equations(correction)
    if np.max(abs(update)) > tolerance:
        raise ValueError("registration refinement did not converge")
    if np.max(abs(correction - initial)) > maximum_disagreement:
        raise ValueError("correlation and robust estimates disagree")
    correlation = np.mean(target * value)
    if correlation < minimum_correlation:
        raise ValueError("registration correlation below minimum")
    covariance = np.linalg.inv(info) * np.dot(weights, residual**2) / max(weights.sum() - 4, 1)
    return dict(
        correction_pixel=correction,
        covariance_pixel2=covariance,
        correlation=float(correlation),
        sample_count=len(rr),
    )


def estimate_row_displacements(
    reference: ArrayInput,
    moving: ArrayInput,
    reference_support: ArrayInput,
    moving_support: ArrayInput,
    *,
    knot_spacing: int,
    regularization_sigma: float,
    window_half_width: int = 8,
    row_minimum_samples: int = 2048,
    maximum_row_residual: float = 0.25,
    **options: Unpack[TranslationOptions],
) -> RowRegistrationEstimate:
    """Fit source joint row-knot displacement, gain and offset after constant fit.

    Row coordinates and corrections are zero-based pixels. Regularization is the
    RMS prior on knot-slope differences times knot spacing. Returns field and full
    conditional knot covariance; downstream source row uncertainty remains a prior.
    Raises ValueError for support, bounds, texture, correlation or convergence failure.

    Args:
        reference: Positive reference radiance image (row,column).
        moving: Corresponding moving image, in consistent radiance units.
        reference_support: Boolean reference-image support.
        moving_support: Boolean moving-image support.
        knot_spacing: Positive integer row-knot spacing, pixels.
        regularization_sigma: Positive RMS prior on scaled knot-slope differences.
        window_half_width: Row-window half width, pixels.
        row_minimum_samples: Minimum supported observations per row window.
        maximum_row_residual: Maximum row correction minus mean correction, pixels.
        options: Constant-translation controls documented by estimate_translation.

    Returns:
        Owned mean and row corrections, knot coordinates/full conditional covariance,
        photometric gain/offset, correlation and fixed sample count.

    Raises:
        ValueError: Controls, support, bounds, texture or robust convergence fail.
        numpy.linalg.LinAlgError: Joint normal equations are singular.
    """
    initial = estimate_translation(reference, moving, reference_support, moving_support, **options)
    kernel = options.get("kernel", "lanczos5")
    smoothing = options.get("smoothing_sigma", 0.7)
    maximum_shift = options.get("maximum_shift", 0.5)
    tolerance = options.get("tolerance", 1e-4)
    maximum_iterations = options.get("maximum_iterations", 8)
    robust = options.get("robust_threshold", 3.0)
    min_correlation = options.get("minimum_correlation", 0.5)
    if knot_spacing < 1 or regularization_sigma <= 0:
        raise ValueError("positive row controls required")
    ref, rs = highpass_log_radiance(reference, reference_support, smoothing)
    mov, ms = highpass_log_radiance(moving, moving_support, smoothing)
    rows = ref.shape[0]
    extra = {"keys_cubic": 0, "lanczos3": 1, "lanczos5": 3}[kernel]
    margin = int(np.ceil(3 * smoothing)) + extra
    first = window_half_width + margin
    last = rows - window_half_width - margin - 1
    if first >= last:
        raise ValueError("insufficient complete row windows")
    knots = np.arange(first, last + 1, knot_spacing, dtype=np.int64)
    if knots[-1] != last:
        knots = np.append(knots, np.int64(last))
    basis = np.stack(
        [np.interp(np.arange(rows), knots, np.eye(len(knots))[k]) for k in range(len(knots))],
        axis=1,
    )
    n = len(knots)
    penalty = np.zeros((2 * n + 2, 2 * n + 2))
    for k in range(1, n - 1):
        weights = np.array(
            [
                knot_spacing / (knots[k] - knots[k - 1]),
                -knot_spacing / (knots[k] - knots[k - 1])
                - knot_spacing / (knots[k + 1] - knots[k]),
                knot_spacing / (knots[k + 1] - knots[k]),
            ]
        )
        for coordinate in range(2):
            indices = np.array(
                [2 * (k - 1) + coordinate, 2 * k + coordinate, 2 * (k + 1) + coordinate]
            )
            penalty[np.ix_(indices, indices)] += (
                np.outer(weights, weights) / regularization_sigma**2
            )
    radius = {"keys_cubic": 2, "lanczos3": 3, "lanczos5": 5}[kernel]
    support = rs & minimum_filter(ms, size=2 * radius + 1, mode="constant", cval=0)
    rr, cc = np.nonzero(support)
    if len(rr) < options.get("minimum_samples", 256) or any(
        np.sum(abs(rr - k) <= window_half_width) < row_minimum_samples for k in knots
    ):
        raise ValueError("insufficient joint row-window support")
    target = _standardize(ref[rr, cc])
    mean = mov[rr, cc].mean()
    scale = mov[rr, cc].std()
    p = np.r_[np.tile(initial["correction_pixel"], n), 1.0, 0.0]
    # Each row depends on at most two knots; retain only their nonzero basis entries.
    local_basis = csr_matrix(basis)[rr].tocoo()
    samples = np.arange(len(rr))
    jac_rows = np.concatenate((local_basis.row, local_basis.row, samples, samples))
    jac_columns = np.concatenate(
        (
            2 * local_basis.col,
            2 * local_basis.col + 1,
            np.full(len(rr), 2 * n),
            np.full(len(rr), 2 * n + 1),
        )
    )

    def evaluate(
        parameters: FloatArray, *, derivatives: bool = True
    ) -> tuple[FloatArray, csr_matrix | None]:
        field = basis @ parameters[: 2 * n].reshape(n, 2)
        value, dr, dc, valid = _sample_image(
            mov, rr + field[rr, 0], cc + field[rr, 1], ms, kernel, derivatives=derivatives
        )
        if not valid.all():
            raise ValueError("joint field lost fixed support")
        value = (value - mean) / scale
        if not derivatives:
            return parameters[-2] * value + parameters[-1], None
        assert dr is not None and dc is not None
        jac = csr_matrix(
            (
                np.concatenate(
                    (
                        local_basis.data * dr[local_basis.row] * parameters[-2] / scale,
                        local_basis.data * dc[local_basis.row] * parameters[-2] / scale,
                        value,
                        np.ones(len(rr)),
                    )
                ),
                (jac_rows, jac_columns),
            ),
            shape=(len(rr), 2 * n + 2),
        )
        return parameters[-2] * value + parameters[-1], jac

    def equations(
        parameters: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float, float, FloatArray, FloatArray]:
        predicted, jac = evaluate(parameters)
        assert jac is not None
        residual = target - predicted
        sigma = max(
            1.4826 * np.median(abs(residual - np.median(residual))), np.sqrt(np.finfo(float).eps)
        )
        threshold = robust * sigma
        weight = (
            np.minimum(1, threshold / np.maximum(abs(residual), np.finfo(float).eps)) / sigma**2
        )
        hessian = penalty + csr_matrix(jac.T @ jac.multiply(weight[:, None])).toarray()
        rhs = jac.T @ (weight * residual) - penalty @ parameters
        return (
            predicted,
            residual,
            float(sigma),
            float(threshold),
            hessian,
            np.linalg.solve(hessian, rhs),
        )

    converged = False
    for _ in range(maximum_iterations):
        _, residual, sigma, threshold, hessian, update = equations(p)
        if np.max(abs(update)) <= tolerance:
            converged = True
            break
        loss = _huber(residual, threshold) / sigma**2 + 0.5 * p @ penalty @ p
        accepted = False
        for backtrack in range(17):
            trial = p + 0.5**backtrack * update
            if np.max(abs(trial[: 2 * n])) >= maximum_shift or trial[-2] <= 0:
                continue
            value = evaluate(trial, derivatives=False)[0]
            candidate = _huber(target - value, threshold) / sigma**2 + 0.5 * trial @ penalty @ trial
            if candidate <= loss + 32 * np.finfo(float).eps * max(loss, np.finfo(float).eps):
                p = trial
                accepted = True
                break
        if not accepted:
            raise ValueError("joint row refinement failed to decrease objective")
    if not converged:
        raise ValueError("joint row refinement did not converge")
    predicted, _, _, _, hessian, _ = equations(p)
    covariance = np.linalg.inv(hessian)[: 2 * n, : 2 * n]
    field = basis @ p[: 2 * n].reshape(n, 2)
    mean_field = field.mean(axis=0)
    if np.max(abs(field - mean_field)) >= maximum_row_residual:
        raise ValueError("row field exceeds residual bound")
    correlation = np.corrcoef(target, predicted)[0, 1]
    if correlation < min_correlation or any(
        np.corrcoef(
            target[abs(rr - k) <= window_half_width], predicted[abs(rr - k) <= window_half_width]
        )[0, 1]
        < min_correlation
        for k in knots
    ):
        raise ValueError("joint registration correlation below minimum")
    mean_map = np.kron(basis.mean(axis=0)[None, :], np.eye(2))
    return dict(
        correction_pixel=mean_field,
        row_correction_pixel=field,
        row_knot_rows=knots,
        row_joint_covariance_pixel2=covariance,
        covariance_pixel2=np.asarray(mean_map @ covariance @ mean_map.T, dtype=np.float64),
        correlation=float(correlation),
        sample_count=len(rr),
        gain=float(p[-2]),
        offset=float(p[-1]),
    )


def _sample_image(
    image: ArrayInput,
    row: ArrayInput,
    column: ArrayInput,
    support: ArrayInput | None,
    kernel: str,
    *,
    derivatives: bool,
) -> tuple[FloatArray, FloatArray | None, FloatArray | None, BoolArray]:
    """Evaluate bounded patches, optionally omitting the two image-gradient contractions.

    Returns:
        Values, optional coordinate derivatives and complete-stencil support.

    Raises:
        ValueError: Image axes, matched coordinate shapes or selected kernel fail.
    """
    image = np.asarray(image, float)
    row, column = np.asarray(row, float), np.asarray(column, float)
    if image.ndim != 2 or min(image.shape) < 1 or row.shape != column.shape:
        raise ValueError("2-D image and matched coordinates required")
    if kernel not in ("keys_cubic", "lanczos3", "lanczos5"):
        raise ValueError("supported kernel and finite coordinates required")
    support = (
        np.isfinite(image) if support is None else np.asarray(support, bool) & np.isfinite(image)
    )
    value = np.empty(row.shape)
    dr = np.empty(row.shape) if derivatives else None
    dc = np.empty(row.shape) if derivatives else None
    valid = np.empty(row.shape, bool)
    for start in range(0, row.size, STENCIL_BLOCK_CELLS):
        chunk = slice(start, start + STENCIL_BLOCK_CELLS)
        rd = cd = None
        if derivatives:
            rr, rw, rd, rv = interpolation_stencil(row.flat[chunk], image.shape[0], kernel)
            cc, cw, cd, cv = interpolation_stencil(column.flat[chunk], image.shape[1], kernel)
        else:
            rr, rw, rv = interpolation_weights(row.flat[chunk], image.shape[0], kernel)
            cc, cw, cv = interpolation_weights(column.flat[chunk], image.shape[1], kernel)
        rclip, cclip = np.clip(rr, 0, image.shape[0] - 1), np.clip(cc, 0, image.shape[1] - 1)
        patch = image[rclip[..., :, None], cclip[..., None, :]]
        good = rv & cv & np.all(support[rclip[..., :, None], cclip[..., None, :]], axis=(-2, -1))
        value.flat[chunk] = np.where(good, np.einsum("...i,...ij,...j->...", rw, patch, cw), np.nan)
        valid.flat[chunk] = good
        if derivatives:
            assert dr is not None and dc is not None and rd is not None and cd is not None
            dr.flat[chunk] = np.where(
                good, np.einsum("...i,...ij,...j->...", rd, patch, cw), np.nan
            )
            dc.flat[chunk] = np.where(
                good, np.einsum("...i,...ij,...j->...", rw, patch, cd), np.nan
            )
    return value, dr, dc, valid


def _gaussian(sigma: float, radius: int) -> FloatArray:
    v = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2 * sigma**2))
    v /= v.sum()
    return v[:, None] * v


def _standardize(values: FloatArray) -> FloatArray:
    if values.std() <= 100 * np.finfo(float).eps:
        raise ValueError("degenerate radiometric contrast")
    return (values - values.mean()) / values.std()


def _huber(residual: FloatArray, threshold: float) -> float:
    magnitude = abs(residual)
    return np.sum(
        np.where(
            magnitude <= threshold, 0.5 * residual**2, threshold * (magnitude - 0.5 * threshold)
        )
    )
