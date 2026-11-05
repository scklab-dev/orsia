"""Solve bounded linear inverse problems with explicit observable-subspace diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import factor_covariance, whiten


@dataclass(frozen=True, eq=False)
class LinearEstimate:
    """Weighted minimum-norm state and its identifiable-subspace information.

    Factory outputs own writable arrays. Null-space components are set to zero
    by the minimum-norm convention; their physical uncertainty is unconstrained.
    The covariance describes noise in this estimator, so zero null-space
    variance must not be interpreted as knowledge of the true state.

    Attributes:
        state: (state,) estimate in caller-defined units and coordinate order.
        covariance: (state,state) estimator sampling covariance in squared units.
        resolution: Signed (state,state) expected-state response, gain times model.
        gain: (state,read) linear estimator; excluded observations have zero columns.
        prediction: (read,) model prediction, including excluded reads.
        used: (read,) Boolean mask of supplied usable observations.
        identifiable: (state,) coordinates individually resolved at the retained rank.
        singular_values: Whitened model singular values in decreasing order.
        rank: Number of singular values retained above the relative threshold.
        condition: Largest/smallest retained singular ratio, infinity if rank deficient.
        weighted_residual_norm: Euclidean norm of the whitened usable residual.
    """

    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    resolution: NDArray[np.float64]
    gain: NDArray[np.float64]
    prediction: NDArray[np.float64]
    used: NDArray[np.bool_]
    identifiable: NDArray[np.bool_]
    singular_values: NDArray[np.float64]
    rank: int
    condition: float
    weighted_residual_norm: float


def solve_linear(
    matrix: NDArray[np.float64],
    observations: NDArray[np.float64],
    *,
    covariance: NDArray[np.float64],
    valid: NDArray[np.bool_] | None = None,
    rcond: float = 1e-10,
    max_bytes: int,
) -> LinearEstimate:
    """Estimate a static linear state using covariance whitening and truncated SVD.

    The covariance is fixed and supplied independently of the observed noise.
    Excluded observations can be NaN. Selection takes the covariance principal
    submatrix, retaining correlations among usable reads. No prior, positivity
    constraint or automatic regularization is imposed. State scaling affects
    numerical rank and the minimum-norm convention; callers choose state units.
    Dense arrays are bounded before factorization using a conservative declared
    workspace estimate; backend allocator overhead is outside this accounting.

    Args:
        matrix: Finite (reads,state) physical or signed linear model.
        observations: (reads,) measured values in the model's output units.
        covariance: Float64 (reads,) variances or (reads,reads) covariance in squared units.
        valid: Optional Boolean (reads,) quality mask; true entries must be finite.
        rcond: Relative positive singular-value cutoff strictly below one.
        max_bytes: Positive explicit upper bound for estimated dense working storage.

    Returns:
        Minimum-norm estimate, covariance, signed resolution, rank and validity diagnostics.
        No usable reads yields zero state/resolution and no identifiable coordinates.

    Raises:
        ValueError: Shapes, finite values, covariance, quality mask or cutoff are invalid.
        MemoryError: Estimated dense working storage exceeds the caller's budget.
    """
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("matrix must be nonempty (reads,state)")
    m, n = matrix.shape
    if observations.shape != (m,) or covariance.shape not in ((m,), (m, m)):
        raise ValueError("observations and covariance must follow matrix read order")
    if not np.isfinite(rcond) or not 0 < rcond < 1 or max_bytes < 1:
        raise ValueError("positive memory budget and relative cutoff in (0,1) required")
    # Include caller inputs, selected matrices, SVD, gain, covariance and resolution.
    required = 8 * (8 * m * n + 8 * n * n + (8 * m * m if covariance.ndim == 2 else 8 * m))
    if required > max_bytes:
        raise MemoryError(
            f"linear solve requires estimated {required} bytes; budget is {max_bytes}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    used = np.ones(m, dtype=np.bool_) if valid is None else np.copy(valid)
    if used.shape != (m,) or used.dtype != np.bool_ or not np.isfinite(observations[used]).all():
        raise ValueError("valid must be Boolean and usable observations finite")
    gain = np.zeros((n, m))
    resolution = np.zeros((n, n))
    state = np.zeros(n)
    sampling = np.zeros((n, n))
    singular = np.empty(0)
    rank = 0
    residual_norm = 0.0
    condition = np.inf
    if used.any():
        selected_covariance = (
            covariance[used] if covariance.ndim == 1 else covariance[np.ix_(used, used)]
        )
        factor = factor_covariance(selected_covariance, method="cholesky")
        model = matrix[used]
        whitened = whiten(factor, model)
        u, singular, vt = np.linalg.svd(whitened, full_matrices=False)
        keep = singular > rcond * singular[0]
        rank = int(keep.sum())
        inverse = (vt[keep].T / singular[keep]) @ u[:, keep].T
        selected_gain = whiten(factor, inverse.T, transpose=True).T
        gain[:, used] = selected_gain
        state = selected_gain @ observations[used]
        resolution = selected_gain @ model
        sampling = inverse @ inverse.T
        residual_norm = float(np.linalg.norm(whiten(factor, model @ state - observations[used])))
        if rank == n:
            condition = float(singular[0] / singular[rank - 1])
    # Coordinate identity is independent of the user-selected singular cutoff.
    identifiable = np.max(np.abs(resolution - np.eye(n)), axis=1) <= 1e-8
    return LinearEstimate(
        state,
        sampling,
        resolution,
        gain,
        matrix @ state,
        used,
        identifiable,
        singular,
        rank,
        condition,
        residual_norm,
    )
