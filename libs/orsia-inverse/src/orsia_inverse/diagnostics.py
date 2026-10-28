"""Local Gauss-Newton information, separate from optimisation and scientific qualification."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .estimation import GaussianEstimate
from .weighting import covariance_factor, whiten


@dataclass(frozen=True, eq=False)
class RetrievalDiagnostics:
    """Local state information and measurement sensitivity at an estimated state.

    Ranks/condition numbers concern whitened Jacobians, not their squared normal
    matrices. An unavailable diagnostic retains ranks but no posterior/gain/AK.
    Gain has shape (n,m) in state/measurement order, AK (n,n), and covariance
    shape (n,n) in squared dimensionless state units. Arrays returned by
    retrieval_diagnostics are writable copies.

    Attributes:
        available: Whether the unconstrained local Gaussian matrices were computed.
        reason: Cause of unavailability, or None when available.
        measurement_rank: Rank of the whitened measurement Jacobian.
        augmented_rank: Rank after including the prior Jacobian, if provided.
        measurement_condition: Spectral condition of measurement sensitivity.
        augmented_condition: Spectral condition after prior augmentation.
        posterior_covariance: Optional (state, state) local covariance.
        gain: Optional (state, measurement) sensitivity to observations.
        averaging_kernel: Optional (state, state) retrieval response matrix.
        degrees_of_freedom: Trace of the averaging kernel when available.
    """

    available: bool
    reason: str | None
    measurement_rank: int
    augmented_rank: int
    measurement_condition: float
    augmented_condition: float
    posterior_covariance: NDArray[np.float64] | None
    gain: NDArray[np.float64] | None
    averaging_kernel: NDArray[np.float64] | None
    degrees_of_freedom: float | None


def retrieval_diagnostics(
    estimate: GaussianEstimate,
    *,
    include_gain: bool = False,
    include_averaging_kernel: bool = False,
    rank_rtol: float = 1e-12,
) -> RetrievalDiagnostics:
    """Compute undamped local Gaussian information at the retained final linearisation.

    SVD of B=[We K; Wa] gives posterior (B.T B)^-1 without forming the normal
    matrix. Singular values <= rank_rtol*s_max are rank deficient; no truncated
    pseudoinverse is returned. Measurement rank is reported separately from prior
    stabilisation. Non-success, backend-active bounds, and deficient augmented
    rank make covariance/gain/AK unavailable. Near-but-inactive bounds and strong
    nonlinearity still need caller perturbation/ensemble checks: this is neither
    a constrained posterior nor scientific validation. Gain uses assumed fixed
    measurement weighting; propagate actual source covariances and signed biases
    separately with G @ C @ G.T and G @ bias, rather than relabelling posterior as noise.
    DOFS=trace(A) is computed even when the optional full AK is not retained.

    Args:
        estimate: Retained MAP state, Jacobian, covariance and termination details.
        include_gain: Retain the full state-to-measurement sensitivity when available.
        include_averaging_kernel: Retain the full state-to-state response when available.
        rank_rtol: Relative singular-value threshold for rank decisions.

    Returns:
        Availability, rank and optional local covariance/sensitivity diagnostics.

    Raises:
        ValueError: The rank threshold is invalid or local calculations overflow.
    """
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must be finite and between zero and one")
    n = estimate.state.size
    factor = covariance_factor(
        estimate.observation_covariance, estimate.prediction.size, "observation_covariance"
    )
    measured = whiten(factor, estimate.jacobian)
    augmented = measured
    if estimate.prior_covariance is not None:
        prior_factor = covariance_factor(estimate.prior_covariance, n, "prior_covariance")
        augmented = np.vstack((measured, whiten(prior_factor, np.eye(n))))
    if not np.all(np.isfinite(augmented)):
        raise ValueError("whitened diagnostic Jacobian must be finite")
    measurement_singular = np.linalg.svd(measured, compute_uv=False)
    _, singular, right = np.linalg.svd(augmented, full_matrices=False)
    measurement_rank, measurement_condition = _rank(measurement_singular, n, rank_rtol)
    augmented_rank, augmented_condition = _rank(singular, n, rank_rtol)
    reason = None
    if not estimate.success:
        reason = "numerical termination was not successful"
    elif np.any(estimate.active_mask):
        reason = "active bounds: unconstrained Gaussian diagnostics are unavailable"
    elif augmented_rank < n:
        reason = "augmented Jacobian is rank deficient at the selected tolerance"
    posterior, gain, averaging_kernel, dofs = None, None, None, None
    if reason is None:
        scaled_right = right.T / singular
        matrix = scaled_right @ scaled_right.T
        posterior = matrix
        # Use the whitened sensitivity to avoid an m-by-m whitening matrix.
        sensitivity = scaled_right @ (measured @ scaled_right).T
        dofs = float(np.sum(sensitivity * measured.T))
        if include_averaging_kernel:
            averaging_kernel = np.copy(sensitivity @ measured)
        if include_gain:
            gain = whiten(factor, sensitivity.T, transpose=True).T.copy()
        for values in (posterior, gain, averaging_kernel, dofs):
            if values is not None and not np.all(np.isfinite(values)):
                raise ValueError("local diagnostics overflow")
    return RetrievalDiagnostics(
        reason is None,
        reason,
        measurement_rank,
        augmented_rank,
        measurement_condition,
        augmented_condition,
        posterior,
        gain,
        averaging_kernel,
        dofs,
    )


def _rank(singular: NDArray[np.float64], columns: int, rtol: float) -> tuple[int, float]:
    rank = int(np.count_nonzero(singular > rtol * singular[0]))
    condition = float(singular[0] / singular[-1]) if rank == columns else float("inf")
    return rank, condition
