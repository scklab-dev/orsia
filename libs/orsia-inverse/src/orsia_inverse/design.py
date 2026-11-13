"""Compute Gaussian design information and deterministic discrete channel exchanges."""

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import CovarianceFactor

from orsia_inverse.diagnostics import local_gaussian_information
from orsia_inverse.priors import QuadraticPrior

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]


class LinearTargetUncertainty(TypedDict):
    """Conditional Gaussian posterior and target measurement weights.

    Attributes:
        covariance: Owned (parameter,parameter) posterior in squared state units.
        target_weights: Owned (measurement,) gain for the supplied linear target.
        target_sigma: Target standard uncertainty in its caller-defined unit.
    """

    covariance: FloatArray
    target_weights: FloatArray
    target_sigma: float


class ChannelSearchResult[T](TypedDict):
    """Initial and locally optimal selections evaluated by the supplied objective.

    Attributes:
        initial: Initial caller-defined evaluation.
        optimized: Best accepted evaluation, with no global-optimality claim.
        iterations: Number of accepted exchanges.
        converged: Whether a complete sweep found no improving exchange.
    """

    initial: T
    optimized: T
    iterations: int
    converged: bool


def linear_target_uncertainty(
    design: FloatArray,
    observation_factor: CovarianceFactor,
    target: FloatArray,
    *,
    prior: QuadraticPrior | None = None,
    rank_rtol: float = 1e-12,
    max_bytes: int = 256 * 1024 * 1024,
) -> LinearTargetUncertainty:
    """Compute local uncertainty and measurement gain for an explicit linear target.

    For target coefficients a, the target is a.T @ state. Rank handling and
    covariance arithmetic are shared with retrieval diagnostics. A coordinate
    target uses its unit basis vector. Numerical damping contributes no information.

    Args:
        design: Float64 response derivatives (measurement,state), in matching units.
        observation_factor: Conditioned covariance factor in measurement order.
        target: Finite float64 (state,) coefficients in target-unit/state-unit.
        prior: Optional quadratic penalty in the same fitted coordinates.
        rank_rtol: Relative singular-value threshold for augmented rank.
        max_bytes: Estimated dense array storage budget, excluding native SVD scratch.

    Returns:
        Owned posterior covariance, target measurement weights and standard uncertainty.

    Raises:
        ValueError: Dimensions, values or numerical controls are invalid.
        MemoryError: Estimated dense storage exceeds the supplied budget.
        numpy.linalg.LinAlgError: Augmented information is rank deficient.
    """
    if (
        design.ndim != 2
        or target.shape != (design.shape[1],)
        or target.dtype != np.float64
        or not np.isfinite(target).all()
    ):
        raise ValueError("target must be a finite float64 vector in design state order")
    information = local_gaussian_information(
        design,
        observation_factor,
        prior=prior,
        include_gain=True,
        rank_rtol=rank_rtol,
        max_bytes=max_bytes,
    )
    if not information.available:
        raise np.linalg.LinAlgError(information.reason)
    posterior, gain = information.posterior_covariance, information.gain
    assert posterior is not None and gain is not None
    variance = float(target @ posterior @ target)
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("target variance must be finite and nonnegative")
    return dict(
        covariance=posterior, target_weights=target @ gain, target_sigma=float(np.sqrt(variance))
    )


def exchange_channels[T](
    candidates: Sequence[FloatArray],
    minimum_spacing: Sequence[float],
    initial_selection: Sequence[FloatArray],
    evaluate: Callable[[list[FloatArray]], tuple[float, T]],
    *,
    maximum_iterations: int = 25,
    improvement_rtol: float = 1e-8,
) -> ChannelSearchResult[T]:
    """Search best improving single-channel exchanges in fixed deterministic order.

    Args:
        candidates: Increasing candidate coordinates for each window, caller units.
        minimum_spacing: Minimum within-window spacing in the same units.
        initial_selection: Ordered distinct selected coordinates for each window.
        evaluate: Pure callback returning objective to minimize and caller evaluation.
            Singular candidate information may raise numpy.linalg.LinAlgError and
            is skipped. Other exceptions propagate. Callback inputs stay unchanged.
        maximum_iterations: Positive maximum accepted-exchange sweeps.
        improvement_rtol: Required relative objective improvement in [0,1).

    Returns:
        Initial and best evaluations, accepted iterations and local convergence.

    Raises:
        ValueError: Search controls or window counts are invalid.
    """
    if (
        maximum_iterations < 1
        or not 0 <= improvement_rtol < 1
        or len(candidates) != len(initial_selection)
        or len(minimum_spacing) != len(candidates)
    ):
        raise ValueError("invalid optimization controls")
    selection = [np.array(v, dtype=float, copy=True) for v in initial_selection]
    current_score, current = evaluate(selection)
    initial = current
    iterations, converged = 0, False
    for iteration in range(1, maximum_iterations + 1):
        best, best_score, best_selection = current, current_score, selection
        for w, grid in enumerate(candidates):
            selected = selection[w]
            for position in range(len(selected)):
                for candidate in grid:
                    if candidate in selected:
                        continue
                    replacement = selected.copy()
                    replacement[position] = candidate
                    replacement.sort()
                    if np.any(np.diff(replacement) < minimum_spacing[w]):
                        continue
                    proposed = [v.copy() for v in selection]
                    proposed[w] = replacement
                    try:
                        score, evaluation = evaluate(proposed)
                    except np.linalg.LinAlgError:
                        continue
                    if score < best_score * (1 - improvement_rtol):
                        best, best_score, best_selection = evaluation, score, proposed
        if best is current:
            converged = True
            break
        current, current_score, selection = best, best_score, best_selection
        iterations = iteration
    return dict(initial=initial, optimized=current, iterations=iterations, converged=converged)
