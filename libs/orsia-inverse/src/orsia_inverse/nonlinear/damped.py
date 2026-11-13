"""Fit Gaussian objectives with diagonal damping and explicit trial rejection."""

from typing import cast

import numpy as np
from orsia_uncertainty import CovarianceFactor
from scipy.linalg import cho_factor, cho_solve

from orsia_inverse.nonlinear.estimation import (
    Evaluation,
    FitTermination,
    GaussianEstimate,
    GaussianObjective,
    Prediction,
    TrialDomainError,
    Vector,
)
from orsia_inverse.priors import QuadraticPrior


def solve_damped_gaussian(
    fun: Prediction | None = None,
    jac: Prediction | None = None,
    observations: Vector | None = None,
    initial_state: Vector | None = None,
    *,
    evaluate: Evaluation | None = None,
    observation_factor: CovarianceFactor,
    prior: QuadraticPrior | None = None,
    lower_bounds: Vector | None = None,
    upper_bounds: Vector | None = None,
    max_iterations: int = 15,
    step_tolerance: float = 1e-8,
    cost_tolerance: float = 5e-11,
    damping: float = 1e-6,
) -> GaussianEstimate:
    """Minimize a half-squared Gaussian objective with a distinct damped method.

    Supply either separate fun/jac or one fused evaluate callable. The objective,
    fixed weighting, prior and output contract match solve_map. This algorithm
    retains its own diagonal damping, decreasing-cost trials and undamped
    stationary-step checks. It supports unbounded fitted coordinates only.

    Only TrialDomainError rejects an unsupported trial. Other model exceptions
    propagate, including ValueError for malformed outputs. Initial model errors
    always propagate. Callbacks must be deterministic and non-mutating.

    Args:
        fun: Prediction callable from finite float64 state (n,) to values (m,).
        jac: Prediction derivative callable returning (m,n), paired with fun.
        observations: Finite nonempty float64 measured vector (m,).
        initial_state: Finite nonempty float64 starting state (n,).
        evaluate: Fused prediction/Jacobian callable, exclusive with fun/jac.
        observation_factor: Conditioned immutable covariance factor, size m.
        prior: Optional quadratic penalty, including nonzero mean or partial rank.
        lower_bounds: Optional (n,) bounds; all entries must be negative infinity.
        upper_bounds: Optional (n,) bounds; all entries must be positive infinity.
        max_iterations: Positive maximum attempted outer iterations.
        step_tolerance: Positive undamped stationary-step norm tolerance.
        cost_tolerance: Positive half-squared objective-drop threshold; convergence
            also requires an undamped stationary step. Default equals half the
            former full-square tolerance of 1e-10.
        damping: Positive initial and minimum diagonal damping, at most 1e8.

    Returns:
        Immutable estimate with final linearization and numerical termination.
        Exhaustion and failed trial search return explicit nonconvergence.

    Raises:
        ValueError: Inputs, controls, finite bounds or callback outputs are invalid.
        TrialDomainError: The initial state or terminal linearization is unsupported.
    """
    objective = GaussianObjective(
        fun,
        jac,
        observations,
        initial_state,
        evaluate=evaluate,
        observation_factor=observation_factor,
        prior=prior,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    if np.any(np.isfinite(objective.lower_bounds)) or np.any(np.isfinite(objective.upper_bounds)):
        raise ValueError("damped Gaussian estimation does not support finite bounds")
    controls = np.array([step_tolerance, cost_tolerance, damping])
    if (
        max_iterations < 1
        or not np.isfinite(controls).all()
        or np.any(controls <= 0)
        or damping > 1e8
    ):
        raise ValueError("finite positive solver controls and damping <= 1e8 required")
    state = objective.initial_state.copy()
    residual = objective.residual(state)
    derivative = objective.derivative(state)
    value = float(residual @ residual / 2)
    current_damping = damping
    reason, success, iterations = "maximum_iterations", False, 0
    accepted_steps = rejected_steps = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        information, gradient = _information(derivative, residual)
        step = _damped_step(information, gradient, current_damping)
        if step is None:
            reason = "singular_information"
            break
        stationary = _damped_step(information, gradient, 0)
        if stationary is not None and np.linalg.norm(stationary) < step_tolerance:
            reason, success = "step_tolerance", True
            break
        trial_damping, accepted = current_damping, False
        trial_state, trial_residual, trial_value = state, residual, value
        while trial_damping <= 1e8:
            trial_state = state + step
            try:
                trial_residual = objective.residual(trial_state)
                trial_value = float(trial_residual @ trial_residual / 2)
            except TrialDomainError:
                trial_value = np.inf
            if trial_value <= value:
                accepted = True
                accepted_steps += 1
                break
            rejected_steps += 1
            trial_damping *= 10
            step = _damped_step(information, gradient, trial_damping)
            if step is None:
                break
        if not accepted:
            stationary = _damped_step(information, gradient, 0)
            success = stationary is not None and np.linalg.norm(stationary) < step_tolerance
            reason = "stationary" if success else "line_search_failed"
            break
        drop = value - trial_value
        state, residual, value = trial_state, trial_residual, trial_value
        derivative = objective.derivative(state)
        current_damping = max(damping, trial_damping / 10)
        if abs(drop) < cost_tolerance:
            information, gradient = _information(derivative, residual)
            stationary = _damped_step(information, gradient, 0)
            if stationary is not None and np.linalg.norm(stationary) < step_tolerance:
                reason, success = "cost_tolerance", True
                break
    # A rejected trial can occupy the adapter cache; restore the terminal state
    # before recording actual callback counts or constructing the estimate.
    residual = objective.residual(state)
    derivative = objective.derivative(state)
    _, gradient = _information(derivative, residual)
    termination = FitTermination(
        "damped_gaussian",
        bool(success),
        reason,
        None,
        float(np.linalg.norm(gradient, ord=np.inf)),
        objective.prediction_calls,
        objective.linearization_calls,
        iterations=iterations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )
    return objective.estimate(state, np.zeros(objective.n, dtype=np.int64), termination)


def _information(derivative: Vector, residual: Vector) -> tuple[Vector, Vector]:
    information = derivative.T @ derivative
    gradient = -(derivative.T @ residual)
    if not np.isfinite(information).all() or not np.isfinite(gradient).all():
        raise ValueError("Gaussian information or gradient overflow")
    return information, gradient


def _damped_step(information: Vector, gradient: Vector, damping: float) -> Vector | None:
    matrix = information.copy()
    matrix[np.diag_indices_from(matrix)] += damping * np.maximum(np.abs(np.diag(matrix)), 1)
    try:
        step = cast(Vector, cho_solve(cho_factor(matrix), gradient))
    except np.linalg.LinAlgError:
        return None
    return step if np.isfinite(step).all() else None
