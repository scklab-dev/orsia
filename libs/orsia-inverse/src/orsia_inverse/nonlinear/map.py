"""Fit fixed-weight Gaussian models with explicit SciPy TRF or LM selection."""

from typing import Literal

import numpy as np
from orsia_uncertainty import CovarianceFactor
from scipy.optimize import least_squares

from orsia_inverse.nonlinear.estimation import (
    Evaluation,
    FitTermination,
    GaussianEstimate,
    GaussianObjective,
    Prediction,
    Vector,
)
from orsia_inverse.priors import QuadraticPrior


def solve_map(
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
    method: Literal["trf", "lm"] = "trf",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 1000,
) -> GaussianEstimate:
    """Minimize a half-squared Gaussian objective in caller-defined coordinates.

    Supply either separate fun/jac or one fused evaluate callable. Callbacks are
    deterministic and non-mutating. Separate trials compute values only; fused
    evaluations compute both outputs. Outputs are snapshotted at the last exact
    state. Covariance weighting remains fixed. Numerical damping is excluded
    from the optional prior and subsequent posterior diagnostics.

    Args:
        fun: Prediction callable from float64 state (n,) to values (m,).
        jac: Prediction derivative callable returning (m,n), paired with fun.
        observations: Finite nonempty float64 measured vector (m,).
        initial_state: Finite nonempty float64 state (n,) in fitted coordinates.
        evaluate: Fused callable returning prediction and its Jacobian.
        observation_factor: Conditioned immutable covariance factor, size m.
        prior: Optional quadratic penalty in the fitted state coordinates.
        lower_bounds: Optional float64 state lower bounds (n,); omitted is unbounded.
        upper_bounds: Optional float64 state upper bounds (n,); omitted is unbounded.
        method: Trust-region reflective (trf) or Levenberg-Marquardt (lm).
        ftol: Positive relative cost tolerance above machine epsilon.
        xtol: Positive relative state-step tolerance above machine epsilon.
        gtol: Positive gradient tolerance above machine epsilon.
        max_nfev: Positive backend residual-call budget, excluding final adapter calls.

    Returns:
        Immutable terminal estimate with shared costs and method-specific termination.
        Exhaustion returns unsuccessful termination with a valid final linearization.

    Raises:
        ValueError: Inputs, weights, bounds, method or callback results are invalid.
            LM rejects finite bounds and fewer augmented residuals than states.
        TrialDomainError: Any explicitly unsupported model state, including trials.
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
    if method not in ("trf", "lm"):
        raise ValueError("method must be 'trf' or 'lm'")
    tolerances = np.array([ftol, xtol, gtol])
    if not np.isfinite(tolerances).all() or np.any(tolerances <= np.finfo(float).eps):
        raise ValueError("tolerances must be finite and greater than machine epsilon")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if method == "lm":
        if np.any(np.isfinite(objective.lower_bounds)) or np.any(
            np.isfinite(objective.upper_bounds)
        ):
            raise ValueError("LM does not support finite bounds")
        if objective.m + objective.prior_root.shape[0] < objective.n:
            raise ValueError("LM requires at least as many augmented residuals as parameters")
    fitted = least_squares(
        objective.residual,
        objective.initial_state,
        jac=objective.derivative,
        bounds=(objective.lower_bounds, objective.upper_bounds),
        method=method,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        loss="linear",
        x_scale=1.0,
    )
    objective.evaluate(fitted.x, derivative=True)
    termination = FitTermination(
        method,
        bool(fitted.success),
        str(fitted.message),
        int(fitted.status),
        float(fitted.optimality),
        objective.prediction_calls,
        objective.linearization_calls,
        backend_nfev=int(fitted.nfev),
        backend_njev=None if fitted.njev is None else int(fitted.njev),
    )
    return objective.estimate(fitted.x, np.asarray(fitted.active_mask, dtype=np.int64), termination)
