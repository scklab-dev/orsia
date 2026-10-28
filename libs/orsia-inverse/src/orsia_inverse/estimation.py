"""Fixed-weight Gaussian MAP estimation; physical parameterisation stays with callers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .weighting import covariance_factor, whiten

type Vector = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class GaussianEstimate:
    """Estimated dimensionless state, final linearisation, fixed weights and termination.

    Created by solve_map; floating arrays are writable copies, not backend workspaces.
    State arrays and Jacobian columns follow the initial_state order;
    observations and Jacobian rows follow the observations order. Residuals are F(q)-y in
    observation units, Jacobian columns differentiate dimensionless q. Bounds
    and prior mean use q coordinates. success is numerical termination only.
    nfev/njev are backend counts; prediction_calls/linearization_calls count
    adapter calls to fun/jac, including final evaluation. Step counts are
    unavailable, not zero.

    Attributes:
        state: Fitted dimensionless state vector (n,).
        prediction: Final model observations (m,) in measurement units.
        residual: Prediction minus observations (m,) in measurement units.
        jacobian: Final (m,n) derivative of prediction with respect to state.
        observation_covariance: Fixed (m,) variances or (m,m) covariance.
        prior_mean: Optional dimensionless prior state (n,).
        prior_covariance: Optional (n,) variances or (n,n) prior covariance.
        lower_bounds: Dimensionless (n,) lower state limits.
        upper_bounds: Dimensionless (n,) upper state limits.
        active_mask: Backend (n,) indicator for active fitted bounds.
        measurement_cost: Final half-squared whitened observation mismatch.
        prior_cost: Final half-squared whitened departure from the prior.
        method: Backend trust-region (trf) or Levenberg-Marquardt (lm) method.
        success: Whether numerical termination met the backend's success criterion.
        status: Backend termination code.
        message: Backend termination explanation.
        optimality: Backend optimality measure at the fitted state.
        nfev: Backend residual evaluation count.
        njev: Backend Jacobian evaluation count.
        prediction_calls: Calls to the supplied prediction function.
        linearization_calls: Calls to the supplied Jacobian function.
        accepted_steps: Unavailable backend step count, always None.
        rejected_steps: Unavailable backend step count, always None.
    """

    state: Vector
    prediction: Vector
    residual: Vector
    jacobian: Vector
    observation_covariance: Vector
    prior_mean: Vector | None
    prior_covariance: Vector | None
    lower_bounds: Vector
    upper_bounds: Vector
    active_mask: NDArray[np.int64]
    measurement_cost: float
    prior_cost: float
    method: Literal["trf", "lm"]
    success: bool
    status: int
    message: str
    optimality: float
    nfev: int
    njev: int
    prediction_calls: int
    linearization_calls: int
    accepted_steps: None = None
    rejected_steps: None = None


def solve_map(
    fun: Callable[[Vector], Vector],
    jac: Callable[[Vector], Vector],
    observations: Vector,
    initial_state: Vector,
    *,
    observation_covariance: Vector,
    prior_mean: Vector | None = None,
    prior_covariance: Vector | None = None,
    lower_bounds: Vector | None = None,
    upper_bounds: Vector | None = None,
    method: Literal["trf", "lm"] = "trf",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 1000,
) -> GaussianEstimate:
    """Minimise 0.5||We(F(q)-y)||² + 0.5||Wa(q-qa)||² with fixed covariances.

    fun(q) returns model values F(q), not residuals, with shape (m,).
    jac(q) returns only their Jacobian dF/dq with shape (m,n), not a tuple.
    Both take q with shape (n,); inputs/outputs are finite float64 in the
    supplied observation/state vector orders. Covariances are float64 variance
    vectors (m,) / (n,) or dense matrices (m,m) / (n,n), not standard deviations.
    They must be positive definite with minimum/maximum eigenvalue ratio >1e-12.
    This function constructs the whitened residuals and prior term internally.
    q, its prior and bounds are dimensionless caller-defined coordinates.
    Observation covariance entries have the product units of their row/column
    observations. Units and ordering are caller responsibilities, not metadata.
    No unit conversion, parameter transform, finite differencing or physical
    model is inferred. Prior mean and covariance must be supplied together.
    Every bound may be infinite; omitted sides are unbounded, equal bounds are
    rejected (remove fixed parameters). LM requires no finite bounds and at
    least n augmented residuals. Backend x_scale=1 and loss='linear' are fixed.

    Callables must be deterministic, not modify their q input, and return stable
    arrays. Derivative correctness is the caller's responsibility. The adapter
    snapshots outputs and caches only the exact last state within this call.
    Trials use fun alone; jac computes derivatives at the requested state.
    Model exceptions propagate; invalid/nonfinite model
    values raise ValueError, never an artificial residual penalty. A spent
    budget returns success=False. max_nfev budgets backend residual evaluations,
    not total physical calls; separate counts include final linearisation.
    Positive finite tolerances above machine epsilon are required.

    Args:
        fun: Prediction from a dimensionless (n,) state to (m,) observations.
        jac: Prediction Jacobian of shape (m,n) for the same state order.
        observations: Finite observed values (m,) in model output units.
        initial_state: Finite dimensionless starting state (n,).
        observation_covariance: Fixed (m,) variances or (m,m) covariance.
        prior_mean: Optional dimensionless prior state, paired with its covariance.
        prior_covariance: Optional (n,) variances or (n,n) covariance.
        lower_bounds: Optional (n,) lower state limits; default is unbounded.
        upper_bounds: Optional (n,) upper state limits; default is unbounded.
        method: Trust-region reflective (trf) or Levenberg-Marquardt (lm).
        ftol: Relative cost convergence threshold for the backend.
        xtol: Relative state-step convergence threshold for the backend.
        gtol: Gradient convergence threshold for the backend.
        max_nfev: Maximum backend residual evaluations, excluding final adapter calls.

    Returns:
        Fitted state, retained linearization, costs and termination information.

    Raises:
        ValueError: Inputs, covariance, bounds or callable results violate the contract.
    """
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a nonempty vector")
    if initial_state.ndim != 1 or initial_state.size == 0:
        raise ValueError("initial_state must be a nonempty vector")
    n, m = initial_state.size, observations.size
    _validate_array(observations, (m,), "observations")
    _validate_array(initial_state, (n,), "initial_state")
    if (prior_mean is None) != (prior_covariance is None):
        raise ValueError("prior mean and covariance must be supplied together")
    if method not in ("trf", "lm"):
        raise ValueError("method must be 'trf' or 'lm'")
    tolerances = np.array([ftol, xtol, gtol])
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= np.finfo(float).eps):
        raise ValueError("tolerances must be finite and greater than machine epsilon")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    lower = _bounds(lower_bounds, n, -np.inf)
    upper = _bounds(upper_bounds, n, np.inf)
    if np.any(lower >= upper):
        raise ValueError("lower bounds must be strictly below upper bounds")
    if np.any(initial_state < lower) or np.any(initial_state > upper):
        raise ValueError("initial_state is outside bounds")
    if method == "lm" and (np.any(np.isfinite(lower)) or np.any(np.isfinite(upper))):
        raise ValueError("LM does not support finite bounds")
    if method == "lm" and prior_mean is None and m < n:
        raise ValueError("LM requires at least as many augmented residuals as parameters")
    y = np.copy(observations)
    observation_covariance = np.copy(observation_covariance)
    observation_factor = covariance_factor(observation_covariance, m, "observation_covariance")
    prior_factor = np.empty((0, n), dtype=np.float64)
    qa: Vector | None = None
    if prior_mean is not None and prior_covariance is not None:
        _validate_array(prior_mean, (n,), "prior_mean")
        qa = np.copy(prior_mean)
        prior_covariance = np.copy(prior_covariance)
        factor = covariance_factor(prior_covariance, n, "prior_covariance")
        prior_factor = whiten(factor, np.eye(n))
        _validate_array(prior_factor, (n, n), "prior whitening factor")

    cached_state: Vector | None = None
    cached_prediction: Vector | None = None
    cached_jacobian: Vector | None = None
    prediction_calls = 0
    linearization_calls = 0

    def evaluate(q: Vector, derivative: bool) -> tuple[Vector, Vector | None]:
        nonlocal cached_state, cached_prediction, cached_jacobian
        nonlocal prediction_calls, linearization_calls
        _validate_array(q, (n,), "trial state")
        if cached_state is None or not np.array_equal(q, cached_state):
            cached_state = np.copy(q)
            cached_prediction = None
            cached_jacobian = None
        if cached_prediction is None:
            prediction_calls += 1
            values = fun(cached_state)
            _validate_array(values, (m,), "prediction")
            cached_prediction = np.copy(values)
        if derivative and cached_jacobian is None:
            linearization_calls += 1
            jacobian = jac(cached_state)
            _validate_array(jacobian, (m, n), "jacobian")
            cached_jacobian = np.copy(jacobian)
        return cached_prediction, cached_jacobian

    def residual(q: Vector) -> Vector:
        values, _ = evaluate(q, False)
        measurement = whiten(observation_factor, values - y)
        prior = prior_factor @ (q - qa) if qa is not None else np.empty(0)
        result = np.concatenate((measurement, prior))
        _validate_array(result, (m + prior_factor.shape[0],), "whitened residual")
        if not np.isfinite(result @ result):
            raise ValueError("residual cost overflow")
        return result

    def derivative(q: Vector) -> Vector:
        _, jacobian = evaluate(q, True)
        assert jacobian is not None
        result = np.vstack((whiten(observation_factor, jacobian), prior_factor))
        _validate_array(result, (m + prior_factor.shape[0], n), "whitened jacobian")
        return result

    fitted = least_squares(
        residual,
        initial_state.copy(),
        jac=derivative,
        bounds=(lower, upper),
        method=method,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        loss="linear",
        x_scale=1.0,
    )
    prediction, jacobian = evaluate(fitted.x, True)
    assert jacobian is not None
    final_residual = residual(fitted.x)
    active_mask = np.asarray(fitted.active_mask, dtype=np.int64)
    active_mask = np.frombuffer(active_mask.tobytes(), dtype=np.int64)
    return GaussianEstimate(
        state=np.copy(fitted.x),
        prediction=prediction,
        residual=np.copy(prediction - y),
        jacobian=jacobian,
        observation_covariance=observation_covariance,
        prior_mean=qa,
        prior_covariance=prior_covariance,
        lower_bounds=lower,
        upper_bounds=upper,
        active_mask=active_mask,
        measurement_cost=float(final_residual[:m] @ final_residual[:m] / 2),
        prior_cost=float(final_residual[m:] @ final_residual[m:] / 2),
        method=method,
        success=bool(fitted.success),
        status=int(fitted.status),
        message=str(fitted.message),
        optimality=float(fitted.optimality),
        nfev=int(fitted.nfev),
        njev=int(fitted.njev),
        prediction_calls=prediction_calls,
        linearization_calls=linearization_calls,
    )


def _validate_array(values: Vector, shape: tuple[int, ...], name: str) -> None:
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape {shape}")


def _bounds(values: Vector | None, size: int, default: float) -> Vector:
    if values is None:
        return np.full(size, default)
    if values.shape != (size,) or np.any(np.isnan(values)):
        raise ValueError("bounds must be (n,) arrays without NaN")
    return np.copy(values)
