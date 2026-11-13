"""Share Gaussian objectives, model evaluation and immutable terminal estimates."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import CovarianceFactor, whiten

from orsia_inverse.priors import QuadraticPrior

type Vector = NDArray[np.float64]
type Prediction = Callable[[Vector], Vector]
type Evaluation = Callable[[Vector], tuple[Vector, Vector]]


class TrialDomainError(ValueError):
    """Signal a known unsupported trial state to the damped solver.

    Application adapters raise this exception deliberately. Invalid initial
    states and exceptions from other model or programming failures propagate.
    """


@dataclass(frozen=True)
class FitTermination:
    """Record numerical termination independently of the retained physical model.

    Counts describe actual supplied callback invocations, including failed trials
    and final evaluations. One fused call counts as one prediction and one
    linearization. Backend counts may follow a different accounting convention.

    Attributes:
        method: Selected numerical method: trf, lm or damped_gaussian.
        success: Whether the method met a numerical stopping condition.
        reason: Method-specific termination explanation.
        status: Backend status code, or None when unavailable.
        optimality: Final objective-gradient indicator, in fitted coordinates.
        prediction_calls: Actual prediction callback calls.
        linearization_calls: Actual derivative callback calls.
        backend_nfev: Backend residual calls, or None when unavailable.
        backend_njev: Backend Jacobian calls, or None when unavailable.
        iterations: Attempted outer iterations, or None when unavailable.
        accepted_steps: Accepted trial steps, or None when unavailable.
        rejected_steps: Rejected trial steps, or None when unavailable.
    """

    method: str
    success: bool
    reason: str
    status: int | None
    optimality: float
    prediction_calls: int
    linearization_calls: int
    backend_nfev: int | None = None
    backend_njev: int | None = None
    iterations: int | None = None
    accepted_steps: int | None = None
    rejected_steps: int | None = None


@dataclass(frozen=True, eq=False)
class GaussianEstimate:
    """Retain a Gaussian-objective fit and its mutually consistent linearization.

    Arrays are owned immutable float64 snapshots, except integer bound activity.
    Observation order follows prediction/residual/Jacobian rows. State order
    follows state, bounds, prior and Jacobian columns. Numerical success does
    not establish posterior availability or physical model validity.

    Attributes:
        state: Terminal state (n,) in caller-defined fitted coordinates.
        prediction: Terminal model observations (m,), in measurement units.
        residual: Prediction minus supplied observations, shape (m,).
        jacobian: Terminal prediction derivative, shape (m,n).
        observation_factor: Immutable covariance factor used for fitting.
        prior: Optional immutable quadratic penalty in state coordinates.
        lower_bounds: State lower limits (n,), including negative infinity.
        upper_bounds: State upper limits (n,), including positive infinity.
        active_mask: Integer bound activity (n,): -1 lower, 0 free, 1 upper.
        measurement_cost: Half-squared whitened observation mismatch.
        prior_cost: Half-squared prior penalty.
        termination: Numerical method, reason and evaluation counts.
    """

    state: Vector
    prediction: Vector
    residual: Vector
    jacobian: Vector
    observation_factor: CovarianceFactor
    prior: QuadraticPrior | None
    lower_bounds: Vector
    upper_bounds: Vector
    active_mask: NDArray[np.int64]
    measurement_cost: float
    prior_cost: float
    termination: FitTermination

    def __post_init__(self) -> None:
        """Snapshot the retained arrays independently of model and solver workspaces."""
        for name in (
            "state",
            "prediction",
            "residual",
            "jacobian",
            "lower_bounds",
            "upper_bounds",
            "active_mask",
        ):
            values = getattr(self, name)
            snapshot = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
            object.__setattr__(self, name, snapshot)

    @property
    def objective(self) -> float:
        """Return the sum of measurement and prior half-squared costs."""
        return self.measurement_cost + self.prior_cost


class GaussianObjective:
    """Adapt one deterministic model and fixed weighting to a Gaussian objective.

    This per-solve owner retains snapshots of observations, initial state and
    bounds, one exact-state cache and callback counts. Use a separate instance
    for each solve; its mutable cache is not shared between threads. Model
    callbacks receive independent writable state copies and must not mutate them.
    Outputs are copied before another callback can reuse its scratch buffers.
    """

    def __init__(
        self,
        fun: Prediction | None,
        jac: Prediction | None,
        observations: Vector | None,
        initial_state: Vector | None,
        *,
        evaluate: Evaluation | None,
        observation_factor: CovarianceFactor,
        prior: QuadraticPrior | None,
        lower_bounds: Vector | None,
        upper_bounds: Vector | None,
    ) -> None:
        """Validate a numerical problem before calling its supplied model.

        Args:
            fun: Separate prediction callable, paired with jac.
            jac: Separate derivative callable, paired with fun.
            observations: Nonempty finite float64 observations (m,).
            initial_state: Nonempty finite float64 initial state (n,).
            evaluate: Fused prediction/Jacobian callable, exclusive with fun/jac.
            observation_factor: Conditioned fixed covariance factor, size m.
            prior: Optional quadratic penalty, with n state columns.
            lower_bounds: Optional state lower limits (n,).
            upper_bounds: Optional state upper limits (n,).

        Raises:
            ValueError: Model forms, dimensions, finite values or weighting fail.
        """
        if (evaluate is None and (fun is None or jac is None)) or (
            evaluate is not None and (fun is not None or jac is not None)
        ):
            raise ValueError("supply exactly one model form: fun and jac, or evaluate")
        for name, values in (("observations", observations), ("initial_state", initial_state)):
            if values is None or values.ndim != 1 or values.size == 0:
                raise ValueError(f"{name} must be a nonempty vector")
            _validate_array(values, values.shape, name)
        assert observations is not None and initial_state is not None
        self.observations = observations.copy()
        self.initial_state = initial_state.copy()
        self.n, self.m = initial_state.size, observations.size
        if observation_factor.size != self.m:
            raise ValueError("observation_factor size must match observations")
        whiten(observation_factor, np.zeros(self.m))
        if prior is not None and prior.mean.size != self.n:
            raise ValueError("prior size must match state")
        self.observation_factor, self.prior = observation_factor, prior
        self.prior_root = np.empty((0, self.n)) if prior is None else prior.precision_root
        self.lower_bounds = _bounds(lower_bounds, self.n, -np.inf)
        self.upper_bounds = _bounds(upper_bounds, self.n, np.inf)
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("lower bounds must be strictly below upper bounds")
        if np.any(initial_state < self.lower_bounds) or np.any(initial_state > self.upper_bounds):
            raise ValueError("initial_state is outside bounds")
        self.fun, self.jac, self.fused = fun, jac, evaluate
        self.prediction_calls = self.linearization_calls = 0
        self._state: Vector | None = None
        self._prediction: Vector | None = None
        self._jacobian: Vector | None = None

    def evaluate(self, state: Vector, *, derivative: bool) -> tuple[Vector, Vector | None]:
        """Evaluate a model at one exact state, snapshotting its outputs.

        Args:
            state: Finite float64 state (n,) in fitted coordinates.
            derivative: Request a Jacobian; fused callbacks always compute both outputs.

        Returns:
            Cached prediction and optional Jacobian, borrowed until the next call.

        Raises:
            ValueError: State or callback outputs violate shape/dtype/finite requirements.
        """
        _validate_array(state, (self.n,), "trial state")
        if self._state is None or not np.array_equal(state, self._state):
            self._state = state.copy()
            self._prediction = self._jacobian = None
        if self._prediction is None:
            self.prediction_calls += 1
            if self.fused is not None:
                self.linearization_calls += 1
                values, jacobian = self.fused(self._state.copy())
                _validate_array(jacobian, (self.m, self.n), "jacobian")
                self._jacobian = jacobian.copy()
            else:
                assert self.fun is not None
                values = self.fun(self._state.copy())
            _validate_array(values, (self.m,), "prediction")
            self._prediction = values.copy()
        if derivative and self._jacobian is None:
            assert self.jac is not None
            self.linearization_calls += 1
            jacobian = self.jac(self._state.copy())
            _validate_array(jacobian, (self.m, self.n), "jacobian")
            self._jacobian = jacobian.copy()
        return self._prediction, self._jacobian

    def residual(self, state: Vector) -> Vector:
        """Return the augmented whitened residual for a model state.

        Args:
            state: Finite fitted state (n,).

        Returns:
            Observation residual followed by prior residual, shape (m+penalty,).

        Raises:
            ValueError: Model values or weighted cost are invalid or overflow.
        """
        values, _ = self.evaluate(state, derivative=False)
        measured = whiten(self.observation_factor, values - self.observations)
        penalty = self.prior_root @ (state - self.prior.mean) if self.prior else np.empty(0)
        residual = np.concatenate((measured, penalty))
        _validate_array(residual, (self.m + self.prior_root.shape[0],), "whitened residual")
        if not np.isfinite(residual @ residual):
            raise ValueError("residual cost overflow")
        return residual

    def derivative(self, state: Vector) -> Vector:
        """Return the augmented whitened prediction derivative.

        Args:
            state: Finite fitted state (n,).

        Returns:
            Jacobian of augmented residual, shape (m+penalty,n).

        Raises:
            ValueError: Model derivatives or whitening fail numerical checks.
        """
        _, jacobian = self.evaluate(state, derivative=True)
        assert jacobian is not None
        result = np.vstack((whiten(self.observation_factor, jacobian), self.prior_root))
        _validate_array(result, (self.m + self.prior_root.shape[0], self.n), "whitened jacobian")
        return result

    def estimate(
        self, state: Vector, active_mask: NDArray[np.int64], termination: FitTermination
    ) -> GaussianEstimate:
        """Snapshot a terminal state after its final values and counts are established.

        Args:
            state: Already evaluated terminal state (n,).
            active_mask: Method-supplied bound activity (n,).
            termination: Termination including final callback counts.

        Returns:
            Immutable estimate with separate half-squared costs.
        """
        prediction, jacobian = self.evaluate(state, derivative=True)
        assert jacobian is not None
        residual = self.residual(state)
        return GaussianEstimate(
            state,
            prediction,
            prediction - self.observations,
            jacobian,
            self.observation_factor,
            self.prior,
            self.lower_bounds,
            self.upper_bounds,
            active_mask,
            float(residual[: self.m] @ residual[: self.m] / 2),
            float(residual[self.m :] @ residual[self.m :] / 2),
            termination,
        )


def _validate_array(values: Vector, shape: tuple[int, ...], name: str) -> None:
    if values.dtype != np.float64 or values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite float64 with shape {shape}")


def _bounds(values: Vector | None, size: int, default: float) -> Vector:
    if values is None:
        return np.full(size, default)
    if values.dtype != np.float64 or values.shape != (size,) or np.any(np.isnan(values)):
        raise ValueError("bounds must be float64 (n,) arrays without NaN")
    return values.copy()
