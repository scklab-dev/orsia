"""Gaussian estimator contracts using independent small dense references."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_inverse import GaussianEstimate, retrieval_diagnostics, solve_map

type Vector = NDArray[np.float64]


@pytest.mark.parametrize("method", ["trf", "lm"])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("with_prior", [False, True])
def test_affine_reference(method: Any, diagonal: bool, with_prior: bool) -> None:
    k = np.array([[1.0, 2.0], [-1.0, 1.0], [2.0, 0.5]])
    y, qa = np.array([1.0, 2.0, -1.0]), np.array([0.4, -0.1])
    se = (
        np.diag([2.0, 3.0, 1.0])
        if diagonal
        else np.array([[2.0, 0.4, 0.1], [0.4, 3.0, 0.2], [0.1, 0.2, 1.0]])
    )
    sa = np.array([[1.0, 0.2], [0.2, 2.0]])
    observation_covariance = np.diag(se).copy() if diagonal else se
    weights = np.linalg.inv(se)
    precision = k.T @ weights @ k
    rhs = k.T @ weights @ y
    options: dict[str, Any] = {}
    if with_prior:
        precision += np.linalg.inv(sa)
        rhs += np.linalg.solve(sa, qa)
        options = {"prior_mean": qa, "prior_covariance": sa}
    expected = np.linalg.solve(precision, rhs)
    result = _solve(k, y, observation_covariance=observation_covariance, method=method, **options)
    assert result.success and result.status > 0
    np.testing.assert_allclose(result.state, expected, atol=1e-10)
    np.testing.assert_allclose(result.prediction, k @ expected, atol=1e-10)
    np.testing.assert_allclose(result.residual, k @ expected - y, atol=1e-10)
    residual = k @ expected - y
    np.testing.assert_allclose(result.measurement_cost, residual @ weights @ residual / 2)
    np.testing.assert_allclose(
        result.prior_cost,
        (expected - qa) @ np.linalg.solve(sa, expected - qa) / 2 if with_prior else 0,
    )
    diagnostics = retrieval_diagnostics(result, include_gain=True, include_averaging_kernel=True)
    assert diagnostics.available and diagnostics.posterior_covariance is not None
    assert diagnostics.gain is not None and diagnostics.averaging_kernel is not None
    assert diagnostics.degrees_of_freedom is not None
    assert diagnostics.measurement_rank == diagnostics.augmented_rank == 2
    posterior = np.linalg.inv(precision)
    gain = posterior @ k.T @ weights
    np.testing.assert_allclose(diagnostics.posterior_covariance, posterior, atol=1e-12)
    np.testing.assert_allclose(diagnostics.gain, gain, atol=1e-12)
    np.testing.assert_allclose(diagnostics.averaging_kernel, gain @ k, atol=1e-12)
    np.testing.assert_allclose(diagnostics.degrees_of_freedom, np.trace(gain @ k), atol=1e-12)
    if with_prior:
        prior_gain = np.eye(2) - gain @ k
        np.testing.assert_allclose(gain @ se @ gain.T + prior_gain @ sa @ prior_gain.T, posterior)
    lean = retrieval_diagnostics(result)
    assert lean.gain is lean.averaging_kernel is None
    assert lean.degrees_of_freedom == diagnostics.degrees_of_freedom
    assert result.accepted_steps is result.rejected_steps is None
    assert result.prediction_calls >= result.nfev and result.linearization_calls >= result.njev


def test_mixed_bounds_coupled_state_and_active_bound_policy() -> None:
    k = np.array(
        [[2.0, 0.1, 0.2, 0.3], [0.2, 2.0, 0.1, 0.2], [0.1, 0.3, 2.0, 0.1], [0.1, 0.2, 0.1, 2.0]]
    )
    truth = np.array([0.2, 0.4, -0.3, 0.2])
    lower = np.array([-np.inf, 0.0, -np.inf, -1.0])
    upper = np.array([np.inf, np.inf, 0.0, 1.0])
    result = _solve(
        k,
        k @ truth,
        initial_state=np.array([0.0, 0.2, -0.1, 0.0]),
        lower_bounds=lower,
        upper_bounds=upper,
    )
    np.testing.assert_allclose(result.state, truth, atol=1e-7)
    assert retrieval_diagnostics(result).available
    # Independent constrained optimum: fix q0 at its upper bound then solve q1.
    k = np.array([[2.0, 1.0], [0.5, 2.0], [1.0, -1.0]])
    y = k @ np.array([3.0, 0.2])
    result = _solve(k, y, upper_bounds=np.array([1.0, np.inf]))
    q1 = np.linalg.lstsq(k[:, 1:], y - k[:, 0], rcond=None)[0][0]
    np.testing.assert_allclose(result.state, [1.0, q1], atol=1e-7)
    assert result.active_mask[0] == 1
    diagnostics = retrieval_diagnostics(result, include_gain=True)
    assert not diagnostics.available and diagnostics.reason is not None
    assert "active bounds" in diagnostics.reason
    assert diagnostics.posterior_covariance is diagnostics.gain is None


def test_nonlinear_derivatives_multiple_starts_and_final_freshness() -> None:
    def fun(q: Vector) -> Vector:
        return np.array([np.exp(q[0]), q[0] + q[1] ** 3, q[1]])

    def jac(q: Vector) -> Vector:
        return np.array([[np.exp(q[0]), 0.0], [1.0, 3 * q[1] ** 2], [0.0, 1.0]])

    truth = np.array([0.4, -0.3])
    for q in (truth, np.array([-1.0, 1.0]), np.array([1.5, -2.0])):
        finite = np.column_stack(
            [(fun(q + step) - fun(q - step)) / (2e-6) for step in np.eye(2) * 1e-6]
        )
        np.testing.assert_allclose(jac(q), finite, atol=2e-9, rtol=2e-9)
        result = solve_map(fun, jac, fun(truth), q, **_options(3, 2))
        assert result.success
        np.testing.assert_allclose(result.state, truth, atol=1e-7)
        np.testing.assert_allclose(result.jacobian, jac(result.state))


def test_affine_coordinate_transform_preserves_physical_solution_and_information() -> None:
    k = np.array([[1.0, 2.0], [-1.0, 1.0], [2.0, 0.5]])
    y, qa = np.array([1.0, 2.0, -1.0]), np.array([0.4, -0.1])
    sa = np.array([[1.0, 0.2], [0.2, 2.0]])
    base = _solve(k, y, prior_mean=qa, prior_covariance=sa)
    offset, d = np.array([0.3, -0.5]), np.diag([0.05, 7.0])
    invd = np.linalg.inv(d)
    transformed = solve_map(
        lambda q: k @ (offset + d @ q),
        lambda q: k @ d,
        y,
        invd @ (-offset),
        **_options(3, 2),
        prior_mean=invd @ (qa - offset),
        prior_covariance=invd @ sa @ invd,
    )
    np.testing.assert_allclose(offset + d @ transformed.state, base.state, atol=1e-10)
    a = retrieval_diagnostics(base, include_gain=True, include_averaging_kernel=True)
    b = retrieval_diagnostics(transformed, include_gain=True, include_averaging_kernel=True)
    assert a.posterior_covariance is not None and b.posterior_covariance is not None
    assert a.gain is not None and b.gain is not None
    assert a.averaging_kernel is not None and b.averaging_kernel is not None
    np.testing.assert_allclose(d @ b.posterior_covariance @ d, a.posterior_covariance)
    np.testing.assert_allclose(d @ b.gain, a.gain, atol=1e-12)
    np.testing.assert_allclose(d @ b.averaging_kernel @ invd, a.averaging_kernel, atol=1e-12)


def test_confusion_prior_information_and_underdetermined_states() -> None:
    k = np.array([[1.0, 1.0], [2.0, 2.0]])
    free = _solve(k, np.array([1.0, 2.0]))
    diagnosis = retrieval_diagnostics(free)
    assert not diagnosis.available and diagnosis.augmented_rank == 1
    assert np.isinf(diagnosis.measurement_condition)
    assert diagnosis.reason is not None and "rank deficient" in diagnosis.reason
    for variance in (1.0, 100.0):
        estimate = _solve(
            k,
            np.array([1.0, 2.0]),
            prior_mean=np.zeros(2),
            prior_covariance=np.full(2, variance),
        )
        diagnosis = retrieval_diagnostics(estimate, include_averaging_kernel=True)
        assert (
            diagnosis.available
            and diagnosis.measurement_rank == 1
            and diagnosis.augmented_rank == 2
        )
        assert diagnosis.posterior_covariance is not None
        difference = np.array([1.0, -1.0]) / np.sqrt(2)
        np.testing.assert_allclose(
            difference @ diagnosis.posterior_covariance @ difference, variance
        )
        assert diagnosis.degrees_of_freedom is not None and 0 < diagnosis.degrees_of_freedom < 1
    under = _solve(np.array([[1.0, 0.0]]), np.array([1.0]))
    assert not retrieval_diagnostics(under).available
    prior_only = _solve(
        np.zeros((2, 2)),
        np.ones(2),
        prior_mean=np.array([0.2, 0.3]),
        prior_covariance=np.ones(2),
    )
    diagnosis = retrieval_diagnostics(prior_only)
    assert diagnosis.available and diagnosis.measurement_rank == 0
    assert diagnosis.degrees_of_freedom == 0


def test_actual_error_and_signed_bias_transport_with_misspecified_weighting() -> None:
    k = np.array([[1.0, 0.3], [2.0, -1.0], [0.2, 1.0]])
    y = np.array([1.0, 2.0, 3.0])
    prior = np.ones(2)
    estimate = _solve(k, y, prior_mean=np.zeros(2), prior_covariance=prior)
    diagnostics = retrieval_diagnostics(estimate, include_gain=True)
    assert diagnostics.gain is not None
    actual = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]])
    bias = np.array([0.1, -0.2, 0.3])
    transported = diagnostics.gain @ bias
    shifted = _solve(k, y + bias, prior_mean=np.zeros(2), prior_covariance=prior)
    np.testing.assert_allclose(shifted.state - estimate.state, transported, atol=1e-12)
    propagated = diagnostics.gain @ actual @ diagnostics.gain.T
    draws = np.random.default_rng(307).multivariate_normal(np.zeros(3), actual, size=8000)
    errors = draws @ diagnostics.gain.T
    np.testing.assert_allclose(np.cov(errors, rowvar=False), propagated, rtol=0.06, atol=0.01)
    assert diagnostics.posterior_covariance is not None
    assert not np.allclose(propagated, diagnostics.posterior_covariance)


def test_nonlinear_small_noise_monte_carlo_and_large_noise_limit() -> None:
    # Invert exp(q); log(1+e) has a curvature bias absent from the local gain.
    errors = np.random.default_rng(411).uniform(-1.0, 1.0, 300)
    for amplitude in (0.01, 0.8):
        recovered = []
        for e in errors * amplitude:
            result = solve_map(
                lambda q: np.exp(q),
                lambda q: np.diag(np.exp(q)),
                np.array([1.0 + e]),
                np.zeros(1),
                **_options(1, 1),
            )
            assert result.success
            recovered.append(result.state[0])
        np.testing.assert_allclose(recovered, np.log1p(errors * amplitude), atol=2e-8)
        # Compare to linearized errors on the same draws (no sampling ambiguity).
        relative = abs(np.var(recovered) / np.var(errors * amplitude) - 1)
        assert relative < 0.003 if amplitude < 0.1 else relative > 0.25


def test_budget_exhaustion_and_snapshot_ownership() -> None:
    q0, y = np.array([0.0]), np.array([4.0])
    result = solve_map(
        lambda q: np.exp(q),
        lambda q: np.diag(np.exp(q)),
        y,
        q0,
        **_options(1, 1),
        max_nfev=1,
    )
    assert not result.success and result.status == 0 and result.nfev == 1
    assert result.prediction_calls == result.linearization_calls == 1
    diagnostics = retrieval_diagnostics(result)
    assert not diagnostics.available and diagnostics.reason is not None
    assert "not successful" in diagnostics.reason
    q0.fill(99)
    y.fill(99)
    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.residual, [-3.0])
    for values in (
        result.state,
        result.prediction,
        result.residual,
        result.jacobian,
        result.lower_bounds,
        result.upper_bounds,
    ):
        assert values.flags.writeable
        assert not np.shares_memory(values, q0)
        assert not np.shares_memory(values, y)
    result.state[0] = 2.0
    assert q0[0] == 99
    assert not result.active_mask.flags.writeable


def test_nonlinear_gaussian_ensemble_matches_local_posterior_at_small_noise() -> None:
    options = _options(1, 1)
    variance = 1e-4
    options["observation_covariance"] = np.array([variance])
    fun = np.exp

    def jac(q: Vector) -> Vector:
        return np.diag(np.exp(q))

    reference = solve_map(fun, jac, np.ones(1), np.zeros(1), **options)
    diagnosis = retrieval_diagnostics(reference)
    assert diagnosis.posterior_covariance is not None
    posterior = diagnosis.posterior_covariance[0, 0]
    np.testing.assert_allclose(posterior, variance)
    perturbations = np.random.default_rng(812).normal(0.0, np.sqrt(variance), size=600)
    states = np.array(
        [
            solve_map(fun, jac, np.array([1.0 + e]), np.zeros(1), **options).state[0]
            for e in perturbations
        ]
    )
    np.testing.assert_allclose(states, np.log1p(perturbations), atol=1e-9)
    # Six hundred draws: a broad bounded test, not a coverage claim.
    np.testing.assert_allclose(np.var(states, ddof=1), posterior, rtol=0.2)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"prior_mean": np.zeros(1)}, "together"),
        ({"prior_covariance": np.ones(1)}, "together"),
        ({"method": "dogbox"}, "method"),
        ({"ftol": 0.0}, "tolerances"),
        ({"xtol": np.nan}, "tolerances"),
        ({"max_nfev": 0}, "positive"),
        ({"lower_bounds": np.array([1.0]), "upper_bounds": np.array([1.0])}, "strictly"),
        ({"upper_bounds": np.array([-1.0])}, "outside"),
        ({"lower_bounds": np.array([np.nan])}, "bounds"),
        ({"lower_bounds": np.zeros(2)}, "bounds"),
        ({"method": "lm", "upper_bounds": np.ones(1)}, "finite bounds"),
        ({"initial_state": np.array([np.inf])}, "initial_state"),
    ],
)
def test_invalid_arguments(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _solve(np.eye(1), np.ones(1), **options)


def test_lm_underdetermined_and_invalid_rank_tolerance() -> None:
    with pytest.raises(ValueError, match="augmented residuals"):
        _solve(np.ones((1, 2)), np.ones(1), method="lm")
    for invalid in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="rank_rtol"):
            retrieval_diagnostics(_solve(np.eye(1), np.ones(1)), rank_rtol=invalid)


def test_model_errors_invalid_shapes_and_nonfinite_derivatives() -> None:
    for fun, jac, match in (
        (lambda q: np.array([np.nan]), lambda q: np.eye(1), "prediction"),
        (lambda q: np.zeros(2), lambda q: np.eye(1), "prediction"),
        (lambda q: q, lambda q: np.ones((2, 1)), "jacobian"),
        (lambda q: q, lambda q: np.ones(1), "jacobian"),
        (lambda q: q, lambda q: np.array([[np.nan]]), "jacobian"),
        (lambda q: q, lambda q: np.array([[np.inf]]), "jacobian"),
        (lambda q: np.full(1, 1e200), lambda q: np.eye(1), "overflow"),
    ):
        with np.errstate(over="ignore"), pytest.raises(ValueError, match=match):
            solve_map(fun, jac, np.ones(1), np.zeros(1), **_options(1, 1))

    def physical_failure(q: Vector) -> Vector:
        raise RuntimeError("outside physical support")

    with pytest.raises(RuntimeError, match="physical support"):
        solve_map(physical_failure, lambda q: np.eye(1), np.ones(1), np.zeros(1), **_options(1, 1))
    with pytest.raises(RuntimeError, match="physical support"):
        solve_map(lambda q: q, physical_failure, np.ones(1), np.zeros(1), **_options(1, 1))


def test_model_receives_writable_independent_state_copy() -> None:
    initial = np.zeros(1)

    def fun(q: Vector) -> Vector:
        assert q.flags.writeable
        assert not np.shares_memory(q, initial)
        return q

    result = solve_map(fun, lambda q: np.eye(1), np.ones(1), initial, **_options(1, 1))
    np.testing.assert_allclose(result.state, [1.0])
    np.testing.assert_array_equal(initial, [0.0])


@pytest.mark.parametrize("method", ["trf", "lm"])
@pytest.mark.parametrize("max_nfev", [1, 100])
def test_fun_jac_keywords_counts_cache_and_output_snapshots(method: Any, max_nfev: int) -> None:
    fun_states: list[Vector] = []
    jac_states: list[Vector] = []
    values = np.empty(1)
    derivatives = np.empty((1, 1))

    def fun(q: Vector) -> Vector:
        fun_states.append(q.copy())
        values[:] = np.exp(q)
        return values

    def jac(q: Vector) -> Vector:
        jac_states.append(q.copy())
        derivatives[:] = np.exp(q)
        return derivatives

    result = solve_map(
        fun=fun,
        jac=jac,
        observations=np.array([4.0]),
        initial_state=np.zeros(1),
        method=method,
        max_nfev=max_nfev,
        **_options(1, 1),
    )
    assert result.prediction_calls == len(fun_states)
    assert result.linearization_calls == len(jac_states)
    assert all(not np.array_equal(a, b) for a, b in zip(fun_states, fun_states[1:], strict=False))
    for states in (fun_states, jac_states):
        np.testing.assert_array_equal(states[-1], result.state)
    if max_nfev == 1 and method == "trf":
        assert result.prediction_calls == result.linearization_calls == 1
    if max_nfev == 100:
        assert result.success
        np.testing.assert_allclose(result.state, [np.log(4.0)])
    values.fill(-99)
    derivatives.fill(-99)
    np.testing.assert_allclose(result.prediction, np.exp(result.state))
    np.testing.assert_allclose(result.jacobian, np.diag(np.exp(result.state)))


@pytest.mark.parametrize("relative_path", ["README.md", "src/orsia_inverse/README.md"])
def test_documented_example(relative_path: str) -> None:
    document = Path(__file__).parents[1] / relative_path
    text = document.read_text()
    namespace: dict[str, Any] = {}
    for block in text.split("```python\n")[1:]:
        code = block.split("```", 1)[0]
        exec(compile(code, str(document), "exec"), namespace)
    assert text.count("```") % 2 == 0
    assert all(line == line.rstrip() for line in text.splitlines())


def _options(m: int, n: int) -> dict[str, Any]:
    return {"observation_covariance": np.ones(m)}


def _solve(k: Vector, y: Vector, **kwargs: Any) -> GaussianEstimate:
    options = _options(*k.shape)
    options.update(kwargs)
    initial = options.pop("initial_state", np.zeros(k.shape[1]))
    return solve_map(lambda q: k @ q, lambda q: k, y, initial, **options)
