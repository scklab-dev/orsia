"""Shared nonlinear contracts and independent Gaussian-information oracles."""

import importlib.util
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_inverse import (
    GaussianEstimate,
    QuadraticPrior,
    TrialDomainError,
    local_gaussian_information,
    prior_from_covariance,
    prior_from_precision,
    retrieval_diagnostics,
    solve_damped_gaussian,
    solve_map,
)
from orsia_inverse.design import linear_target_uncertainty
from orsia_inverse.detection import scan_compensated_apertures
from orsia_uncertainty import factor_covariance


@pytest.mark.parametrize("method", ["trf", "lm", "damped"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("root", [np.empty((0, 2)), np.array([[0.0, 2.0]]), np.eye(2)])
def test_shared_linear_oracle_and_owned_estimates(method, fused, root):
    matrix = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 1.0]])
    observed = np.array([1.0, -2.0, 0.5])
    covariance = np.array([[2.0, 0.2, 0.0], [0.2, 3.0, 0.1], [0.0, 0.1, 1.0]])
    mean = np.array([0.4, -0.3])
    prior = QuadraticPrior(mean, root)
    factor = factor_covariance(covariance, method="cholesky")
    values, derivative = np.empty(3), np.empty((3, 2))
    counts = [0, 0]

    def fun(q):
        counts[0] += 1
        values[:] = matrix @ q
        return values

    def jac(q):
        counts[1] += 1
        derivative[:] = matrix
        return derivative

    def evaluate(q):
        return fun(q), jac(q)

    options: dict[str, Any] = dict(
        observations=observed, initial_state=np.zeros(2), observation_factor=factor, prior=prior
    )
    options.update(dict(evaluate=evaluate) if fused else dict(fun=fun, jac=jac))
    result = (
        solve_damped_gaussian(**options)
        if method == "damped"
        else solve_map(**options, method=method)
    )
    precision = root.T @ root
    information = matrix.T @ np.linalg.solve(covariance, matrix) + precision
    expected = np.linalg.solve(
        information, matrix.T @ np.linalg.solve(covariance, observed) + precision @ mean
    )
    assert isinstance(result, GaussianEstimate)
    assert result.termination.success
    assert_allclose(result.state, expected, atol=1e-8)
    residual = matrix @ expected - observed
    expected_cost = (
        residual @ np.linalg.solve(covariance, residual) + np.sum((root @ (expected - mean)) ** 2)
    ) / 2
    assert_allclose(result.objective, expected_cost, atol=1e-14)
    assert result.termination.prediction_calls == counts[0]
    assert result.termination.linearization_calls == counts[1]
    if fused:
        assert counts[0] == counts[1]
    if method == "damped":
        assert result.termination.backend_nfev is result.termination.backend_njev is None
        assert not result.active_mask.any()
    diagnosis = retrieval_diagnostics(result, include_gain=True, include_averaging_kernel=True)
    direct = local_gaussian_information(matrix, factor, prior=prior, include_gain=True)
    assert diagnosis.posterior_covariance is not None
    assert diagnosis.gain is not None and direct.gain is not None
    assert result.prior is not None
    assert_allclose(diagnosis.posterior_covariance, np.linalg.inv(information), atol=1e-12)
    assert_allclose(diagnosis.gain, direct.gain, atol=1e-12)
    values.fill(99)
    derivative.fill(99)
    observed.fill(99)
    mean.fill(99)
    assert_allclose(result.prediction, matrix @ result.state)
    assert_array_equal(result.jacobian, matrix)
    for array in (
        result.state,
        result.prediction,
        result.jacobian,
        result.prior.mean,
        result.prior.precision_root,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    writable = result.state.copy()
    writable[:] = 123
    assert not np.array_equal(writable, result.state)


@pytest.mark.parametrize("solver", [solve_map, solve_damped_gaussian])
def test_exact_model_form_required_and_invalid_results_propagate(solver):
    options = dict(
        observations=np.ones(1),
        initial_state=np.zeros(1),
        observation_factor=factor_covariance(np.ones(1), method="cholesky"),
    )
    for forms in (
        {},
        {"fun": lambda q: q},
        {"jac": lambda q: np.eye(1)},
        {"fun": lambda q: q, "jac": lambda q: np.eye(1), "evaluate": lambda q: (q, np.eye(1))},
    ):
        with pytest.raises(ValueError, match="exactly one"):
            solver(**options, **forms)
    for result in (
        (np.ones(2), np.eye(1)),
        (np.ones(1), np.ones(1)),
        (np.ones(1), np.array([[np.nan]])),
    ):
        with pytest.raises(ValueError, match="prediction|jacobian"):
            solver(**options, evaluate=lambda q, result=result: result)

    def unsupported(q):
        raise TrialDomainError("initial domain")

    with pytest.raises(TrialDomainError, match="initial domain"):
        solver(**options, evaluate=unsupported)


def test_trial_rejection_distinguishes_contract_errors_and_counts():
    calls = []

    def evaluate(q):
        calls.append(q.copy())
        if np.any(q != 0):
            raise TrialDomainError("known trial support")
        return q, np.eye(1)

    options = dict(
        observations=np.ones(1),
        initial_state=np.zeros(1),
        observation_factor=factor_covariance(np.ones(1), method="cholesky"),
    )
    result = solve_damped_gaussian(**options, evaluate=evaluate)
    assert result.termination.reason == "line_search_failed"
    assert result.termination.rejected_steps is not None
    assert result.termination.rejected_steps > 0
    assert (
        result.termination.prediction_calls == result.termination.linearization_calls == len(calls)
    )
    assert_array_equal(calls[-1], result.state)
    assert not retrieval_diagnostics(result).available
    for error in (ValueError, RuntimeError):

        def malformed(q, error=error):
            if q[0] != 0:
                raise error("programming failure")
            return q, np.eye(1)

        with pytest.raises(error, match="programming failure"):
            solve_damped_gaussian(**options, evaluate=malformed)
    with pytest.raises(TrialDomainError):
        solve_map(**options, evaluate=evaluate)


def test_priors_preserve_mean_precision_rank_and_scale():
    mean = np.array([0.4, -0.3])
    for precision in (np.diag([0.0, 2.0]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2))):
        prior = prior_from_precision(mean, precision)
        assert_allclose(prior.precision_root.T @ prior.precision_root, precision, atol=1e-15)
    for scale in (1e-200, 1.0, 1e200):
        prior = prior_from_precision(mean, scale * np.array([1.0, 2.0]))
        assert_allclose(prior.precision_root.T @ prior.precision_root / scale, np.diag([1.0, 2.0]))
    for precision in (
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.diag([1.0, -1.0]),
        np.full((2, 2), np.nan),
        np.ones(3),
    ):
        with pytest.raises(ValueError):
            prior_from_precision(mean, precision)
    for mu, root in (
        (np.empty(0), np.empty((0, 0))),
        (np.zeros(2), np.eye(3)),
        (np.zeros(2), np.ones(2)),
        (np.zeros(2), np.full((1, 2), np.nan)),
        (np.zeros(2, dtype=np.float32), np.eye(2)),
    ):
        with pytest.raises(ValueError):
            QuadraticPrior(cast(Any, mu), root)
    with pytest.raises(ValueError, match="size"):
        prior_from_covariance(mean, factor_covariance(np.ones(3), method="cholesky"))


@pytest.mark.parametrize("control", ["step_tolerance", "cost_tolerance", "damping"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_damped_control_validation(control, value):
    with pytest.raises(ValueError, match="controls"):
        solve_damped_gaussian(
            lambda q: q,
            lambda q: np.eye(1),
            np.ones(1),
            np.zeros(1),
            observation_factor=factor_covariance(np.ones(1), method="cholesky"),
            **{control: value},
        )


def test_bounds_underdetermined_lm_and_numerical_damping_exclusion():
    factor = factor_covariance(np.ones(1), method="cholesky")
    with pytest.raises(ValueError, match="finite bounds"):
        solve_damped_gaussian(
            lambda q: q,
            lambda q: np.eye(1),
            np.ones(1),
            np.zeros(1),
            observation_factor=factor,
            lower_bounds=np.array([-1.0]),
        )
    for root in (np.empty((0, 3)), np.zeros((1, 3))):
        with pytest.raises(ValueError, match="augmented residuals"):
            solve_map(
                lambda q: q[:1],
                lambda q: np.array([[1.0, 0.0, 0.0]]),
                np.ones(1),
                np.zeros(3),
                observation_factor=factor,
                prior=QuadraticPrior(np.zeros(3), root),
                method="lm",
            )
    for damping in (1e-8, 0.1):
        estimate = solve_damped_gaussian(
            lambda q: q,
            lambda q: np.eye(1),
            np.ones(1),
            np.zeros(1),
            observation_factor=factor,
            damping=damping,
            max_iterations=40,
        )
        assert estimate.termination.success
        posterior = retrieval_diagnostics(estimate).posterior_covariance
        assert posterior is not None
        assert_allclose(posterior, [[1.0]])
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="overflow"):
        solve_damped_gaussian(
            lambda q: q,
            lambda q: np.array([[1e200]]),
            np.ones(1),
            np.zeros(1),
            observation_factor=factor,
        )


def test_local_information_target_permutation_and_budgets():
    matrix = np.array([[1.0, 2.0], [-1.0, 1.0], [3.0, -0.5]])
    covariance = np.array([[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]])
    factor = factor_covariance(covariance, method="cholesky")
    prior = QuadraticPrior(np.zeros(2), np.array([[0.0, 0.5]]))
    target = np.array([0.3, -2.0])
    result = linear_target_uncertainty(matrix, factor, target, prior=prior)
    posterior = np.linalg.inv(
        matrix.T @ np.linalg.solve(covariance, matrix)
        + prior.precision_root.T @ prior.precision_root
    )
    assert_allclose(result["covariance"], posterior)
    assert_allclose(
        result["target_weights"], target @ posterior @ np.linalg.solve(covariance, matrix).T
    )
    assert_allclose(result["target_sigma"] ** 2, target @ posterior @ target)
    permuted = linear_target_uncertainty(
        matrix[:, ::-1],
        factor,
        target[::-1],
        prior=QuadraticPrior(prior.mean[::-1], prior.precision_root[:, ::-1]),
    )
    assert_allclose(permuted["target_weights"], result["target_weights"])
    assert_allclose(permuted["target_sigma"], result["target_sigma"])
    for bad in (np.ones(3), np.array([np.nan, 1.0])):
        with pytest.raises(ValueError, match="target"):
            linear_target_uncertainty(matrix, factor, bad)
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        linear_target_uncertainty(np.ones((3, 2)), factor, target)
    with pytest.raises(MemoryError):
        local_gaussian_information(matrix, factor, max_bytes=1)
    for options in (
        {"max_bytes": 0},
        {"rank_rtol": 0},
        {"prior": QuadraticPrior(np.zeros(3), np.eye(3))},
    ):
        with pytest.raises(ValueError):
            local_gaussian_information(matrix, factor, **cast(dict[str, Any], options))
    with pytest.raises(ValueError, match="jacobian"):
        local_gaussian_information(np.ones((2, 2)), factor)


def test_aperture_dense_covariance_signed_terms_c_order_and_missing_values():
    shape = (11, 12)
    signal, valid = np.zeros(shape), np.ones(shape, bool)
    valid[3, 3] = False
    signal[3, 3] = np.nan
    rng = np.random.default_rng(751)
    modes = rng.normal(size=(signal.size, 2)) * 0.1
    dense = np.eye(signal.size) * 0.25 + modes @ modes.T
    calls = []

    def covariance(indices, weights):
        assert np.all(valid.ravel()[indices])
        assert np.all(np.diff(indices) > 0)
        assert_allclose(weights.sum(), 0, atol=1e-15)
        value = float(weights @ dense[np.ix_(indices, indices)] @ weights)
        calls.append((indices.copy(), weights.copy(), value))
        return value

    result = scan_compensated_apertures(
        signal,
        valid,
        covariance,
        variance_terms={"cross": lambda i, w: -0.01},
        core_radius=1,
        background_inner=2,
        background_outer=3,
        minimum_valid_fraction=0.5,
    )
    assert (result["best_row"], result["best_column"]) == (3, 3)
    assert result["variance_contributions"]["cross"] == -0.01
    assert_allclose(result["best_sigma"] ** 2, calls[0][2])
    assert_array_equal(result["score_sigma"][result["valid_support_mask"]], 0)
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="variance"):
            scan_compensated_apertures(
                signal,
                valid,
                lambda i, w, bad=bad: bad,
                core_radius=1,
                background_inner=2,
                background_outer=3,
                minimum_valid_fraction=0.5,
            )
    with pytest.raises(ValueError, match="contributions"):
        scan_compensated_apertures(
            signal,
            valid,
            covariance,
            variance_terms={"bad": lambda i, w: np.nan},
            core_radius=1,
            background_inner=2,
            background_outer=3,
            minimum_valid_fraction=0.5,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        scan_compensated_apertures(signal, np.ones((2, 2), bool), covariance)
    with pytest.raises(ValueError, match="invalid compensated"):
        scan_compensated_apertures(signal, valid, covariance, core_radius=4, background_inner=2)


def test_removed_inverse_modules_have_one_authoritative_successor():
    assert importlib.util.find_spec("orsia_inverse.estimation") is None
    assert importlib.util.find_spec("orsia_inverse.damped") is None
    assert solve_map.__module__ == "orsia_inverse.nonlinear.map"
    assert solve_damped_gaussian.__module__ == "orsia_inverse.nonlinear.damped"
