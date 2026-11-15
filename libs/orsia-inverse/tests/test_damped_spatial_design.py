"""Numerical-method oracles independent of application unknowns and policies."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_inverse import TrialDomainError, prior_from_precision, solve_damped_gaussian
from orsia_inverse.contrast import estimate_shared_contrast_offset
from orsia_inverse.detection import (
    aperture_scorer,
    calibrate_null_threshold,
    scan_compensated_apertures,
)
from orsia_inverse.diagnostics import linear_target_uncertainty
from orsia_inverse.spatial import masked_second_difference_operator, solve_correlated_field
from orsia_uncertainty import factor_covariance
from orsia_uncertainty.spatial import covariance_from_marginals
from scipy.sparse import csr_matrix


def test_damped_linear_model_and_rejected_trial_diagnostics() -> None:
    design = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    observed = np.array([0.3, 0.8, 0.5])
    prior = np.diag([0.1, 0.2])

    def model(q: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return design @ q, design.copy()

    result = solve_damped_gaussian(
        observations=observed,
        initial_state=np.zeros(2),
        evaluate=model,
        observation_factor=factor_covariance(np.ones(3), method="cholesky"),
        prior=prior_from_precision(np.zeros(2), prior),
    )
    expected = np.linalg.solve(design.T @ design + prior, design.T @ observed)
    assert_allclose(result.state, expected, atol=1e-8)
    assert result.termination.success and result.termination.iterations is not None
    assert result.termination.iterations > 0
    exhausted = solve_damped_gaussian(
        observations=observed,
        initial_state=np.zeros(2),
        evaluate=model,
        observation_factor=factor_covariance(np.ones(3), method="cholesky"),
        prior=prior_from_precision(np.zeros(2), prior),
        max_iterations=1,
        step_tolerance=1e-30,
    )
    assert not exhausted.termination.success

    def reject(q: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if np.any(q != 0):
            raise TrialDomainError("unsupported trial")
        return np.zeros(3), design.copy()

    failed = solve_damped_gaussian(
        observations=observed,
        initial_state=np.zeros(2),
        evaluate=reject,
        observation_factor=factor_covariance(np.ones(3), method="cholesky"),
        prior=prior_from_precision(np.zeros(2), prior),
    )
    assert not failed.termination.success and failed.termination.reason == "line_search_failed"
    stationary = solve_damped_gaussian(
        observations=np.zeros(3),
        initial_state=np.zeros(2),
        evaluate=model,
        observation_factor=factor_covariance(np.ones(3), method="cholesky"),
        prior=prior_from_precision(np.zeros(2), prior),
    )
    assert stationary.termination.success and stationary.termination.reason == "step_tolerance"
    with pytest.raises(ValueError):
        solve_damped_gaussian(
            observations=observed,
            initial_state=np.zeros(2),
            evaluate=model,
            observation_factor=factor_covariance(np.ones(3), method="cholesky"),
            damping=0,
        )


def test_shared_offset_is_common_and_explicit_fallback() -> None:
    prediction = np.broadcast_to(np.array([1.0, 1.2, 0.9, 1.1]), (8, 4)).copy()
    observed = prediction + np.array([0.003, -0.001, 0.002, -0.004])
    covariance = np.broadcast_to(np.eye(4) * 0.001, (8, 4, 4)).copy()
    calibration = estimate_shared_contrast_offset(
        observed, prediction, covariance, minimum_pixels=2
    )
    assert calibration["converged"]
    assert abs(calibration["channel_relative_correction"].sum()) < 1e-16
    assert np.linalg.eigvalsh(calibration["contrast_posterior_covariance"]).min() > 0
    fallback = estimate_shared_contrast_offset(observed, prediction, covariance)
    assert not fallback["converged"] and np.isnan(fallback["contrast_posterior_covariance"]).all()
    with pytest.raises(ValueError):
        estimate_shared_contrast_offset(observed, prediction[:, :-1], covariance)


def test_correlated_field_matches_dense_masked_map() -> None:
    mask = np.ones((5, 6), bool)
    mask[2, 2] = False
    d, indices = masked_second_difference_operator(mask)
    assert d.shape[1] == len(indices)
    rng = np.random.default_rng(13)
    modes = rng.normal(size=(len(indices), 2)) * 0.01
    detector = csr_matrix(np.diag(np.full(len(indices), 0.02)))
    marginal = 0.03 + (modes**2).sum(axis=1)
    apply, diagonal, independent = covariance_from_marginals(marginal, detector, modes)
    observation = rng.normal(size=len(indices)) * 0.01
    result = solve_correlated_field(observation, apply, diagonal, d, curvature_rms=0.05)
    dense = detector.toarray() + np.diag(independent) + modes @ modes.T
    precision = np.linalg.inv(dense)
    expected = np.linalg.solve(precision + (d.T @ d).toarray() / 0.05**2, precision @ observation)
    assert_allclose(result["state"], expected, atol=1e-11)
    assert result["converged"]
    assert result["inner_iterations"] >= result["maximum_inner_iterations"]
    exhausted = solve_correlated_field(
        observation, apply, diagonal, d, curvature_rms=0.05, maximum_iterations=0
    )
    assert not exhausted["converged"]
    with pytest.raises(ValueError):
        masked_second_difference_operator(np.zeros((3, 3), bool))
    with pytest.raises(ValueError):
        solve_correlated_field(observation, apply, -diagonal, d, curvature_rms=0.05)
    with pytest.raises(ValueError, match="did not converge"):
        solve_correlated_field(observation, lambda v: -v, diagonal, d, curvature_rms=0.05)


def test_linear_target_information() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    variance = np.array([1.0, 2.0, 3.0])
    result = linear_target_uncertainty(
        matrix,
        factor_covariance(variance, method="cholesky"),
        np.array([1.0, 0.0]),
        prior=prior_from_precision(np.zeros(2), np.array([0.0, 1 / 0.3**2])),
    )
    expected = np.linalg.inv(matrix.T @ (matrix / variance[:, None]) + np.diag([0.0, 1 / 0.3**2]))
    assert_allclose(result["covariance"], expected, atol=1e-15)
    assert_allclose(result["target_weights"], (expected @ matrix.T / variance)[0], atol=1e-15)

    with pytest.raises(ValueError):
        linear_target_uncertainty(
            matrix, factor_covariance(-variance, method="cholesky"), np.array([1.0, 0.0])
        )


def test_aperture_covariance_score_and_independent_null_threshold() -> None:
    shape = (17, 18)
    y = np.zeros(shape)
    y[8, 9] = 2
    valid = np.ones(shape, bool)
    valid[5, 5] = False
    variance = np.full(shape, 0.25)

    def covariance(indices, weights):
        return float(np.sum(variance.ravel()[indices] * weights**2) + 3 * weights.sum() ** 2)

    terms = {"offset": lambda indices, weights: float(3 * weights.sum() ** 2)}
    result = scan_compensated_apertures(
        y,
        valid,
        covariance,
        variance_terms=terms,
        core_radius=1,
        background_inner=3,
        background_outer=4,
        minimum_valid_fraction=0.5,
    )
    score = aperture_scorer(valid, (1, 3, 4), result["valid_support_mask"], result["sigma"])
    assert_allclose(score(y), result["score_sigma"], atol=1e-14, equal_nan=True)
    assert result["variance_contributions"]["offset"] < 1e-15
    assert result["best_row"] > 0 and result["best_column"] > 0
    threshold, rank = calibrate_null_threshold(np.arange(100.0), 0.1)
    assert rank == 91 and threshold == 90
    assert not 90 > threshold
    with pytest.raises(ValueError):
        calibrate_null_threshold(np.arange(2.0), 0.01)
    with pytest.raises(ValueError):
        scan_compensated_apertures(
            y,
            np.zeros(shape, bool),
            covariance,
            core_radius=1,
            background_inner=3,
            background_outer=4,
        )
