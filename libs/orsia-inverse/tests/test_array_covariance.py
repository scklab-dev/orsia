"""Array covariance validation, representation equivalence and ownership contracts."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_inverse import retrieval_diagnostics, solve_map


@pytest.mark.parametrize("name", ["observation_covariance", "prior_covariance"])
@pytest.mark.parametrize(
    "covariance, message",
    [
        (np.ones(3), "shape"),
        (np.ones((2, 1)), "shape"),
        (np.ones((2, 2, 1)), "shape"),
        (np.array(1.0), "shape"),
        (np.ones(2, dtype=np.float32), "float64"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.zeros(2), "positive definite"),
        (np.zeros((2, 2)), "positive definite"),
        (np.array([1.0, -1.0]), "positive definite"),
        (np.array([1.0, 0.0]), "positive definite"),
        (np.ones((2, 2)), "positive definite"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (np.array([1.0, 1e-12]), "well-conditioned"),
        (np.diag([1.0, 1e-12]), "well-conditioned"),
        (np.array([[1.0, 0.1], [0.2, 1.0]]), "symmetric"),
    ],
)
def test_invalid_covariance(name: str, covariance: NDArray[np.float64], message: str) -> None:
    options: dict[str, Any] = {"observation_covariance": np.ones(2)}
    if name == "prior_covariance":
        options["prior_mean"] = np.zeros(2)
    options[name] = covariance
    with pytest.raises(ValueError, match=f"{name}.*{message}"):
        solve_map(lambda q: q, lambda q: np.eye(2), np.ones(2), np.zeros(2), **options)


@pytest.mark.parametrize("diagonal_observation", [False, True])
@pytest.mark.parametrize("diagonal_prior", [False, True])
def test_variance_vectors_and_dense_diagonals_agree(
    diagonal_observation: bool, diagonal_prior: bool
) -> None:
    variances = np.array([2.0, 5.0, 3.0])
    prior_variances = np.array([4.0, 9.0])
    k = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 1.0]])
    y, qa = np.array([1.0, -2.0, 0.5]), np.array([0.2, -0.3])
    result = solve_map(
        lambda q: k @ q,
        lambda q: k,
        y,
        np.zeros(2),
        observation_covariance=variances if diagonal_observation else np.diag(variances),
        prior_mean=qa,
        prior_covariance=prior_variances if diagonal_prior else np.diag(prior_variances),
    )
    precision = k.T @ np.diag(1 / variances) @ k + np.diag(1 / prior_variances)
    expected = np.linalg.solve(precision, k.T @ (y / variances) + qa / prior_variances)
    np.testing.assert_allclose(result.state, expected, atol=1e-12)
    diagnostics = retrieval_diagnostics(result, include_gain=True)
    assert diagnostics.available
    assert diagnostics.posterior_covariance is not None and diagnostics.gain is not None
    np.testing.assert_allclose(diagnostics.posterior_covariance, np.linalg.inv(precision))
    np.testing.assert_allclose(
        diagnostics.gain, np.linalg.solve(precision, k.T / variances), atol=1e-12
    )


@pytest.mark.parametrize("dense", [False, True])
def test_covariance_snapshots_and_writable_diagnostics(dense: bool) -> None:
    se = np.array([2.0, 3.0])
    sa = np.array([4.0, 5.0])
    if dense:
        se, sa = np.diag(se), np.diag(sa)
    expected_se, expected_sa = se.copy(), sa.copy()
    qa = np.zeros(2)

    def fun(q: NDArray[np.float64]) -> NDArray[np.float64]:
        # Mutation of caller storage after entry must not change fixed fit weights.
        se.fill(99)
        sa.fill(99)
        qa.fill(99)
        return q.copy()

    estimate = solve_map(
        fun,
        lambda q: np.eye(2),
        np.ones(2),
        np.zeros(2),
        observation_covariance=se,
        prior_mean=qa,
        prior_covariance=sa,
    )
    np.testing.assert_allclose(estimate.state, [4 / 6, 5 / 8])
    for retained, original, expected in (
        (estimate.observation_covariance, se, expected_se),
        (estimate.prior_covariance, sa, expected_sa),
        (estimate.prior_mean, qa, np.zeros(2)),
    ):
        assert retained is not None and retained.flags.writeable
        assert not np.shares_memory(retained, original)
        np.testing.assert_array_equal(retained, expected)
    diagnostics = retrieval_diagnostics(estimate, include_gain=True, include_averaging_kernel=True)
    for values in (
        diagnostics.posterior_covariance,
        diagnostics.gain,
        diagnostics.averaging_kernel,
    ):
        assert values is not None and values.flags.writeable
        assert values.dtype == np.float64 and values.shape == (2, 2)
        assert not np.shares_memory(values, estimate.observation_covariance)
        assert not np.shares_memory(values, estimate.jacobian)
    assert diagnostics.posterior_covariance is not None
    diagnostics.posterior_covariance.fill(99)
    repeated = retrieval_diagnostics(estimate)
    assert repeated.posterior_covariance is not None
    np.testing.assert_allclose(repeated.posterior_covariance, np.diag([8 / 6, 15 / 8]))


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
@pytest.mark.parametrize("dense", [False, True])
def test_covariance_scale_is_not_an_absolute_cutoff(scale: float, dense: bool) -> None:
    variances = scale * np.array([1.0, 2.0])
    covariance = np.diag(variances) if dense else variances
    # Start at the exact solution; this tests weights/diagnostics, not optimizer scaling.
    estimate = solve_map(
        lambda q: q,
        lambda q: np.eye(2),
        np.ones(2),
        np.ones(2),
        observation_covariance=covariance,
    )
    diagnostics = retrieval_diagnostics(estimate, include_gain=True)
    assert diagnostics.available and diagnostics.posterior_covariance is not None
    assert diagnostics.gain is not None
    np.testing.assert_allclose(diagnostics.posterior_covariance / scale, np.diag([1.0, 2.0]))
    np.testing.assert_allclose(diagnostics.gain, np.eye(2), atol=1e-14)


def test_roundoff_asymmetry_is_averaged_without_modifying_input() -> None:
    covariance = np.array([[2.0, 0.3 + 1e-13], [0.3, 1.0]])
    original = covariance.copy()
    estimate = solve_map(
        lambda q: q,
        lambda q: np.eye(2),
        np.ones(2),
        np.zeros(2),
        observation_covariance=covariance,
    )
    np.testing.assert_array_equal(covariance, original)
    diagnostics = retrieval_diagnostics(estimate)
    assert diagnostics.posterior_covariance is not None
    np.testing.assert_allclose(
        diagnostics.posterior_covariance,
        (original + original.T) / 2,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("name", ["observations", "initial_state"])
@pytest.mark.parametrize("values", [np.empty(0), np.ones((2, 1)), np.array([np.nan, 1.0])])
def test_invalid_vectors(name: str, values: NDArray[np.float64]) -> None:
    options: dict[str, Any] = {"observations": np.ones(2), "initial_state": np.zeros(2)}
    options[name] = values
    with pytest.raises(ValueError, match=name):
        solve_map(lambda q: q, lambda q: np.eye(2), observation_covariance=np.ones(2), **options)


@pytest.mark.parametrize(
    "keyword", ["observation_layout", "state_layout", "observation_unit_squared"]
)
def test_removed_metadata_keywords_are_rejected(keyword: str) -> None:
    options: dict[str, Any] = {keyword: None}
    with pytest.raises(TypeError, match=keyword):
        solve_map(
            lambda q: q,
            lambda q: np.eye(1),
            np.ones(1),
            np.zeros(1),
            observation_covariance=np.ones(1),
            **options,
        )
