"""Array covariance validation, representation equivalence and ownership contracts."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_inverse import prior_from_covariance, retrieval_diagnostics, solve_map
from orsia_uncertainty import factor_covariance


@pytest.mark.parametrize("name", ["observation_factor", "prior_factor"])
def test_factor_dimensions_and_inverse_support(name: str) -> None:
    for covariance in (np.ones(3), np.array([1.0, 1e-12]), np.array([1.0, 0.0])):
        options: dict[str, Any] = {
            "observation_factor": factor_covariance(np.ones(2), method="eigh")
        }
        with pytest.raises(ValueError, match="size|ill-conditioned"):
            factor = factor_covariance(covariance, method="eigh")
            if name == "prior_factor":
                options["prior"] = prior_from_covariance(np.zeros(2), factor)
            else:
                options[name] = factor
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
        observation_factor=factor_covariance(
            variances if diagonal_observation else np.diag(variances), method="cholesky"
        ),
        prior=prior_from_covariance(
            qa,
            factor_covariance(
                prior_variances if diagonal_prior else np.diag(prior_variances), method="cholesky"
            ),
        ),
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
        observation_factor=factor_covariance(se, method="cholesky"),
        prior=prior_from_covariance(qa, factor_covariance(sa, method="cholesky")),
    )
    np.testing.assert_allclose(estimate.state, [4 / 6, 5 / 8])
    assert estimate.prior is not None
    for retained, original in (
        (estimate.observation_factor.root, se),
        (estimate.prior.precision_root, sa),
    ):
        assert not retained.flags.writeable
        assert not np.shares_memory(retained, original)
    np.testing.assert_array_equal(estimate.prior.mean, np.zeros(2))
    diagnostics = retrieval_diagnostics(estimate, include_gain=True, include_averaging_kernel=True)
    for values in (
        diagnostics.posterior_covariance,
        diagnostics.gain,
        diagnostics.averaging_kernel,
    ):
        assert values is not None and values.flags.writeable
        assert values.dtype == np.float64 and values.shape == (2, 2)
        assert not np.shares_memory(values, estimate.observation_factor.root)
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
        observation_factor=factor_covariance(covariance, method="cholesky"),
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
        observation_factor=factor_covariance(covariance, method="cholesky"),
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
        solve_map(
            lambda q: q,
            lambda q: np.eye(2),
            observation_factor=factor_covariance(np.ones(2), method="cholesky"),
            **options,
        )
