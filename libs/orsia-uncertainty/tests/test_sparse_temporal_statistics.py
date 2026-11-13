"""Sparse covariance, chronological draws and finite-sample summary contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from orsia_uncertainty.sparse import sparse_row_product, validate_sparse_covariance
from orsia_uncertainty.spatial import covariance_from_marginals
from orsia_uncertainty.statistics import (
    finite_interval_sensitivity,
    half_probability_crossing,
    wilson_interval,
)
from orsia_uncertainty.temporal import sample_linear_process
from scipy.sparse import csr_matrix


def test_sparse_row_products_preserve_signed_duplicates_and_source_domain() -> None:
    left = csr_matrix(([1.0, -2.0, 3.0], [1, 1, 900], [0, 2, 3]), shape=(2, 1000000))
    right = csr_matrix(([4.0, -1.0], [1, 900], [0, 1, 2]), shape=(2, 1000000))
    assert_allclose(sparse_row_product(left, right).toarray(), [[-4, 0], [0, -3]])
    compact_left = csr_matrix([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    assert_allclose(sparse_row_product(compact_left, compact_left).toarray(), [[5, 0], [0, 9]])
    assert sparse_row_product(csr_matrix((0, 3)), compact_left).shape == (0, 2)
    with pytest.raises(ValueError):
        sparse_row_product(left, csr_matrix((2, 3)))


@pytest.mark.parametrize(
    "matrix", [np.zeros((2, 2)), np.eye(3), np.array([[1.0, 0.8], [0.8, 1.0]])]
)
def test_sparse_psd_contract(matrix: NDArray[np.float64]) -> None:
    validate_sparse_covariance(csr_matrix(matrix))
    for invalid in [
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.diag([-1.0, 1.0]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ]:
        with pytest.raises(ValueError):
            validate_sparse_covariance(csr_matrix(invalid))
    with pytest.raises(ValueError):
        covariance_from_marginals([0.1, 0.1], csr_matrix(np.eye(2)))
    with pytest.raises(ValueError):
        covariance_from_marginals([0.1, 0.1], common_modes=np.full((2, 1), np.nan))


def test_linear_process_preserves_draw_shape_order_and_rng_advancement() -> None:
    transition = np.array([[0.8, 0.1], [0.0, 0.6]])
    stationary = np.array([[2.0, 0.0], [0.3, 1.0]])
    innovation = np.eye(2) * 0.2
    expected_rng, actual_rng = np.random.default_rng(134), np.random.default_rng(134)
    state = stationary @ expected_rng.standard_normal(2)
    expected = [state]
    for _ in range(7):
        state = transition @ state + innovation @ expected_rng.standard_normal(2)
        expected.append(state)
    actual = sample_linear_process(transition, stationary, innovation, 8, actual_rng)
    assert_array_equal(actual, expected)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    with pytest.raises(ValueError):
        sample_linear_process(transition, stationary, innovation, 0, actual_rng)


def test_metric_probability_and_binomial_domains() -> None:
    assert half_probability_crossing(np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.4, 0.7])) == (
        4 / 3,
        [1.0, 2.0],
    )
    assert half_probability_crossing(np.array([0.0, 1.0]), np.array([0.1, 0.2])) == (None, None)
    assert half_probability_crossing(np.array([0.0, 1.0]), np.array([0.6, 0.7])) == (None, None)
    summary = finite_interval_sensitivity(2, 1, 3, 4, 2, 6)
    assert summary["slope"] == 2 and summary["elasticity"] == 1
    assert finite_interval_sensitivity(0, -1, 1, 1, 1, 1)["elasticity"] is None
    with pytest.raises(ValueError):
        finite_interval_sensitivity(2, 3, 1, 1, 1, 1)
    assert wilson_interval(0, 10)[0] == 0
    assert_allclose(wilson_interval(10, 10)[1], 1, rtol=0, atol=np.finfo(float).eps)
    assert wilson_interval(5, 10)[0] < 0.5 < wilson_interval(5, 10)[1]
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_projected_gaussian_statistics_preserve_sampling_law() -> None:
    from orsia_uncertainty.statistics import projected_gaussian_statistics

    weights = np.array([0.3, -0.2, 0.5])
    covariance = np.array([[2.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 3.0]])
    result = projected_gaussian_statistics(
        weights, covariance, np.random.default_rng(451), sample_count=10000
    )
    assert_allclose(result["analytical_sigma"] ** 2, weights @ covariance @ weights, rtol=1e-15)
    assert abs(result["sigma_ratio"] - 1) < 0.04
    assert abs(result["coverage_one_sigma"] - 0.68268949) < 0.025
    with pytest.raises(ValueError):
        projected_gaussian_statistics(weights, covariance, np.random.default_rng(0), sample_count=1)
    with pytest.raises(ValueError):
        projected_gaussian_statistics(
            np.zeros(3), covariance, np.random.default_rng(0), sample_count=2
        )
