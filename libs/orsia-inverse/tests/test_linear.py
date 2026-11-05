"""Independent weighted linear references, missing reads and rank-aware uncertainty."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_inverse.linear import solve_linear


def solve(matrix, observations, **kwargs):
    parameters = dict(covariance=np.ones(len(observations)), max_bytes=10_000_000)
    parameters.update(kwargs)
    return solve_linear(
        np.asarray(matrix, dtype=float), np.asarray(observations, dtype=float), **parameters
    )


def test_weighted_scalar_reference_and_correlated_noise():
    matrix = np.array([[1.0], [2.0], [3.0]])
    observed = np.array([2.0, 3.0, 7.0])
    variances = np.array([1.0, 4.0, 9.0])
    result = solve(matrix, observed, covariance=variances)
    assert_allclose(result.state, [np.sum(matrix[:, 0] * observed / variances) / 3])
    assert_allclose(result.covariance, [[1 / 3]])
    assert_allclose(result.resolution, [[1]])
    assert result.condition == 1
    covariance = np.diag(variances) + 0.3
    result = solve(matrix, observed, covariance=covariance)
    precision = np.linalg.inv(covariance)
    expected_gain = np.linalg.solve(matrix.T @ precision @ matrix, matrix.T @ precision)
    assert_allclose(result.gain, expected_gain)
    assert_allclose(result.covariance, expected_gain @ covariance @ expected_gain.T)
    rng = np.random.default_rng(11)
    noise = rng.multivariate_normal(np.zeros(3), covariance, size=12000)
    assert_allclose(np.var(noise @ result.gain.T), result.covariance[0, 0], rtol=0.04)


def test_rank_deficiency_missing_data_and_resolution_are_explicit():
    result = solve([[1, 1], [2, 2]], [2, 4])
    assert result.rank == 1
    assert np.isinf(result.condition)
    assert not result.identifiable.any()
    assert_allclose(result.state, [1, 1])
    assert_allclose(result.resolution, np.full((2, 2), 0.5))
    result = solve([[1, 0], [0, 1]], [3, np.nan], valid=np.array([True, False]))
    assert result.identifiable.tolist() == [True, False]
    assert_allclose(result.state, [3, 0])
    assert_allclose(result.gain[:, 1], 0)
    empty = solve([[1, 0], [0, 1]], [np.nan, np.nan], valid=np.zeros(2, dtype=bool))
    assert empty.rank == 0
    assert not empty.identifiable.any()
    assert empty.weighted_residual_norm == 0
    zero = solve([[0, 0]], [0])
    assert zero.rank == 0
    assert_allclose(zero.covariance, 0)


@pytest.mark.parametrize(
    "kwargs,match,error",
    [
        ({"max_bytes": 1}, "budget", MemoryError),
        ({"max_bytes": 0}, "positive", ValueError),
        ({"rcond": 1}, "cutoff", ValueError),
        ({"covariance": np.ones(3)}, "order", ValueError),
        ({"covariance": np.zeros(2)}, "singular", ValueError),
        ({"valid": np.array([1, 1])}, "Boolean", ValueError),
    ],
)
def test_linear_invalid_contracts(kwargs, match, error):
    with pytest.raises(error, match=match):
        solve(np.eye(2), [1, 2], **kwargs)


def test_invalid_model_and_observations():
    with pytest.raises(ValueError, match="matrix"):
        solve(np.ones(2), [1, 2])
    with pytest.raises(ValueError, match="finite"):
        solve([[np.nan]], [1])
    with pytest.raises(ValueError, match="finite"):
        solve([[1]], [np.nan])


@pytest.mark.parametrize("rcond", [0.1, 0.5, 0.99])
def test_large_rank_cutoff_never_marks_null_coordinates_identifiable(rcond):
    zero = solve(np.zeros((2, 3)), [0, 0], rcond=rcond)
    assert zero.rank == 0
    assert not zero.identifiable.any()
    partial = solve(np.diag([1.0, 0.01]), [1, 0.01], rcond=rcond)
    assert partial.identifiable.tolist() == [True, False]
