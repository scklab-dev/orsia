"""Independent references for source sums, signed transport and ensembles."""

from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_uncertainty import (
    combine_contributions,
    empirical_covariance,
    propagate_bias,
    propagate_covariance,
    propagate_vector,
    sample_gaussian,
)


def test_dense_diagonal_and_sequential_transport_match_direct_reference() -> None:
    matrix = np.array([[4.0, -1.0], [-1.0, 9.0]])
    first = np.array([[2.0, -1.0], [-0.5, 3.0]])
    second = np.array([[-1.0, 0.25], [1.0, 2.0]])
    for covariance in (matrix, np.diag(matrix).copy()):
        saved = covariance.copy()
        reference = np.diag(covariance) if covariance.ndim == 1 else covariance
        one = propagate_covariance(covariance, first)
        two = propagate_covariance(one, second)
        direct = propagate_covariance(covariance, second @ first)
        np.testing.assert_allclose(one, first @ reference @ first.T, rtol=1e-14)
        np.testing.assert_allclose(two, direct, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(covariance, saved)
        assert one.flags.writeable and not np.shares_memory(one, covariance)
    bias = np.array([2.0, -3.0])
    transported = propagate_bias(propagate_bias(bias, first), second)
    np.testing.assert_allclose(transported, (second @ first) @ bias, rtol=1e-14)
    np.testing.assert_allclose(transported, propagate_vector(bias, second @ first))
    assert not np.shares_memory(transported, bias)


def test_three_sources_preserve_nonsymmetric_cross_blocks_and_zero_pairs() -> None:
    contributions = {
        "a": np.array([[2.0, 0.25], [0.25, 1.0]]),
        "b": np.array([3.0, 2.0]),
        "c": np.array([[1.0, -0.1], [-0.1, 2.0]]),
    }
    ab = np.array([[0.1, 0.2], [-0.05, 0.15]])
    bc = np.array([[0.2, 0.0], [0.1, -0.1]])
    cross = {("a", "b"): ab, ("a", "c"): np.zeros((2, 2)), ("b", "c"): bc}
    expected = (
        contributions["a"]
        + np.diag(contributions["b"])
        + contributions["c"]
        + ab
        + ab.T
        + bc
        + bc.T
    )
    combined = combine_contributions(contributions, cross)
    np.testing.assert_allclose(combined, expected)
    np.testing.assert_allclose(
        combine_contributions(contributions, dict(reversed(tuple(cross.items())))), expected
    )
    jacobian = np.array([[1.0, 2.0], [-1.0, 1.0]])
    np.testing.assert_allclose(
        propagate_covariance(combined, jacobian), jacobian @ expected @ jacobian.T
    )
    ab.fill(100)
    contributions["a"].fill(100)
    np.testing.assert_allclose(combined, expected)


def test_sources_reject_missing_pairs_bad_blocks_and_impossible_joint_law() -> None:
    contributions = {"a": np.ones(2), "b": np.ones(2)}
    for pairs in (
        {},
        {("b", "a"): np.zeros((2, 2))},
        {("a", "b"): np.zeros((2, 2)), ("b", "a"): np.zeros((2, 2))},
    ):
        with pytest.raises(ValueError, match="cross covariance"):
            combine_contributions(contributions, pairs)
    for bad in (np.ones((2, 1)), np.full((2, 2), np.nan), np.ones((2, 2), dtype=np.float32)):
        with pytest.raises(ValueError, match="finite float64 matrix"):
            combine_contributions(contributions, {("a", "b"): cast(NDArray[np.float64], bad)})
    # The covariance of the sum is PSD, but the joint source law is impossible.
    with pytest.raises(ValueError, match="joint source covariance"):
        combine_contributions(contributions, {("a", "b"): 2 * np.eye(2)})
    with pytest.raises(ValueError, match="dimensions"):
        combine_contributions({"a": np.ones(2), "b": np.ones(3)}, {("a", "b"): np.zeros((2, 2))})
    with pytest.raises(ValueError, match="at least one"):
        combine_contributions({}, {})


@pytest.mark.parametrize("dense", [True, False])
def test_exact_independence_avoids_joint_decomposition_and_returns_dense(dense: bool) -> None:
    blocks = tuple(np.diag([float(i), float(i + 1)]) for i in range(1, 5))
    contributions = {
        str(i): block if dense else np.diag(block).copy() for i, block in enumerate(blocks)
    }
    cross = {(str(i), str(j)): np.zeros((2, 2)) for i in range(4) for j in range(i + 1, 4)}
    with patch("orsia_uncertainty.operations._minimum_eigenvalue") as spectrum:
        combined = combine_contributions(contributions, cross)
    spectrum.assert_not_called()
    assert isinstance(combined, np.ndarray) and combined.shape == (2, 2)
    np.testing.assert_array_equal(combined, sum(blocks))
    single = combine_contributions({"a": contributions["0"]}, {})
    np.testing.assert_array_equal(single, blocks[0])


def test_any_nonzero_cross_block_still_validates_joint_law() -> None:
    from orsia_uncertainty.covariance import _minimum_eigenvalue

    cross = np.full((2, 2), np.nextafter(0.0, 1.0))
    with patch(
        "orsia_uncertainty.operations._minimum_eigenvalue", wraps=_minimum_eigenvalue
    ) as spectrum:
        combine_contributions({"a": np.ones(2), "b": np.ones(2)}, {("a", "b"): cross})
    assert spectrum.call_args.args[0].shape == (4, 4)


def test_gaussian_seeded_centred_singular_laws_and_rng_ownership() -> None:
    covariance = np.array([[2.0, 2.0], [2.0, 2.0]])
    draws = sample_gaussian(covariance, np.random.default_rng(99), 20000)
    np.testing.assert_array_equal(
        draws, sample_gaussian(covariance, np.random.default_rng(99), 20000)
    )
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    np.testing.assert_allclose(draws.mean(axis=0), [0.0, 0.0], atol=0.04)
    np.testing.assert_allclose(empirical_covariance(draws), covariance, atol=0.06)
    assert draws.flags.writeable and not np.shares_memory(draws, covariance)
    rng = np.random.default_rng(101)
    np.testing.assert_array_equal(sample_gaussian(np.zeros(2), rng, 3), 0)
    second = sample_gaussian(np.ones(2), rng, 3)
    control = np.random.default_rng(101)
    control.standard_normal((3, 2))
    np.testing.assert_array_equal(second, control.standard_normal((3, 2)))
    with pytest.raises(ValueError, match="positive"):
        sample_gaussian(covariance, rng, 0)


def test_empirical_covariance_centres_rows_and_uses_n_minus_one() -> None:
    samples = np.array([[2.0, -1.0], [4.0, 3.0], [6.0, 5.0]])
    expected = np.cov(samples, rowvar=False, ddof=1)
    empirical = empirical_covariance(samples)
    np.testing.assert_allclose(empirical, expected)
    samples.fill(100)
    np.testing.assert_allclose(empirical, expected)
    assert empirical.flags.writeable and not np.shares_memory(empirical, samples)
    np.testing.assert_allclose(empirical_covariance(np.array([[1.0], [3.0]])), [[2.0]])


@pytest.mark.parametrize(
    "samples",
    [
        np.ones((1, 2)),
        np.empty((3, 0)),
        np.ones(2),
        np.array([[np.inf, 1.0], [2.0, 3.0]]),
        np.ones((3, 2), dtype=np.float32),
    ],
)
def test_empirical_rejects_invalid_numerical_inputs(samples: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="ensemble"):
        empirical_covariance(samples)
