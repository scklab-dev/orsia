"""Numerical covariance contracts without coordinate or unit metadata."""

from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_uncertainty import (
    covariance_product,
    covariance_solver,
    gaussian_sampler,
    propagate_covariance,
    propagate_vector,
    sample_gaussian,
    solve_covariance,
    validate_covariance,
    whitener,
)


@pytest.mark.parametrize("diagonal", [False, True])
def test_reference_apply_solve_snapshot_and_noncontiguous_inputs(diagonal: bool) -> None:
    backing = np.array([[4.0, 8.0, 1.0, 8.0], [1.0, 8.0, 3.0, 8.0]])
    source = np.array([4.0, 8.0, 3.0, 8.0])[::2] if diagonal else backing[:, ::2]
    saved = source.copy()
    covariance = validate_covariance(source)
    vector = np.array([2.0, -1.0])
    applied = covariance_product(covariance, vector)
    np.testing.assert_array_equal(applied, [8.0, -3.0] if diagonal else [7.0, -1.0])
    solved = solve_covariance(covariance, applied)
    np.testing.assert_allclose(solved, vector, atol=1e-15)
    np.testing.assert_allclose(covariance_solver(covariance)(applied), vector, atol=1e-15)
    for result in (covariance, applied, solved):
        assert result.flags.writeable
        assert not np.shares_memory(result, source)
        assert not np.shares_memory(result, vector)
    source.fill(100)
    np.testing.assert_array_equal(covariance, saved)
    covariance.fill(0)
    np.testing.assert_array_equal(source, 100)


@pytest.mark.parametrize(
    "covariance, message",
    [
        (np.array(1.0), "square matrix"),
        (np.empty(0), "square matrix"),
        (np.empty((0, 0)), "square matrix"),
        (np.ones((2, 1)), "square matrix"),
        (np.ones((2, 2, 1)), "square matrix"),
        (np.eye(2, dtype=np.float32), "float64"),
        (np.array([1, 2]), "float64"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), "finite"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.array([1.0, -1.0]), "nonnegative"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_numerical_validation_at_every_covariance_boundary(
    covariance: NDArray[np.float64], message: str
) -> None:
    for operation in (
        validate_covariance,
        covariance_solver,
        gaussian_sampler,
        whitener,
        lambda c: sample_gaussian(c, np.random.default_rng(1), 2),
        lambda c: covariance_product(c, np.ones(2)),
        lambda c: solve_covariance(c, np.ones(2)),
        lambda c: propagate_covariance(c, np.eye(2)),
    ):
        with pytest.raises(ValueError, match=message):
            operation(covariance)


@pytest.mark.parametrize("diagonal", [False, True])
def test_singular_and_ill_conditioned_can_apply_transport_and_sample(diagonal: bool) -> None:
    for values in (np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, 1e-12])):
        covariance = values if diagonal else np.diag(values)
        np.testing.assert_array_equal(covariance_product(covariance, np.ones(2)), values)
        np.testing.assert_allclose(propagate_covariance(covariance, np.eye(2)), np.diag(values))
        assert sample_gaussian(covariance, np.random.default_rng(1), 3).shape == (3, 2)
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            solve_covariance(covariance, np.ones(2))
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            covariance_solver(covariance)
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            whitener(covariance)


@pytest.mark.parametrize("diagonal", [False, True])
def test_apply_and_solve_reject_invalid_vectors(diagonal: bool) -> None:
    covariance = np.ones(2) if diagonal else np.eye(2)
    for vector in (np.ones(3), np.ones((2, 1)), np.array([1.0, np.nan]), np.full(2, np.inf)):
        for operation in (covariance_product, solve_covariance):
            with pytest.raises(ValueError, match="vector"):
                operation(covariance, vector)


def test_transport_checks_dimensions_finiteness_and_new_numerical_anomalies() -> None:
    for jacobian in (np.ones(2), np.ones((2, 3)), np.empty((0, 2)), np.full((2, 2), np.inf)):
        for operation in (propagate_covariance, propagate_vector):
            with pytest.raises(ValueError, match="jacobian"):
                operation(np.ones(2), jacobian)
    for vector in (np.array(1.0), np.empty(0), np.ones((2, 1)), np.array([1.0, np.nan])):
        with pytest.raises(ValueError, match="vector"):
            propagate_vector(vector, np.eye(2))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            propagate_covariance(np.eye(2), np.eye(2) * 1e200)
        with pytest.raises(ValueError, match="finite"):
            propagate_vector(np.ones(2) * 1e200, np.eye(2) * 1e200)


def test_units_and_index_meanings_are_not_part_of_the_core_api() -> None:
    import orsia_uncertainty as uncertainty

    for name in (
        "ArrayLayout",
        "Covariance",
        "DiagonalCovariance",
        "SignedBias",
        "Contribution",
        "UncertaintySource",
    ):
        assert not hasattr(uncertainty, name)
    # Equal-shaped arrays are computed in their supplied order, without semantic checks.
    covariance = np.diag([2.0, 5.0])
    jacobian = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(propagate_covariance(covariance, jacobian), [[22.0]])
    np.testing.assert_allclose(propagate_covariance(covariance[::-1, ::-1], jacobian), [[13.0]])
    with pytest.raises(TypeError):
        propagate_covariance(covariance, jacobian, target_layout=None)  # type: ignore[unexpected-keyword]


def test_backend_failure_is_explicit() -> None:
    with patch(
        "orsia_uncertainty.covariance._minimum_eigenvalue", side_effect=np.linalg.LinAlgError
    ):
        with pytest.raises(ValueError, match="eigenvalue check"):
            validate_covariance(np.eye(2))
    with patch.object(np.linalg, "solve", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="solve failed"):
            solve_covariance(np.eye(2), np.ones(2))
    with patch.object(np.linalg, "eigh", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="factorisation failed"):
            covariance_solver(np.eye(2))
