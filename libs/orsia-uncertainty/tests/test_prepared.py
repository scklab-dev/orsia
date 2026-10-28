"""Prepared callables retain numerical contracts, not hidden shared work buffers."""

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_uncertainty import (
    covariance_solver,
    gaussian_sampler,
    sample_gaussian,
    solve_covariance,
    validate_covariance,
    whitener,
)


def test_prepared_dense_solve_reuses_decomposition_and_has_small_backward_error() -> None:
    rng = np.random.default_rng(54)
    basis, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    for condition in (1.0, 1e4, 1e10):
        matrix = (basis * np.geomspace(1 / condition, 1, 12)) @ basis.T
        covariance = validate_covariance(matrix)
        saved = covariance.copy()
        with patch.object(np.linalg, "eigh", wraps=np.linalg.eigh) as decomposition:
            solve = covariance_solver(covariance)
            vectors = tuple(rng.standard_normal(12) for _ in range(4))
            results = tuple(solve(vector) for vector in vectors)
            assert decomposition.call_count == 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = tuple(pool.map(solve, vectors))
        for vector, result, parallel in zip(vectors, results, concurrent, strict=True):
            np.testing.assert_array_equal(result, parallel)
            reference = np.linalg.solve(matrix, vector)
            np.testing.assert_allclose(
                result, reference, rtol=condition * 2e-14, atol=condition * 2e-14
            )
            residual = np.linalg.norm(matrix @ result - vector)
            scale = np.linalg.norm(matrix) * np.linalg.norm(result) + np.linalg.norm(vector)
            assert residual / scale < 3e-15
            assert result.flags.writeable and not np.shares_memory(result, vector)
        np.testing.assert_array_equal(covariance, saved)


@pytest.mark.parametrize("diagonal", [False, True])
def test_prepared_solve_rejects_singular_and_ill_conditioned_laws(diagonal: bool) -> None:
    for variances in (
        np.zeros(2),
        np.array([0.0, 1.0]),
        np.array([1e-13, 1.0]),
        np.array([1e-12, 1.0]),
    ):
        covariance = (
            validate_covariance(variances) if diagonal else validate_covariance(np.diag(variances))
        )
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            covariance_solver(covariance)


@pytest.mark.parametrize("diagonal", [False, True])
def test_prepared_draws_match_one_off_seed_and_support_singular_covariances(diagonal: bool) -> None:
    covariance = (
        validate_covariance(np.array([2.0, 0.0]))
        if diagonal
        else validate_covariance(np.array([[2.0, 2.0], [2.0, 2.0]]))
    )
    rng = np.random.default_rng(54)
    control = np.random.default_rng(54)
    with patch.object(np.linalg, "eigh", wraps=np.linalg.eigh) as decomposition:
        draw = gaussian_sampler(covariance)
        for count in (1, 5, 100):
            actual = draw(rng, count)
            assert actual.shape == (count, 2) and actual.flags.writeable
        assert decomposition.call_count == (0 if diagonal else 1)
    rng = np.random.default_rng(54)
    for count in (1, 5, 100):
        np.testing.assert_array_equal(draw(rng, count), sample_gaussian(covariance, control, count))
    with pytest.raises(ValueError, match="positive"):
        draw(rng, 0)
    zeros = gaussian_sampler(validate_covariance(np.zeros(2)))
    np.testing.assert_array_equal(zeros(np.random.default_rng(1), 3), 0)


def test_prepared_diagonal_solve_and_vector_validation() -> None:
    covariance = validate_covariance(np.array([2.0, 4.0]))
    vector = np.array([1.0, 2.0])
    solve = covariance_solver(covariance)
    np.testing.assert_array_equal(solve(vector), solve_covariance(covariance, vector))
    dense = covariance_solver(validate_covariance(np.diag([2.0, 4.0])))
    for operation in (solve, dense):
        for bad in (
            np.ones(3),
            np.ones((2, 1)),
            np.array([np.nan, 1.0]),
        ):
            with pytest.raises(ValueError, match="vector"):
                operation(cast(NDArray[np.float64], bad))


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("method", ["eigh", "cholesky"])
def test_prepared_operations_capture_covariance_values(
    diagonal: bool, method: Literal["eigh", "cholesky"]
) -> None:
    covariance = (
        validate_covariance(np.array([2.0, 4.0]))
        if diagonal
        else validate_covariance(np.diag([2.0, 4.0]))
    )
    solve = covariance_solver(covariance, method=method)
    whiten = whitener(covariance)
    draw = gaussian_sampler(covariance)
    vector = np.array([1.0, 2.0])
    expected_solve = solve(vector)
    expected_whiten = whiten(vector)
    expected_draw = draw(np.random.default_rng(51), 3)
    covariance[:] *= 2
    np.testing.assert_array_equal(solve(vector), expected_solve)
    np.testing.assert_array_equal(whiten(vector), expected_whiten)
    np.testing.assert_array_equal(draw(np.random.default_rng(51), 3), expected_draw)
