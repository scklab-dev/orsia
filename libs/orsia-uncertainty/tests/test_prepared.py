"""Factor ownership, one-time decomposition, PSD sampling and stable solves."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
from orsia_uncertainty import (
    factor_covariance,
    sample_gaussian,
    solve_covariance,
    validate_covariance,
    whiten,
)


@pytest.mark.parametrize("method", ["eigh", "cholesky"])
@pytest.mark.parametrize("condition", [1.0, 1e4, 1e10])
def test_factor_reuse_backward_error_and_concurrent_calls(
    method: Literal["eigh", "cholesky"], condition: float
) -> None:
    rng = np.random.default_rng(54)
    basis, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    matrix = (basis * np.geomspace(1 / condition, 1, 12)) @ basis.T
    saved = matrix.copy()
    backend = "eigh" if method == "eigh" else "eigvalsh"
    with patch.object(np.linalg, backend, wraps=getattr(np.linalg, backend)) as spectrum:
        factor = factor_covariance(matrix, method=method)
        vectors = tuple(rng.standard_normal(12) for _ in range(4))
        results = tuple(solve_covariance(factor, v) for v in vectors)
        sample_gaussian(factor, rng, 2)
        whiten(factor, np.eye(12))
        assert spectrum.call_count == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = tuple(pool.map(partial(solve_covariance, factor), vectors))
    for vector, result, concurrent in zip(vectors, results, parallel, strict=True):
        np.testing.assert_array_equal(result, concurrent)
        np.testing.assert_allclose(
            result, np.linalg.solve(matrix, vector), rtol=condition * 2e-14, atol=condition * 2e-14
        )
        scale = np.linalg.norm(matrix) * np.linalg.norm(result) + np.linalg.norm(vector)
        assert np.linalg.norm(matrix @ result - vector) / scale < 3e-15
        assert result.flags.writeable and not np.shares_memory(result, vector)
    for values in (factor.root, factor.eigenvalues):
        assert not values.flags.writeable and not np.shares_memory(values, matrix)
        with pytest.raises(ValueError):
            values.setflags(write=True)
    matrix.fill(99)
    np.testing.assert_array_equal(solve_covariance(factor, vectors[0]), results[0])
    assert np.isfinite(saved).all()


@pytest.mark.parametrize("diagonal", [True, False])
def test_psd_cutoff_and_tolerated_negatives(diagonal: bool) -> None:
    for small in (0.0, 1e-13, 1e-12):
        law = np.array([small, 1.0])
        factor = factor_covariance(law if diagonal else np.diag(law), method="eigh")
        assert sample_gaussian(factor, np.random.default_rng(1), 5).shape == (5, 2)
        for operation in (whiten, solve_covariance):
            with pytest.raises(ValueError, match="ill-conditioned"):
                operation(factor, np.ones(2))
    accepted = factor_covariance(np.array([np.nextafter(1e-12, np.inf), 1.0]), method="eigh")
    assert np.isfinite(whiten(accepted, np.ones(2))).all()
    factor = factor_covariance(np.zeros(2) if diagonal else np.zeros((2, 2)), method="eigh")
    rng = np.random.default_rng(101)
    np.testing.assert_array_equal(sample_gaussian(factor, rng, 3), 0)
    control = np.random.default_rng(101)
    control.standard_normal((3, 2))
    np.testing.assert_array_equal(rng.standard_normal(2), control.standard_normal(2))
    with pytest.raises(ValueError, match="positive"):
        sample_gaussian(factor, rng, 0)
    if not diagonal:
        tiny = factor_covariance(np.diag([-0.5e-12, 1.0]), method="eigh")
        assert tiny.eigenvalues[0] < 0
        np.testing.assert_array_equal(sample_gaussian(tiny, rng, 5)[:, 0], 0)
        with pytest.raises(ValueError, match="ill-conditioned"):
            whiten(tiny, np.ones(2))
        with pytest.raises(ValueError, match="positive definite"):
            factor_covariance(np.zeros((2, 2)), method="cholesky")


def test_factor_invalid_backend_and_failures() -> None:
    with pytest.raises(ValueError, match="method"):
        factor_covariance(np.eye(2), method=cast(Literal["eigh", "cholesky"], "bad"))
    with patch("orsia_uncertainty.covariance.cholesky", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="factorisation failed"):
            factor_covariance(np.eye(2), method="cholesky")


@pytest.mark.parametrize("method", ["eigh", "cholesky"])
def test_diagonal_extreme_dynamic_range_preserves_small_draws(method) -> None:
    variances = np.array([1e200, 1e-200])
    saved = variances.copy()
    factor = factor_covariance(variances, method=method)
    np.testing.assert_array_equal(validate_covariance(variances), saved)
    np.testing.assert_allclose(factor.root * np.sqrt(factor.scale), np.sqrt(saved), rtol=2e-15)
    draws = sample_gaussian(factor, np.random.default_rng(81), 4)
    normal = np.random.default_rng(81).standard_normal((4, 2))
    np.testing.assert_allclose(draws / np.sqrt(saved), normal, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(variances, saved)
    for operation in (whiten, solve_covariance):
        with pytest.raises(ValueError, match="ill-conditioned"):
            operation(factor, np.ones(2))


@pytest.mark.parametrize("operation", ["validate", "eigh", "cholesky"])
def test_diagonal_rejects_negative_variance_before_normalization(operation) -> None:
    variances = np.array([1e200, -1e-200])
    saved = variances.copy()
    with pytest.raises(ValueError, match="nonnegative"):
        if operation == "validate":
            validate_covariance(variances)
        else:
            factor_covariance(variances, method=operation)
    np.testing.assert_array_equal(variances, saved)


@pytest.mark.parametrize("method", ["eigh", "cholesky"])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("operation", ["whiten", "transpose", "solve"])
def test_inverse_root_avoids_overflow_for_finite_results(method, columns, operation) -> None:
    factor = factor_covariance(np.diag([1e-10, 1.0]), method=method)
    values = np.array([1e295 if operation == "solve" else 1e300, 0.0])
    expected = np.array([1e305, 0.0])
    if columns:
        values = np.column_stack((values, -values))
        expected = np.column_stack((expected, -expected))
    saved = values.copy()
    with np.errstate(over="raise", invalid="raise"):
        actual = (
            solve_covariance(factor, values)
            if operation == "solve"
            else whiten(factor, values, transpose=operation == "transpose")
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(values, saved)
    assert actual.flags.writeable and not np.shares_memory(actual, values)
