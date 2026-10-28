"""Selected spectra and opt-in Cholesky retain covariance contracts."""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_uncertainty import (
    combine_contributions,
    covariance_solver,
    validate_covariance,
)
from orsia_uncertainty import covariance as covariance_module


def test_selected_minimum_validates_psd_without_a_full_spectrum() -> None:
    rng = np.random.default_rng(63)
    for size in (1, 4, 32, 127, 128):
        basis, _ = np.linalg.qr(rng.standard_normal((size, size)))
        spectrum = np.linspace(0.01, 1.0, size)
        matrix = (basis * spectrum) @ basis.T
        saved = matrix.copy()
        with patch.object(covariance_module, "eigvalsh", wraps=covariance_module.eigvalsh) as eigen:
            law = validate_covariance(matrix)
        if size >= 128:
            assert eigen.call_args.kwargs["subset_by_index"] == [0, 0]
        else:
            eigen.assert_not_called()
        np.testing.assert_array_equal(matrix, saved)
        assert law.flags.writeable
        np.testing.assert_allclose(np.linalg.eigvalsh(law), spectrum, atol=3e-15)
        if size > 1:
            spectrum[0] = -1e-3
            with pytest.raises(ValueError, match="positive semidefinite"):
                validate_covariance((basis * spectrum) @ basis.T)
    # The PSD tolerance and singular-law support are not replaced by Cholesky.
    validate_covariance(np.diag([-0.5e-12, 1.0]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance(np.diag([-2e-12, 1.0]))


def test_partial_spectrum_checks_large_joint_law_not_only_the_sum() -> None:
    covariance = validate_covariance(np.ones(64))
    sources = tuple((name, covariance) for name in ("first", "second"))
    cross = {("first", "second"): np.eye(64) * 0.25}
    with patch.object(covariance_module, "eigvalsh", wraps=covariance_module.eigvalsh) as eigen:
        combined = combine_contributions(dict(sources), cross)
    assert eigen.call_count == 1
    assert eigen.call_args.args[0].shape == (128, 128)
    assert eigen.call_args.kwargs["subset_by_index"] == [0, 0]
    np.testing.assert_array_equal(combined, 2.5 * np.eye(64))
    # The sum remains positive while the joint law has negative eigenvalues.
    with pytest.raises(ValueError, match="joint source covariance"):
        combine_contributions(dict(sources), {("first", "second"): 1.1 * np.eye(64)})


@pytest.mark.parametrize("condition", [1.0, 1e4, 1e10])
def test_cholesky_preparation_reuses_immutable_factor_and_solves_concurrently(
    condition: float,
) -> None:
    rng = np.random.default_rng(71)
    basis, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    matrix = (basis * np.geomspace(1 / condition, 1, 20)) @ basis.T
    covariance = validate_covariance(matrix)
    vectors = [rng.standard_normal(20) for _ in range(6)]
    with patch.object(
        covariance_module, "cho_factor", wraps=covariance_module.cho_factor
    ) as factor:
        operation = covariance_solver(covariance, method="cholesky")
        results = [operation(vector) for vector in vectors]
        assert factor.call_count == 1
    with patch.object(covariance_module, "cho_solve", wraps=covariance_module.cho_solve) as solve:
        operation(vectors[0])
        triangle, lower = solve.call_args.args[0]
        assert lower and triangle.flags.f_contiguous and not triangle.flags.writeable
        with pytest.raises(ValueError):
            triangle.setflags(write=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel = list(pool.map(operation, vectors))
    for vector, result, concurrent in zip(vectors, results, parallel, strict=True):
        np.testing.assert_array_equal(result, concurrent)
        scale = np.linalg.norm(matrix) * np.linalg.norm(result) + np.linalg.norm(vector)
        assert np.linalg.norm(matrix @ result - vector) / scale < 3e-15
        assert result.flags.writeable and not np.shares_memory(result, vector)
    np.testing.assert_array_equal(covariance, (matrix + matrix.T) / 2)
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("diagonal", [True, False])
def test_cholesky_preserves_spectral_conditioning_rejection(diagonal: bool) -> None:
    for variances in (
        np.zeros(2),
        np.array([0.0, 1.0]),
        np.array([1e-12, 1.0]),
        np.array([1e-13, 1.0]),
    ):
        law = (
            validate_covariance(variances) if diagonal else validate_covariance(np.diag(variances))
        )
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            covariance_solver(law, method="cholesky")


def test_cholesky_vector_validation_diagonal_dispatch_and_lapack_failures() -> None:
    diagonal = validate_covariance(np.array([2.0, 4.0]))
    dense = validate_covariance(np.diag([2.0, 4.0]))
    np.testing.assert_array_equal(
        covariance_solver(diagonal, method="cholesky")(np.array([2.0, 8.0])), [1.0, 2.0]
    )
    operation = covariance_solver(dense, method="cholesky")
    for bad in (np.ones(3), np.ones((2, 1)), np.array([np.nan, 1.0])):
        with pytest.raises(ValueError, match="vector"):
            operation(cast(NDArray[np.float64], bad))
    with pytest.raises(ValueError, match="method"):
        covariance_solver(dense, method=cast(Literal["eigh", "cholesky"], "bad"))
    with patch.object(covariance_module, "cho_factor", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="factorisation failed"):
            covariance_solver(dense, method="cholesky")
    with patch.object(np.linalg, "eigvalsh", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="eigenvalue check"):
            covariance_solver(dense, method="cholesky")
    with patch.object(covariance_module, "_minimum_eigenvalue", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="eigenvalue check"):
            validate_covariance(np.eye(2))
