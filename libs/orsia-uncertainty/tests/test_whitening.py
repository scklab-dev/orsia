"""Independent square-root weighting and ownership contracts."""

from concurrent.futures import ThreadPoolExecutor
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_uncertainty import (
    validate_covariance,
    whitener,
)


@pytest.mark.parametrize("diagonal", [False, True])
def test_whitening_vectors_matrices_ownership_and_reuse(diagonal: bool) -> None:
    matrix = (
        np.diag([2.0, 5.0, 3.0])
        if diagonal
        else np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    )
    covariance = (
        validate_covariance(np.diag(matrix).copy()) if diagonal else validate_covariance(matrix)
    )
    with patch("orsia_uncertainty.covariance.cholesky", wraps=np.linalg.cholesky) as factor:
        # numpy's test wrapper has a different keyword interface.
        factor.side_effect = lambda a, **kwargs: np.linalg.cholesky(a)
        whiten = whitener(covariance)
        vectors = np.arange(12.0, dtype=np.float64).reshape(3, 4)
        snapshot = vectors.copy()
        result = whiten(vectors)
        np.testing.assert_allclose(result.T @ result, vectors.T @ np.linalg.solve(matrix, vectors))
        np.testing.assert_allclose(
            whiten(np.eye(3)) @ matrix @ whiten(np.eye(3)).T, np.eye(3), atol=1e-14
        )
        for i in range(4):
            np.testing.assert_allclose(result[:, i], whiten(vectors[:, i]))
        assert factor.call_count == (0 if diagonal else 1)
    np.testing.assert_array_equal(vectors, snapshot)
    assert result.flags.writeable and not np.shares_memory(result, vectors)
    result.fill(0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(whiten, [vectors, vectors]))
    np.testing.assert_array_equal(parallel[0], parallel[1])
    assert not np.shares_memory(parallel[0], parallel[1])
    assert whiten(np.empty((3, 0))).shape == (3, 0)


@pytest.mark.parametrize("diagonal", [False, True])
def test_whitening_rejects_bad_covariances_and_inputs(diagonal: bool) -> None:
    for values in (np.zeros(3), np.array([1.0, 0.0, 1.0]), np.array([1.0, 1e-12, 1.0])):
        covariance = (
            validate_covariance(values) if diagonal else validate_covariance(np.diag(values))
        )
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            whitener(covariance)
    covariance = validate_covariance(np.ones(3)) if diagonal else validate_covariance(np.eye(3))
    whiten = whitener(covariance)
    for invalid in (
        np.array(1.0),
        np.ones(2),
        np.ones((3, 1, 1)),
        np.array([1.0, np.nan, 1.0]),
        np.full((3, 2), np.inf),
    ):
        with pytest.raises(ValueError, match="whitening input"):
            whiten(cast(NDArray[np.float64], invalid))


def test_factorisation_failure_is_explicit() -> None:
    covariance = validate_covariance(np.eye(3))
    with patch("orsia_uncertainty.covariance.cholesky", side_effect=np.linalg.LinAlgError):
        with pytest.raises(ValueError, match="factorisation failed"):
            whitener(covariance)
