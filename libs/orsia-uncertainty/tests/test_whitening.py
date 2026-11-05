"""Independent whitening identities, transpose support and extreme scales."""

from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_uncertainty import factor_covariance, solve_covariance, whiten


@pytest.mark.parametrize("method", ["eigh", "cholesky"])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_whitening_transpose_extreme_scale(
    method: Literal["eigh", "cholesky"], diagonal: bool, scale: float
) -> None:
    matrix = (
        np.diag([2.0, 3.0, 4.0])
        if diagonal
        else np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    )
    covariance = scale * (np.diag(matrix).copy() if diagonal else matrix)
    factor = factor_covariance(covariance, method=method)
    values = np.arange(12.0).reshape(3, 4) * np.sqrt(scale)
    saved = values.copy()
    result = whiten(factor, cast(NDArray[np.float64], values))
    np.testing.assert_allclose(
        result.T @ result,
        (values / np.sqrt(scale)).T @ np.linalg.solve(matrix, values / np.sqrt(scale)),
    )
    w = whiten(factor, np.eye(3) * np.sqrt(scale))
    np.testing.assert_allclose(w @ matrix @ w.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(
        whiten(factor, values, transpose=True), w.T @ (values / np.sqrt(scale)), atol=1e-14
    )
    np.testing.assert_allclose(
        solve_covariance(factor, values) * np.sqrt(scale),
        np.linalg.solve(matrix, values / np.sqrt(scale)),
        atol=1e-14,
    )
    for i in range(4):
        np.testing.assert_allclose(result[:, i], whiten(factor, values[:, i]))
    np.testing.assert_array_equal(values, saved)
    assert result.flags.writeable and not np.shares_memory(result, values)
    assert whiten(factor, np.empty((3, 0))).shape == (3, 0)


def test_whitening_rejects_invalid_values() -> None:
    factor = factor_covariance(np.eye(3), method="cholesky")
    for values in (
        np.array(1.0),
        np.ones(2),
        np.ones((3, 1, 1)),
        np.full((3, 2), np.inf),
        np.ones(3, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="whitening input"):
            whiten(factor, cast(NDArray[np.float64], values))
