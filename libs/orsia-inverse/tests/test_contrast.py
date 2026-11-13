"""Normalization invariants, analytical derivatives and covariance contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_inverse.contrast import helmert_basis, normalized_contrast


@pytest.mark.parametrize("count", [3, 4, 10])
def test_helmert_basis_and_normalization(count: int) -> None:
    h = helmert_basis(count)
    assert_allclose(h.T @ h, np.eye(count - 1), atol=3e-16)
    assert_allclose(h.sum(axis=0), 0, atol=3e-16)
    signal = np.linspace(-0.5, 3, count)
    covariance = np.eye(count) * 0.1 + np.ones((count, count)) * 0.03
    before = signal.copy(), covariance.copy()
    z, derivative, cov = normalized_contrast(signal, covariance)
    assert cov is not None
    assert_allclose(z, h.T @ (signal / signal.mean()), rtol=0, atol=0)
    assert_allclose(cov, derivative @ covariance @ derivative.T, atol=2e-16)
    assert_allclose(normalized_contrast(signal * 5)[0], z, atol=3e-16)
    assert_allclose(derivative @ signal, 0, atol=4e-16)
    step = 1e-5
    for index in range(count):
        delta = np.eye(count)[index] * step
        finite = (
            normalized_contrast(signal + delta)[0] - normalized_contrast(signal - delta)[0]
        ) / (2 * step)
        assert_allclose(derivative[:, index], finite, rtol=2e-9, atol=2e-11)
    for array in (z, derivative, cov):
        assert array.flags.owndata and array.flags.writeable
    assert_allclose(signal, before[0], rtol=0, atol=0)
    assert_allclose(covariance, before[1], rtol=0, atol=0)
    z[:] = 9
    assert not np.any(normalized_contrast(signal)[0] == 9)
    assert normalized_contrast([1.0, 2.0, 3.0])[2] is None


@pytest.mark.parametrize(
    "signal", [np.ones((1, 3)), np.array([1, np.nan, 2]), np.array([-1, 0, 1]), np.array([1, 2])]
)
def test_invalid_signal(signal: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        normalized_contrast(signal)


@pytest.mark.parametrize(
    "covariance",
    [
        np.ones((2, 2)),
        np.diag([1.0, -1, 1]),
        np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]]),
        np.full((3, 3), np.nan),
    ],
)
def test_invalid_covariance(covariance: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        normalized_contrast([1.0, 2.0, 3.0], covariance)


def test_singular_covariance_preserved() -> None:
    _, _, covariance = normalized_contrast([1.0, 2.0, 3.0], np.zeros((3, 3)))
    assert covariance is not None
    assert_allclose(covariance, np.zeros((2, 2)), atol=0)
