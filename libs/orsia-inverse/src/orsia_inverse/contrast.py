"""Mean-normalized Helmert contrasts with derivatives and covariance transport."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty.covariance import validate_covariance

type FloatArray = NDArray[np.float64]


def helmert_basis(channel_count: int) -> FloatArray:
    """Construct an orthonormal basis for zero-mean channel contrasts.

    Args:
        channel_count: Number of channels, at least three for this method.

    Returns:
        Fresh writable float64 (channel, channel-1) basis. Column k-1 has
        its first k entries equal to 1/sqrt(k*(k+1)), followed by
        -k/sqrt(k*(k+1)) at entry k; subsequent entries are zero.

    Raises:
        ValueError: Fewer than three channels are supplied.
    """
    if channel_count < 3:
        raise ValueError("normalized contrasts require at least three channels")
    basis = np.zeros((channel_count, channel_count - 1))
    for k in range(1, channel_count):
        basis[:k, k - 1] = 1 / np.sqrt(k * (k + 1))
        basis[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return basis


def normalized_contrast(
    signal: FloatArray | Sequence[float],
    covariance: FloatArray | Sequence[Sequence[float]] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Remove a common signal amplitude and propagate its local measurement errors.

    Args:
        signal: Finite (channel,) measurements with positive mean and at least
            three channels. All channels use the same measurement unit.
        covariance: Optional symmetric positive-semidefinite (channel, channel)
            covariance in squared measurement units, in the signal's channel order.

    Returns:
        Dimensionless Helmert contrasts, their (channel-1, channel) derivative
        in inverse measurement units, and the dimensionless contrast covariance.
        Covariance is None when omitted. Outputs own independent writable float64
        arrays. Inputs are borrowed during the call and remain unchanged.
        The covariance is a first-order transport through mean normalization.

    Raises:
        ValueError: Signal shape, finiteness, mean, channel count or covariance
            shape, symmetry, finiteness or positive-semidefiniteness is invalid.
    """
    y = np.asarray(signal, float)
    if y.ndim != 1 or not np.isfinite(y).all() or y.mean() <= 0:
        raise ValueError("finite channel vector with positive mean required")
    h = helmert_basis(len(y))
    jacobian = h.T @ (np.eye(len(y)) / y.mean() - y[:, None] / (len(y) * y.mean() ** 2))
    cz = None
    if covariance is not None:
        c = np.asarray(covariance, float)
        if c.shape != (len(y), len(y)):
            raise ValueError("channel covariance axes differ from signal")
        c = validate_covariance(c)
        cz = jacobian @ c @ jacobian.T
        cz = (cz + cz.T) / 2
    return h.T @ (y / y.mean()), jacobian, cz
