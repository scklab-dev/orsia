"""Construct quadratic penalties in the caller's fitted state coordinates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import CovarianceFactor, factor_covariance, whiten


@dataclass(frozen=True, eq=False)
class QuadraticPrior:
    """Retain the penalty ``0.5 * ||precision_root @ (q - mean)||**2``.

    A rectangular or rank-deficient root permits partial regularization. Such a
    penalty need not define a proper probability distribution. Arrays are owned
    immutable float64 snapshots. State coordinates and their units are caller-defined.

    Attributes:
        mean: Finite reference state, shape (state,).
        precision_root: Finite root, shape (penalty,state), including zero rows.
            Its product with a state difference is dimensionless.
    """

    mean: NDArray[np.float64]
    precision_root: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate dimensions and snapshot the penalty coefficients.

        Raises:
            ValueError: State or root dimensions, dtype or finite values are invalid.
        """
        if (
            self.mean.ndim != 1
            or self.mean.size == 0
            or self.precision_root.ndim != 2
            or self.precision_root.shape[1] != self.mean.size
            or self.mean.dtype != np.float64
            or self.precision_root.dtype != np.float64
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.precision_root).all()
        ):
            raise ValueError("prior mean and precision_root must be aligned finite float64 arrays")
        for name in ("mean", "precision_root"):
            values = getattr(self, name)
            snapshot = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, snapshot)


def prior_from_covariance(mean: NDArray[np.float64], factor: CovarianceFactor) -> QuadraticPrior:
    """Construct a quadratic prior from an already prepared covariance factor.

    Args:
        mean: Finite float64 reference state, shape (state,).
        factor: Matching covariance factor admitting inverse operations.

    Returns:
        Immutable prior with a whitening matrix as its precision root.

    Raises:
        ValueError: Dimensions disagree or the covariance is ill-conditioned.
    """
    if mean.ndim != 1 or factor.size != mean.size:
        raise ValueError("prior covariance size must match mean")
    return QuadraticPrior(mean, whiten(factor, np.eye(mean.size)))


def prior_from_precision(
    mean: NDArray[np.float64], precision: NDArray[np.float64]
) -> QuadraticPrior:
    """Construct a quadratic prior from symmetric positive-semidefinite precision.

    Spectral factorization uses Uncertainty's scale-relative symmetry and PSD
    tolerance (1e-12). Tolerated negative roundoff eigenvalues become zero.
    Singular precision is supported without introducing inverse regularization.

    Args:
        mean: Finite float64 reference state, shape (state,).
        precision: Float64 diagonal precision (state,) or matrix (state,state).

    Returns:
        Immutable prior with the selected spectral precision root.

    Raises:
        ValueError: Dimensions, symmetry, finite values or PSD requirements fail.
    """
    factor = factor_covariance(precision, method="eigh")
    if mean.ndim != 1 or factor.size != mean.size:
        raise ValueError("prior precision size must match mean")
    root = np.diag(factor.root) if factor.root.ndim == 1 else factor.root.T
    return QuadraticPrior(mean, np.sqrt(factor.scale) * root)
