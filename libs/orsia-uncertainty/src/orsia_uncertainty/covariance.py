"""Prepare reusable covariance factors and transport uncertainty in caller order.

Arrays use float64. Coordinate order and unit consistency belong to the caller.
Preparation owns immutable snapshots; operations allocate independent writable results.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, eigvalsh, solve_triangular


@dataclass(frozen=True, eq=False)
class CovarianceFactor:
    """Scaled square root and spectrum of a covariance in supplied coordinate order.

    Construct with factor_covariance. All buffers have independent immutable storage.
    The normalized root R satisfies C = scale * R R.T, after clipping tolerated
    negative eigenvalues for spectral sampling only. Inverse operations require
    the original minimum/maximum eigenvalue ratio strictly greater than 1e-12.

    Attributes:
        method: Diagonal, lower Cholesky or spectral square-root representation.
        scale: Largest absolute covariance entry, or one for a zero covariance.
        eigenvalues: Normalized spectrum (size,), retaining tolerated negatives.
        root: Normalized diagonal scales (size,) or square root (size,size).
    """

    method: Literal["diagonal", "cholesky", "eigh"]
    scale: float
    eigenvalues: NDArray[np.float64]
    root: NDArray[np.float64]

    @property
    def size(self) -> int:
        """Return the number of covariance coordinates."""
        return self.eigenvalues.size


def factor_covariance(
    covariance: NDArray[np.float64], *, method: Literal["eigh", "cholesky"]
) -> CovarianceFactor:
    """Prepare one independent covariance snapshot for whitening, solves and draws.

    Normalize before symmetry and spectral checks to support extreme finite scales.
    Dense PSD tolerance is 1e-12 times the largest absolute entry. Spectral roots
    clip only tolerated negative eigenvalues to zero when drawing; the retained
    spectrum keeps their sign. Cholesky requires positive definite covariance.
    Diagonal variances remain nonnegative and use linear storage for either method.
    No preparation consumes RNG or introduces regularization or a pseudoinverse.

    Args:
        covariance: Finite float64 variances (n,) or symmetric covariance (n,n).
        method: Dense square-root backend, 'eigh' or 'cholesky'.

    Returns:
        Factor with immutable owned buffers and one spectral validation per snapshot.

    Raises:
        ValueError: Shape, dtype, finiteness, symmetry, PSD or factorization is invalid.
    """
    if method not in ("eigh", "cholesky"):
        raise ValueError("factor method must be 'eigh' or 'cholesky'")
    matrix, scale, snapshot = _normalized_snapshot(covariance)
    if matrix.ndim == 1:
        root = np.sqrt(snapshot) / np.sqrt(scale)
        return CovarianceFactor("diagonal", scale, _immutable(matrix), _immutable(root))
    try:
        if method == "eigh":
            eigenvalues, basis = np.linalg.eigh(matrix)
        else:
            eigenvalues = np.linalg.eigvalsh(matrix)
            basis = None
        if eigenvalues[0] < -1e-12:
            raise ValueError("covariance matrix must be positive semidefinite")
        if basis is None:
            if eigenvalues[0] <= 0:
                raise ValueError("Cholesky requires positive definite covariance; use eigh for PSD")
            root = cholesky(matrix, lower=True, check_finite=False)
        else:
            root = basis * np.sqrt(np.maximum(eigenvalues, 0))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance factorisation failed") from exc
    return CovarianceFactor(method, scale, _immutable(eigenvalues), _immutable(root))


def whiten(
    factor: CovarianceFactor, values: NDArray[np.float64], *, transpose: bool = False
) -> NDArray[np.float64]:
    """Apply R^-1 / sqrt(scale), or its transpose, to column right-hand sides.

    Spectral whitening components follow eigenvector order; Cholesky components
    follow triangular elimination. Both define the same covariance quadratic form.
    Buffers are borrowed and unchanged; calls share no writable workspace.

    Args:
        factor: Prepared covariance factor with min/max eigenvalue ratio >1e-12.
        values: Finite float64 (size,) or (size,columns) array.
        transpose: Apply the transpose whitening operator for gain propagation.

    Returns:
        Fresh writable whitened vector or matrix.

    Raises:
        ValueError: Values have invalid support, or covariance is ill-conditioned.
    """
    _require_conditioned(factor.eigenvalues, "whiten")
    _validate_whitening_input(values, factor.size)
    scaled = values / np.sqrt(factor.scale)
    if factor.method == "diagonal":
        return scaled / (factor.root if values.ndim == 1 else factor.root[:, None])
    if factor.method == "cholesky":
        return solve_triangular(
            factor.root, scaled, lower=True, trans="T" if transpose else "N", check_finite=False
        )
    inverse_root = factor.root / factor.eigenvalues
    return inverse_root @ scaled if transpose else inverse_root.T @ scaled


def solve_covariance(factor: CovarianceFactor, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve C x = values using an existing factor without spectral preparation.

    Args:
        factor: Prepared covariance with min/max eigenvalue ratio >1e-12.
        values: Finite float64 vector (size,) or column right-hand sides (size,k).

    Returns:
        Independent writable solution in the supplied coordinate order.

    Raises:
        ValueError: Right-hand sides are invalid or the covariance is ill-conditioned.
    """
    return whiten(factor, whiten(factor, values), transpose=True)


def sample_gaussian(
    factor: CovarianceFactor, rng: np.random.Generator, count: int
) -> NDArray[np.float64]:
    """Draw centred perturbations from a prepared covariance with explicit RNG.

    Singular and zero laws use diagonal or spectral roots. Even a zero law consumes
    count*size standard normals in row order. Roots may change seeded components
    across backends; each backend preserves its own deterministic draw convention.

    Args:
        factor: Prepared covariance factor, including supported singular PSD laws.
        rng: Caller-owned generator consumed for ordered standard normal draws.
        count: Positive number of realizations.

    Returns:
        Fresh writable float64 perturbations (count,size), realizations in rows.

    Raises:
        ValueError: Count is not positive.
    """
    if count < 1:
        raise ValueError("sample count must be positive")
    normal = rng.standard_normal((count, factor.size))
    values = normal * factor.root if factor.method == "diagonal" else normal @ factor.root.T
    return values * np.sqrt(factor.scale)


def covariance_product(
    covariance: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Multiply a vector by dense or diagonal covariance in caller order.

    Args:
        covariance: Float64 (n,) variances or (n,n) symmetric PSD matrix.
        vector: Finite float64 (n,) input in matching coordinate order.

    Returns:
        Covariance-weighted (n,) vector in squared units times vector units.

    Raises:
        ValueError: Covariance or vector fails numerical validation.
    """
    covariance = validate_covariance(covariance)
    _validate_vector(vector, covariance.shape[0])
    return covariance * vector if covariance.ndim == 1 else covariance @ vector


def propagate_covariance(
    covariance: NDArray[np.float64], jacobian: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return dense ``J C J.T`` with J shaped (target, source).

    Covariance is (source,) variances or (source,source). Singular PSD inputs
    are supported. No unit conversion, reordering or diagonalisation occurs.

    Args:
        covariance: Float64 (source,) variances or (source,source) PSD matrix.
        jacobian: Finite float64 (target,source) coordinate derivative.

    Returns:
        Dense (target,target) output covariance in caller-defined target order.

    Raises:
        ValueError: Input dimensions or numerical covariance properties are invalid.
    """
    covariance = validate_covariance(covariance)
    _validate_jacobian(jacobian, covariance.shape[0])
    matrix = (
        (jacobian * covariance) @ jacobian.T
        if covariance.ndim == 1
        else jacobian @ covariance @ jacobian.T
    )
    return validate_covariance(matrix)


def propagate_vector(
    vector: NDArray[np.float64], jacobian: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Transport a signed offset to target coordinates through a Jacobian.

    Args:
        vector: Finite float64 signed offsets of shape (source,).
        jacobian: Finite float64 derivatives of shape (target,source).

    Returns:
        Signed float64 target offsets of shape (target,).

    Raises:
        ValueError: Offset or Jacobian shape or finite-value validation fails.
    """
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be nonempty and one-dimensional")
    _validate_vector(vector, vector.size)
    _validate_jacobian(jacobian, vector.size)
    result = jacobian @ vector
    _validate_vector(result, jacobian.shape[0])
    return result


def validate_covariance(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a numerically validated, independent writable covariance snapshot.

    Accept float64 nonnegative variances (n,) or finite symmetric PSD (n,n), n>=1.
    Dense symmetry and PSD tolerance is 1e-12 times the largest absolute entry;
    accepted dense matrices are symmetrised. No coordinate or unit checks occur.

    Args:
        covariance: Float64 (n,) variances or finite symmetric PSD (n,n) matrix.

    Returns:
        Independent writable covariance, symmetrised if dense.

    Raises:
        ValueError: Shape, dtype, finiteness, symmetry or PSD validation fails.
    """
    matrix, _, snapshot = _normalized_snapshot(covariance)
    if matrix.ndim == 2:
        try:
            minimum = _minimum_eigenvalue(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance matrix eigenvalue check failed") from exc
        if minimum < -1e-12:
            raise ValueError("covariance matrix must be positive semidefinite")
    return snapshot if snapshot.ndim == 1 else snapshot * 0.5 + snapshot.T * 0.5


def _normalized_snapshot(
    covariance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    snapshot = np.copy(covariance)
    if (
        snapshot.dtype != np.float64
        or snapshot.ndim not in (1, 2)
        or snapshot.shape[0] == 0
        or (snapshot.ndim == 2 and snapshot.shape[0] != snapshot.shape[1])
    ):
        raise ValueError("covariance must be float64 variances (n,) or a square matrix (n,n)")
    if not np.all(np.isfinite(snapshot)):
        raise ValueError("covariance must be finite")
    if snapshot.ndim == 1 and np.any(snapshot < 0):
        raise ValueError("variances must be nonnegative")
    scale = float(np.max(np.abs(snapshot))) or 1.0
    normalized = snapshot / scale
    if snapshot.ndim == 2:
        if not np.all(np.abs(normalized - normalized.T) <= 1e-12):
            raise ValueError("covariance matrix must be symmetric")
        normalized = normalized * 0.5 + normalized.T * 0.5
    return normalized, scale, snapshot


def _immutable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


def _require_conditioned(eigenvalues: NDArray[np.float64], operation: str) -> None:
    if np.min(eigenvalues) <= 1e-12 * np.max(eigenvalues):
        raise ValueError(f"cannot {operation} a singular or ill-conditioned covariance")


def _validate_jacobian(jacobian: NDArray[np.float64], source_size: int) -> None:
    if jacobian.ndim != 2 or jacobian.shape[0] == 0 or jacobian.shape[1] != source_size:
        raise ValueError("jacobian shape must be (target, source)")
    if jacobian.dtype != np.float64 or not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian must be finite float64")


def _validate_whitening_input(values: NDArray[np.float64], size: int) -> None:
    if values.ndim not in (1, 2) or values.shape[0] != size:
        raise ValueError("whitening input must have shape (size,) or (size, columns)")
    if values.dtype != np.float64 or not np.all(np.isfinite(values)):
        raise ValueError("whitening input must be finite float64")


def _minimum_eigenvalue(matrix: NDArray[np.float64]) -> float:
    """Find the smallest eigenvalue of a finite symmetric matrix.

    Returns:
        Minimum eigenvalue as a Python float; the input remains unchanged.
    """
    if matrix.shape[0] < 128:
        return float(np.linalg.eigvalsh(matrix)[0])
    return float(eigvalsh(matrix, subset_by_index=[0, 0], check_finite=False)[0])


def _validate_vector(vector: NDArray[np.float64], size: int) -> None:
    if vector.shape != (size,):
        raise ValueError("vector must have the covariance vector length")
    if vector.dtype != np.float64 or not np.all(np.isfinite(vector)):
        raise ValueError("vector must be finite float64")
