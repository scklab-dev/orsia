"""Validate, solve and propagate covariance in caller-defined vector order.

Arrays are float64; callers own coordinate ordering and unit consistency.
Inputs are never modified. Prepared functions snapshot their numerical factors.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, cholesky, eigvalsh, solve_triangular


def whitener(
    covariance: NDArray[np.float64],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Prepare a reusable whitening transform for covariance-weighted residuals.

    The returned callable computes ``W b``, where ``C = L L.T`` and ``W = L^-1``.
    Covariance is a variance vector (n,) or dense matrix (n,n). The callable
    accepts finite float64 (n,) or (n,k) arrays and returns fresh writable arrays.
    Reject eigenvalue ratios <=1e-12, without regularisation or pseudoinverse.
    Factors are independent snapshots; calls share no writable work buffers.

    Args:
        covariance: Float64 (n,) variances or positive-definite (n,n) covariance.

    Returns:
        Callable whitening a (n,) vector or (n,k) matrix with owned factors.

    Raises:
        ValueError: Covariance shape, finiteness, PSD or conditioning fails validation.
    """
    covariance = validate_covariance(covariance)
    size = covariance.shape[0]
    if covariance.ndim == 1:
        _require_conditioned(covariance, "whiten")
        scales = np.sqrt(covariance)

        def whiten_diagonal(values: NDArray[np.float64]) -> NDArray[np.float64]:
            _validate_whitening_input(values, size)
            return values / (scales if values.ndim == 1 else scales[:, None])

        return whiten_diagonal
    try:
        _require_conditioned(np.linalg.eigvalsh(covariance), "whiten")
        factor = cholesky(covariance, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance whitening factorisation failed") from exc

    def whiten_dense(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _validate_whitening_input(values, size)
        return solve_triangular(factor, values, lower=True, check_finite=False)

    return whiten_dense


def covariance_solver(
    covariance: NDArray[np.float64],
    *,
    method: Literal["eigh", "cholesky"] = "eigh",
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Prepare a reusable covariance solver for finite float64 vectors (n,).

    The returned callable solves ``C x = b`` for each supplied vector.

    Variance vectors retain O(n) storage; dense arrays retain O(n**2) factors.
    The default eigensystem and explicit Cholesky backend both reject eigenvalue
    ratios <=1e-12. No regularisation, clipping or pseudoinverse is used.
    Preparation snapshots inputs; results are independent writable arrays.

    Args:
        covariance: Float64 (n,) variances or (n,n) covariance of a vector.
        method: Dense factorization choice; diagonal variances use direct division.

    Returns:
        Callable solving covariance times a finite (n,) vector for that vector.

    Raises:
        ValueError: The method or covariance is invalid or poorly conditioned.
    """
    if method not in ("eigh", "cholesky"):
        raise ValueError("solve method must be 'eigh' or 'cholesky'")
    covariance = validate_covariance(covariance)
    size = covariance.shape[0]
    if covariance.ndim == 1:
        _require_conditioned(covariance, "solve with")

        def solve_diagonal(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            _validate_vector(vector, size)
            return vector / covariance

        return solve_diagonal
    try:
        if method == "cholesky":
            _require_conditioned(np.linalg.eigvalsh(covariance), "solve with")
            factor, lower = cho_factor(covariance, lower=True, check_finite=False)
            factor = np.frombuffer(factor.tobytes(order="F"), dtype=np.float64).reshape(
                factor.shape, order="F"
            )

            def solve_cholesky(vector: NDArray[np.float64]) -> NDArray[np.float64]:
                _validate_vector(vector, size)
                return cho_solve((factor, lower), vector, check_finite=False)

            return solve_cholesky
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance factorisation failed") from exc
    _require_conditioned(eigenvalues, "solve with")

    def solve_dense(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        _validate_vector(vector, size)
        return eigenvectors @ ((eigenvectors.T @ vector) / eigenvalues)

    return solve_dense


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


def solve_covariance(
    covariance: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve ``C x = vector`` without regularisation or pseudoinverse.

    Dense one-off solves use NumPy LU after checking spectral conditioning;
    variance vectors use direct division. Output is fresh and writable.

    Args:
        covariance: Float64 (n,) variances or invertible (n,n) covariance.
        vector: Finite float64 (n,) right-hand side in matching coordinates.

    Returns:
        Fresh solution vector satisfying ``C x = vector``.

    Raises:
        ValueError: Covariance or vector is invalid, singular or ill-conditioned.
    """
    covariance = validate_covariance(covariance)
    _validate_vector(vector, covariance.shape[0])
    if covariance.ndim == 1:
        _require_conditioned(covariance, "solve with")
        return vector / covariance
    try:
        _require_conditioned(np.linalg.eigvalsh(covariance), "solve with")
        return np.linalg.solve(covariance, vector)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance solve failed") from exc


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
    if (
        covariance.dtype != np.float64
        or covariance.ndim not in (1, 2)
        or covariance.shape[0] == 0
        or (covariance.ndim == 2 and covariance.shape[0] != covariance.shape[1])
    ):
        raise ValueError("covariance must be float64 variances (n,) or a square matrix (n,n)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if covariance.ndim == 1:
        if np.any(covariance < 0):
            raise ValueError("variances must be nonnegative")
        return np.copy(covariance)
    scale = np.max(np.abs(covariance))
    tolerance = 1e-12 * scale
    if not np.all(np.abs(covariance - covariance.T) <= tolerance):
        raise ValueError("covariance matrix must be symmetric")
    matrix = covariance * 0.5 + covariance.T * 0.5
    try:
        minimum = _minimum_eigenvalue(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance matrix eigenvalue check failed") from exc
    if minimum < -tolerance:
        raise ValueError("covariance matrix must be positive semidefinite")
    return matrix


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
