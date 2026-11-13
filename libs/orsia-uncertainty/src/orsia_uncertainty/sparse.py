"""Multiply and validate sparse covariance terms with storage-bounded scratch."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from orsia_uncertainty.covariance import factor_covariance


def sparse_row_product(left: csr_matrix, right: csr_matrix) -> csr_matrix:
    """Multiply local registration rows without a source-cube-sized transpose.

    Args:
        left: Borrowed CSR rows, possibly already weighted by source variance.
        right: Borrowed CSR rows with the same original source-cell columns.

    Returns:
        Owned CSR left @ right.T, preserving signed coefficients and duplicate
        contributions. Scratch depends on stored entries in these local rows,
        rather than the declared number of source cells. Inputs stay unchanged.

    Raises:
        ValueError: Source-column dimensions differ.
    """
    if left.shape[1] != right.shape[1]:
        raise ValueError("registration rows must share source-cell columns")
    if not left.nnz or not right.nnz:
        return csr_matrix((left.shape[0], right.shape[0]))
    first = min(int(left.indices.min()), int(right.indices.min()))
    last = max(int(left.indices.max()), int(right.indices.max()))
    columns = last - first + 1
    if columns <= left.nnz + right.nnz:
        # Nearby geometric stencils need only an offset, avoiding a sort per pair.
        left_columns, right_columns = left.indices - first, right.indices - first
    else:
        sources, inverse = np.unique(
            np.concatenate((left.indices, right.indices)), return_inverse=True
        )
        columns = len(sources)
        left_columns, right_columns = inverse[: left.nnz], inverse[left.nnz :]
    compact_left = csr_matrix(
        (left.data, left_columns, left.indptr), shape=(left.shape[0], columns), copy=True
    )
    compact_right = csr_matrix(
        (right.data, right_columns, right.indptr), shape=(right.shape[0], columns), copy=True
    )
    return csr_matrix(compact_left @ compact_right.T)


def validate_sparse_covariance(covariance: csr_matrix) -> None:
    """Check sparse PSD covariance without allocating a dense scene matrix.

    Args:
        covariance: Square CSR covariance, borrowed unchanged.

    Raises:
        ValueError: Shape, finite support, symmetry or PSD validation fails,
            including failure of the sparse extreme-eigenvalue calculation.
    """
    n, m = covariance.shape
    if n != m or n == 0 or not np.isfinite(covariance.data).all():
        raise ValueError("finite square detector covariance required")
    scale = float(np.max(np.abs(covariance.data), initial=0))
    if scale == 0:
        return
    normalized = covariance / scale
    difference = normalized - normalized.T
    if np.max(np.abs(difference.data), initial=0) > 1e-12:
        raise ValueError("symmetric detector covariance required")
    diagonal = normalized.diagonal()
    if np.any(diagonal < -1e-12):
        raise ValueError("positive semidefinite detector covariance required")
    # Symmetric diagonally dominant matrices with nonnegative diagonal are PSD.
    radius = np.asarray(abs(normalized).sum(axis=1)).ravel() - np.abs(diagonal)
    if np.all(diagonal >= radius):
        return
    if n <= 64:
        factor_covariance(normalized.toarray(), method="eigh")
        return
    try:
        smallest = eigsh(
            (normalized + normalized.T) * 0.5,
            k=1,
            which="SA",
            return_eigenvectors=False,
            v0=np.random.default_rng(0).standard_normal(n),
            tol=1e-13,
        )[0]
    except ArpackNoConvergence as error:
        raise ValueError("detector covariance PSD check did not converge") from error
    if smallest < -1e-12:
        raise ValueError("positive semidefinite detector covariance required")
