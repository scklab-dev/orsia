"""Keep source errors and their linear projections without an output covariance allocation."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, issparse

from orsia_uncertainty.covariance import factor_covariance, solve_covariance, validate_covariance

type ProjectionMatrix = NDArray[np.float64] | csr_array


@dataclass(frozen=True, eq=False)
class ProjectedCovariance:
    """Covariance from explicitly uncorrelated source blocks: sum(M Q M.T).

    Put correlated sources together in one Q, including their cross terms.
    Singular source blocks are supported. Construction snapshots all arrays;
    snapshots remain writable, but mutation invalidates the representation: rebuild
    it before further operations. Construction rejects nonfinite or negative
    computed marginal variances, including tiny negatives tolerated in source Q.
    Sources are preserved without clipping. This marginal check does not certify
    the full output spectrum. Sparse support is CSR projections only; Q is diagonal
    or dense. No general solve is implied.

    Attributes:
        projections: Nonempty tuple of finite (observations,sources) float64 dense
            or CSR maps, in one common output order and output/source units.
        source_covariances: Matching float64 variance vectors or dense PSD source
            matrices in squared source units. Distinct blocks have zero covariance.
    """

    projections: tuple[ProjectionMatrix, ...]
    source_covariances: tuple[NDArray[np.float64], ...]

    def __post_init__(self) -> None:
        if not self.projections or len(self.projections) != len(self.source_covariances):
            raise ValueError("declare matching nonempty projection and source tuples")
        projections = tuple(_matrix_snapshot(matrix) for matrix in self.projections)
        covariances = tuple(validate_covariance(q) for q in self.source_covariances)
        size = cast(tuple[int, int], projections[0].shape)[0]
        if any(
            m.shape != (size, q.shape[0]) for m, q in zip(projections, covariances, strict=True)
        ):
            raise ValueError("source and projection dimensions must agree")
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "source_covariances", covariances)
        projected_variances(self)


def projected_covariance_product(
    covariance: ProjectedCovariance, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply covariance to a vector or RHS columns through its source projections.

    Relies on construction-time marginal validation and unchanged source buffers.
    The operation does not repeat that check or certify the output spectrum.

    Args:
        covariance: Stable source blocks and maps in output coordinate order.
        values: Finite float64 (observations,) or (observations,columns) operand.

    Returns:
        Owned array of matching shape in covariance units times operand units.

    Raises:
        ValueError: Operand dimensions, dtype or finite support are invalid.
    """
    _operand(values, cast(tuple[int, int], covariance.projections[0].shape)[0])
    result = np.zeros_like(values)
    for m, q in zip(covariance.projections, covariance.source_covariances, strict=True):
        source = m.T @ values
        weighted = (
            (q * source if source.ndim == 1 else q[:, None] * source) if q.ndim == 1 else q @ source
        )
        result += m @ weighted
    if not np.isfinite(result).all():
        raise ValueError("covariance product is not finite")
    return result


def transport_projected_covariance(
    covariance: ProjectedCovariance, transform: ProjectionMatrix
) -> ProjectedCovariance:
    """Transport every source through the same supplied linear map.

    Args:
        covariance: Stable source representation.
        transform: Finite dense or CSR (target,observations) map in target/input units.

    Returns:
        Independent source snapshots with maps T M, without output covariance.

    Raises:
        ValueError: Transform support or dimensions are invalid, or transformed
            marginal variances are negative or nonfinite.
    """
    transform = _matrix_snapshot(transform)
    if (
        cast(tuple[int, int], transform.shape)[1]
        != cast(tuple[int, int], covariance.projections[0].shape)[0]
    ):
        raise ValueError("transform must have one column per observation")
    return ProjectedCovariance(
        tuple(transform @ m for m in covariance.projections), covariance.source_covariances
    )


def select_projected_covariance(
    covariance: ProjectedCovariance, indices: NDArray[np.int64]
) -> ProjectedCovariance:
    """Select or permute output coordinates while preserving all source dependencies.

    Args:
        covariance: Stable source representation.
        indices: Nonempty int64 indices in range; repeated indices are allowed.

    Returns:
        Independent snapshots in exactly the supplied order.

    Raises:
        ValueError: Indices are empty, nonintegral, non-vector or out of range,
            or selected marginal variances are negative or nonfinite.
    """
    size = cast(tuple[int, int], covariance.projections[0].shape)[0]
    if (
        indices.dtype != np.int64
        or indices.ndim != 1
        or not indices.size
        or np.any((indices < 0) | (indices >= size))
    ):
        raise ValueError("selection requires nonempty int64 indices in range")
    return ProjectedCovariance(
        tuple(cast(ProjectionMatrix, m[indices, :]) for m in covariance.projections),
        covariance.source_covariances,
    )


def projected_variances(covariance: ProjectedCovariance) -> NDArray[np.float64]:
    """Compute marginal variances without an observation-square array.

    Every computed negative variance raises, including source-tolerance and
    roundoff negatives. No clipping or regularization changes the source model.

    Args:
        covariance: Stable dense/CSR source maps and diagonal/dense source covariance.

    Returns:
        Owned (observations,) nonnegative float64 marginal variances.

    Raises:
        ValueError: Computed variances are negative or overflow finite float64 support.
    """
    result = np.zeros(cast(tuple[int, int], covariance.projections[0].shape)[0])
    for m, q in zip(covariance.projections, covariance.source_covariances, strict=True):
        if q.ndim == 1:
            if isinstance(m, csr_array):
                weighted = m.copy()
                weighted.data *= q[m.indices]
                weighted.data *= m.data
                result += np.asarray(weighted.sum(axis=1)).ravel()
            else:
                result += np.einsum("ij,j,ij->i", m, q, m)
        elif isinstance(m, csr_array):
            for i in range(cast(tuple[int, int], m.shape)[0]):
                start, end = m.indptr[i : i + 2]
                columns, row = m.indices[start:end], m.data[start:end]
                result[i] += row @ q[np.ix_(columns, columns)] @ row
        else:
            # Bound the dense source-space product independently of output size.
            rows = max(1, 262_144 // q.shape[0])
            for start in range(0, len(m), rows):
                block = m[start : start + rows]
                result[start : start + rows] += np.einsum("ij,ij->i", block @ q, block)
    if not np.isfinite(result).all():
        raise ValueError("marginal variance is not finite")
    if np.any(result < 0):
        raise ValueError("projected marginal variance is negative; resolve the source model")
    return result


def materialize_covariance(
    covariance: ProjectedCovariance, *, max_bytes: int
) -> NDArray[np.float64]:
    """Explicitly allocate a dense output covariance within an output-byte budget.

    Uses one output matrix and vector workspaces; source storage and validation
    workspaces are outside this output budget. No identity matrix is allocated.
    Checks the computed diagonal for negativity without a full output PSD check.

    Args:
        covariance: Stable source representation, including singular support.
        max_bytes: Maximum bytes for the float64 output matrix, checked before allocation.

    Returns:
        Owned symmetric (observations,observations) covariance.

    Raises:
        MemoryError: Output matrix exceeds the declared nonnegative budget.
        ValueError: Numerical products are nonfinite or a computed diagonal entry is negative.
    """
    size = cast(tuple[int, int], covariance.projections[0].shape)[0]
    if max_bytes < size * size * 8:
        raise MemoryError("dense covariance exceeds max_bytes")
    result = np.empty((size, size))
    unit = np.zeros(size)
    for j in range(size):
        unit[j] = 1
        result[:, j] = projected_covariance_product(covariance, unit)
        unit[j] = 0
    # Column products can differ by rounding; avoid a second square workspace.
    for j in range(size):
        row = result[j, :j] * 0.5 + result[:j, j] * 0.5
        result[j, :j] = result[:j, j] = row
    if np.any(result.diagonal() < 0):
        raise ValueError("materialized marginal variance is negative; resolve the source model")
    return result


def solve_diagonal_low_rank(
    variances: NDArray[np.float64], modes: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve (diag(variances) + modes modes.T) x = values without output densification.

    Modes include their source standard deviations or square-root covariance.
    The Woodbury solve uses the existing conditioned dense factorization for the
    small mode-space matrix. No jitter or singular pseudo-inverse is added.

    Args:
        variances: Strictly positive finite float64 (observations,) diagonal.
        modes: Finite float64 (observations,rank) weighted error modes; rank may be zero.
        values: Finite float64 vector or RHS columns with matching first dimension.

    Returns:
        Owned solution with the same shape as values.

    Raises:
        ValueError: Inputs are invalid, diagonal support is singular, or the
            small mode-space solve is ill-conditioned or overflows.
    """
    diagonal = validate_covariance(variances)
    if diagonal.ndim != 1 or np.any(diagonal <= 0):
        raise ValueError("solve requires strictly positive diagonal variances")
    _operand(values, diagonal.size)
    if (
        modes.dtype != np.float64
        or modes.ndim != 2
        or modes.shape[0] != diagonal.size
        or not np.isfinite(modes).all()
    ):
        raise ValueError("modes must be finite float64 (observations,rank)")
    scaled = values / (diagonal if values.ndim == 1 else diagonal[:, None])
    weighted = modes / diagonal[:, None]
    if modes.shape[1]:
        small = np.eye(modes.shape[1]) + modes.T @ weighted
        factor = factor_covariance(small, method="cholesky")
        scaled = scaled - weighted @ solve_covariance(factor, modes.T @ scaled)
    if not np.isfinite(scaled).all():
        raise ValueError("solution is not finite")
    recovered = (diagonal if values.ndim == 1 else diagonal[:, None]) * scaled
    recovered += modes @ (modes.T @ scaled)
    residual = np.max(np.abs(recovered - values), initial=0)
    if not np.isfinite(residual) or residual > 1e-9 * np.max(np.abs(values), initial=0):
        raise ValueError("low-rank solve lost numerical accuracy")
    return scaled


def _matrix_snapshot(matrix: ProjectionMatrix) -> ProjectionMatrix:
    if issparse(matrix) and not isinstance(matrix, csr_array):
        raise ValueError("only CSR sparse arrays are supported")
    if (
        matrix.dtype != np.float64
        or matrix.ndim != 2
        or min(cast(tuple[int, int], matrix.shape)) < 1
    ):
        raise ValueError("projection must be nonempty two-dimensional float64")
    result = matrix.copy()
    if isinstance(result, csr_array):
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        finite = np.isfinite(result.data).all()
    else:
        finite = np.isfinite(result).all()
    if not finite:
        raise ValueError("projection must be finite")
    return result


def _operand(values: NDArray[np.float64], size: int) -> None:
    if (
        values.dtype != np.float64
        or values.ndim not in (1, 2)
        or values.shape[0] != size
        or not np.isfinite(values).all()
    ):
        raise ValueError("operand must be finite float64 with matching first dimension")
