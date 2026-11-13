"""Estimate scalar fields with sparse regularization and checked nested PCG.

The caller selects coordinates, masks, covariance and prior strength. No sensor
representation or physical forward implementation enters this numerical method.
"""

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

type FloatArray = NDArray[np.float64]
type ArrayInput = NDArray[np.floating | np.integer] | Sequence[float] | Sequence[Sequence[float]]

type BoolArray = NDArray[np.bool_]


class CorrelatedFieldEstimate(TypedDict):
    """Scalar coordinate estimate and nested-solve diagnostics.

    Attributes:
        state: Owned (selected_pixel,) field coordinates.
        iterations: Outer PCG iteration count.
        converged: Whether outer relative residual met tolerance.
        relative_residual: Outer relative residual norm.
        data_cost: Half residual precision quadratic form.
        curvature_cost: Half scaled regularization quadratic form.
        inner_iterations: Sum of inner covariance-solve iterations.
        maximum_inner_iterations: Largest inner iteration count.
    """

    state: FloatArray
    iterations: int
    converged: bool
    relative_residual: float
    data_cost: float
    curvature_cost: float
    inner_iterations: int
    maximum_inner_iterations: int


def masked_second_difference_operator(mask: BoolArray) -> tuple[csr_matrix, NDArray[np.int64]]:
    """Return mask-contained second differences and valid C-order cell indices.

    Boolean mask has (row,column) axes. Each CSR row contains [1,-2,1] along
    one spatial axis, only when all three cells are valid. No bridge crosses holes.

    Args:
        mask: Nonempty Boolean (row,column) selected field support.

    Returns:
        Owned CSR second differences and int64 selected C-order spatial indices.
        Every [1,-2,1] triplet lies inside the mask; holes are never bridged.

    Raises:
        ValueError: Mask shape, dtype or nonempty support fails.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool or not mask.any():
        raise ValueError("nonempty Boolean spatial support required")
    indices = np.flatnonzero(mask.ravel())
    compact = np.full(mask.shape, -1, int)
    compact.ravel()[indices] = np.arange(len(indices))
    triplets = []
    for r, c in np.argwhere(mask):
        if 0 < r < mask.shape[0] - 1 and mask[r - 1, c] and mask[r + 1, c]:
            triplets.append([compact[r - 1, c], compact[r, c], compact[r + 1, c]])
        if 0 < c < mask.shape[1] - 1 and mask[r, c - 1] and mask[r, c + 1]:
            triplets.append([compact[r, c - 1], compact[r, c], compact[r, c + 1]])
    rows = np.repeat(np.arange(len(triplets)), 3)
    columns = np.asarray(triplets, int).ravel()
    return csr_matrix(
        (np.tile([1.0, -2.0, 1.0], len(triplets)), (rows, columns)),
        shape=(len(triplets), len(indices)),
    ), indices


def solve_correlated_field(
    observation: FloatArray,
    apply_covariance: Callable[[FloatArray], FloatArray],
    diagonal: FloatArray,
    d: csr_matrix,
    *,
    curvature_rms: float,
    relative_tolerance: float = 1e-10,
    maximum_iterations: int = 1000,
) -> CorrelatedFieldEstimate:
    """Estimate an explicit scalar coordinate field with a supplied covariance action.

    Args:
        observation: Finite selected scalar coordinates, (pixel,).
        apply_covariance: Pure covariance action in the same selected coordinate order.
        diagonal: Positive covariance marginal variances, (pixel,).
        d: Sparse regularization operator, (constraint,pixel).
        curvature_rms: Positive RMS scale of d applied to the coordinate field.
        relative_tolerance: Outer relative residual tolerance.
        maximum_iterations: Maximum outer PCG iterations.

    Returns:
        Owned selected-coordinate estimate, costs and nested convergence diagnostics.
        Outer exhaustion remains explicit in converged=False.

    Raises:
        ValueError: Axes/domains fail or an inner covariance solve does not converge.
    """
    if (
        observation.ndim != 1
        or diagonal.shape != observation.shape
        or d.shape[1] != len(observation)
        or not np.isfinite(observation).all()
        or not np.isfinite(diagonal).all()
        or np.any(diagonal <= 0)
        or curvature_rms <= 0
    ):
        raise ValueError("aligned finite scalar coordinates and positive variances required")
    inner_iterations = []

    def precision(vector: FloatArray) -> FloatArray:
        solution, iterations, converged, residual = _pcg(
            apply_covariance, vector, diagonal, vector / diagonal, 1e-12, max(2 * len(vector), 32)
        )
        inner_iterations.append(iterations)
        if not converged:
            raise ValueError(f"posterior covariance solve did not converge: {residual}")
        return solution

    prior = (d.T @ d) / curvature_rms**2
    rhs = precision(observation)

    def apply(v: FloatArray) -> FloatArray:
        return precision(v) + prior @ v

    solution, iterations, converged, residual = _pcg(
        apply,
        rhs,
        1 / diagonal + prior.diagonal(),
        observation,
        relative_tolerance,
        maximum_iterations,
    )

    data_residual = solution - observation
    return dict(
        state=solution,
        iterations=iterations,
        converged=converged,
        relative_residual=residual,
        data_cost=float(0.5 * data_residual @ precision(data_residual)),
        curvature_cost=float(0.5 * np.sum((d @ solution) ** 2) / curvature_rms**2),
        inner_iterations=sum(inner_iterations),
        maximum_inner_iterations=max(inner_iterations),
    )


def _pcg(
    apply: Callable[[FloatArray], FloatArray],
    rhs: FloatArray,
    diagonal: FloatArray,
    initial: FloatArray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[FloatArray, int, bool, float]:
    solution = initial.copy()
    residual = rhs - apply(solution)
    scale = max(np.linalg.norm(rhs), np.finfo(float).eps)
    relative = np.linalg.norm(residual) / scale
    if relative <= tolerance:
        return solution, 0, True, float(relative)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    product = residual @ preconditioned
    for iteration in range(1, maximum_iterations + 1):
        action = apply(direction)
        denominator = direction @ action
        if not np.isfinite(denominator) or denominator <= 0:
            return solution, iteration - 1, False, float(relative)
        step = product / denominator
        solution += step * direction
        residual -= step * action
        relative = np.linalg.norm(residual) / scale
        if relative <= tolerance:
            return solution, iteration, True, float(relative)
        preconditioned = residual / diagonal
        updated = residual @ preconditioned
        if not np.isfinite(updated) or updated <= 0:
            return solution, iteration, False, float(relative)
        direction = preconditioned + updated / product * direction
        product = updated
    return solution, maximum_iterations, False, float(relative)
