"""Adapt general linear maps and materialize bounded reference matrices."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

type Product = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def linear_operator(shape: tuple[int, int], jvp: Product, vjp: Product) -> LinearOperator:
    """Wrap a linear map and its transpose in a checked float64 operator.

    Callbacks represent one fixed linear map, with real Euclidean transpose
    convention ``dot(w, jvp(v)) == dot(vjp(w), v)``. Units and coordinates belong
    to the caller. The caller retains explicit immutable state or keeps borrowed
    state stable for the operator lifetime. No tracing or numerical derivatives
    are inferred. Calls own their buffers; concurrent use requires callbacks with
    call-local accumulation and stable retained inputs.

    Args:
        shape: Positive (output_size, input_size) dimensions in C flattening order.
        jvp: Forward product callback; accepts (input_size, k) and returns
            (output_size, k), including k=0.
        vjp: Transpose product callback; accepts (output_size, k) and returns
            (input_size, k), including k=0.

    Returns:
        SciPy operator supporting vectors and direction-last blocks. Callback
        inputs are owned read-only copies; outputs are fresh writable copies.

    Raises:
        ValueError: Dimensions, product shapes, dtype or finite support are invalid.
    """
    m, n = shape
    if min(m, n) < 1:
        raise ValueError("operator dimensions must be positive")

    def apply(values: NDArray[np.float64], reverse: bool) -> NDArray[np.float64]:
        rows, columns = (n, m) if reverse else (m, n)
        if values.dtype != np.float64 or values.ndim not in (1, 2):
            raise ValueError("products require native float64 vectors or blocks")
        if values.shape[0] != columns or not np.isfinite(values).all():
            raise ValueError("product input support mismatch")
        block = np.array(values.reshape(columns, -1), order="C", copy=True)
        block.flags.writeable = False
        result = (vjp if reverse else jvp)(block)
        if (
            result.dtype != np.float64
            or result.shape != (rows, block.shape[1])
            or not np.isfinite(result).all()
        ):
            raise ValueError("product callback must return finite float64 direction-last blocks")
        return np.array(result, order="C", copy=True)

    return LinearOperator(
        shape,
        matvec=lambda v: apply(v, False).reshape(m),
        rmatvec=lambda w: apply(w, True).reshape(n),
        matmat=lambda v: apply(v, False),
        rmatmat=lambda w: apply(w, True),
        dtype=np.float64,
    )


def materialize_operator(
    operator: LinearOperator, *, max_bytes: int, block_size: int = 16
) -> NDArray[np.float64]:
    """Materialize a small reference matrix under an explicit allocation budget.

    Args:
        operator: Real float64 operator with fixed state and deterministic products.
        max_bytes: Positive budget for output plus seed and returned product blocks.
            Callback-internal workspace, adapter copies and later diagnostic SVD,
            covariance or sensitivity arrays are outside this budget.
        block_size: Positive maximum number of identity columns per product.

    Returns:
        Owned C-contiguous (m,n) matrix in operator coordinate order.

    Raises:
        ValueError: Controls, dtype or returned products are invalid.
        MemoryError: Output plus explicit block estimate exceeds max_bytes.
    """
    m, n = operator.shape
    if min(m, n) < 1 or max_bytes < 1 or block_size < 1 or operator.dtype != np.float64:
        raise ValueError("positive budgets and a float64 operator required")
    width = min(n, block_size)
    required = 8 * (m * n + (m + n) * width)
    if required > max_bytes:
        raise MemoryError(f"matrix and explicit blocks require {required} bytes")
    result = np.empty((m, n), dtype=np.float64)
    for start in range(0, n, width):
        stop = min(n, start + width)
        seeds = np.zeros((n, stop - start), dtype=np.float64)
        seeds[np.arange(start, stop), np.arange(stop - start)] = 1
        block = operator.matmat(seeds)
        if (
            block.dtype != np.float64
            or block.shape != (m, stop - start)
            or not np.isfinite(block).all()
        ):
            raise ValueError("invalid operator product")
        result[:, start:stop] = block
        del block, seeds
    return result


__all__ = ["linear_operator", "materialize_operator"]
