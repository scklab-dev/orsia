"""Assemble explicit direction-last tangents and Jacobians."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

type TangentFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]


def jacobian(
    function: TangentFunction,
    x: NDArray[np.float64],
    *,
    block_size: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a value and its dense Jacobian using identity seed columns.

    The callback computes ``(F(x), J @ directions)``. The direction axis is
    final, including when the value is scalar. Each call receives a read-only
    owned state snapshot and fresh writable seeds. The callback must keep the
    state and seeds unchanged and return the same value for every seed block.
    Inputs and callback outputs are never retained in the returned arrays.
    There is no finite differencing, tracing, or retained derivative state.

    Args:
        function: Explicit tangent callback taking state ``(n,)`` and seeds
            ``(n, k)`` and returning value and tangent ``value.shape + (k,)``.
            Both outputs must contain finite native float64 values.
        x: Finite native float64 state vector. Coordinates and units belong to
            the caller; the Jacobian has output units per state coordinate unit.
        block_size: Positive maximum number of identity columns per call.
            None evaluates all columns together. Empty states evaluate once
            with seeds of shape ``(0, 0)``.

    Returns:
        Fresh writable C-contiguous float64 value and Jacobian arrays. The
        Jacobian has shape ``value.shape + (n,)`` in state-coordinate order.

    Raises:
        ValueError: State, block size, callback shape, dtype or finite support
            is invalid, or the value changes between blocks. Callback exceptions
            propagate unchanged.
    """
    state = _state_snapshot(x)
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer or None")
    n = state.size
    width = max(n, 1) if block_size is None else block_size
    value: NDArray[np.float64] | None = None
    derivatives: NDArray[np.float64] | None = None
    for start in range(0, max(n, 1), width):
        stop = min(n, start + width)
        seeds = np.zeros((n, stop - start), dtype=np.float64)
        seeds[np.arange(start, stop), np.arange(stop - start)] = 1.0
        current, tangent = _evaluate_tangent(function, state, seeds)
        if value is None:
            value = np.array(current, dtype=np.float64, order="C", copy=True)
            derivatives = np.empty(value.shape + (n,), dtype=np.float64)
        elif not np.array_equal(current, value):
            raise ValueError("callback value changed between seed blocks")
        assert derivatives is not None
        derivatives[..., start:stop] = tangent
    assert value is not None and derivatives is not None
    return value, derivatives


def directional_derivative(
    function: TangentFunction,
    x: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a value and explicit derivatives along supplied directions.

    The callback receives a read-only owned state snapshot and fresh writable
    seeds of shape ``(n, k)``. It must keep both inputs unchanged. The callback
    runs once, including for empty states, outputs or direction blocks.

    Args:
        function: Explicit tangent callback taking state ``(n,)`` and seeds
            ``(n, k)`` and returning finite native float64 arrays for the value
            and tangent ``value.shape + (k,)``. Callback exceptions propagate.
        x: Finite native float64 state vector ``(n,)``. Coordinates and units
            belong to the caller.
        direction: Finite native float64 vector ``(n,)`` or block ``(n, k)``.
            Columns give state-coordinate perturbations per unit step. A vector
            is passed to the callback as one seed column. Empty blocks are valid.

    Returns:
        Fresh writable C-contiguous float64 value and derivative arrays. A
        vector direction gives derivative shape ``value.shape``; a block gives
        ``value.shape + (k,)``. Derivatives have output units per step unit.
        Returned arrays share no storage with inputs or callback outputs.

    Raises:
        ValueError: State, direction or callback shape, dtype or finite support
            is invalid.
    """
    state = _state_snapshot(x)
    if (
        direction.ndim not in (1, 2)
        or direction.shape[0] != state.size
        or direction.dtype != np.float64
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("direction must be a finite native float64 vector or direction-last block")
    vector = direction.ndim == 1
    seeds = np.array(direction[:, None] if vector else direction, order="C", copy=True)
    value, tangent = _evaluate_tangent(function, state, seeds)
    derivative = tangent[..., 0] if vector else tangent
    return (
        np.array(value, dtype=np.float64, order="C", copy=True),
        np.array(derivative, dtype=np.float64, order="C", copy=True),
    )


def _state_snapshot(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy a valid state into an owned read-only snapshot.

    Returns:
        Independent native float64 state vector with write protection.

    Raises:
        ValueError: State shape, dtype or finite support is invalid.
    """
    if x.ndim != 1 or x.dtype != np.float64 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite native float64 vector")
    state = x.copy()
    state.flags.writeable = False
    return state


def _evaluate_tangent(
    function: TangentFunction, state: NDArray[np.float64], seeds: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate and check the common finite direction-last callback contract.

    Returns:
        Callback value and tangent arrays with their original storage.

    Raises:
        ValueError: Callback shape, dtype or finite support is invalid.
    """
    value, tangent = function(state, seeds)
    if (
        value.dtype != np.float64
        or tangent.dtype != np.float64
        or tangent.shape != value.shape + (seeds.shape[1],)
        or not np.all(np.isfinite(value))
        or not np.all(np.isfinite(tangent))
    ):
        raise ValueError("callback must return finite float64 values and last-axis tangents")
    return value, tangent


__all__ = ["jacobian", "directional_derivative"]
