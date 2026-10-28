"""Locate atmospheric coordinate samples for linear spatial interpolation."""

import numpy as np
from numpy.typing import NDArray


def linear_axis_weights(
    axis: NDArray[np.float64], point: float, *, periodic: bool
) -> list[tuple[int, float]]:
    """Return original indices and linear weights on a strict monotone 1D axis.

    A periodic axis has period 360 and unique nodes spanning less than 360;
    caller must establish global coverage. No extrapolation or nearest fallback.
    Exact nodes do not read adjacent cells. Neither input array is retained or
    mutated. A singleton axis only supports its exact coordinate.

    Args:
        axis: Finite strictly monotone one-dimensional coordinate values.
        point: Requested coordinate in the same unit as ``axis``.
        periodic: Wrap coordinates with a 360-degree period, for global longitude.

    Returns:
        One exact-node or two bracketing (original index, linear weight) pairs.

    Raises:
        ValueError: Coordinates are invalid, the periodic axis spans at least
            360 degrees, or the requested point lies outside covered coordinates.
    """
    if axis.ndim != 1 or not len(axis) or not np.all(np.isfinite(axis)) or not np.isfinite(point):
        raise ValueError("invalid interpolation axis or requested point")
    order = np.arange(len(axis))
    if len(axis) > 1 and np.all(np.diff(axis) < 0):
        axis, order = axis[::-1], order[::-1]
    if np.any(np.diff(axis) <= 0):
        raise ValueError("interpolation coordinates must be strictly monotone")
    if periodic:
        if len(axis) < 2 or axis[-1] - axis[0] >= 360:
            raise ValueError("periodic longitude needs unique nodes spanning less than 360 degrees")
        point = (point - axis[0]) % 360 + axis[0]
        axis, order = np.append(axis, axis[0] + 360), np.append(order, order[0])
    if not axis[0] <= point <= axis[-1]:
        raise ValueError("requested point is outside coordinate coverage")
    i = int(np.searchsorted(axis, point))
    if axis[i] == point:
        return [(int(order[i]), 1.0)]
    fraction = (point - axis[i - 1]) / (axis[i] - axis[i - 1])
    return [(int(order[i - 1]), float(1 - fraction)), (int(order[i]), float(fraction))]
