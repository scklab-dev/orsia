"""Validate finite cell partitions and assign sampled coordinates consistently."""

import numpy as np
from numpy.typing import NDArray


def cell_edges(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Snapshot a finite strictly increasing cell-edge vector.

    Args:
        values: One-dimensional edges in one caller-declared coordinate unit.

    Returns:
        Independent writable edges with at least one cell.

    Raises:
        ValueError: Edges are nonfinite, unordered or have fewer than two entries.
    """
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("cell edges must be finite strictly increasing vectors")
    return np.copy(values)


def cell_indices(edges: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Assign samples to half-open cells with an inclusive final upper edge.

    Args:
        edges: Validated increasing cell-edge vector.
        values: Sample coordinates with arbitrary shape, in the same units.

    Returns:
        Same-shaped indices; -1 and len(edges)-1 identify exterior samples.
    """
    indices = np.searchsorted(edges, values, side="right") - 1
    return np.where(values == edges[-1], len(edges) - 2, indices)
