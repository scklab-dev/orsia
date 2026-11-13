"""Integrate declared piecewise-constant sources into finite midpoint releases."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


def release_intervals(
    edges_s: FloatArray,
    rates_kg_s: FloatArray,
    partition_times_s: FloatArray | None,
    stop_time_s: float | None,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Partition an emission history while retaining its declared mass integral.

    Args:
        edges_s: Strictly increasing finite interval edges (n+1,), seconds.
        rates_kg_s: Finite nonnegative rates (n,), kg/s.
        partition_times_s: Optional increasing finite partition times, seconds.
        stop_time_s: Optional finite end time; clips included intervals.

    Returns:
        Midpoint times, positive included masses and original interval indices,
        each with one entry per retained subinterval. Zero-rate intervals are omitted.

    Raises:
        ValueError: Times, rates or integrated masses are invalid.
    """
    edges = np.asarray(edges_s, dtype=np.float64)
    rates = np.asarray(rates_kg_s, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges_s must be a finite vector with at least two edges")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges_s must increase strictly")
    if rates.shape != (len(edges) - 1,) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("rates must be finite nonnegative values per interval")
    cuts = np.empty(0) if partition_times_s is None else np.asarray(partition_times_s)
    if cuts.ndim != 1 or not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("partition_times_s must be a finite increasing vector")
    stop = edges[-1] if stop_time_s is None else stop_time_s
    if not np.isfinite(stop):
        raise ValueError("stop_time_s must be finite")
    partition = np.unique(np.concatenate((edges, cuts, [stop])))
    partition = partition[(partition >= edges[0]) & (partition <= min(stop, edges[-1]))]
    left, right = partition[:-1], partition[1:]
    indices = np.searchsorted(edges, left, side="right") - 1
    masses = rates[indices] * (right - left)
    if not np.isfinite(masses).all():
        raise ValueError("integrated emission masses must be finite")
    keep = masses > 0
    return left[keep] + (right[keep] - left[keep]) / 2, masses[keep], indices[keep]
