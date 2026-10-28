"""Sample a nonnegative spectral point density onto a covered target grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.spectral.grid import (
    SpectralGrid,
    require_wavenumber_grid,
)


def interpolate_point_density(
    source_grid: SpectralGrid,
    source_values: NDArray[np.float64],
    target_grid: SpectralGrid,
) -> NDArray[np.float64]:
    """Piecewise-linear nonnegative point density on increasing cm-1 grids.

    Values have shape (source_points,) and must be finite float64. Every target
    point must be bracketed by source samples in the same declared source-domain
    interval. Exact knots (including a singleton source) are copied unchanged.
    With unknown source domain, only the sample envelope is checked; undeclared
    gaps cannot be detected. No extrapolation, bin averaging, convolution or
    uncertainty propagation. Return fresh writable values; inputs are unmodified.
    A finer target grid cannot restore missing source spectral structure.

    Args:
        source_grid: Increasing source samples in wavenumber cm-1 with optional
            declared physical coverage.
        source_values: Finite nonnegative float64 density (source_points,).
        target_grid: Increasing target wavenumber samples within source coverage.

    Returns:
        Fresh writable density (target_points,) in the source density unit.

    Raises:
        ValueError: Coordinate units or densities are invalid, interpolation
            would cross a declared gap or target points require extrapolation.
    """
    require_wavenumber_grid(source_grid)
    require_wavenumber_grid(target_grid)
    if (
        source_values.shape != source_grid.points.shape
        or not np.all(np.isfinite(source_values))
        or np.any(source_values < 0)
    ):
        raise ValueError("source density must be finite nonnegative on source_grid")
    source = source_grid.points
    target = target_grid.points
    right = np.searchsorted(source, target, side="left")
    exact = (right < source.size) & (source[np.minimum(right, source.size - 1)] == target)
    if np.any((right == 0) & ~exact) or np.any(right == source.size):
        raise ValueError("target grid requires extrapolation beyond source points")
    if source_grid.domain is not None:
        bounds = np.asarray(source_grid.domain.intervals)
        intervals = np.searchsorted(bounds[:, 0], target, side="right") - 1
        lower = bounds[np.maximum(intervals, 0), 0]
        upper = bounds[np.maximum(intervals, 0), 1]
        if np.any(intervals < 0) or np.any(target > upper):
            raise ValueError("target points lie in a declared spectral gap")
        if np.any(source[np.maximum(right - 1, 0)][~exact] < lower[~exact]) or np.any(
            source[right][~exact] > upper[~exact]
        ):
            raise ValueError("interpolation stencil crosses a declared spectral gap")
    result = np.interp(target, source, source_values)
    if not np.all(np.isfinite(result)):
        raise ValueError("interpolated point density must be finite")
    return result
