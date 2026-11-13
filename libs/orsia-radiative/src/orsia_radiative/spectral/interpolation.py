"""Sample a nonnegative spectral point density onto a covered target grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


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
    left, right, wl, wr = spectral_interpolation_weights(source_grid, target_grid)
    result = wl * source_values[left] + wr * source_values[right]
    if not np.all(np.isfinite(result)):
        raise ValueError("interpolated point density must be finite")
    return result


def spectral_interpolation_weights(
    source_grid: SpectralGrid,
    target_grid: SpectralGrid,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Locate nonnegative linear stencils within declared spectral intervals.

    Exact nodes use identical left/right indices with weights one/zero, including
    a singleton source. Unknown domains enforce the sampled envelope only.

    Args:
        source_grid: Increasing source samples with optional physical domain.
        target_grid: Increasing same-coordinate target samples requiring in-domain brackets.

    Returns:
        Owned left/right int64 indices and left/right float64 weights on target points.

    Raises:
        ValueError: Coordinates mismatch, extrapolation is needed or a stencil crosses a gap.
    """
    if source_grid.coordinate != target_grid.coordinate:
        raise ValueError("spectral interpolation coordinates must match")
    source, target = source_grid.points, target_grid.points
    right = np.searchsorted(source, target, side="left").astype(np.int64)
    exact = (right < len(source)) & (source[np.minimum(right, len(source) - 1)] == target)
    if np.any((right == 0) & ~exact) or np.any(right == len(source)):
        raise ValueError("spectral extrapolation beyond source points is not supported")
    left = np.where(exact, right, right - 1)
    if source_grid.domain is not None:
        bounds = np.asarray(source_grid.domain.intervals)
        intervals = np.searchsorted(bounds[:, 0], target, side="right") - 1
        lower, upper = bounds[np.maximum(intervals, 0)].T
        if np.any(intervals < 0) or np.any(target > upper):
            raise ValueError("target points lie outside declared spectral support in a gap")
        if np.any(source[left][~exact] < lower[~exact]) or np.any(
            source[right][~exact] > upper[~exact]
        ):
            raise ValueError("interpolation stencil crosses a declared spectral gap")
    width = np.where(exact, 1.0, source[right] - source[left])
    wl = np.where(exact, 1.0, (source[right] - target) / width)
    wr = np.where(exact, 0.0, (target - source[left]) / width)
    return left, right, wl, wr
