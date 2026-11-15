"""Reduce gas and scattering extinction on independently supplied straight paths.

Segment molecular columns already include geometric path length. Cross sections
are fixed within each segment. Scattering depths describe direct-beam losses only.
Inputs are borrowed and stable for a call; outputs own writable float64 storage.
"""

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.optics.cross_sections import (
    cross_section_optical_depth,
    cross_section_optical_depth_tangent,
)


def integrate_path_optical_depth(
    segment_columns_molecules_m2: NDArray[np.float64],
    segment_cross_sections_m2: NDArray[np.float64],
    *,
    path_offsets: NDArray[np.int64],
    segment_scattering_optical_depth: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Sum species absorption and optional scattering losses separately for each path.

    Args:
        segment_columns_molecules_m2: Nonnegative (segments,species) slant amounts.
        segment_cross_sections_m2: Nonnegative (segments,species,spectral) m2/molecule.
        path_offsets: Nondecreasing (paths+1,) offsets, from zero through segments.
            Repeated offsets denote empty paths with zero optical depth.
        segment_scattering_optical_depth: Optional nonnegative (segments,spectral)
            dimensionless extinction from scattering, added once.

    Returns:
        Fresh dimensionless (paths,spectral) total extinction depths.

    Raises:
        ValueError: Shapes, values or offsets are invalid, or arithmetic overflows.
    """
    _validate(segment_columns_molecules_m2, segment_cross_sections_m2, path_offsets)
    n, species, points = segment_cross_sections_m2.shape
    depth = np.zeros((n, points))
    if n:
        for gas in range(species):
            depth += cross_section_optical_depth(
                segment_cross_sections_m2[:, gas], segment_columns_molecules_m2[:, gas]
            )
    if segment_scattering_optical_depth is not None:
        if (
            segment_scattering_optical_depth.shape != depth.shape
            or not np.isfinite(segment_scattering_optical_depth).all()
            or np.any(segment_scattering_optical_depth < 0)
        ):
            raise ValueError("scattering depth must be finite nonnegative (segments,spectral)")
        depth += segment_scattering_optical_depth
    return _sum_segments(depth, path_offsets)


def integrate_path_optical_depth_tangent(
    segment_columns_molecules_m2: NDArray[np.float64],
    segment_cross_sections_m2: NDArray[np.float64],
    columns_tangent_molecules_m2: NDArray[np.float64],
    *,
    path_offsets: NDArray[np.int64],
    segment_scattering_optical_depth: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate signed gas-column directions at fixed paths and spectroscopy.

    Geometry, scattering losses and cross sections remain fixed. No additional
    air-mass factor or segment length multiplies the supplied amounts.

    Args:
        segment_columns_molecules_m2: Nonnegative (segments,species) molecular amounts.
        segment_cross_sections_m2: Nonnegative (segments,species,spectral) m2/molecule.
        columns_tangent_molecules_m2: Signed finite (segments,species,k) molecular
            directions per caller-defined parameter; k may be zero.
        path_offsets: Nondecreasing (paths+1,) offsets from zero through segments.
        segment_scattering_optical_depth: Optional fixed (segments,spectral) losses.

    Returns:
        Fresh depths (paths,spectral) and tangents (paths,spectral,k), dimensionless
        and dimensionless per parameter, respectively.

    Raises:
        ValueError: Physical inputs, tangent shape or finiteness are invalid.
    """
    primal = integrate_path_optical_depth(
        segment_columns_molecules_m2,
        segment_cross_sections_m2,
        path_offsets=path_offsets,
        segment_scattering_optical_depth=segment_scattering_optical_depth,
    )
    if (
        columns_tangent_molecules_m2.ndim != 3
        or columns_tangent_molecules_m2.shape[:2] != segment_columns_molecules_m2.shape
        or not np.isfinite(columns_tangent_molecules_m2).all()
    ):
        raise ValueError("column tangents must be finite (segments,species,k)")
    n, species, points = segment_cross_sections_m2.shape
    tangent = np.zeros((n, points, columns_tangent_molecules_m2.shape[-1]))
    if n:
        for gas in range(species):
            _, derivative = cross_section_optical_depth_tangent(
                segment_cross_sections_m2[:, gas],
                segment_columns_molecules_m2[:, gas],
                columns_tangent_molecules_m2[:, gas],
            )
            tangent += derivative
    return primal, _sum_segments(tangent, path_offsets)


def _validate(columns, sections, offsets) -> None:
    if (
        columns.ndim != 2
        or sections.ndim != 3
        or sections.shape[:2] != columns.shape
        or sections.shape[1] < 1
        or sections.shape[2] < 1
        or not np.isfinite(columns).all()
        or not np.isfinite(sections).all()
        or np.any(columns < 0)
        or np.any(sections < 0)
    ):
        raise ValueError("finite nonnegative segment/species columns and cross sections required")
    if (
        offsets.ndim != 1
        or len(offsets) < 1
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets[0] != 0
        or offsets[-1] != len(columns)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("path offsets must partition all segments in order")


def _sum_segments(values, offsets):
    result = np.zeros((len(offsets) - 1, *values.shape[1:]))
    for path, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        result[path] = values[start:stop].sum(axis=0)
    if not np.isfinite(result).all():
        raise ValueError("path optical depth arithmetic overflow")
    return result
