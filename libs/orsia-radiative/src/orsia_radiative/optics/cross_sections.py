"""Apply fixed absorption cross sections to molecular columns and their derivatives.

These array operations preserve layer and spectral order. The caller supplies
physically current cross sections; spectroscopy and physical composition remain
separate. Derivatives are with respect to columns at fixed cross sections.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.optics import absorption_native


def cross_section_optical_depth(
    cross_sections_m2: NDArray[np.float64],
    columns_molecules_m2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute vertical optical depth from cross sections and molecular columns.

    Evaluate tau[l,p] = sigma[l,p] * N[l] in serial native code, allocating only
    the depth output. Inputs must remain stable during calls, including concurrent
    calls. Strided, unaligned and read-only native float64 arrays are supported.

    Args:
        cross_sections_m2: Finite nonnegative (layers, points) cross sections in
            m2/molecule, on caller-supplied bottom-to-top layers and spectral points.
        columns_molecules_m2: Finite nonnegative (layers,) molecular columns in
            molecules/m2 for one absorber.

    Returns:
        Owned writable C-contiguous float64 (layers, points) dimensionless depths.
        Inputs are preserved and share no storage with the output.

    Raises:
        ValueError: Dtype, shape, finite values or physical bounds are invalid,
            layer/point support is empty, or a computed value overflows.
        OverflowError: Output dimensions exceed addressable storage.
    """
    return absorption_native.cross_section_optical_depth(cross_sections_m2, columns_molecules_m2)


def cross_section_optical_depth_tangent(
    cross_sections_m2: NDArray[np.float64],
    columns_molecules_m2: NDArray[np.float64],
    columns_tangent_molecules_m2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate molecular columns and directions through fixed absorption cross sections.

    Compute tau[l,p] = sigma[l,p] * N[l] and dtau[l,p,d] = sigma[l,p] * dN[l,d]
    analytically in one serial native call. Cross sections, layer conditions and
    spectral sampling are fixed. The caller checks their physical freshness;
    temperature, pressure and cross-section derivatives are outside this API.
    Inputs must remain stable during calls, including concurrent calls.
    Strided, unaligned and read-only native float64 arrays are supported.

    Args:
        cross_sections_m2: Finite nonnegative (layers, points) cross sections in
            m2/molecule, evaluated on fixed bottom-to-top layers and spectral points.
        columns_molecules_m2: Finite nonnegative (layers,) molecular columns in
            molecules/m2 for one absorber.
        columns_tangent_molecules_m2: Finite signed (layers, k) column directions
            in molecules/m2 per caller-defined direction. Runtime k may be zero.

    Returns:
        Owned writable C-contiguous float64 arrays: dimensionless optical depth
        (layers, points) and directional tangent (layers, points, k). Inputs are
        preserved; outputs share storage with neither inputs nor each other.

    Raises:
        ValueError: Dtype, shape, finite values or physical bounds are invalid,
            layer/point support is empty, or a computed value overflows.
        OverflowError: Output dimensions exceed addressable storage.
    """
    return absorption_native.cross_section_optical_depth_tangent(
        cross_sections_m2, columns_molecules_m2, columns_tangent_molecules_m2
    )
