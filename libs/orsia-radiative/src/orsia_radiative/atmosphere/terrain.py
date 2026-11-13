"""Retain atmospheric material above terrain under an explicit height reconstruction."""

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere.layers import Absorber, LayerAtmosphere


def clip_atmosphere_height(background: LayerAtmosphere, surface_height_m: float) -> LayerAtmosphere:
    """Reconstruct full and partial layers above a horizontal surface.

    Molecular number densities and temperature are constant within each source
    layer. Pressure is linear in height between interfaces; a clipped layer uses
    its new midpoint pressure. Uncut layer pressures are preserved. This explicit
    reconstruction need not be hydrostatic. Recompute state-dependent spectroscopy
    and scattering after clipping. Profile perturbation bases require remapping by
    their owner and are rejected here rather than silently discarded.

    Args:
        background: Stable bottom-to-top atmospheric state; buffers are borrowed.
        surface_height_m: Finite height in the background datum, in [bottom,top).

    Returns:
        Independent atmosphere with actual retained layer count, scaled gas/air/water
        amounts, reconstructed pressure and retained temperatures. No plume is added.

    Raises:
        ValueError: Surface support is invalid or a retained perturbation basis exists.
    """
    edges = background.height_boundaries_m
    if not np.isfinite(surface_height_m) or not edges[0] <= surface_height_m < edges[-1]:
        raise ValueError("surface height must lie in [atmosphere bottom, atmosphere top)")
    if (
        background.temperature_basis is not None
        or background.water_vapour_basis is not None
        or any(gas.profile_basis is not None for gas in background.absorbers)
    ):
        raise ValueError("remap perturbation bases explicitly before terrain clipping")
    first = int(np.searchsorted(edges, surface_height_m, side="right") - 1)
    height = edges[first:].copy()
    height[0] = surface_height_m
    fraction = retained_layer_fractions(edges, surface_height_m)[first:]
    pressure_edges = background.pressure_boundaries_pa[first:].copy()
    pressure_edges[0] = np.interp(surface_height_m, edges, background.pressure_boundaries_pa)
    pressure = background.pressure_pa[first:].copy()
    if surface_height_m != edges[first]:
        pressure[0] = (pressure_edges[0] + pressure_edges[1]) / 2
    return LayerAtmosphere(
        height,
        pressure_edges,
        pressure,
        background.temperature_k[first:],
        background.total_air_columns_molecules_m2[first:] * fraction,
        background.dry_air_columns_molecules_m2[first:] * fraction,
        background.water_vapour_columns_molecules_m2[first:] * fraction,
        tuple(
            Absorber(gas.species, gas.columns_molecules_m2[first:] * fraction, gas.air_broadener)
            for gas in background.absorbers
        ),
    )


def retained_layer_fractions(
    height_edges_m: NDArray[np.float64], surface_height_m: float
) -> NDArray[np.float64]:
    """Return source-layer retained fractions for constant-density extensive quantities.

    Use these fractions for fixed scattering constituents or independent extensive
    fields sharing the atmosphere's height reconstruction. Zero fractions identify
    removed layers; callers slice them out before constructing a transfer column.

    Args:
        height_edges_m: Finite increasing (layers+1,) metre interfaces.
        surface_height_m: Finite height in [bottom,top), in the same datum.

    Returns:
        Fresh (layers,) dimensionless fractions between zero and one.

    Raises:
        ValueError: Height edges or surface support are invalid.
    """
    if (
        height_edges_m.ndim != 1
        or len(height_edges_m) < 2
        or not np.isfinite(height_edges_m).all()
        or np.any(np.diff(height_edges_m) <= 0)
        or not np.isfinite(surface_height_m)
        or not height_edges_m[0] <= surface_height_m < height_edges_m[-1]
    ):
        raise ValueError("increasing edges and surface within [bottom,top) required")
    return np.maximum(
        0, height_edges_m[1:] - np.maximum(height_edges_m[:-1], surface_height_m)
    ) / np.diff(height_edges_m)
