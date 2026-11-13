"""Normalized, layer-specific molecular cross-section file reading."""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import LayerCrossSections, wavenumber_grid


def read_cross_section_table(
    path: Path,
    *,
    species: str,
    isotopologue: str | None,
    kind: Literal["cross_section", "continuum_excluding_lines"],
    height_edges_m: NDArray[np.float64],
) -> LayerCrossSections[np.float64]:
    """Read a normalized, pickle-free NPZ of fixed layer cross sections.

    Required keys carry explicit units: wavenumbers_cm1,temperature_k,
    pressure_pa,cross_sections_m2_per_molecule, with axes (layer, point).
    The caller supplies physical layer interfaces because the preserved NPZ schema
    contains no geometry. File values retain their declared T/p; this reader performs
    no condition interpolation or continuum evaluation. Source and normalization
    records remain caller-owned. The radiative value owns immutable snapshots.

    Args:
        path: Local normalized NPZ file containing the declared grids and fields.
        species: Gas identity attached to the returned cross-section field.
        isotopologue: Optional isotopologue identity attached to the field.
        kind: Cross-section or line-excluding continuum classification.
        height_edges_m: Increasing physical interfaces (layer+1,), metres, supplied
            in the same layer order as the file. No synthetic interfaces are inferred.

    Returns:
        Layer-specific cross-section field with wavenumbers in cm^-1 and values
        in m^2 per molecule, using copied buffers from the radiative contract.

    Raises:
        ValueError: If required arrays or physical field support are invalid.
    """
    with np.load(path, allow_pickle=False) as dataset:
        try:
            arrays = [
                np.asarray(dataset[name], dtype=np.float64)
                for name in (
                    "wavenumbers_cm1",
                    "temperature_k",
                    "pressure_pa",
                    "cross_sections_m2_per_molecule",
                )
            ]
        except (KeyError, ValueError) as error:
            raise ValueError("invalid normalized cross-section arrays") from error
    return LayerCrossSections(
        species,
        isotopologue,
        kind,
        wavenumber_grid(arrays[0]),
        *arrays[1:],
        height_edges_m=height_edges_m,
    )
