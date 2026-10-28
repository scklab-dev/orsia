"""Normalized, layer-specific molecular cross-section file reading."""

from pathlib import Path
from typing import Literal

import numpy as np
from orsia_radiative import CrossSectionField, wavenumber_grid


def read_cross_section_table(
    path: Path,
    *,
    species: str,
    isotopologue: str | None,
    kind: Literal["cross_section", "continuum_excluding_lines"],
) -> CrossSectionField:
    """Read a normalized, pickle-free NPZ of fixed layer cross sections.

    Required keys carry explicit units: wavenumbers_cm1,temperature_k,
    pressure_pa,cross_sections_m2_per_molecule, with axes (layer, point).
    This is a decoded-file exchange, NOT automatic T/p resampling, conversion
    of arbitrary measured spectra or evaluation of a continuum model. The
    caller retains any source and normalization records externally. All numerical
    buffers are snapshotted by the public radiative contract.

    Args:
        path: Local normalized NPZ file containing the declared grids and fields.
        species: Gas identity attached to the returned cross-section field.
        isotopologue: Optional isotopologue identity attached to the field.
        kind: Cross-section or line-excluding continuum classification.

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
    return CrossSectionField(species, isotopologue, kind, wavenumber_grid(arrays[0]), *arrays[1:])
