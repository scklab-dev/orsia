"""Materialise direct beam-normal spectral irradiance with explicit spectral support.

Values retain their evaluated grid and independent writable snapshots. Source
geometry, provider access and transfer calculations remain separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


@dataclass(frozen=True, eq=False)
class SpectralIrradiance:
    """Direct spectral irradiance normal to a collimated incident beam.

    Values use W m-2 per (cm-1) on an increasing cm-1 point grid. The record
    is materialised input, not a provider lookup or an implicit unit conversion.
    Axis and values have shape (n_points,). Values are independent writable
    float64 snapshots. Coordinates share safely immutable contiguous bytes-backed
    storage or snapshot other caller buffers; a read-only flag alone is not enough.

    Attributes:
        grid: Evaluated wavenumber grid in cm-1.
        values: Owned (points,) beam-normal irradiance in W m-2 per cm-1.
    """

    grid: SpectralGrid
    values: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if self.values.shape != (self.wavenumbers_cm1.size,):
            raise ValueError("irradiance values must match the spectral axis")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("irradiance values must be finite and nonnegative")
        object.__setattr__(self, "values", np.copy(self.values))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the irradiance grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


__all__ = ["SpectralIrradiance"]
