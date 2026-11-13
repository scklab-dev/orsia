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

    Values use W m-2 per declared spectral coordinate (nm or cm-1) on an
    increasing point grid. Construction retains the supplied density measure.
    Axis and values have shape (n_points,). Values are independent writable
    float64 snapshots. Coordinates share safely immutable contiguous bytes-backed
    storage or snapshot other caller buffers; a read-only flag alone is not enough.

    Attributes:
        grid: Evaluated point grid in nm or cm-1, defining the density measure.
        values: Owned (points,) beam-normal irradiance in W m-2 per grid coordinate.
    """

    grid: SpectralGrid
    values: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        if self.values.shape != (self.grid.points.size,):
            raise ValueError("irradiance values must match the spectral axis")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("irradiance values must be finite and nonnegative")
        object.__setattr__(self, "values", np.copy(self.values))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose cm-1 samples when the irradiance already uses that coordinate.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.

        Raises:
            ValueError: The record uses nm; convert grid and density explicitly.
        """
        require_wavenumber_grid(self.grid)
        return self.grid.points


__all__ = ["SpectralIrradiance"]
