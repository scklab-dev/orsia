"""Materialise directional spectral radiance with explicit spectral and viewing support.

Values retain their evaluated grid, local geometry and independent writable
snapshots. Transfer diagnostics and input lineage belong to the producing calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.geometry import ObservationGeometry
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


@dataclass(frozen=True, eq=False)
class Radiance:
    """Directional spectral radiance at one evaluated local observation geometry.

    ``values`` are W m-2 sr-1 per declared spectral coordinate (nm or cm-1).
    The public contract
    supports directional, unpolarised radiance with an explicit local geometry;
    it does not represent an angular field, a detector realisation or covariance.
    It deliberately contains no transfer-method identity, source provenance,
    component decomposition or diagnostic arrays. Those are calculation-local
    products and may be returned separately by a calculation that needs them.
    No spatial footprint or time interval is supplied; acquisition callers must
    establish those supports themselves.
    Axis retains immutable support, shared when already safely immutable. Values
    are independent writable float64 copies with shape (n_points,).

    Attributes:
        grid: Evaluated point grid in nm or cm-1, defining the density measure.
        values: Owned (points,) directional radiance in W m-2 sr-1 per grid coordinate.
        geometry: Local outgoing observation direction of this radiance.
    """

    grid: SpectralGrid
    values: NDArray[np.float64]
    geometry: ObservationGeometry
    __hash__ = None

    def __post_init__(self) -> None:
        if self.values.shape != (self.grid.points.size,):
            raise ValueError("radiance values must match the spectral axis")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("radiance values must be finite and nonnegative")
        object.__setattr__(self, "values", np.copy(self.values))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose cm-1 samples when the radiance already uses that coordinate.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.

        Raises:
            ValueError: The record uses nm; convert grid and density explicitly.
        """
        require_wavenumber_grid(self.grid)
        return self.grid.points


__all__ = ["Radiance"]
