"""Materialised spectral irradiance and radiance values.

These records carry physical spectral values and the support needed to use
them. Transfer-method labels, decomposition, diagnostics and input lineage
belong to the calculation that produced a value, not to the value itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.geometry import ObservationGeometry
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


@dataclass(frozen=True, eq=False)
class Radiance:
    """Directional spectral radiance at one evaluated local observation geometry.

    ``values`` are W m-2 sr-1 per (cm-1) point samples. The public contract
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
        grid: Evaluated wavenumber grid in cm-1.
        values: Owned (points,) directional radiance in W m-2 sr-1 per cm-1.
        geometry: Local outgoing observation direction of this radiance.
    """

    grid: SpectralGrid
    values: NDArray[np.float64]
    geometry: ObservationGeometry
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if self.values.shape != (self.wavenumbers_cm1.size,):
            raise ValueError("radiance values must match the spectral axis")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("radiance values must be finite and nonnegative")
        object.__setattr__(self, "values", np.copy(self.values))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the radiance grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


__all__ = ["Radiance", "SpectralIrradiance"]
