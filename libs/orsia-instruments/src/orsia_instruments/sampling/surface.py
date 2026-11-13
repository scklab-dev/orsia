"""Retain surface radiance samples on explicit point and spectral supports."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import PointLattice
from orsia_radiative import SpectralGrid


@dataclass(frozen=True, eq=False)
class SurfaceRadianceSamples:
    """Own nonnegative spectral radiance densities on a declared surface point lattice.

    Attributes:
        lattice: Shared immutable increasing x/y sample positions and local frame.
            Points imply neither finite cell area nor terrain.
        spectral_grid: Shared immutable spectral samples and declared density coordinate.
        values: Independent writable float64 (y,x,spectral) radiance values in
            W/m2/sr per spectral coordinate. Keep these buffers stable while an
            instrument response uses them. This record adds no atmospheric law,
            temporal evolution, interpolation choice or coverage beyond its samples.
    """

    lattice: PointLattice
    spectral_grid: SpectralGrid
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Snapshot values and validate their exact sample axes and physical domain.

        Raises:
            ValueError: Sample dimensions or finite nonnegative radiance values fail.
        """
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.shape != (*self.lattice.shape, self.spectral_grid.points.size):
            raise ValueError("surface radiance must be (y,x,spectral) on its supports")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("surface radiance must be finite and nonnegative")
        object.__setattr__(self, "values", values)
