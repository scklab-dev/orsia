"""Materialised opaque Lambertian spectral surface boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.profiles import LayerProfileBasis
from orsia_radiative.profiles import check_profile_bounds as _check_bounds
from orsia_radiative.profiles import validate_profile_values as _finite
from orsia_radiative.spectral.grid import (
    SpectralGrid,
    require_wavenumber_grid,
)


@dataclass(frozen=True, eq=False)
class SurfaceState:
    """Spectral opaque, Lambertian boundary; no directional BRDF or terrain.

    Reflectance is hemispherical reflectance for isotropic diffuse reflection;
    emissivity is hemispherical thermal emissivity. If both are supplied at the
    same spectral points, opaque-surface Kirchhoff consistency is required.

    Attributes:
        grid: Sampled surface wavenumber grid in cm-1.
        reflectance: Optional (points,) hemispherical diffuse reflectance in [0,1].
        emissivity: Optional (points,) hemispherical thermal emissivity in [0,1].
        temperature_k: Positive kelvin surface temperature paired with emissivity.
        reflectance_basis: Optional matching spectral perturbation basis.
        emissivity_basis: Optional matching spectral perturbation basis.
    """

    grid: SpectralGrid
    reflectance: NDArray[np.float64] | None = None
    emissivity: NDArray[np.float64] | None = None
    temperature_k: float | None = None
    reflectance_basis: LayerProfileBasis | None = None
    emissivity_basis: LayerProfileBasis | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if self.reflectance is None and self.emissivity is None:
            raise ValueError("surface needs reflectance or emissivity")
        if (self.emissivity is None) != (self.temperature_k is None):
            raise ValueError("thermal emissivity and temperature must be supplied together")
        if self.temperature_k is not None and (
            not isfinite(self.temperature_k) or self.temperature_k <= 0
        ):
            raise ValueError("surface temperature must be positive Kelvin")
        for name, values in (("reflectance", self.reflectance), ("emissivity", self.emissivity)):
            if values is not None:
                _finite(values, len(self.wavenumbers_cm1), name, minimum=0)
                _check_bounds(values, None, 1)
                object.__setattr__(self, name, np.copy(values))
        for basis, values in (
            (self.reflectance_basis, self.reflectance),
            (self.emissivity_basis, self.emissivity),
        ):
            if basis is not None:
                if (
                    basis.domain != "spectral"
                    or basis.unit != "1"
                    or values is None
                    or not np.array_equal(values, basis.baseline)
                ):
                    raise ValueError("surface basis baseline must equal the materialised state")
        if (
            self.reflectance is not None
            and self.emissivity is not None
            and any(
                not isclose(r + e, 1.0, rel_tol=0, abs_tol=1e-12)
                for r, e in zip(self.reflectance, self.emissivity, strict=True)
            )
        ):
            raise ValueError("opaque Lambertian reflectance plus emissivity must equal one")

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the surface grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points
