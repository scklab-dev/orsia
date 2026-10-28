"""Materialised Rayleigh and aerosol scattering properties and role adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


def rayleigh_component(properties: ScatteringProperties) -> RayleighComponent:
    """Use validated scalar Rayleigh moments as a transfer optical component.

    Args:
        properties: Untruncated (1,0,0.1) Rayleigh phase moments and depths.

    Returns:
        Conservative Rayleigh depth on the same spectral sampling.

    Raises:
        ValueError: Moments or truncation differ from supported scalar Rayleigh.
    """
    if (
        properties.truncation != "source_untruncated"
        or properties.phase_moments.shape[-1] != 3
        or np.any(np.abs(properties.phase_moments[..., 1]) > 1e-12)
        or np.any(np.abs(properties.phase_moments[..., 2] - 0.1) > 1e-12)
    ):
        raise ValueError("Rayleigh requires untruncated scalar moments (1, 0, 0.1)")
    return RayleighComponent(
        properties.grid,
        properties.optical_depth,
    )


def aerosol_component(optics: AerosolOptics) -> AerosolComponent:
    """Expose aerosol scattering and absorption consistently to transfer code.

    Args:
        optics: Evaluated aerosol extinction, albedo and HG asymmetry by layer.

    Returns:
        Aerosol component with scattering depth equal to extinction times albedo.
    """
    return AerosolComponent(
        optics.grid,
        optics.optical_depth,
        optics.optical_depth * optics.single_scattering_albedo,
        optics.asymmetry,
    )


@dataclass(frozen=True, eq=False)
class ScatteringProperties:
    """Vertical tau and scalar Legendre moments per [layer][spectral point].

    Phase expansion: P(cos(theta)) = sum_l (2l+1) * a_l * P_l(cos(theta));
    a_0=1, integral over 4pi is 4pi. The supplied finite moments do NOT
    guarantee pointwise nonnegative phase; callers must qualify their inputs.
    This record does not describe a polarisation phase matrix.
    Depths have shape (n_layers, n_points); moments have shape
    (n_layers, n_points, n_moments). Arrays are independent writable float64 copies.

    Attributes:
        grid: Evaluated wavenumber grid in cm-1.
        optical_depth: (layers, points) vertical scattering optical depths.
        phase_moments: (layers, points, moments) scalar Legendre coefficients.
        truncation: Declared source truncation convention for these moments.
    """

    grid: SpectralGrid
    optical_depth: NDArray[np.float64]
    phase_moments: NDArray[np.float64]
    truncation: Literal["source_truncated", "source_untruncated"]
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if self.truncation not in ("source_truncated", "source_untruncated"):
            raise ValueError("scattering needs a supported truncation declaration")
        s = self.wavenumbers_cm1.size
        if (
            self.optical_depth.ndim != 2
            or self.optical_depth.shape[0] == 0
            or self.optical_depth.shape[1] != s
        ):
            raise ValueError("scattering depth must have shape (layers, spectral points)")
        if (
            self.phase_moments.ndim != 3
            or self.phase_moments.shape[:2] != self.optical_depth.shape
            or self.phase_moments.shape[2] == 0
        ):
            raise ValueError("phase moments must have shape (layers, points, moments)")
        if not np.all(np.isfinite(self.optical_depth)) or np.any(self.optical_depth < 0):
            raise ValueError("scattering optical depth must be finite and nonnegative")
        if np.any(np.abs(self.phase_moments[..., 0] - 1.0) > 1e-12):
            raise ValueError("phase zeroth Legendre moment must be one")
        if not np.all(np.abs(self.phase_moments) <= 1):
            raise ValueError("phase moments must be finite and bounded by one")
        object.__setattr__(self, "optical_depth", np.copy(self.optical_depth))
        object.__setattr__(self, "phase_moments", np.copy(self.phase_moments))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the scattering grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


@dataclass(frozen=True, eq=False)
class AerosolOptics:
    """Aerosol extinction, single-scattering albedo and HG asymmetry per layer/point.

    Optical depths are vertical and ordered bottom to top. The untruncated
    Henyey-Greenstein scalar phase has 4π angular integral 4π; |g| <= 0.8
    bounds the angular quadrature used by the aerosol-scattering model. Aerosol absorption is
    (1 - albedo) * extinction, not an additional gas component.
    Arrays have shape (n_layers, n_points) and are independent writable float64 copies.
    This extinction/albedo parameterisation is a materialised input to
    ``aerosol_component``, not a second transfer state or a provenance record.

    Attributes:
        grid: Evaluated wavenumber grid in cm-1.
        optical_depth: (layers, points) aerosol extinction optical depths.
        single_scattering_albedo: (layers, points) scattering fraction of extinction.
        asymmetry: (layers, points) HG asymmetry coefficient in [-0.8,0.8].
    """

    grid: SpectralGrid
    optical_depth: NDArray[np.float64]
    single_scattering_albedo: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if (
            self.optical_depth.ndim != 2
            or self.optical_depth.shape[0] == 0
            or self.optical_depth.shape[1] != self.wavenumbers_cm1.size
        ):
            raise ValueError("aerosol extinction must have shape (layers, points)")
        for name in ("single_scattering_albedo", "asymmetry"):
            values = getattr(self, name)
            if values.shape != self.optical_depth.shape:
                raise ValueError(f"{name} must match the extinction shape")
        if not np.all(np.isfinite(self.optical_depth)) or np.any(self.optical_depth < 0):
            raise ValueError("aerosol extinction must be finite and nonnegative")
        if not np.all((self.single_scattering_albedo >= 0) & (self.single_scattering_albedo <= 1)):
            raise ValueError("aerosol single-scattering albedo must be in [0, 1]")
        if not np.all(np.abs(self.asymmetry) <= 0.8):
            raise ValueError("aerosol HG asymmetry must be finite and in [-0.8, 0.8]")
        for name in ("optical_depth", "single_scattering_albedo", "asymmetry"):
            object.__setattr__(self, name, np.copy(getattr(self, name)))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the aerosol grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


@dataclass(frozen=True, eq=False)
class RayleighComponent:
    """Conservative scalar Rayleigh depth [layer][point], bottom to top.

    Extinction equals scattering depth; the phase is fixed, not a caller choice.
    Grid is immutable support, shared when already safely immutable. Depths are
    independent writable float64 copies of shape (n_layers, n_points).

    Attributes:
        grid: Evaluated wavenumber grid in cm-1.
        optical_depth: (layers, points) conservative Rayleigh scattering depths.
    """

    grid: SpectralGrid
    optical_depth: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if (
            self.optical_depth.ndim != 2
            or self.optical_depth.shape[0] == 0
            or self.optical_depth.shape[1] != self.wavenumbers_cm1.size
        ):
            raise ValueError("Rayleigh depth must have shape (layers, spectral points)")
        if not np.all(np.isfinite(self.optical_depth)) or np.any(self.optical_depth < 0):
            raise ValueError("Rayleigh optical depth must be finite and nonnegative")
        object.__setattr__(self, "optical_depth", np.copy(self.optical_depth))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the Rayleigh grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


@dataclass(frozen=True, eq=False)
class AerosolComponent:
    """HG aerosol extinction, scattering depth and g [layer][point], bottom to top.

    Depths obey 0 <= scattering <= extinction and |g| <= 0.8. The difference
    already contains aerosol absorption. Grid is immutable support, shared when
    already safely immutable. Numerical arrays are independent writable float64
    snapshots of shape (n_layers, n_points).

    Attributes:
        grid: Evaluated wavenumber grid in cm-1.
        optical_depth: (layers, points) total aerosol extinction depths.
        scattering_depth: (layers, points) aerosol scattering part of extinction.
        asymmetry: (layers, points) HG asymmetry coefficient in [-0.8,0.8].
    """

    grid: SpectralGrid
    optical_depth: NDArray[np.float64]
    scattering_depth: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if (
            self.optical_depth.ndim != 2
            or self.optical_depth.shape[0] == 0
            or self.optical_depth.shape[1] != self.wavenumbers_cm1.size
        ):
            raise ValueError("aerosol extinction must have shape (layers, points)")
        for name in ("scattering_depth", "asymmetry"):
            values = getattr(self, name)
            if values.shape != self.optical_depth.shape:
                raise ValueError(f"{name} must match the extinction shape")
        if not np.all(np.isfinite(self.optical_depth)) or not np.all(
            (self.scattering_depth >= 0) & (self.scattering_depth <= self.optical_depth)
        ):
            raise ValueError("finite depths must satisfy 0 <= scattering <= extinction")
        if not np.all(np.abs(self.asymmetry) <= 0.8):
            raise ValueError("HG asymmetry must be finite and in [-0.8, 0.8]")
        for name in ("optical_depth", "scattering_depth", "asymmetry"):
            object.__setattr__(self, name, np.copy(getattr(self, name)))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the aerosol component's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points
