"""Materialised Rayleigh and aerosol scattering properties and role adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.spectral.conversions import wavenumber_to_wavelength
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


def rayleigh_from_air_columns(
    grid: SpectralGrid,
    dry_air_columns_molecules_m2: NDArray[np.float64],
    reference_cross_section_m2: float,
    reference_wavelength_um: float,
) -> RayleighComponent:
    """Build approximate dry-air Rayleigh depths from layer molecular columns.

    Evaluate sigma(lambda) = sigma_ref * (lambda_ref / lambda)**4, then
    multiply by each dry-air column. This prescribed inverse-fourth-power law
    uses the caller's reference cross section; it omits refractive-index and
    depolarization corrections. The component uses conservative scalar Rayleigh
    scattering with its existing fixed phase convention.

    Args:
        grid: Increasing wavenumber-cm-1 sampling and optional coverage domain.
        dry_air_columns_molecules_m2: Finite nonnegative float64 columns with
            shape (layers,), at least one layer, ordered bottom to top.
        reference_cross_section_m2: Finite nonnegative reference cross section
            in m2 per molecule.
        reference_wavelength_um: Finite positive reference wavelength in um.

    Returns:
        Rayleigh component retaining the exact grid and domain, with independent
        writable float64 dimensionless depths of shape (layers, points).
        Layer and spectral order are preserved. Inputs remain unchanged and
        must remain stable during the call; calls share no mutable state.

    Raises:
        ValueError: The grid uses wavelength coordinates, columns have invalid
            shape or values, reference parameters are invalid, or evaluation
            produces nonfinite wavelengths, cross sections or optical depths.
    """
    require_wavenumber_grid(grid)
    columns = dry_air_columns_molecules_m2
    if columns.ndim != 1 or columns.size == 0:
        raise ValueError("dry-air columns must have nonempty shape (layers,)")
    if not np.all(np.isfinite(columns)) or np.any(columns < 0):
        raise ValueError("dry-air columns must be finite and nonnegative")
    if not np.isfinite(reference_cross_section_m2) or reference_cross_section_m2 < 0:
        raise ValueError("reference cross section must be finite and nonnegative")
    if not np.isfinite(reference_wavelength_um) or reference_wavelength_um <= 0:
        raise ValueError("reference wavelength must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        wavelengths_um = wavenumber_to_wavelength(grid.points)[::-1] / 1000.0
        cross_section = reference_cross_section_m2 * (reference_wavelength_um / wavelengths_um) ** 4
        depth = columns[:, None] * cross_section[None, :]
    if not np.all(np.isfinite(cross_section)):
        raise ValueError("evaluated Rayleigh cross section must be finite")
    return RayleighComponent(grid, depth)


def angstrom_optical_depth(
    grid: SpectralGrid,
    reference_optical_depth: float,
    reference_wavelength_nm: float,
    angstrom_exponent: float,
) -> NDArray[np.float64]:
    """Evaluate spectral column extinction depth with a prescribed Angstrom law.

    Evaluate tau(lambda) = tau_ref * (lambda / lambda_ref)**(-alpha).
    The caller supplies vertical allocation, single-scattering albedo and phase
    separately. Optical depth is dimensionless, so coordinate conversion applies
    no spectral-density Jacobian.

    Args:
        grid: Increasing wavelength-nm or wavenumber-cm-1 sampling. The optional
            coverage domain is retained by the caller and does not affect values.
        reference_optical_depth: Finite nonnegative dimensionless column
            extinction optical depth at the reference wavelength.
        reference_wavelength_nm: Finite positive reference wavelength in nm.
        angstrom_exponent: Finite dimensionless exponent alpha. Positive values
            decrease extinction with wavelength; zero and negative values are valid.

    Returns:
        Independent writable float64 dimensionless column extinction depths with
        shape (points,), in the supplied grid's point order. Inputs remain
        unchanged and must remain stable during the call; calls share no mutable
        state. This function performs no vertical allocation or scattering split.

    Raises:
        ValueError: Reference parameters are invalid or evaluation produces
            nonfinite wavelengths or optical depths.
    """
    if not np.isfinite(reference_optical_depth) or reference_optical_depth < 0:
        raise ValueError("reference optical depth must be finite and nonnegative")
    if not np.isfinite(reference_wavelength_nm) or reference_wavelength_nm <= 0:
        raise ValueError("reference wavelength must be finite and positive")
    if not np.isfinite(angstrom_exponent):
        raise ValueError("Angstrom exponent must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        wavelengths_nm = (
            wavenumber_to_wavelength(grid.points)[::-1]
            if grid.coordinate == "wavenumber_cm1"
            else grid.points
        )
        depth = reference_optical_depth * (wavelengths_nm / reference_wavelength_nm) ** (
            -angstrom_exponent
        )
    if not np.all(np.isfinite(depth)):
        raise ValueError("evaluated Angstrom optical depth must be finite")
    return depth


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
