"""Fixed layer cross sections, explicit spectral preparation and column application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid, wavenumber_grid
from orsia_radiative.spectral.interpolation import spectral_interpolation_weights
from orsia_radiative.spectroscopy.components import AbsorptionComponent, _gas


def table_absorption(
    atmosphere: Atmosphere,
    wavenumbers_cm1: NDArray[np.float64] | SpectralGrid,
    inputs: CrossSectionField,
) -> AbsorptionComponent:
    """Compute layer optical depths from tabulated cross sections and current columns.

    Args:
        atmosphere: Atmospheric gas columns and layer temperature/pressure.
        wavenumbers_cm1: Target spectral grid or wavenumbers in cm-1.
        inputs: Fixed layer cross sections and their declared layer conditions.

    Returns:
        Gas absorption component with (layers,points) dimensionless optical depths.

    Raises:
        ValueError: Spectral interpolation exceeds table support or layer conditions differ.
    """
    return evaluate_table_absorption(atmosphere, cross_sections(wavenumbers_cm1, inputs))


def cross_sections(
    wavenumbers_cm1: NDArray[np.float64] | SpectralGrid, inputs: CrossSectionField
) -> PreparedCrossSections:
    """Interpolate fixed tabulated cross sections once, without extrapolation.

    The result belongs to the caller and can be applied to different columns
    only at the table's exact T/p and layer support. No hidden cache is created.

    Args:
        wavenumbers_cm1: Target spectral grid or wavenumbers in cm-1.
        inputs: Source cross sections in m2/molecule on declared spectral support.

    Returns:
        Fixed cross sections interpolated onto the target grid as an owned
        (layers,points) float64 array, with the original table attached.

    Raises:
        ValueError: Target extrapolates or interpolation crosses a declared gap.
    """
    sampling = wavenumber_grid(wavenumbers_cm1)
    wavenumbers_cm1 = sampling.points
    grid = wavenumbers_cm1
    left_indices, right_indices, left_weights, fractions = spectral_interpolation_weights(
        inputs.grid, sampling
    )
    rows = np.empty((inputs.temperature_k.size, grid.size), dtype=np.float64)
    for row, source_values in zip(rows, inputs.cross_sections_m2_per_molecule, strict=True):
        # Nonnegative weighted terms avoid cancellation across large dynamic
        # ranges; exact nodes have weights 0/1 and preserve the tabulated value.
        row[:] = (
            left_weights * source_values[left_indices] + fractions * source_values[right_indices]
        )
    return PreparedCrossSections(inputs, sampling, rows)


def evaluate_table_absorption(
    atmosphere: Atmosphere, prepared: PreparedCrossSections
) -> AbsorptionComponent:
    """Convert prepared cross sections into optical depths for current gas columns.

    Args:
        atmosphere: Gas amounts in molecules/m2 and current layer conditions.
        prepared: Fixed per-layer cross sections and their evaluation grid.

    Returns:
        Dimensionless (layers,points) absorption component on the prepared grid.

    Raises:
        ValueError: Layer counts or temperature/pressure differ from the fixed table.
    """
    inputs = prepared.inputs
    amounts, _ = _gas(atmosphere, inputs.species)
    if not np.array_equal(inputs.temperature_k, atmosphere.temperature_k) or not np.array_equal(
        inputs.pressure_pa, atmosphere.pressure_pa
    ):
        raise ValueError("cross-section layer conditions must match atmosphere exactly")
    if amounts.shape != inputs.temperature_k.shape:
        raise ValueError("cross-section rows must align with atmospheric layers")
    rows = amounts[:, None] * prepared.cross_sections_m2_per_molecule
    return AbsorptionComponent(
        prepared.grid,
        inputs.species,
        inputs.isotopologue,
        inputs.kind,
        rows,
    )


@dataclass(frozen=True, eq=False)
class CrossSectionField:
    """Pre-materialised layer cross sections [m2/molecule] on a cm-1 axis.

    Temperature/pressure must *exactly* match the supplied atmosphere; only
    spectral interpolation is performed here. Continuum excludes any line-wing
    contributions already present in the supplied LineSet(s).
    Layer conditions have shape (n_layers,), cross sections
    (n_layers, n_points); numerical values are independent writable float64 copies.

    Attributes:
        species: Absorbing molecular species.
        isotopologue: Optional identity for isotope-specific cross sections.
        kind: Fixed cross-section or continuum-excluding-lines interpretation.
        grid: Materialised wavenumber support in cm-1.
        temperature_k: (layers,) exact table temperature conditions in kelvin.
        pressure_pa: (layers,) exact table pressure conditions in pascals.
        cross_sections_m2_per_molecule: (layers,points) absorption cross sections.
    """

    species: str
    isotopologue: str | None
    kind: Literal["cross_section", "continuum_excluding_lines"]
    grid: SpectralGrid
    temperature_k: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    cross_sections_m2_per_molecule: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        if not self.species or self.kind not in ("cross_section", "continuum_excluding_lines"):
            raise ValueError("cross sections need species and supported kind")
        if self.kind == "cross_section" and not self.isotopologue:
            raise ValueError("cross sections require isotopologue identity")
        require_wavenumber_grid(self.grid)
        grid = self.grid.points
        if len(grid) < 2:
            raise ValueError("cross-section interpolation requires at least two points")
        if (
            self.temperature_k.ndim != 1
            or self.temperature_k.size == 0
            or self.pressure_pa.shape != self.temperature_k.shape
        ):
            raise ValueError(
                "tabulated layer conditions must have matching nonempty shape (layers,)"
            )
        if self.cross_sections_m2_per_molecule.shape != (self.temperature_k.size, grid.size):
            raise ValueError("cross sections must have shape (layers, spectral points)")
        if (
            not np.all(np.isfinite(self.temperature_k))
            or not np.all(np.isfinite(self.pressure_pa))
            or np.any(self.temperature_k <= 0)
            or np.any(self.pressure_pa <= 0)
        ):
            raise ValueError("tabulated layer conditions must be finite and positive")
        if not np.all(np.isfinite(self.cross_sections_m2_per_molecule)) or np.any(
            self.cross_sections_m2_per_molecule < 0
        ):
            raise ValueError("cross section must be finite and nonnegative")
        for name in ("temperature_k", "pressure_pa", "cross_sections_m2_per_molecule"):
            object.__setattr__(self, name, np.copy(getattr(self, name)))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the tabulated cross section's sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


@dataclass(frozen=True, eq=False)
class PreparedCrossSections:
    """Fixed, interpolated cross sections, not a reference optical-depth basis.

    The table declares T/p, species and provenance; the retained target
    grid and writable float64 (n_layers, n_points) array contain m2/molecule.
    Safely immutable target coordinates are shared, not copied or reselected.
    Columns are supplied at evaluation.
    No line-profile, broadening or atmospheric-state update is approximated.

    Attributes:
        inputs: Original species, source grid and layer-condition declarations.
        grid: Target wavenumber grid after spectral interpolation.
        cross_sections_m2_per_molecule: (layers,points) interpolated cross sections.
    """

    inputs: CrossSectionField
    grid: SpectralGrid
    cross_sections_m2_per_molecule: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        values = self.cross_sections_m2_per_molecule
        if (
            values.shape != (self.inputs.temperature_k.size, self.wavenumbers_cm1.size)
            or values.size == 0
        ):
            raise ValueError("prepared cross sections must have shape (layers, points)")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("prepared cross section must be finite and nonnegative")
        object.__setattr__(self, "cross_sections_m2_per_molecule", np.copy(values))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the prepared cross section's sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points
