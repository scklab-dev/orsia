"""Total vertical gas absorption and explicit upstream contribution composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid
from orsia_radiative.spectroscopy.components import (
    AbsorptionComponent,
    validate_absorption_freshness,
)


def compose_absorption(
    atmosphere: Atmosphere,
    source: SolarSource,
    components: tuple[AbsorptionComponent, ...],
    *,
    reference_atmospheres: tuple[Atmosphere, ...] | None = None,
) -> Absorption:
    """Validate and sum current contributions before entering radiative transfer.

    The atmosphere defines layers/state; ``source.irradiance.wavenumbers_cm1``
    defines the exact spectral points in cm-1. No resampling is performed. Only
    the source's grid is used or retained, not its irradiance, direction or ID;
    same-grid illumination changes do not invalidate the total.

    Reuse components only when their actual T/p, column and broadener dependencies
    remain valid. Supply ``reference_atmospheres`` in component order to check
    separately retained evaluation atmospheres. Without them, physical freshness
    is the caller's responsibility. No abundance-only
    scaling, fixed background, cache or dependency graph is inferred. Inputs are
    never mutated; the result owns writable total depths, not component arrays.
    Empty components produce zero absorption on the atmosphere/source support.

    Args:
        atmosphere: Absorber species and bottom-to-top atmospheric layers.
        source: Solar irradiance providing the target spectral grid in cm-1.
        components: Species-resolved absorption terms to sum on that support.
        reference_atmospheres: Optional original evaluation atmospheres, one per
            component, for an explicit dependency freshness check.

    Returns:
        Total dimensionless (layers,points) absorption with owned writable depths.

    Raises:
        ValueError: Layer/grid support, species, overlap or reference freshness fails.
    """
    return compose_absorption_on_grid(
        atmosphere, source.irradiance.grid, components, reference_atmospheres=reference_atmospheres
    )


def compose_absorption_on_grid(
    atmosphere: Atmosphere,
    grid: SpectralGrid,
    components: tuple[AbsorptionComponent, ...],
    *,
    reference_atmospheres: tuple[Atmosphere, ...] | None = None,
) -> Absorption:
    """Compose absorption on explicit sampling, independently of illumination.

    The caller materialises one positive increasing float64 cm-1 point grid.
    Layers, explicit freshness, overlap checks and ownership are identical to
    ``compose_absorption``. No interpolation or source object is introduced.
    Empty components produce zero depth on the supplied atmosphere/grid.

    Args:
        atmosphere: Absorber identities and bottom-to-top layer support.
        grid: Explicit evaluation wavenumber grid in cm-1.
        components: Absorption contributions evaluated at the grid points.
        reference_atmospheres: Optional per-component evaluation atmospheres for
            comparison against current physical dependencies.

    Returns:
        Total dimensionless (layers,points) absorption with fresh depths.

    Raises:
        ValueError: Incompatible grid/layers, duplicate or undeclared species,
            overlapping line/table terms, or stale reference dependencies.
    """
    require_wavenumber_grid(grid)
    wavenumbers_cm1 = grid.points
    n = len(atmosphere.pressure_pa)
    declared_species = {gas.species for gas in atmosphere.absorbers}
    seen: set[tuple[str, str | None, str]] = set()
    kinds: dict[str, set[str]] = {}
    if reference_atmospheres is not None and len(reference_atmospheres) != len(components):
        raise ValueError("reference atmospheres must align with absorption components")
    for index, component in enumerate(components):
        if not component.grid.same_points(grid) or len(component.optical_depth) != n:
            raise ValueError("absorption axis and layer order must match the atmosphere")
        # Lorentz and Voigt are alternative profiles of the same supplied lines,
        # not independent contributors for one species/isotopologue.
        contribution_kind = (
            "lines" if component.kind in ("lorentz_lines", "voigt_lines") else component.kind
        )
        key = (component.species, component.isotopologue, contribution_kind)
        if key in seen or component.species not in declared_species:
            raise ValueError("duplicate component or undeclared absorber species")
        if reference_atmospheres is not None:
            validate_absorption_freshness(
                atmosphere, component, reference_atmosphere=reference_atmospheres[index]
            )
        seen.add(key)
        kinds.setdefault(component.species, set()).add(component.kind)
    if any(
        "cross_section" in entries and bool(entries & {"lorentz_lines", "voigt_lines"})
        for entries in kinds.values()
    ):
        raise ValueError("lines and complete cross section overlap for one species")
    for component in components:
        if (
            component.kind == "continuum_excluding_lines"
            and component.line_wing_convention == "mt_ckd_local_25"
            and any(
                other.species == component.species
                and other.kind in ("lorentz_lines", "voigt_lines")
                and other.line_wing_convention != "mt_ckd_local_25"
                for other in components
            )
        ):
            raise ValueError(
                "MT_CKD water continuum requires local 25 cm-1 pedestal-subtracted lines"
            )
    total = np.zeros((n, wavenumbers_cm1.size), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for component in components:
            total += component.optical_depth
    # The public constructor checks the newly summed depths, including overflow.
    return Absorption(grid, total)


@dataclass(frozen=True, eq=False)
class Absorption:
    """Total dimensionless vertical absorption [bottom-to-top layer, spectral point].

    Grid (cm-1) retains original evaluation support, shared when already safely
    immutable, otherwise snapshotted. Depths are independent writable float64 copies.
    Contributor identities and atmosphere references are deliberately not stored
    on this physical value. Composition validates its inputs before constructing
    the total; external suppliers are responsible for physical consistency.
    Aerosol absorption already included in aerosol extinction must not be added.

    Attributes:
        grid: Evaluated spectral sampling in wavenumbers (cm-1).
        optical_depth: (bottom-to-top layer, spectral point) absorption depth.
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
            raise ValueError("absorption must have shape (layers, spectral points)")
        if not np.all(np.isfinite(self.optical_depth)):
            raise ValueError("absorption optical depth must be finite; possible overflow")
        if np.any(self.optical_depth < 0):
            raise ValueError("absorption must be finite and nonnegative")
        object.__setattr__(self, "optical_depth", np.copy(self.optical_depth))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the absorption grid's canonical sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points
