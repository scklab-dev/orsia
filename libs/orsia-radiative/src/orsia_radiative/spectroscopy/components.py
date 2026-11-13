"""Species-resolved absorption contributions and explicit dependency freshness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import LayerAtmosphere
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


def values_equal(left: object, right: object) -> bool:
    """Compare scientific dependency values, including nested arrays and tuples.

    Args:
        left: First scalar, NumPy array or nested tuple of values.
        right: Candidate value with matching structure and elements.

    Returns:
        True when array elements and nested tuple entries compare exactly;
        otherwise the scalar equality result.
    """
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(right, np.ndarray):
        return False
    if isinstance(left, tuple):
        return (
            isinstance(right, tuple)
            and len(left) == len(right)
            and all(values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def validate_absorption_freshness(
    atmosphere: LayerAtmosphere,
    component: AbsorptionComponent,
    *,
    reference_atmosphere: LayerAtmosphere,
) -> None:
    """Compare dependencies with a separately retained evaluation atmosphere.

    The reference must be the atmosphere used to compute the component. This
    explicit check neither establishes input provenance nor validates a supplier's
    calculation. No ID-based or abundance-only shortcut is used.

    Args:
        atmosphere: Current gas columns, layer conditions and broadener amounts.
        component: Contribution whose species and model select dependencies.
        reference_atmosphere: LayerAtmosphere used at evaluation, retained by the caller.

    Raises:
        ValueError: Required species is absent or relevant dependencies changed.
    """
    if not values_equal(
        _absorption_dependencies(
            atmosphere, component.species, component.kind, component.includes_water_continuum
        ),
        _absorption_dependencies(
            reference_atmosphere,
            component.species,
            component.kind,
            component.includes_water_continuum,
        ),
    ):
        raise ValueError(
            f"stale absorption for species {component.species}; recompute its dependencies"
        )


@dataclass(frozen=True, eq=False)
class AbsorptionComponent:
    """Vertical optical depths, ordered [bottom-to-top layer][spectral point].

    This physical contribution contains no input provenance or atmosphere
    reference. Callers retain evaluation dependencies separately and may use
    ``validate_absorption_freshness`` explicitly before reuse.
    Optical depth has shape (n_layers, n_points), in independent writable float64 storage.
    The grid retains original evaluation support, sharing an already safely
    immutable numerical grid without depending on source illumination/direction.

    Attributes:
        grid: Evaluated wavenumber sampling in cm-1.
        species: Gas identity for this absorption contribution.
        isotopologue: Optional local isotopologue identity for the selected source.
        kind: Line, cross-section or continuum absorption model identity.
        optical_depth: (layers,points) vertical dimensionless absorption depths.
        line_wing_convention: Optional local line cutoff and pedestal convention.
        includes_water_continuum: Explicit inclusion for a supplied H2O cross section;
            None means unknown. Separate continuum requires a declared exclusion.
    """

    grid: SpectralGrid
    species: str
    isotopologue: str | None
    kind: Literal["lorentz_lines", "voigt_lines", "cross_section", "continuum_excluding_lines"]
    optical_depth: NDArray[np.float64]
    line_wing_convention: Literal["unbounded", "truncated", "mt_ckd_local_25"] | None = None
    includes_water_continuum: bool | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        if (
            self.optical_depth.ndim != 2
            or self.optical_depth.shape[0] == 0
            or self.optical_depth.shape[1] != self.wavenumbers_cm1.size
        ):
            raise ValueError("optical depth must have shape (layers, spectral points)")
        if not self.species:
            raise ValueError("absorption requires species and layer values")
        if self.isotopologue == "":
            raise ValueError("isotopologue identity cannot be empty")
        if self.kind not in (
            "lorentz_lines",
            "voigt_lines",
            "cross_section",
            "continuum_excluding_lines",
        ):
            raise ValueError("unsupported absorption kind")
        if not np.all(np.isfinite(self.optical_depth)) or np.any(self.optical_depth < 0):
            raise ValueError("optical depth must be finite and nonnegative")
        object.__setattr__(self, "optical_depth", np.copy(self.optical_depth))

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the component's canonical spectral sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


def validate_water_continuum_pairing(
    includes_water_continuum: bool | None,
    table_line_convention: str | None,
    continuum_line_convention: str | None,
) -> None:
    """Require explicit nonoverlap when combining a water table with continuum.

    Args:
        includes_water_continuum: Declared inclusion in the fixed H2O table, or unknown.
        table_line_convention: Supplied table's local-line cutoff/pedestal convention.
        continuum_line_convention: Convention required by the supplied continuum.

    Raises:
        ValueError: Continuum inclusion is true/unknown or the required line convention differs.
    """
    if includes_water_continuum is not False:
        raise ValueError("water cross-section tables must explicitly exclude separate continuum")
    if continuum_line_convention is not None and table_line_convention != continuum_line_convention:
        raise ValueError("water table and continuum require matching local line conventions")


def _gas(atmosphere: LayerAtmosphere, species: str) -> tuple[NDArray[np.float64], str]:
    for gas in atmosphere.absorbers:
        if gas.species == species:
            return gas.columns_molecules_m2, gas.air_broadener
    raise ValueError(f"no declared absorber for species {species}")


def _absorption_dependencies(
    atmosphere: LayerAtmosphere, species: str, kind: str, includes_water_continuum: bool | None
) -> tuple[object, ...]:
    amounts, broadener = _gas(atmosphere, species)
    common = (
        atmosphere.height_boundaries_m,
        atmosphere.pressure_boundaries_pa,
        atmosphere.pressure_pa,
        atmosphere.temperature_k,
        amounts,
    )
    if kind == "continuum_excluding_lines" or includes_water_continuum:
        return (
            *common,
            atmosphere.total_air_columns_molecules_m2,
            atmosphere.water_vapour_columns_molecules_m2,
            atmosphere.dry_air_columns_molecules_m2,
        )
    if kind not in ("lorentz_lines", "voigt_lines"):
        return common
    return (
        *common,
        broadener,
        atmosphere.total_air_columns_molecules_m2,
        atmosphere.dry_air_columns_molecules_m2 if broadener == "dry_air" else (),
    )
