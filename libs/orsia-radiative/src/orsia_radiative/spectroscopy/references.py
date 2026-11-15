"""Reusable isotope spectroscopy with exact, potentially disconnected line support."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

import numpy as np

from orsia_radiative.atmosphere import LayerAtmosphere
from orsia_radiative.spectral.grid import SpectralGrid, convert_spectral_grid
from orsia_radiative.spectroscopy.components import AbsorptionComponent
from orsia_radiative.spectroscopy.cross_sections import LayerCrossSections
from orsia_radiative.spectroscopy.line_tables import (
    PartitionFunctionTable,
    SpectralLineTable,
    line_set,
)
from orsia_radiative.spectroscopy.lines import LineSet, line_absorption, line_cross_sections


@dataclass(frozen=True, eq=False)
class LineSpectroscopy:
    """Transitions, partition sums and mass for one molecular isotopologue.

    Each database declares a separate inclusive complete line-centre interval.
    Intervals must be ordered and disjoint; gaps carry no completeness claim.
    Intensities include natural isotopic abundance relative to whole-species columns
    at 296 K, with widths and shifts referenced to 101325 Pa. Partition samples
    are independent read-only float64 copies; lines are immutable scalar records.

    Attributes:
        species: Atmospheric whole-species identity.
        databases: Nonempty ordered tuple of disjoint complete line intervals, cm-1.
        partition_table: Matching molecule/local-isotope Q(T) samples, K and dimensionless.
        molecular_mass_kg: Positive isotope mass in kg per molecule.
        intensity_convention: Whole-species natural-abundance intensity convention.
    """

    species: str
    databases: tuple[SpectralLineTable, ...]
    partition_table: PartitionFunctionTable
    molecular_mass_kg: float
    intensity_convention: Literal["hitran_natural_abundance"] = "hitran_natural_abundance"
    __hash__ = None

    def __post_init__(self) -> None:
        if not self.species or not self.databases:
            raise ValueError("line spectroscopy requires species and complete intervals")
        if not isfinite(self.molecular_mass_kg) or self.molecular_mass_kg <= 0:
            raise ValueError("molecular mass must be positive and finite")
        if self.intensity_convention != "hitran_natural_abundance":
            raise ValueError("unsupported line intensity convention")
        table = self.partition_table
        previous = -1.0
        for database in self.databases:
            if (database.molecule_id, database.isotopologue_id) != (
                table.molecule_id,
                table.isotopologue_id,
            ):
                raise ValueError("line intervals and partition table must share isotope identity")
            if database.coverage_cm1[0] <= previous:
                raise ValueError("line intervals must be ordered and disjoint")
            previous = database.coverage_cm1[1]
        copied = PartitionFunctionTable(
            table.molecule_id, table.isotopologue_id, table.temperatures_k, table.partition_sums
        )
        for name in ("temperatures_k", "partition_sums"):
            array = getattr(copied, name)
            object.__setattr__(copied, name, np.frombuffer(array.tobytes(), dtype=array.dtype))
        object.__setattr__(self, "partition_table", copied)


def evaluate_line_absorption(
    atmosphere: LayerAtmosphere,
    grid: SpectralGrid,
    reference: LineSpectroscopy,
    *,
    profile: Literal["voigt", "lorentz"],
    pressure_range_pa: tuple[float, float],
    wing_cutoff_cm1: float,
    subtract_lorentz_pedestal: bool,
) -> AbsorptionComponent:
    """Evaluate reusable spectroscopy with explicit pressure and local-wing physics.

    Select a single complete interval covering the grid, cutoff and maximum
    pressure shift. Disconnected intervals never imply support across a gap.
    Q(296 K)/Q(T) uses the existing linear partition interpolation and kernels.

    Args:
        atmosphere: Current bottom-to-top layer conditions, columns and broadener pools.
        grid: Increasing target wavenumbers in cm-1.
        reference: Isotope transitions, Q(T), exact support and mass.
        profile: Explicit thermal Voigt or pressure-only Lorentz profile.
        pressure_range_pa: Inclusive declared pressure support in Pa.
        wing_cutoff_cm1: Positive finite wing radius in cm-1, without renormalization.
        subtract_lorentz_pedestal: Select H2O local 25 cm-1 MT_CKD pedestal subtraction.

    Returns:
        Existing species-resolved absorption component with owned writable depths.

    Raises:
        EmptyLineSelection: A covered selection has no transitions.
        ValueError: Grid, wing margin, identity, temperatures, pressures or profile are invalid.
    """
    inputs = _reference_lines(
        atmosphere,
        grid,
        reference,
        profile,
        pressure_range_pa,
        wing_cutoff_cm1,
        subtract_lorentz_pedestal,
    )
    return line_absorption(atmosphere, grid, inputs)


def evaluate_line_cross_sections(
    atmosphere: LayerAtmosphere,
    grid: SpectralGrid,
    reference: LineSpectroscopy,
    *,
    profile: Literal["voigt", "lorentz"],
    pressure_range_pa: tuple[float, float],
    wing_cutoff_cm1: float,
    subtract_lorentz_pedestal: bool,
) -> LayerCrossSections[np.float64]:
    """Prepare reusable cross sections without dividing by absorber abundance.

    Args:
        atmosphere: Explicit layer temperatures, pressures, height interfaces and
            actual gas/air columns defining self and foreign broadening fractions.
            Zero target columns are allowed with valid broadening conditions.
        grid: Increasing target samples in nm or cm-1, with declared support.
        reference: Isotope transitions, partition sums, mass and complete intervals.
        profile: Explicit Voigt or Lorentz profile.
        pressure_range_pa: Inclusive pressure validity in Pa.
        wing_cutoff_cm1: Positive finite wing radius, cm-1.
        subtract_lorentz_pedestal: Select local H2O 25 cm-1 pedestal subtraction.

    Returns:
        Immutable layer cross sections in m2/molecule, with exact T/p and heights.
        Coefficients apply at the supplied composition; reuse at another broadening
        composition requires reevaluation. No water continuum is included.

    Raises:
        EmptyLineSelection: A complete covered interval contains no selected lines.
        ValueError: Physical conditions, reference support or profile choices fail.
    """
    spectral = grid if grid.coordinate == "wavenumber_cm1" else convert_spectral_grid(grid)
    inputs = _reference_lines(
        atmosphere,
        spectral,
        reference,
        profile,
        pressure_range_pa,
        wing_cutoff_cm1,
        subtract_lorentz_pedestal,
    )
    values = line_cross_sections(atmosphere, spectral, inputs)
    if grid.coordinate == "wavelength_nm":
        values = values[:, ::-1]
    return LayerCrossSections(
        reference.species,
        inputs.isotopologue,
        "cross_section",
        grid,
        atmosphere.temperature_k,
        atmosphere.pressure_pa,
        values,
        height_edges_m=atmosphere.height_boundaries_m,
        line_wing_convention="mt_ckd_local_25" if subtract_lorentz_pedestal else "truncated",
        includes_water_continuum=False,
    )


def _reference_lines(
    atmosphere: LayerAtmosphere,
    grid: SpectralGrid,
    reference: LineSpectroscopy,
    profile: Literal["voigt", "lorentz"],
    pressure_range_pa: tuple[float, float],
    wing_cutoff_cm1: float,
    subtract_lorentz_pedestal: bool,
) -> LineSet:
    if profile not in ("voigt", "lorentz"):
        raise ValueError("profile must be voigt or lorentz")
    # Try only intervals that can contain the requested grid; propagate physical
    # errors from materialisation rather than interpreting them as empty selections.
    for database in reference.databases:
        if (
            database.coverage_cm1[0]
            <= grid.points[0]
            <= grid.points[-1]
            <= database.coverage_cm1[1]
        ):
            inputs = line_set(
                database,
                reference.partition_table,
                atmosphere,
                grid,
                species=reference.species,
                molecular_mass_kg=reference.molecular_mass_kg,
                pressure_range_pa=pressure_range_pa,
                wing_cutoff_cm1=wing_cutoff_cm1,
                subtract_lorentz_pedestal=subtract_lorentz_pedestal,
            )
            if profile == "lorentz":
                inputs = replace(inputs, molecular_mass_kg=None)
            return inputs
    raise ValueError("requested grid outside declared complete line intervals")
