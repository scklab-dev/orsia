"""Pure materialisation of bounded line databases and partition-function tables."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.spectral.grid import SpectralGrid, wavenumber_grid
from orsia_radiative.spectroscopy.lines import Line, LineSet


@dataclass(frozen=True)
class LineDatabase:
    """One molecule/local-isotope's lines within declared centre coverage [cm-1].

    Empty lines are valid. Coverage describes completeness supplied by the caller,
    not the first/last line centres or authenticated provider coverage. Line
    strengths include isotope abundance relative to the whole-species column.

    Attributes:
        lines: Supplied lines for one molecule/local isotopologue.
        molecule_id: Positive molecular catalogue identifier.
        isotopologue_id: Positive molecule-local isotope identifier.
        coverage_cm1: Caller-declared inclusive line-centre support in cm-1.
    """

    lines: tuple[Line, ...]
    molecule_id: int
    isotopologue_id: int
    coverage_cm1: tuple[float, float]

    def __post_init__(self) -> None:
        lo, hi = self.coverage_cm1
        if self.molecule_id <= 0 or self.isotopologue_id <= 0:
            raise ValueError("molecule and local isotope identifiers must be positive")
        if not all(map(isfinite, (lo, hi))) or lo < 0 or hi < lo:
            raise ValueError("line coverage must be finite and increasing")
        if any(not lo <= line.centre_cm1 <= hi for line in self.lines):
            raise ValueError("line centre outside declared coverage")


@dataclass(frozen=True, eq=False)
class PartitionFunctionTable:
    """Dimensionless Q at increasing temperatures [K] for one molecule/local isotope.

    Both arrays have shape (n_knots,), n_knots >= 2; values are positive finite
    float64 independent writable copies. No file or provider identity is stored.

    Attributes:
        molecule_id: Positive molecular catalogue identifier.
        isotopologue_id: Positive molecule-local isotope identifier.
        temperatures_k: Increasing sampled temperatures in kelvin.
        partition_sums: Matching dimensionless partition sums.
    """

    molecule_id: int
    isotopologue_id: int
    temperatures_k: NDArray[np.float64]
    partition_sums: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        knots = self.temperatures_k
        sums = self.partition_sums
        if (
            knots.ndim != 1
            or sums.shape != knots.shape
            or not np.all(np.isfinite(knots))
            or not np.all(np.isfinite(sums))
        ):
            raise ValueError("partition temperatures and sums must be matching finite vectors")
        if (
            self.molecule_id <= 0
            or self.isotopologue_id <= 0
            or knots.size < 2
            or np.any(knots <= 0)
            or np.any(np.diff(knots) <= 0)
            or np.any(sums <= 0)
        ):
            raise ValueError("partition table needs positive isotope, increasing knots and sums")
        object.__setattr__(self, "temperatures_k", np.copy(knots))
        object.__setattr__(self, "partition_sums", np.copy(sums))


class EmptyLineSelection(ValueError):
    """A declared covered line selection has no entries."""


def evaluate_partition_function(
    table: PartitionFunctionTable, temperatures_k: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return dimensionless Q by linear T interpolation, without extrapolation.

    Input is a nonempty finite float64 vector; output is fresh writable storage.
    This approximation is not HAPI/TIPS interpolation; refine knots explicitly.

    Args:
        table: Isotope-specific temperature knots and dimensionless partition sums.
        temperatures_k: Nonempty temperature vector in kelvin within table support.

    Returns:
        Fresh writable dimensionless partition sums shaped like temperatures_k.

    Raises:
        ValueError: Temperatures have invalid shape/values or leave knot support.
    """
    knots = table.temperatures_k
    if (
        temperatures_k.ndim != 1
        or temperatures_k.size == 0
        or not np.all(np.isfinite(temperatures_k))
        or np.any(temperatures_k < knots[0])
        or np.any(temperatures_k > knots[-1])
    ):
        raise ValueError("temperature outside partition-table support or invalid shape")
    return np.interp(temperatures_k, knots, table.partition_sums)


def line_set(
    database: LineDatabase,
    partition_table: PartitionFunctionTable,
    atmosphere: Atmosphere,
    grid: SpectralGrid | NDArray[np.float64],
    *,
    species: str,
    molecular_mass_kg: float,
    pressure_range_pa: tuple[float, float],
    wing_cutoff_cm1: float,
    subtract_lorentz_pedestal: bool = False,
) -> LineSet:
    """Build layer-specific Voigt inputs from materialised scientific tables.

    Require centre coverage beyond the target by cutoff plus the maximum absolute
    pressure shift over declared pressure support. No files, downloads, identity
    generation or caches. Mass is kg/molecule; pressure Pa; grid/cutoff cm-1.
    Q(296 K)/Q(T_layer) uses explicit linear interpolation. Empty covered selection
    raises EmptyLineSelection, distinct from insufficient coverage.

    Args:
        database: Line strengths and declared centre coverage for one isotopologue.
        partition_table: Matching isotope partition sums at temperature knots in K.
        atmosphere: Layer temperatures, pressures and declared absorber species.
        grid: Requested wavenumber grid or points in cm-1.
        species: Atmospheric absorber identity for the selected lines.
        molecular_mass_kg: Positive molecular mass in kg per molecule for Voigt widths.
        pressure_range_pa: Inclusive pressure support in pascals.
        wing_cutoff_cm1: Positive line-wing radius in cm-1, included in coverage checks.
        subtract_lorentz_pedestal: Apply the H2O 25 cm-1 MT_CKD line convention.

    Returns:
        Selected line set with layer partition ratios and declared pressure support.

    Raises:
        EmptyLineSelection: The covered spectral interval contains zero selected lines.
        ValueError: Isotope, species, pressure, temperature, wing or coverage conflicts.
    """
    points = wavenumber_grid(grid).points
    lo, hi = pressure_range_pa
    if (
        not all(map(isfinite, (lo, hi, molecular_mass_kg, wing_cutoff_cm1)))
        or lo <= 0
        or hi < lo
        or molecular_mass_kg <= 0
        or wing_cutoff_cm1 <= 0
        or np.any(atmosphere.pressure_pa < lo)
        or np.any(atmosphere.pressure_pa > hi)
    ):
        raise ValueError("invalid mass/cutoff or atmosphere outside pressure support")
    if species not in {gas.species for gas in atmosphere.absorbers}:
        raise ValueError("materialisation requires a declared atmospheric absorber")
    if (database.molecule_id, database.isotopologue_id) != (
        partition_table.molecule_id,
        partition_table.isotopologue_id,
    ):
        raise ValueError("line database and partition table must describe the same isotope")
    if subtract_lorentz_pedestal and (species != "H2O" or wing_cutoff_cm1 != 25.0):
        raise ValueError("MT_CKD local pedestal requires H2O and a 25 cm-1 cutoff")
    margin = (
        wing_cutoff_cm1
        + max((abs(line.pressure_shift_ref_cm1) for line in database.lines), default=0.0)
        * hi
        / 101325.0
    )
    interval = (float(points[0] - margin), float(points[-1] + margin))
    if not database.coverage_cm1[0] <= interval[0] <= interval[1] <= database.coverage_cm1[1]:
        raise ValueError("requested spectrum/wing margin outside declared line coverage")
    selected = tuple(
        line for line in database.lines if interval[0] <= line.centre_cm1 <= interval[1]
    )
    if not selected:
        raise EmptyLineSelection("covered line selection is empty")
    ratios = evaluate_partition_function(partition_table, np.array([296.0]))[
        0
    ] / evaluate_partition_function(partition_table, atmosphere.temperature_k)
    knots = partition_table.temperatures_k
    return LineSet(
        species,
        str(database.isotopologue_id),
        selected,
        ratios,
        (float(knots[0]), float(knots[-1])),
        pressure_range_pa,
        molecular_mass_kg=molecular_mass_kg,
        wing_cutoff_cm1=wing_cutoff_cm1,
        subtract_lorentz_pedestal=subtract_lorentz_pedestal,
    )
