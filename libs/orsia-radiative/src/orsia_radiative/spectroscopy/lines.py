"""Isolated Lorentz and thermal Voigt lines from materialised inputs.

Output optical depths are vertical, dimensionless and bottom-to-top.
No database, provider or solver access is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, pi, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy import constants
from scipy.special import voigt_profile

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.spectral.grid import SpectralGrid, wavenumber_grid
from orsia_radiative.spectroscopy.components import AbsorptionComponent, _gas

_SECOND_RADIATION_CONSTANT_CM_K = 100 * constants.h * constants.c / constants.k


def line_absorption(
    atmosphere: Atmosphere, wavenumbers_cm1: NDArray[np.float64] | SpectralGrid, inputs: LineSet
) -> AbsorptionComponent:
    """Evaluate isolated Lorentz or explicit thermal Voigt lines.

    Self pressure is p * target column/total-air column. The foreign-air pool
    is either dry air (less target if it is dry) or total air less target.
    Unmodelled foreign gas in dry-air mode (e.g. water) is not folded into air.
    With LineSet.molecular_mass_kg=None (default), retain Lorentz profiles.
    A positive mass in kg per molecule enables Voigt, using Gaussian standard
    deviation centre_cm1*sqrt(k*T/mass)/c (unshifted transition frequency) and
    the same pressure-broadened Lorentz HWHM. Mass is never inferred from species
    or fetched. The default has no wing cutoff. An explicit wing_cutoff_cm1
    truncates without renormalisation. subtract_lorentz_pedestal selects the
    MT_CKD local H2O convention: +/-25 cm-1 about the shifted centre, subtract
    the Lorentz value there (also for Voigt), no renormalisation. This is not
    line mixing, a complete non-water continuum convention, or a continuum.
    Inputs are borrowed; output owns writable float64 layer/point depths.

    Args:
        atmosphere: Bottom-to-top gas columns, pressure, temperature and broadener pools.
        wavenumbers_cm1: Target spectral grid or wavenumber samples in cm-1.
        inputs: Materialised line parameters, validity ranges and profile choices.

    Returns:
        Species-resolved dimensionless (layers,points) absorption component,
        with independently writable optical depths.

    Raises:
        ValueError: Species or broadener is incompatible, layer conditions leave
            declared support, or evaluated line strengths/widths overflow.
    """
    sampling = wavenumber_grid(wavenumbers_cm1)
    wavenumbers_cm1 = sampling.points
    grid = wavenumbers_cm1
    amounts, broadener = _gas(atmosphere, inputs.species)
    n = len(atmosphere.pressure_pa)
    if len(inputs.partition_ratios) != n:
        raise ValueError("partition ratios must match atmospheric layers")
    rows = np.empty((n, grid.size), dtype=np.float64)
    for i, (temperature, pressure, amount) in enumerate(
        zip(atmosphere.temperature_k, atmosphere.pressure_pa, amounts, strict=True)
    ):
        if not inputs.temperature_range_k[0] <= temperature <= inputs.temperature_range_k[1]:
            raise ValueError("layer temperature outside declared line validity")
        if not inputs.pressure_range_pa[0] <= pressure <= inputs.pressure_range_pa[1]:
            raise ValueError("layer pressure outside declared line validity")
        self_fraction = amount / atmosphere.total_air_columns_molecules_m2[i]
        foreign_fraction = (
            (
                atmosphere.dry_air_columns_molecules_m2[i]
                - (amount if inputs.species != "H2O" else 0)
            )
            / atmosphere.total_air_columns_molecules_m2[i]
            if broadener == "dry_air"
            else 1 - self_fraction
        )
        if foreign_fraction < 0:
            raise ValueError("target gas exceeds declared broadener pool")
        values = rows[i]
        values.fill(0)
        # One reusable block buffer per layer, rather than a lines x layers x grid cube.
        profile = np.empty(min(grid.size, 8192), dtype=np.float64)
        for line in inputs.lines:
            c2 = _SECOND_RADIATION_CONSTANT_CM_K
            try:
                strength = (
                    line.strength_ref_cm_per_molecule
                    * inputs.partition_ratios[i]
                    * exp(
                        -c2
                        * line.lower_energy_cm1
                        * (1 / temperature - 1 / inputs.reference_temperature_k)
                    )
                    * (
                        expm1(-c2 * line.centre_cm1 / temperature)
                        / expm1(-c2 * line.centre_cm1 / inputs.reference_temperature_k)
                    )
                )
            except OverflowError as error:
                raise ValueError("line strength overflows supported numerical domain") from error
            if not isfinite(strength):
                raise ValueError("line strength overflows supported numerical domain")
            gamma = (
                (inputs.reference_temperature_k / temperature) ** line.temperature_exponent
                * pressure
                / inputs.reference_pressure_pa
                * (
                    line.air_hwhm_ref_cm1 * foreign_fraction
                    + line.self_hwhm_ref_cm1 * self_fraction
                )
            )
            if gamma <= 0 or not isfinite(gamma):
                raise ValueError("line width outside supported Lorentz domain")
            centre = (
                line.centre_cm1
                + line.pressure_shift_ref_cm1 * pressure / inputs.reference_pressure_pa
            )
            # cm/molecule * cm * 1e-4 m2/cm2 * molecule/m2 => dimensionless.
            amplitude = amount * 1e-4 * strength
            sigma = None
            if inputs.molecular_mass_kg is not None:
                sigma = (
                    line.centre_cm1
                    / constants.c
                    * sqrt(constants.k * temperature / inputs.molecular_mass_kg)
                )
                if sigma <= 0 or not isfinite(sigma):
                    raise ValueError("Doppler width outside supported Voigt domain")
            cutoff = inputs.wing_cutoff_cm1
            first = 0 if cutoff is None else int(np.searchsorted(grid, centre - cutoff))
            last = (
                grid.size
                if cutoff is None
                else int(np.searchsorted(grid, centre + cutoff, side="right"))
            )
            pedestal = (
                gamma / (pi * (25.0**2 + gamma**2)) if inputs.subtract_lorentz_pedestal else 0.0
            )
            for start in range(first, last, profile.size):
                stop = min(start + profile.size, last)
                block = profile[: stop - start]
                np.subtract(grid[start:stop], centre, out=block)
                if sigma is None:
                    np.square(block, out=block)
                    block += gamma**2
                    np.divide(amplitude * gamma / pi, block, out=block)
                else:
                    voigt_profile(block, sigma, gamma, out=block)
                    block *= amplitude
                if inputs.subtract_lorentz_pedestal:
                    block -= pedestal * amplitude
                    if np.any(block < -1e-14 * pedestal * amplitude):
                        raise ValueError(
                            "negative local line profile outside supported pedestal domain"
                        )
                    np.maximum(block, 0.0, out=block)
                values[start:stop] += block
    return AbsorptionComponent(
        sampling,
        inputs.species,
        inputs.isotopologue,
        "lorentz_lines" if inputs.molecular_mass_kg is None else "voigt_lines",
        rows,
        line_wing_convention=(
            "mt_ckd_local_25"
            if inputs.subtract_lorentz_pedestal
            else "unbounded"
            if inputs.wing_cutoff_cm1 is None
            else "truncated"
        ),
    )


@dataclass(frozen=True)
class Line:
    """Isolated line: centre/lower energy in cm-1, S(ref) in cm/molecule.

    Air/self Lorentz HWHM and pressure shift are in cm-1 at reference pressure;
    the shift scales with total layer pressure, not with the broadener pool.
    The dimensionless temperature exponent is signed and finite; a negative
    exponent does not imply negative broadening.
    Intensity must already include the declared isotopologue abundance relative
    to the whole-species molecular column. This is NOT per-isotopologue column.

    Attributes:
        centre_cm1: Reference spectral line centre in cm-1.
        strength_ref_cm_per_molecule: Reference intensity including isotope abundance.
        lower_energy_cm1: Lower-state term energy in cm-1.
        air_hwhm_ref_cm1: Air Lorentz half-width at reference pressure, cm-1.
        self_hwhm_ref_cm1: Self Lorentz half-width at reference pressure, cm-1.
        temperature_exponent: Signed dimensionless temperature scaling exponent.
        pressure_shift_ref_cm1: Signed centre shift at reference pressure, cm-1.
    """

    centre_cm1: float
    strength_ref_cm_per_molecule: float
    lower_energy_cm1: float
    air_hwhm_ref_cm1: float
    self_hwhm_ref_cm1: float
    temperature_exponent: float
    pressure_shift_ref_cm1: float = 0.0

    def __post_init__(self) -> None:
        for name in ("centre_cm1", "air_hwhm_ref_cm1", "self_hwhm_ref_cm1"):
            _positive(getattr(self, name), name)
        for name in ("strength_ref_cm_per_molecule", "lower_energy_cm1"):
            _positive(getattr(self, name), name, zero=True)
        # HITRAN n_air is a signed exponent, not a width; negative values
        # still produce positive broadening at positive temperatures.
        if not isfinite(self.temperature_exponent):
            raise ValueError("temperature exponent must be finite")
        if not isfinite(self.pressure_shift_ref_cm1):
            raise ValueError("pressure shift must be finite")


@dataclass(frozen=True, eq=False)
class LineSet:
    """Physical lines and materialised Q(T_ref)/Q(T_layer) for each layer.

    Isotopologue-specific partition ratios are input, never silently fetched or
    approximated. Temperature and pressure ranges are declared by the caller.
    Partition ratios are writable float64 copies with shape (n_layers,).
    Validity ranges remain fixed (minimum, maximum) pairs, not numerical buffers.
    Optional molecular_mass_kg is the isotopologue mass in kg per molecule;
    supplied mass enables thermal Voigt profiles, None preserves Lorentz.

    Attributes:
        species: Name of the absorbing gas with whole-species column convention.
        isotopologue: Local isotope identity for lines and partition ratios.
        lines: Supplied line parameters for this isotopologue.
        partition_ratios: (layers,) reference-to-layer partition function ratios.
        temperature_range_k: Declared covered layer-temperature interval in kelvin.
        pressure_range_pa: Declared covered layer-pressure interval in pascals.
        reference_temperature_k: Reference line-intensity temperature in kelvin.
        reference_pressure_pa: Reference line-width pressure in pascals.
        molecular_mass_kg: Optional isotopologue mass in kg per molecule for Voigt.
        wing_cutoff_cm1: Optional finite line-wing half-width in cm-1.
        subtract_lorentz_pedestal: Enable declared local-wing pedestal removal.
    """

    species: str
    isotopologue: str
    lines: tuple[Line, ...]
    partition_ratios: NDArray[np.float64]
    temperature_range_k: tuple[float, float]
    pressure_range_pa: tuple[float, float]
    reference_temperature_k: float = 296.0
    reference_pressure_pa: float = 101325.0
    molecular_mass_kg: float | None = None
    wing_cutoff_cm1: float | None = None
    subtract_lorentz_pedestal: bool = False
    __hash__ = None

    def __post_init__(self) -> None:
        if self.partition_ratios.ndim != 1 or self.partition_ratios.size == 0:
            raise ValueError("partition ratios must be a nonempty vector")
        if not self.species or not self.isotopologue or not self.lines:
            raise ValueError("lines need species, isotopologue and entries")
        for name in ("reference_temperature_k", "reference_pressure_pa"):
            _positive(getattr(self, name), name)
        if self.molecular_mass_kg is not None:
            _positive(self.molecular_mass_kg, "molecular mass")
        if self.wing_cutoff_cm1 is not None:
            _positive(self.wing_cutoff_cm1, "line wing cutoff")
        if self.subtract_lorentz_pedestal and (
            self.species != "H2O" or self.wing_cutoff_cm1 != 25.0
        ):
            raise ValueError("MT_CKD local pedestal requires H2O and a 25 cm-1 cutoff")
        for name in ("temperature_range_k", "pressure_range_pa"):
            lo, hi = getattr(self, name)
            _positive(lo, name)
            _positive(hi, name)
            if hi < lo:
                raise ValueError(f"{name} must be increasing")
        for ratio in self.partition_ratios:
            _positive(ratio, "partition ratio")
        object.__setattr__(self, "partition_ratios", np.copy(self.partition_ratios))


def _positive(value: float, name: str, *, zero: bool = False) -> None:
    if not isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
