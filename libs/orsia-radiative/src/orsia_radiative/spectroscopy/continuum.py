"""Pure water-continuum evaluation from caller-materialised MT_CKD ABSCO data.

No file, provider, cache or campaign access. Homogeneous layers use their
explicit molecule columns divided by geometric thickness, not an inferred
hydrostatic column. This is only the water self/foreign continuum; in
particular it does not include the separate nitrogen-H2O continuum.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, h, k

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.spectral.grid import SpectralGrid, wavenumber_grid
from orsia_radiative.spectroscopy.components import AbsorptionComponent


@dataclass(frozen=True, eq=False)
class WaterContinuumReference:
    """MT_CKD water self/foreign ABSCO and temperature exponents on a uniform cm-1 grid.

    ABSCO units are cm2/molecule/(cm-1), normalised to the reference partner
    density; the radiation term is NOT included. Arrays are writable float64
    snapshots. Reference pressure is Pa, temperature K; the declared temperature
    range is the caller's supported modelling envelope, not proof of accuracy.
    Reference nodes may include negative interpolation padding (MT_CKD 4.3
    includes -20 and -10 cm-1). Evaluation grids remain strictly positive.

    Attributes:
        wavenumbers_cm1: Uniform increasing reference knots in cm-1.
        self_absco_ref: Self-broadened ABSCO in cm2/molecule/(cm-1).
        foreign_absco_ref: Foreign-broadened ABSCO in the same units.
        self_temperature_exponent: Dimensionless exponent on the same reference knots.
        reference_pressure_pa: Reference partner pressure in pascals.
        reference_temperature_k: Reference temperature in kelvin.
        temperature_range_k: Declared inclusive temperature support in kelvin.
    """

    wavenumbers_cm1: NDArray[np.float64]
    self_absco_ref: NDArray[np.float64]
    foreign_absco_ref: NDArray[np.float64]
    self_temperature_exponent: NDArray[np.float64]
    reference_pressure_pa: float
    reference_temperature_k: float
    temperature_range_k: tuple[float, float]
    __hash__ = None

    def __post_init__(self) -> None:
        grid = self.wavenumbers_cm1
        if grid.ndim != 1 or not np.all(np.isfinite(grid)):
            raise ValueError("continuum grid must be a finite vector")
        if grid.size < 4 or np.any(np.diff(grid) <= 0):
            raise ValueError("continuum needs at least four increasing reference points")
        if not np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-10, atol=1e-12):
            raise ValueError("MT_CKD interpolation requires a uniform reference grid")
        object.__setattr__(self, "wavenumbers_cm1", np.copy(grid))
        for name in ("self_absco_ref", "foreign_absco_ref", "self_temperature_exponent"):
            values = getattr(self, name)
            if values.shape != grid.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite and match the continuum grid shape")
            if name != "self_temperature_exponent" and np.any(values < 0):
                raise ValueError("continuum reference coefficients must be nonnegative")
            object.__setattr__(self, name, np.copy(values))
        lo, hi = self.temperature_range_k
        if (
            not all(
                map(isfinite, (lo, hi, self.reference_pressure_pa, self.reference_temperature_k))
            )
            or lo <= 0
            or hi < lo
            or self.reference_pressure_pa <= 0
            or self.reference_temperature_k <= 0
        ):
            raise ValueError("continuum requires positive reference conditions/support")


def water_continuum_absorption(
    atmosphere: Atmosphere,
    wavenumbers_cm1: NDArray[np.float64] | SpectralGrid,
    reference: WaterContinuumReference,
) -> AbsorptionComponent:
    """Return dimensionless (bottom-to-top layer, point) water-continuum depths.

    Apply temperature scaling and radiation term on the ORIGINAL coefficient
    grid, then four-point cubic interpolation, as in MT_CKD_H2O 4.3. Targets
    need a complete stencil (reference[1] through reference[-2]); extrapolation
    and clipping negative cubic overshoot are forbidden. This interpolation
    order differs from interpolating ABSCO/exponents first. The stable exact
    tanh radiation expression is used, not the Fortran small/large-x shortcuts.

    Self/foreign density products are N_water**2/dz and
    N_water*(N_total-N_water)/dz [molecule2/m5]. Pressure is not used to replace
    explicit columns. Require the H2O absorber to equal the atmosphere's water
    column. Dependencies include foreign density, geometry and temperature;
    changing a non-inverted gas does not make this contribution constant.
    Inputs are borrowed; output depths are independent writable copies.

    Args:
        atmosphere: Layer geometry, explicit air/water columns and temperatures.
        wavenumbers_cm1: Target wavenumber grid or points in cm-1.
        reference: Caller-materialised water self/foreign coefficients and their
            temperature support, pressure reference and spectral knots.

    Returns:
        H2O continuum contribution with dimensionless (bottom-to-top layer,point)
        optical depths on the requested spectral grid.

    Raises:
        ValueError: A target lacks an interpolation stencil, a layer is outside
            temperature support, or the H2O absorber disagrees with the water column.
    """
    sampling = wavenumber_grid(wavenumbers_cm1)
    wavenumbers_cm1 = sampling.points
    grid = reference.wavenumbers_cm1
    target = wavenumbers_cm1
    if np.any(target < grid[1]) or np.any(target > grid[-2]):
        raise ValueError("continuum spectral support requires a full four-point stencil")
    temperature = atmosphere.temperature_k
    lo, hi = reference.temperature_range_k
    if np.any(temperature < lo) or np.any(temperature > hi):
        raise ValueError("temperature outside declared continuum support")
    water = atmosphere.water_vapour_columns_molecules_m2
    absorber = next((a for a in atmosphere.absorbers if a.species == "H2O"), None)
    if absorber is None or not np.array_equal(absorber.columns_molecules_m2, water):
        raise ValueError("H2O absorber must equal the atmosphere water column")
    thickness = np.diff(atmosphere.height_boundaries_m)
    self_product = water * (water / thickness)
    foreign_product = water * ((atmosphere.total_air_columns_molecules_m2 - water) / thickness)
    reference_density_cm3 = reference.reference_pressure_pa / (
        k * reference.reference_temperature_k * 1e6
    )
    # cm5 -> m5; ABSCO/reference_density supplies the density-pair coefficient.
    scale = 1e-10 / reference_density_cm3
    rows = np.empty((temperature.size, target.size), dtype=np.float64)
    left = np.clip(np.searchsorted(grid, target, side="right") - 1, 1, grid.size - 3)
    fraction = (target - grid[left]) / (grid[1] - grid[0])
    for layer, t in enumerate(temperature):
        with np.errstate(over="ignore", invalid="ignore"):
            radiation = grid * np.tanh(0.5 * (100 * h * c / k) * grid / t)
            self_values = (
                reference.self_absco_ref
                * (reference.reference_temperature_k / t) ** reference.self_temperature_exponent
                * radiation
            )
            foreign_values = reference.foreign_absco_ref * radiation
            rows[layer] = scale * (
                self_product[layer] * _cubic(self_values, left, fraction)
                + foreign_product[layer] * _cubic(foreign_values, left, fraction)
            )
    return AbsorptionComponent(
        sampling,
        "H2O",
        None,
        "continuum_excluding_lines",
        rows,
        line_wing_convention="mt_ckd_local_25",
    )


def _cubic(
    values: NDArray[np.float64], left: NDArray[np.int64], fraction: NDArray[np.float64]
) -> NDArray[np.float64]:
    smooth = fraction**2 * (3 - 2 * fraction)
    slope = fraction * (1 - fraction) / 2
    before = slope * (1 - fraction)
    after = slope * fraction
    return (
        -values[left - 1] * before
        + values[left] * (1 - smooth + after)
        + values[left + 1] * (smooth + before)
        - values[left + 2] * after
    )
