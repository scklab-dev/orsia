"""Pure water-continuum evaluation from caller-materialised MT_CKD ABSCO data.

No file, provider, cache or campaign access. Homogeneous layers use their
explicit molecule columns divided by geometric thickness, not an inferred
hydrostatic column. This is only the water self/foreign continuum; in
particular it does not include the separate nitrogen-H2O continuum.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, h, k

from orsia_radiative.atmosphere import LayerAtmosphere
from orsia_radiative.spectral.grid import SpectralGrid, convert_spectral_grid, wavenumber_grid
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


@dataclass(frozen=True, eq=False)
class WaterContinuumCoefficients:
    """Own separate water density-pair coefficients on declared layer intervals.

    Attributes:
        grid: Increasing spectral samples in nm or cm-1; coefficients are point values.
        height_edges_m: Increasing physical interval interfaces (layer+1,), metres.
        temperature_k: Positive evaluation temperatures (layer,), kelvin.
        pressure_pa: Positive declared layer pressures (layer,), pascals. They identify
            the coefficient pairing context; the MT_CKD density-pair kernel uses T.
        self_coefficients_m5_molecule2: Nonnegative self coefficients (layer,spectral),
            m5/molecule2, including the radiation term.
        foreign_coefficients_m5_molecule2: Matching non-water-air coefficients and units.
        line_wing_convention: Explicit matching local-line and continuum convention.
            The supported MT_CKD preparation requires a 25 cm-1 Lorentz pedestal cut.
            All numerical arrays are independent immutable float64 snapshots.
    """

    grid: SpectralGrid
    height_edges_m: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    self_coefficients_m5_molecule2: NDArray[np.float64]
    foreign_coefficients_m5_molecule2: NDArray[np.float64]
    line_wing_convention: Literal["mt_ckd_local_25"]
    __hash__ = None

    def __post_init__(self) -> None:
        """Validate density-pair units' numerical domains and snapshot all arrays.

        Raises:
            ValueError: Dimensions, support, conditions, convention or coefficients fail.
        """
        t, p, z = self.temperature_k, self.pressure_pa, self.height_edges_m
        if (
            t.ndim != 1
            or not t.size
            or p.shape != t.shape
            or z.shape != (t.size + 1,)
            or not all(np.isfinite(v).all() for v in (t, p, z))
            or np.any(t <= 0)
            or np.any(p <= 0)
            or np.any(np.diff(z) <= 0)
            or self.line_wing_convention != "mt_ckd_local_25"
        ):
            raise ValueError(
                "valid continuum layer support, conditions and line convention required"
            )
        names = ("self_coefficients_m5_molecule2", "foreign_coefficients_m5_molecule2")
        for name in names:
            values = getattr(self, name)
            if (
                values.shape != (t.size, self.grid.points.size)
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError(
                    "continuum coefficients must be finite nonnegative (layer,spectral)"
                )
        for name in ("height_edges_m", "temperature_k", "pressure_pa", *names):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            owned = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, owned)


def water_continuum_coefficients(
    grid: SpectralGrid,
    reference: WaterContinuumReference,
    *,
    height_edges_m: NDArray[np.float64],
    temperature_k: NDArray[np.float64],
    pressure_pa: NDArray[np.float64],
) -> WaterContinuumCoefficients:
    """Prepare MT_CKD water coefficients for either supported spectral coordinate.

    Scale temperature and radiation terms on the native reference grid before
    four-point cubic interpolation. Reciprocal output coordinates only reorder
    point coefficients; no spectral-density Jacobian applies. Molecular density
    products remain independent inputs to optical-depth contraction.

    Args:
        grid: Increasing nm or cm-1 evaluation samples with complete native stencils.
        reference: Caller-supplied MT_CKD reference coefficients and temperature envelope.
        height_edges_m: Increasing physical layer interfaces, metres.
        temperature_k: Explicit positive coefficient evaluation temperatures (layer,), K.
        pressure_pa: Explicit positive layer pressure context (layer,), Pa.

    Returns:
        Immutable separate self/foreign coefficients in m5/molecule2 and their support.

    Raises:
        ValueError: Native support, temperatures, layer declarations or coefficients fail.
    """
    spectral = convert_spectral_grid(grid) if grid.coordinate == "wavelength_nm" else grid
    own, foreign, scale = _coefficient_samples(temperature_k, spectral.points, reference)
    if grid.coordinate == "wavelength_nm":
        own, foreign = own[:, ::-1], foreign[:, ::-1]
    return WaterContinuumCoefficients(
        grid,
        height_edges_m,
        temperature_k,
        pressure_pa,
        own * scale,
        foreign * scale,
        "mt_ckd_local_25",
    )


def water_continuum_absorption(
    atmosphere: LayerAtmosphere,
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
    water = atmosphere.water_vapour_columns_molecules_m2
    absorber = next((a for a in atmosphere.absorbers if a.species == "H2O"), None)
    if absorber is None or not np.array_equal(absorber.columns_molecules_m2, water):
        raise ValueError("H2O absorber must equal the atmosphere water column")
    thickness = np.diff(atmosphere.height_boundaries_m)
    self_product = water * (water / thickness)
    foreign_product = water * ((atmosphere.total_air_columns_molecules_m2 - water) / thickness)
    own, foreign, scale = _coefficient_samples(atmosphere.temperature_k, sampling.points, reference)
    # Retain the homogeneous evaluator's established arithmetic ordering.
    with np.errstate(over="ignore", invalid="ignore"):
        rows = scale * (self_product[:, None] * own + foreign_product[:, None] * foreign)
    return AbsorptionComponent(
        sampling,
        "H2O",
        None,
        "continuum_excluding_lines",
        rows,
        line_wing_convention="mt_ckd_local_25",
    )


def _coefficient_samples(
    temperature: NDArray[np.float64],
    target: NDArray[np.float64],
    reference: WaterContinuumReference,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    grid = reference.wavenumbers_cm1
    if np.any(target < grid[1]) or np.any(target > grid[-2]):
        raise ValueError("continuum spectral support requires a full four-point stencil")
    lo, hi = reference.temperature_range_k
    if (
        temperature.ndim != 1
        or not temperature.size
        or not np.isfinite(temperature).all()
        or np.any(temperature < lo)
        or np.any(temperature > hi)
    ):
        raise ValueError("temperature outside declared continuum support")
    reference_density_cm3 = reference.reference_pressure_pa / (
        k * reference.reference_temperature_k * 1e6
    )
    scale = 1e-10 / reference_density_cm3
    own = np.empty((temperature.size, target.size), dtype=np.float64)
    foreign = np.empty_like(own)
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
            own[layer] = _cubic(self_values, left, fraction)
            foreign[layer] = _cubic(foreign_values, left, fraction)
    return own, foreign, scale


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
