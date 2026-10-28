"""Gas-fraction conversion and hydrostatic atmospheric-column construction.

Inputs are caller-owned arrays, never mutated or retained. Outputs are fresh
writable float64 arrays, including Atmosphere's independent copies. No I/O,
providers, interpolation policy or aerosol optics live here.
"""

from collections.abc import Mapping
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere.layers import (
    _pressure_layer_overlap,
    validate_pressure_boundaries,
    validate_vertical_values,
)
from orsia_radiative.atmosphere.models import Absorber, AirBroadener, Atmosphere

AVOGADRO_MOL1 = 6.02214076e23
MOLAR_MASS_DRY_AIR_KG_MOL = 0.0289647
MOLAR_MASS_WATER_KG_MOL = 0.01801528
GAS_CONSTANT_J_MOL_K = 8.31446261815324
GasFraction = Literal["dry_air_mole", "total_air_mole", "dry_air_mass", "total_air_mass"]


def specific_humidity(ratio: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a water mixing ratio to specific humidity for moist-air columns.

    Args:
        ratio: Nonnegative water mass per dry-air mass at each layer, kg/kg.

    Returns:
        Fresh vector of water mass per total moist-air mass, kg/kg.

    Raises:
        ValueError: If the input layer vector is invalid or contains negative values.
    """
    validate_vertical_values(ratio, None, "water mixing ratio")
    if np.any(ratio < 0):
        raise ValueError("water mixing ratio must be nonnegative")
    return ratio / (1 + ratio)


def gas_columns_from_fraction(
    pressure_boundaries_pa: NDArray[np.float64],
    specific_humidity: NDArray[np.float64],
    fraction: NDArray[np.float64],
    *,
    convention: GasFraction,
    gravity_m_s2: float,
    molar_mass_kg_mol: float | None = None,
) -> NDArray[np.float64]:
    """Return hydrostatic gas columns [molecules/m2] on the source layers.

    q is kg water/kg moist air; fraction is mol/mol dry air, mol/mol total
    (moist) air, or kg gas/kg total air, selected explicitly by convention.
    For total_air_mass or dry_air_mass (kg gas/kg dry air), a positive species
    molar mass [kg/mol] is required.
    Molar fractions are bounded by one; mass fractions must fit in dry mass.
    Constant gravity and fixed mean dry-air molar mass are declared assumptions.
    This does not infer species identities or their broadener convention.

    Args:
        pressure_boundaries_pa: Bottom-to-top source layer interfaces in Pa.
        specific_humidity: Water mass per total moist-air mass at each source layer.
        fraction: Species mole or mass fraction on each source layer.
        convention: Dry/total-air mole or mass fraction interpretation.
        gravity_m_s2: Positive constant gravitational acceleration in m/s^2.
        molar_mass_kg_mol: Species molar mass in kg/mol for mass conventions.

    Returns:
        Fresh source-layer gas column vector in molecules/m^2.

    Raises:
        ValueError: If physical pressure, humidity, fraction, gravity or molar-mass
            support is invalid for the chosen convention.
    """
    mass, dry, water = _air_columns(pressure_boundaries_pa, specific_humidity, gravity_m_s2)
    validate_vertical_values(fraction, len(dry), "gas fraction")
    if np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("gas fractions must lie between zero and one")
    if convention == "dry_air_mole":
        return fraction * dry
    if convention == "total_air_mole":
        return fraction * (dry + water)
    if convention not in ("total_air_mass", "dry_air_mass"):
        raise ValueError("unknown gas fraction convention")
    if molar_mass_kg_mol is None or not isfinite(molar_mass_kg_mol) or molar_mass_kg_mol <= 0:
        raise ValueError("mass fraction conversion requires a positive molar mass")
    if convention == "dry_air_mass":
        mass = mass * (1 - specific_humidity)
    elif np.any(fraction + specific_humidity > 1):
        raise ValueError("gas mass fraction exceeds the dry-air mass")
    return mass * fraction * AVOGADRO_MOL1 / molar_mass_kg_mol


def hydrostatic_atmosphere(
    source_pressure_boundaries_pa: NDArray[np.float64],
    temperature_k: NDArray[np.float64],
    specific_humidity: NDArray[np.float64],
    *,
    target_pressure_boundaries_pa: NDArray[np.float64],
    gravity_m_s2: float,
    bottom_height_m: float,
    gas_columns_molecules_m2: Mapping[str, NDArray[np.float64]],
    air_broadener: AirBroadener,
) -> Atmosphere:
    """Build a radiative Atmosphere by hydrostatic remapping of source layers.

    T [K] and q [kg water/kg moist air] have shape (source_layer,). Gas columns
    are already materialized on this source grid; use explicit remapping first
    when combining products. H2O is inserted from q, never from a trace-gas map.
    Temperature is pressure-mass weighted; dry/water/gas columns are conserved
    by overlap. No pressure-level midpoint-to-interface inference is performed.

    Heights [m] use a constant-gravity ideal-gas hypsometric approximation with
    layer-effective T and composition; bottom_height_m anchors the TARGET grid.
    They are approximate geometric heights, not converted native geopotential.
    All target pressures must be strictly positive. Dry air has fixed molar mass
    0.0289647 kg/mol. Midlayer pressure is the arithmetic interface mean.
    No covariance, uncertainty, aerosol optics or implicit reference gas is added.

    Args:
        source_pressure_boundaries_pa: Bottom-to-top source interfaces in Pa.
        temperature_k: Piecewise-constant source-layer temperatures in kelvin.
        specific_humidity: Source-layer water mass per total moist-air mass.
        target_pressure_boundaries_pa: Positive target interfaces in Pa, covered by source.
        gravity_m_s2: Positive constant gravitational acceleration in m/s^2.
        bottom_height_m: Altitude in metres anchoring the target bottom interface.
        gas_columns_molecules_m2: Source-layer dry-air gas columns in molecules/m^2.
        air_broadener: Broadening convention for the resulting absorbers.

    Returns:
        Independently owned atmosphere with hydrostatic heights and target-layer
        conserved dry, water and trace-gas columns.

    Raises:
        ValueError: If pressure support, temperature, humidity, gas columns or
            gravity violates the hydrostatic input constraints.
    """
    mass, dry, water = _air_columns(source_pressure_boundaries_pa, specific_humidity, gravity_m_s2)
    validate_vertical_values(temperature_k, len(mass), "temperature")
    if np.any(temperature_k <= 0) or not isfinite(bottom_height_m):
        raise ValueError("positive temperature and finite bottom height are required")
    overlap = _pressure_layer_overlap(source_pressure_boundaries_pa, target_pressure_boundaries_pa)
    if target_pressure_boundaries_pa[-1] <= 0:
        raise ValueError("hydrostatic heights require a positive top pressure")
    widths = -np.diff(target_pressure_boundaries_pa)
    remapping = overlap / -np.diff(source_pressure_boundaries_pa)
    target_dry, target_water = remapping @ dry, remapping @ water
    target_temperature = overlap @ temperature_k / widths
    # Preserve the integral of R_specific*T in each merged layer for height.
    specific_r = GAS_CONSTANT_J_MOL_K * (
        (1 - specific_humidity) / MOLAR_MASS_DRY_AIR_KG_MOL
        + specific_humidity / MOLAR_MASS_WATER_KG_MOL
    )
    rt = overlap @ (specific_r * temperature_k) / widths
    height = np.concatenate(
        (
            [bottom_height_m],
            bottom_height_m
            + np.cumsum(
                rt
                / gravity_m_s2
                * np.log(target_pressure_boundaries_pa[:-1] / target_pressure_boundaries_pa[1:])
            ),
        )
    )
    if "H2O" in gas_columns_molecules_m2:
        raise ValueError("H2O is derived from specific humidity, not a separate gas column")
    absorbers = [Absorber("H2O", target_water, air_broadener)]
    for species, columns in gas_columns_molecules_m2.items():
        validate_vertical_values(columns, len(mass), "gas columns")
        if np.any(columns < 0) or np.any(columns > dry):
            raise ValueError("trace-gas columns must fit within source dry-air columns")
        absorbers.append(Absorber(species, remapping @ columns, air_broadener))
    return Atmosphere(
        height,
        target_pressure_boundaries_pa,
        (target_pressure_boundaries_pa[:-1] + target_pressure_boundaries_pa[1:]) / 2,
        target_temperature,
        target_dry + target_water,
        target_dry,
        target_water,
        tuple(absorbers),
    )


def _air_columns(
    pressure: NDArray[np.float64], q: NDArray[np.float64], gravity: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    validate_pressure_boundaries(pressure, "pressure")
    validate_vertical_values(q, len(pressure) - 1, "specific humidity")
    if np.any(q < 0) or np.any(q >= 1):
        raise ValueError("specific humidity must lie in [0, 1)")
    if not isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity must be finite and positive")
    mass = -np.diff(pressure) / gravity
    return (
        mass,
        mass * (1 - q) * AVOGADRO_MOL1 / MOLAR_MASS_DRY_AIR_KG_MOL,
        mass * q * AVOGADRO_MOL1 / MOLAR_MASS_WATER_KG_MOL,
    )
