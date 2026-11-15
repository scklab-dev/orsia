"""Resample level profiles and convert explicit gas-composition conventions."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere.hydrostatic import (
    MOLAR_MASS_DRY_AIR_KG_MOL,
    MOLAR_MASS_WATER_KG_MOL,
    GasFraction,
)
from orsia_radiative.atmosphere.levels import LevelAtmosphere

type FloatArray = NDArray[np.float64]


def resample_pressure_profile(
    pressure_pa: FloatArray,
    values: FloatArray,
    target_pressure_pa: FloatArray,
    *,
    method: Literal["pressure", "log_pressure"],
    extend_surface: bool = False,
) -> FloatArray:
    """Evaluate level values with an explicit pressure-coordinate interpolation law.

    Args:
        pressure_pa: Decreasing positive source pressures (level,), Pa; at least two.
        values: Finite source values (level,) in caller-defined units.
        target_pressure_pa: Positive target pressures (target,), Pa, in caller order.
        method: Linear interpolation in pressure or log-pressure.
        extend_surface: Permit continuation above the largest source pressure using
            the first two source levels in the selected coordinate. The caller must
            restrict this option to intended surface targets. Top extrapolation fails.

    Returns:
        Fresh float64 target values in the input units. Inputs remain unchanged.
        This evaluates levels, separately from pressure-weighted layer means.

    Raises:
        ValueError: Axes, finite values, pressure ordering, method or support fail.
    """
    _pressure_vector(pressure_pa)
    target = target_pressure_pa
    if (
        values.shape != pressure_pa.shape
        or not np.isfinite(values).all()
        or target.ndim != 1
        or not target.size
        or not np.isfinite(target).all()
        or np.any(target <= 0)
        or method not in ("pressure", "log_pressure")
    ):
        raise ValueError("finite aligned values, positive targets and explicit method required")
    if np.any(target < pressure_pa[-1]) or (not extend_surface and np.any(target > pressure_pa[0])):
        raise ValueError("pressure interpolation outside native support")
    x = np.log(pressure_pa) if method == "log_pressure" else pressure_pa
    t = np.log(target) if method == "log_pressure" else target
    result = np.interp(t, x[::-1], values[::-1])
    outside = target > pressure_pa[0]
    result[outside] = values[0] + (t[outside] - x[0]) / (x[1] - x[0]) * (values[1] - values[0])
    if not np.isfinite(result).all():
        raise ValueError("profile interpolation exceeds finite numerical domain")
    return result


def convert_gas_fraction(
    fraction: FloatArray,
    *,
    source: GasFraction,
    target: GasFraction,
    specific_humidity_kg_kg: FloatArray,
    molar_mass_kg_mol: float | None = None,
) -> FloatArray:
    """Convert species ratios between mass/mole and dry/moist-air denominators.

    Args:
        fraction: Nonnegative finite (level,) gas ratios in the source convention.
            Dry-air ratios can exceed one, including water vapour relative to dry air.
        source: Explicit mass or mole ratio with dry-air or total-air denominator.
        target: Requested convention using the same denominator definitions.
        specific_humidity_kg_kg: Matching water mass per moist-air mass in [0,1).
        molar_mass_kg_mol: Positive species molar mass required for mass conventions.

    Returns:
        Fresh float64 ratios using fixed mean dry-air and water molar masses.
        Total-air ratios must lie in [0,1]. Species identity is caller-owned.

    Raises:
        ValueError: Conventions, physical domains, shapes or required mass fail.
    """
    conventions = ("dry_air_mole", "total_air_mole", "dry_air_mass", "total_air_mass")
    q = specific_humidity_kg_kg
    if (
        source not in conventions
        or target not in conventions
        or fraction.ndim != 1
        or not fraction.size
        or q.shape != fraction.shape
        or not np.isfinite(fraction).all()
        or not np.isfinite(q).all()
        or np.any(fraction < 0)
        or np.any((q < 0) | (q >= 1))
        or (source.startswith("total") and np.any(fraction > 1))
    ):
        raise ValueError("finite physical composition vectors and explicit conventions required")
    mass = 1.0
    if source.endswith("mass") or target.endswith("mass"):
        if (
            molar_mass_kg_mol is None
            or not np.isfinite(molar_mass_kg_mol)
            or molar_mass_kg_mol <= 0
        ):
            raise ValueError("mass conversion requires positive finite species molar mass")
        mass = molar_mass_kg_mol
    dry_moles = (1 - q) / MOLAR_MASS_DRY_AIR_KG_MOL
    total_moles = dry_moles + q / MOLAR_MASS_WATER_KG_MOL
    factors = {
        "dry_air_mole": dry_moles,
        "total_air_mole": total_moles,
        "dry_air_mass": (1 - q) / mass,
        "total_air_mass": np.full(q.shape, 1 / mass),
    }
    result = fraction * (factors[source] / factors[target])
    if not np.isfinite(result).all() or (target.startswith("total") and np.any(result > 1)):
        raise ValueError("converted composition exceeds physical or finite domain")
    return result


def resample_level_atmosphere(
    atmosphere: LevelAtmosphere,
    target_pressure_pa: FloatArray,
    *,
    surface_altitude_m: float,
    method: Literal["pressure", "log_pressure"],
    surface_continuation: Literal["reject", "linear_altitude"] = "reject",
) -> LevelAtmosphere:
    """Construct level thermodynamics above an explicitly supplied surface.

    Args:
        atmosphere: Source levels with dry-air species molar ratios and moist-air
            specific humidity. All buffers are borrowed and left unchanged.
        target_pressure_pa: Decreasing pressure levels inside source support, Pa.
        surface_altitude_m: Surface height in the source datum, m. Target levels
            within 1e-6 m of or below this surface are replaced by the surface.
        method: Pressure-coordinate interpolation for the above-surface levels.
        surface_continuation: Reject a surface below source support, or continue
            source log-pressure, T, humidity and species ratios linearly in altitude
            using the bottom two levels. No continuation above the source top.

    Returns:
        Independent immutable LevelAtmosphere with the surface as its first level.
        Specific humidity is interpolated before any later composition conversion.
        Species ratios already in the source are interpolated in their supplied form.

    Raises:
        ValueError: Pressure support, surface selection or resulting physical domains fail.
    """
    _pressure_vector(target_pressure_pa)
    z, p = atmosphere.level_altitude_m, atmosphere.level_pressure_pa
    if (
        surface_continuation not in ("reject", "linear_altitude")
        or not np.isfinite(surface_altitude_m)
        or surface_altitude_m >= z[-1]
        or (surface_continuation == "reject" and surface_altitude_m < z[0])
    ):
        raise ValueError("surface outside selected continuation support")
    heights = resample_pressure_profile(p, z, target_pressure_pa, method=method)
    above = heights > surface_altitude_m + 1e-6
    if not above.any():
        raise ValueError("no requested level remains above surface")

    def resample(values: FloatArray) -> FloatArray:
        surface = _altitude_value(z, values, surface_altitude_m)
        levels = resample_pressure_profile(p, values, target_pressure_pa, method=method)
        return np.r_[surface, levels[above]]

    temperature = resample(atmosphere.level_temperature_k)
    humidity = resample(atmosphere.level_specific_humidity_kg_kg)
    vmr = np.empty((temperature.size, len(atmosphere.species)))
    for index in range(vmr.shape[1]):
        vmr[:, index] = resample(atmosphere.species_vmr[:, index])
    return LevelAtmosphere(
        np.r_[surface_altitude_m, heights[above]],
        np.r_[np.exp(_altitude_value(z, np.log(p), surface_altitude_m)), target_pressure_pa[above]],
        temperature,
        humidity,
        atmosphere.species,
        vmr,
    )


def _pressure_vector(pressure: FloatArray) -> None:
    if (
        pressure.ndim != 1
        or pressure.size < 2
        or not np.isfinite(pressure).all()
        or np.any(pressure <= 0)
        or np.any(np.diff(pressure) >= 0)
    ):
        raise ValueError("decreasing positive finite pressure vector required")


def _altitude_value(z: FloatArray, values: FloatArray, target: float) -> float:
    if target < z[0]:
        return float(values[0] + (target - z[0]) / (z[1] - z[0]) * (values[1] - values[0]))
    return float(np.interp(target, z, values))
