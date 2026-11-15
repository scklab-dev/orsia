"""Reconstruct vertical material from immutable level-sampled atmospheric states.

Gas densities use a log-linear integral, with a trapezoidal near-equal/zero
endpoint branch. Water continuum uses products of linearly interpolated endpoint
number densities. These laws remain distinct from constant-density layer clipping.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere.columns import AtmosphericColumnIntegrals

type FloatArray = NDArray[np.float64]
_BOLTZMANN_J_K = 1.380649e-23
_DRY_AIR_KG_MOL = 0.0289647
_H2O_KG_MOL = 0.01801528


@dataclass(frozen=True, eq=False)
class LevelAtmosphere:
    """Own ordered level thermodynamics and species-labelled dry-air mixing ratios.

    Attributes:
        level_altitude_m: Increasing geometric altitudes (level,), metres in a
            caller-declared common datum; at least two levels.
        level_pressure_pa: Decreasing positive level pressures, pascals.
        level_temperature_k: Positive level temperatures, kelvin.
        level_specific_humidity_kg_kg: Moist-air water mass fraction (level,), in [0,1).
        species: Unique nonempty absorber identities in the declared species order;
            an empty tuple permits air-only material reconstruction.
        species_vmr: Nonnegative dry-air molar ratios (level,species), mol/mol.
            All numerical arrays are independent immutable float64 snapshots.
            Spectroscopy and representative interval T/p are separate supplied inputs.
    """

    level_altitude_m: FloatArray
    level_pressure_pa: FloatArray
    level_temperature_k: FloatArray
    level_specific_humidity_kg_kg: FloatArray
    species: tuple[str, ...]
    species_vmr: FloatArray
    __hash__ = None

    def __post_init__(self) -> None:
        """Validate level axes and physical domains, then snapshot the state.

        Raises:
            ValueError: Species, dimensions, ordering or physical domains are invalid.
        """
        z, p, t, q = (
            self.level_altitude_m,
            self.level_pressure_pa,
            self.level_temperature_k,
            self.level_specific_humidity_kg_kg,
        )
        vmr = self.species_vmr
        if z.ndim != 1 or z.size < 2 or any(v.shape != z.shape for v in (p, t, q)):
            raise ValueError("column requires matching level vectors with at least two levels")
        if (
            len(set(self.species)) != len(self.species)
            or any(not s for s in self.species)
            or vmr.shape != (z.size, len(self.species))
        ):
            raise ValueError("unique species and (level,species) mixing ratios required")
        if any(not np.isfinite(v).all() for v in (z, p, t, q, vmr)) or (
            np.any(np.diff(z) <= 0)
            or np.any(np.diff(p) >= 0)
            or np.any(p <= 0)
            or np.any(t <= 0)
            or np.any(q < 0)
            or np.any(q >= 1)
            or np.any(vmr < 0)
        ):
            raise ValueError("column levels violate finite physical domains or ordering")
        for name in (
            "level_altitude_m",
            "level_pressure_pa",
            "level_temperature_k",
            "level_specific_humidity_kg_kg",
            "species_vmr",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            owned = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "species", tuple(self.species))


def integrate_atmospheric_columns(
    atmosphere: LevelAtmosphere, *, surface_altitude_m: float
) -> AtmosphericColumnIntegrals:
    """Integrate atmospheric material above a surface, including partial layers.

    Gas and dry-air columns use log-linear density reconstruction with the
    near-equal/zero-endpoint rule in ``log_linear_density_integral``. Water
    self/foreign products integrate linear endpoint densities instead.

    Args:
        atmosphere: Immutable bottom-to-top levels and (level,species) dry-air ratios.
        surface_altitude_m: Terrain altitude in the level datum, metres. Distances
            at most 1e-6 m from an interface snap to it; the top must leave a layer.

    Returns:
        Immutable retained material on the original interval axes, with
        (layer,species) gas columns in molecules/m2 and water products in molecules2/m5.
        Partial and first-complete interval indices are zero-based, or None if absent.

    Raises:
        ValueError: Terrain lies outside support or resulting integrals are invalid.
    """
    z, p, t, q = (
        atmosphere.level_altitude_m,
        atmosphere.level_pressure_pa,
        atmosphere.level_temperature_k,
        atmosphere.level_specific_humidity_kg_kg,
    )
    # Preserve the established arithmetic ordering internally; public species axes are last.
    vmr = atmosphere.species_vmr.T
    altitude = float(surface_altitude_m)
    if not np.isfinite(altitude) or altitude < z[0] - 1e-6 or altitude >= z[-1] - 1e-6:
        raise ValueError("surface must leave a radiative layer within prepared support")
    fraction = 0.0
    exact = np.flatnonzero(np.abs(z - altitude) <= 1e-6)
    if len(exact):
        first = int(exact[0])
        altitude, pressure = z[first], p[first]
        partial = None
    else:
        first = int(np.searchsorted(z, altitude))
        lower = first - 1
        fraction = (altitude - z[lower]) / (z[first] - z[lower])
        pressure = np.exp(np.log(p[lower]) + fraction * (np.log(p[first]) - np.log(p[lower])))
        partial = lower
    # Retain all original layer axes, substituting only the intersected lower endpoint.
    lower_p, lower_t, lower_q = p[:-1].copy(), t[:-1].copy(), q[:-1].copy()
    lower_vmr = vmr[:, :-1].copy()
    thickness = np.diff(z)
    if partial is not None:
        lower_p[partial] = pressure
        lower_t[partial] += fraction * (t[first] - t[partial])
        lower_q[partial] += fraction * (q[first] - q[partial])
        lower_vmr[:, partial] += fraction * (vmr[:, first] - vmr[:, partial])
        thickness[partial] = z[first] - altitude
    lower_total, lower_dry, lower_water = moist_air_number_densities(lower_p, lower_t, lower_q)
    upper_total, upper_dry, upper_water = moist_air_number_densities(p[1:], t[1:], q[1:])
    dry = log_linear_density_integral(lower_dry, upper_dry, thickness)
    species = log_linear_density_integral(lower_dry * lower_vmr, upper_dry * vmr[:, 1:], thickness)
    self_product, foreign_product = _water_density_products(
        lower_total, upper_total, lower_water, upper_water, thickness
    )
    start = first if partial is None else partial
    dry[:start] = 0
    species[:, :start] = 0
    self_product[:start] = 0
    foreign_product[:start] = 0
    return AtmosphericColumnIntegrals(
        z,
        atmosphere.species,
        float(altitude),
        float(pressure),
        dry,
        species.T,
        self_product,
        foreign_product,
    )


def moist_air_number_densities(
    pressure: FloatArray, temperature: FloatArray, humidity: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate moist-air molecular number densities at arbitrary thermodynamic samples.

    Inputs follow NumPy broadcasting and remain unchanged. Dry-air/water molar
    masses are 0.0289647 and 0.01801528 kg/mol for this reconstruction convention.

    Args:
        pressure: Positive finite pressure samples in Pa.
        temperature: Positive finite temperatures in K, broadcastable with pressure.
        humidity: Specific humidity in kg H2O/kg moist air, finite in [0,1), with
            a shape broadcastable with pressure and temperature.

    Returns:
        Fresh writable float64 total-air, dry-air and water densities in molecules/m3.
        Total-air shape broadcasts pressure and temperature; dry-air and water
        shapes also include humidity's broadcast axes.

    Raises:
        ValueError: Physical domains are invalid or shapes cannot broadcast.
    """
    if (
        any(not np.isfinite(v).all() for v in (pressure, temperature, humidity))
        or np.any(pressure <= 0)
        or np.any(temperature <= 0)
        or np.any((humidity < 0) | (humidity >= 1))
    ):
        raise ValueError("positive finite pressure/temperature and humidity in [0,1) required")
    total = pressure / (_BOLTZMANN_J_K * temperature)
    water_vmr = humidity / (1 - humidity) * _DRY_AIR_KG_MOL / _H2O_KG_MOL
    dry = total / (1 + water_vmr)
    return total, dry, dry * water_vmr


def log_linear_density_integral(
    lower: FloatArray, upper: FloatArray, thickness: FloatArray
) -> FloatArray:
    """Integrate endpoint number densities with a fixed near-equal convention.

    The logarithmic mean gives the integral for positive distinct endpoints.
    When either endpoint is zero or abs(1-lower/upper) <= 1e-5, use the
    trapezoidal expression. This branch is part of the method's contract.

    Args:
        lower: Finite nonnegative lower-endpoint number densities, molecules/m3.
        upper: Finite nonnegative upper-endpoint densities, molecules/m3.
        thickness: Finite nonnegative layer thicknesses in metres. All three
            arrays broadcast under NumPy rules and remain unchanged.

    Returns:
        Fresh writable float64 integrated columns in molecules/m2 with broadcast axes.

    Raises:
        ValueError: Values are negative/nonfinite or shapes cannot broadcast.
    """
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in (lower, upper, thickness)):
        raise ValueError("finite nonnegative endpoint densities and thickness required")
    lower, upper = np.broadcast_arrays(lower, upper)
    ratio = np.divide(lower, upper, out=np.ones_like(lower), where=upper != 0)
    trapezoid = (lower == 0) | (upper == 0) | (np.abs(1 - ratio) <= 1e-5)
    result = 0.5 * (lower + upper) * thickness
    np.divide(
        thickness * (lower - upper),
        np.log(ratio, where=ratio > 0, out=np.ones_like(ratio)),
        out=result,
        where=~trapezoid,
    )
    return result


def _water_density_products(
    lower_total: FloatArray,
    upper_total: FloatArray,
    lower_water: FloatArray,
    upper_water: FloatArray,
    thickness: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Integrate self/foreign water products from linear endpoint densities.

    Returns:
        Fresh self and foreign product integrals in molecules2/m5 on layer axes.
    """
    self_product = thickness * (lower_water**2 + lower_water * upper_water + upper_water**2) / 3
    lower_foreign = np.maximum(lower_total - lower_water, 0)
    upper_foreign = np.maximum(upper_total - upper_water, 0)
    foreign_product = (
        thickness
        * (
            2 * lower_water * lower_foreign
            + lower_water * upper_foreign
            + upper_water * lower_foreign
            + 2 * upper_water * upper_foreign
        )
        / 6
    )
    return self_product, foreign_product
