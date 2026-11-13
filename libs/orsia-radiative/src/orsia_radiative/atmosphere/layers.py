"""Supply layer atmospheres and reconstruct vertical coordinates and material.

Numerical operations return fresh writable arrays. Layer-state constructors
retain independent numerical snapshots; callers keep those values stable during
calculations and rebuild dependent preparation after a state change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.profiles import LayerProfileBasis
from orsia_radiative.profiles import validate_profile_values as _finite

AirBroadener = Literal["dry_air", "total_air"]


@dataclass(frozen=True, eq=False)
class Absorber:
    """Supply a gas column and broadener convention for each atmospheric layer.

    The pool is a declared column convention, not a partial-pressure calculation:
    spectroscopy decides whether to subtract the target gas and add self broadening.

    Attributes:
        species: Unique gas identifier within the containing atmosphere.
        columns_molecules_m2: Nonnegative (layers,) molecular column densities.
        air_broadener: Select total-air or dry-air column convention.
        profile_basis: Optional layer basis with matching molecular-column baseline.
    """

    species: str
    columns_molecules_m2: NDArray[np.float64]
    air_broadener: AirBroadener
    profile_basis: LayerProfileBasis | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        if self.columns_molecules_m2.size == 0:
            raise ValueError("absorber columns must be nonempty")
        if not self.species or self.air_broadener not in ("dry_air", "total_air"):
            raise ValueError("species and dry_air or total_air broadener are required")
        _finite(self.columns_molecules_m2, self.columns_molecules_m2.size, "absorber", minimum=0)
        if self.profile_basis is not None and (
            self.profile_basis.domain != "layer"
            or not np.array_equal(self.profile_basis.baseline, self.columns_molecules_m2)
            or self.profile_basis.unit != "molecule m-2"
        ):
            raise ValueError("absorber basis must match its materialised molecule column")
        object.__setattr__(self, "columns_molecules_m2", np.copy(self.columns_molecules_m2))


@dataclass(frozen=True, eq=False)
class LayerAtmosphere:
    """Supply bottom-to-top layer state for optical and transfer calculations.

    Boundary arrays have N+1 entries; layer pressures/temperatures and columns
    have N entries. Total air = dry air + water vapour in each layer, as molecule
    columns in m-2. This identity does not impose hydrostatic equilibrium or
    convert columns to partial pressure. No aerosols/clouds are modelled here.

    Attributes:
        height_boundaries_m: Increasing (layers+1,) height edges in metres.
        pressure_boundaries_pa: Decreasing (layers+1,) pressure edges in pascals.
        pressure_pa: (layers,) pressures enclosed by the corresponding boundary pair.
        temperature_k: Positive (layers,) layer temperatures in kelvin.
        total_air_columns_molecules_m2: (layers,) total-air molecular columns.
        dry_air_columns_molecules_m2: (layers,) dry-air molecular columns.
        water_vapour_columns_molecules_m2: (layers,) water-vapour molecular columns.
        absorbers: Unique gas columns sharing this layer ordering.
        temperature_basis: Optional layer perturbation basis at the stored temperatures.
        water_vapour_basis: Optional layer basis at the stored water-vapour columns.
    """

    height_boundaries_m: NDArray[np.float64]
    pressure_boundaries_pa: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    total_air_columns_molecules_m2: NDArray[np.float64]
    dry_air_columns_molecules_m2: NDArray[np.float64]
    water_vapour_columns_molecules_m2: NDArray[np.float64]
    absorbers: tuple[Absorber, ...] = ()
    temperature_basis: LayerProfileBasis | None = None
    water_vapour_basis: LayerProfileBasis | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        n = self.height_boundaries_m.size - 1
        if n < 1:
            raise ValueError("at least one atmospheric layer is required")
        _finite(self.height_boundaries_m, n + 1, "height boundaries")
        if any(
            b <= a
            for a, b in zip(self.height_boundaries_m, self.height_boundaries_m[1:], strict=False)
        ):
            raise ValueError("height boundaries must increase bottom to top")
        _finite(self.pressure_boundaries_pa, n + 1, "pressure boundaries", minimum=0)
        if any(
            b >= a
            for a, b in zip(
                self.pressure_boundaries_pa, self.pressure_boundaries_pa[1:], strict=False
            )
        ):
            raise ValueError("pressure boundaries must decrease bottom to top")
        _finite(self.pressure_pa, n, "layer pressure")
        _finite(self.temperature_k, n, "layer temperature")
        _finite(self.total_air_columns_molecules_m2, n, "total-air columns")
        _finite(self.dry_air_columns_molecules_m2, n, "dry-air columns")
        _finite(self.water_vapour_columns_molecules_m2, n, "water-vapour columns", minimum=0)
        for basis, values, unit in (
            (self.temperature_basis, self.temperature_k, "K"),
            (self.water_vapour_basis, self.water_vapour_columns_molecules_m2, "molecule m-2"),
        ):
            if basis is not None and (
                basis.domain != "layer"
                or not np.array_equal(basis.baseline, values)
                or basis.unit != unit
            ):
                raise ValueError(
                    "atmospheric basis must match its materialised layer values and unit"
                )
        for i in range(n):
            if not (
                self.pressure_boundaries_pa[i]
                >= self.pressure_pa[i]
                >= self.pressure_boundaries_pa[i + 1]
                and self.pressure_pa[i] > 0
            ):
                raise ValueError("layer pressure must lie within its pressure boundaries")
            if (
                self.temperature_k[i] <= 0
                or self.total_air_columns_molecules_m2[i] <= 0
                or self.dry_air_columns_molecules_m2[i] <= 0
            ):
                raise ValueError("temperature and total/dry air columns must be positive")
            if not isclose(
                self.total_air_columns_molecules_m2[i],
                self.dry_air_columns_molecules_m2[i] + self.water_vapour_columns_molecules_m2[i],
                rel_tol=1e-12,
            ):
                raise ValueError("total air must equal dry air plus water vapour")
        if len({gas.species for gas in self.absorbers}) != len(self.absorbers):
            raise ValueError("absorber species must be unique")
        for gas in self.absorbers:
            _finite(gas.columns_molecules_m2, n, "absorber columns", minimum=0)
            limit = (
                self.water_vapour_columns_molecules_m2
                if gas.species == "H2O"
                else self.dry_air_columns_molecules_m2
            )
            for i, amount in enumerate(gas.columns_molecules_m2):
                if gas.species == "H2O" and not isclose(amount, limit[i], rel_tol=1e-12):
                    raise ValueError("absorber column conflicts with water vapour or dry air")
                if gas.species != "H2O" and amount > limit[i]:
                    raise ValueError("absorber column conflicts with water vapour or dry air")
        # Retain independent fields only after validating the physical state.
        for name in (
            "height_boundaries_m",
            "pressure_boundaries_pa",
            "pressure_pa",
            "temperature_k",
            "total_air_columns_molecules_m2",
            "dry_air_columns_molecules_m2",
            "water_vapour_columns_molecules_m2",
        ):
            object.__setattr__(self, name, np.copy(getattr(self, name)))


def geometric_height(
    geopotential_m2_s2: NDArray[np.float64],
    *,
    reference_gravity_m_s2: float,
    earth_radius_m: float,
) -> NDArray[np.float64]:
    """Convert geopotential [m2/s2] to spherical geometric height [m].

    H=geopotential/reference_gravity is geopotential height, not geometric height;
    z=R*H/(R-H) assumes inverse-square gravity on a spherical Earth. Caller chooses
    gravity, radius and vertical datum; this is not a geoid/terrain correction.

    Args:
        geopotential_m2_s2: Finite geopotential values in square metres per second squared.
        reference_gravity_m_s2: Positive reference acceleration for geopotential height.
        earth_radius_m: Positive radius in metres used by the spherical conversion.

    Returns:
        Fresh geometric heights in metres with the input shape.

    Raises:
        ValueError: Input is nonfinite, gravity or radius is invalid, or a
            geopotential height reaches the specified Earth radius.
    """
    validate_vertical_values(geopotential_m2_s2, None, "geopotential")
    if (
        not isfinite(reference_gravity_m_s2)
        or reference_gravity_m_s2 <= 0
        or not isfinite(earth_radius_m)
        or earth_radius_m <= 0
    ):
        raise ValueError("positive finite gravity and Earth radius required")
    height = geopotential_m2_s2 / reference_gravity_m_s2
    if np.any(height >= earth_radius_m):
        raise ValueError("geopotential height must be below Earth radius")
    return earth_radius_m * height / (earth_radius_m - height)


def pressure_level_layer_means(
    pressure_pa: NDArray[np.float64],
    values: NDArray[np.float64],
    pressure_boundaries_pa: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate a point profile linear in pressure into pressure-mass layer means.

    Point knots and interfaces are strictly decreasing [Pa]. Values have one
    finite value per knot (K, kg/kg, etc.). Target interfaces must be bracketed by
    knots, including at the surface: no below-terrain or top extrapolation. This
    is a caller-selected reconstruction, not native model-layer averages.

    Args:
        pressure_pa: Bottom-to-top decreasing point-knot pressures in Pa.
        values: Finite physical values at those pressure knots.
        pressure_boundaries_pa: Target bottom-to-top layer interfaces in Pa,
            bracketed by the point knots.

    Returns:
        Fresh pressure-mass means (target_layers,) in the input value unit.

    Raises:
        ValueError: Pressure support, order or profile shape is invalid.
    """
    validate_pressure_boundaries(pressure_pa, "pressure knots")
    validate_pressure_boundaries(pressure_boundaries_pa, "pressure boundaries")
    validate_vertical_values(values, len(pressure_pa), "point profile")
    if pressure_boundaries_pa[0] > pressure_pa[0] or pressure_boundaries_pa[-1] < pressure_pa[-1]:
        raise ValueError("layer interfaces are outside pressure-knot coverage")
    means = np.empty(len(pressure_boundaries_pa) - 1)
    for i, (bottom, top) in enumerate(
        zip(pressure_boundaries_pa[:-1], pressure_boundaries_pa[1:], strict=True)
    ):
        points = np.sort(
            np.concatenate(
                ([top, bottom], pressure_pa[(pressure_pa > top) & (pressure_pa < bottom)])
            )
        )
        sampled = np.interp(points, pressure_pa[::-1], values[::-1])
        means[i] = np.sum(np.diff(points) * (sampled[:-1] + sampled[1:]) / 2) / (bottom - top)
    return means


def remap_layer_columns(
    source_pressure_boundaries_pa: NDArray[np.float64],
    columns: NDArray[np.float64],
    target_pressure_boundaries_pa: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Conservatively remap extensive layer quantities by pressure overlap.

    Both grids are bottom to top [Pa]. columns has shape (source_layer,) and
    contains nonnegative extensive amounts (molecules/m2, kg/m2, etc.). Assume
    uniform amount per Pa within each source layer. Target support must be fully
    covered: no extrapolation, missing layers or below-terrain padding. Identical
    outer boundaries conserve the integrated amount to floating-point precision;
    a narrower target intentionally retains only its overlapping column.

    Args:
        source_pressure_boundaries_pa: Decreasing source interfaces in Pa.
        columns: Finite nonnegative source-layer amounts (source_layers,).
        target_pressure_boundaries_pa: Decreasing target interfaces within the
            source pressure coverage, in Pa.

    Returns:
        Fresh target-layer amounts (target_layers,) in the input column unit.

    Raises:
        ValueError: Pressure boundaries, target coverage or source columns
            violate the required shape and physical constraints.
    """
    weights = _pressure_layer_overlap(source_pressure_boundaries_pa, target_pressure_boundaries_pa)
    validate_vertical_values(columns, weights.shape[1], "layer columns")
    if np.any(columns < 0):
        raise ValueError("layer columns must be nonnegative")
    return weights @ (columns / -np.diff(source_pressure_boundaries_pa))


def validate_pressure_boundaries(values: NDArray[np.float64], name: str) -> None:
    """Require finite, nonnegative, strictly decreasing pressure interfaces [Pa].

    The one-dimensional vector must contain at least two interfaces. ``name``
    identifies the coordinate in ValueError messages; values remain caller-owned.

    Args:
        values: Pressure interface vector in Pa, bottom to top.
        name: Name identifying the pressure coordinate in errors.

    Raises:
        ValueError: Input is short, nonfinite, negative or out of order.
    """
    validate_vertical_values(values, None, name)
    if len(values) < 2 or np.any(values < 0) or np.any(np.diff(values) >= 0):
        raise ValueError(f"{name} must have at least two nonnegative decreasing boundaries")


def validate_vertical_values(values: NDArray[np.float64], size: int | None, name: str) -> None:
    """Require a finite vector on vertical support, optionally of the given size.

    Units follow the supplied quantity (e.g. Pa, K or kg/kg); ``name`` identifies
    it in ValueError messages. Values remain caller-owned with their original dtype.

    Args:
        values: One-dimensional physical profile values.
        size: Expected vector length, or None for any length.
        name: Quantity name identifying shape or finiteness errors.

    Raises:
        ValueError: Values are nonfinite or fail the vector shape requirement.
    """
    if (
        values.ndim != 1
        or (size is not None and len(values) != size)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} must be a finite vector with the declared shape")


def _pressure_layer_overlap(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return intersecting pressure widths [Pa], shape (target_layer, source_layer).

    Shared internally by column remapping and hydrostatic construction.
    Both interface vectors decrease bottom to top and allow zero pressure at
    the top. Target support must lie within source support. The fresh writable
    array contains nonnegative widths; dividing each column by its source-layer
    width gives conservative redistribution fractions.

    Returns:
        Fresh writable pressure overlap widths in Pa, shape (target_layer, source_layer).

    Raises:
        ValueError: Pressure boundaries are invalid or target support exceeds source coverage.
    """
    validate_pressure_boundaries(source, "source pressure")
    validate_pressure_boundaries(target, "target pressure")
    if target[0] > source[0] or target[-1] < source[-1]:
        raise ValueError("target pressure support is outside the source coverage")
    return np.maximum(
        0.0,
        np.minimum(target[:-1, None], source[None, :-1])
        - np.maximum(target[1:, None], source[None, 1:]),
    )
