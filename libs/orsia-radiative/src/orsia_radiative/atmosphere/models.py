"""Materialised atmospheric columns and validation of their physical properties."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
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
class Atmosphere:
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
