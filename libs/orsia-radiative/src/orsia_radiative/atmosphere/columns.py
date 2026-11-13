"""Retain vertical material integrals and evaluate explicit dry-air diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_AVOGADRO_MOL1 = 6.02214076e23


@dataclass(frozen=True, eq=False)
class AtmosphericColumnIntegrals:
    """Own retained gas/air columns and water-density products above one surface.

    Intervals retain their original bottom-to-top axes. Excluded intervals are
    zero. These values describe vertical material, independently of spectroscopy
    or an arbitrary ray path. Arrays are immutable float64 snapshots.

    Attributes:
        height_edges_m: Increasing original interfaces (layer+1,), metres.
        species: Unique identities in the species axis order.
        surface_altitude_m: Retained lower altitude within the interface support, metres.
        surface_pressure_pa: Positive pressure evaluated at that altitude, pascals.
        layer_dry_air_column_molecules_m2: Retained dry-air amounts (layer,), molecules/m2.
        layer_species_column_molecules_m2: Retained gas amounts (layer,species), molecules/m2.
        layer_water_self_density_product_molecules2_m5: Water-density-squared integrals
            (layer,), molecules2/m5, under the selected endpoint reconstruction.
        layer_water_foreign_density_product_molecules2_m5: Water/non-water-air density
            product integrals on the same layer axes and in the same units.
    """

    height_edges_m: NDArray[np.float64]
    species: tuple[str, ...]
    surface_altitude_m: float
    surface_pressure_pa: float
    layer_dry_air_column_molecules_m2: NDArray[np.float64]
    layer_species_column_molecules_m2: NDArray[np.float64]
    layer_water_self_density_product_molecules2_m5: NDArray[np.float64]
    layer_water_foreign_density_product_molecules2_m5: NDArray[np.float64]
    __hash__ = None

    def __post_init__(self) -> None:
        """Validate support and protect the consistency of retained material arrays.

        Raises:
            ValueError: Support, identities, surface state or material arrays are invalid.
        """
        edges = self.height_edges_m
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("column integrals require increasing finite height edges")
        if (
            len(set(self.species)) != len(self.species)
            or any(not s for s in self.species)
            or not np.isfinite(self.surface_altitude_m)
            or not edges[0] <= self.surface_altitude_m < edges[-1]
            or not np.isfinite(self.surface_pressure_pa)
            or self.surface_pressure_pa <= 0
        ):
            raise ValueError("column identities or retained surface state are invalid")
        names = (
            "layer_dry_air_column_molecules_m2",
            "layer_species_column_molecules_m2",
            "layer_water_self_density_product_molecules2_m5",
            "layer_water_foreign_density_product_molecules2_m5",
        )
        excluded = edges[1:] <= self.surface_altitude_m
        for name in names:
            values = getattr(self, name)
            shape = (edges.size - 1, len(self.species)) if name == names[1] else (edges.size - 1,)
            if (
                values.shape != shape
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values[excluded] != 0)
            ):
                raise ValueError("material integrals must be finite nonnegative on retained layers")
        for name in ("height_edges_m", *names):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            owned = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "species", tuple(self.species))

    @property
    def first_complete_layer_index(self) -> int | None:
        """Return the first complete interval's zero-based index, or None if absent."""
        index = int(np.searchsorted(self.height_edges_m, self.surface_altitude_m, side="left"))
        return index if index < self.height_edges_m.size - 1 else None

    @property
    def partial_layer_index(self) -> int | None:
        """Return the intersected interval's zero-based index, or None at an interface."""
        index = int(np.searchsorted(self.height_edges_m, self.surface_altitude_m, side="left"))
        return None if self.height_edges_m[index] == self.surface_altitude_m else index - 1

    @property
    def dry_air_column_mol_m2(self) -> float:
        """Return the total retained dry-air amount in mol/m2 for display or ratios."""
        return float(self.layer_dry_air_column_molecules_m2.sum() / _AVOGADRO_MOL1)

    @property
    def partial_h2o_self_density_product(self) -> float:
        """Return the partial interval's water self product, or zero, in molecules2/m5."""
        index = self.partial_layer_index
        return (
            0.0
            if index is None
            else float(self.layer_water_self_density_product_molecules2_m5[index])
        )

    @property
    def partial_h2o_foreign_density_product(self) -> float:
        """Return the partial interval's foreign product, or zero, in molecules2/m5."""
        index = self.partial_layer_index
        return (
            0.0
            if index is None
            else float(self.layer_water_foreign_density_product_molecules2_m5[index])
        )


def dry_air_molar_density(
    pressure_pa: NDArray[np.float64],
    temperature_k: NDArray[np.float64],
    specific_humidity_kg_kg: NDArray[np.float64],
    *,
    water_dry_mass_ratio: float,
) -> NDArray[np.float64]:
    """Convert humidity and ideal-gas state to dry-air molar density.

    Args:
        pressure_pa: Positive pressure on arbitrary common axes, Pa.
        temperature_k: Positive temperature on the same axes, K.
        specific_humidity_kg_kg: Moist-air water mass fraction on those axes, [0,1).
        water_dry_mass_ratio: Positive water-to-dry-air molecular-mass ratio.
            The caller selects this convention explicitly for historical reproduction.

    Returns:
        Owned dry-air molar density in mol/m3, using R=8.31446261815324 J/mol/K.

    Raises:
        ValueError: Axes, finiteness or physical domains fail.
    """
    p, t, q = pressure_pa, temperature_k, specific_humidity_kg_kg
    if (
        p.shape != t.shape
        or q.shape != p.shape
        or not all(np.isfinite(v).all() for v in (p, t, q))
        or np.any(p <= 0)
        or np.any(t <= 0)
        or np.any((q < 0) | (q >= 1))
        or not np.isfinite(water_dry_mass_ratio)
        or water_dry_mass_ratio <= 0
    ):
        raise ValueError("aligned physical pressure, temperature and humidity required")
    ratio = q / (1 - q)
    return (p - p * ratio / (water_dry_mass_ratio + ratio)) / (8.31446261815324 * t)
