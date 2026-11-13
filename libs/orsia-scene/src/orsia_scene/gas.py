"""Bind species-labelled mass fields and molecular columns to physical supports."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray
from orsia_dispersion import GaussianPuffField, LayerMassMoments, path_columns
from orsia_geometry import LayeredCells
from orsia_radiative.atmosphere.hydrostatic import AVOGADRO_MOL1
from scipy.constants import Avogadro


@overload
def puff_molecular_columns(
    starts_m: NDArray[np.float64],
    ends_m: NDArray[np.float64],
    layer_indices: NDArray[np.int64],
    *,
    height_edges_m: NDArray[np.float64],
    background_columns_molecules_m2: NDArray[np.float64],
    plumes: Sequence[GaussianPuffField],
    species_indices: NDArray[np.int64],
    molar_masses_kg_mol: NDArray[np.float64],
    scales: NDArray[np.float64],
    scale_tangent: None = None,
) -> NDArray[np.float64]: ...


@overload
def puff_molecular_columns(
    starts_m: NDArray[np.float64],
    ends_m: NDArray[np.float64],
    layer_indices: NDArray[np.int64],
    *,
    height_edges_m: NDArray[np.float64],
    background_columns_molecules_m2: NDArray[np.float64],
    plumes: Sequence[GaussianPuffField],
    species_indices: NDArray[np.int64],
    molar_masses_kg_mol: NDArray[np.float64],
    scales: NDArray[np.float64],
    scale_tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def puff_molecular_columns(
    starts_m: NDArray[np.float64],
    ends_m: NDArray[np.float64],
    layer_indices: NDArray[np.int64],
    *,
    height_edges_m: NDArray[np.float64],
    background_columns_molecules_m2: NDArray[np.float64],
    plumes: Sequence[GaussianPuffField],
    species_indices: NDArray[np.int64],
    molar_masses_kg_mol: NDArray[np.float64],
    scales: NDArray[np.float64],
    scale_tangent: NDArray[np.float64] | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Add prescribed plume excess to layered background on supplied segments.

    Background density is constant within each layer. Plumes preserve their
    declared reflection boundary and optional compact cutoff on retained segments,
    without terrain transport or mass renormalization. Multiple plumes may contribute
    to the same species. Air density, spectroscopy, geometry and each frozen plume
    remain fixed.

    Args:
        starts_m: Finite (segments,3) segment starts in metres.
        ends_m: Matching segment ends in metres, within the same source layer.
        layer_indices: (segments,) indices into the background layer axis.
        height_edges_m: Increasing finite (layers+1,) interfaces in metres.
        background_columns_molecules_m2: Nonnegative (layers,species) vertical
            molecular amounts. Species order is defined by the caller.
        plumes: Frozen puff fields, one for each scale, including boundary and cutoff.
            Bare populations must first be bound to an explicit GaussianPuffField.
        species_indices: (plumes,) indices into the molecular species axis.
        molar_masses_kg_mol: Positive (plumes,) molecular masses in kg/mol.
        scales: Nonnegative finite (plumes,) dimensionless mass multipliers.
        scale_tangent: Finite signed (plumes,k) scale directions per caller
            parameter. Empty plume sequences and k=0 are supported. None computes
            molecular amounts alone, without allocating or computing tangents.

    Returns:
        Fresh total molecular columns (segments,species) in molecules/m2 and
        signed tangents (segments,species,k) in molecules/m2 per parameter when
        scale_tangent is supplied. Otherwise return only the columns array.

    Raises:
        ValueError: Shapes, physical values, layer assignments or plume inputs
            are invalid, or molecular arithmetic exceeds finite float64 support.
    """
    validate_layer_interfaces(height_edges_m)
    background = background_columns_molecules_m2
    if (
        background.ndim != 2
        or background.shape[0] != len(height_edges_m) - 1
        or background.shape[1] < 1
        or not np.isfinite(background).all()
        or np.any(background < 0)
    ):
        raise ValueError("background requires nonnegative finite (layers,species) columns")
    if (
        starts_m.ndim != 2
        or starts_m.shape[1] != 3
        or ends_m.shape != starts_m.shape
        or not np.isfinite(starts_m).all()
        or not np.isfinite(ends_m).all()
        or layer_indices.shape != (len(starts_m),)
        or np.any((layer_indices < 0) | (layer_indices >= len(background)))
    ):
        raise ValueError("segments and source-layer indices must match atmospheric support")
    lower, upper = height_edges_m[layer_indices], height_edges_m[layer_indices + 1]
    for heights in (starts_m[:, 2], ends_m[:, 2]):
        if np.any(
            ((heights < lower) & ~np.isclose(heights, lower, rtol=0, atol=1e-9))
            | ((heights > upper) & ~np.isclose(heights, upper, rtol=0, atol=1e-9))
        ):
            raise ValueError("segment endpoints must lie in their assigned layer")
    count = len(plumes)
    if (
        species_indices.shape != (count,)
        or np.any((species_indices < 0) | (species_indices >= background.shape[1]))
        or molar_masses_kg_mol.shape != (count,)
        or not np.isfinite(molar_masses_kg_mol).all()
        or np.any(molar_masses_kg_mol <= 0)
        or scales.shape != (count,)
        or not np.isfinite(scales).all()
        or np.any(scales < 0)
        or (
            scale_tangent is not None
            and (
                scale_tangent.ndim != 2
                or scale_tangent.shape[0] != count
                or not np.isfinite(scale_tangent).all()
            )
        )
    ):
        raise ValueError("plume species, positive molar masses, scales and tangent axes must match")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        density = background / np.diff(height_edges_m)[:, None]
        columns = density[layer_indices] * np.linalg.norm(ends_m - starts_m, axis=1)[:, None]
        tangent = (
            None if scale_tangent is None else np.zeros((*columns.shape, scale_tangent.shape[1]))
        )
        for index, plume in enumerate(plumes):
            excess = path_columns(plume, starts_m, ends_m) * (Avogadro / molar_masses_kg_mol[index])
            gas = species_indices[index]
            columns[:, gas] += scales[index] * excess
            if tangent is not None and scale_tangent is not None:
                tangent[:, gas] += excess[:, None] * scale_tangent[index]
    if not np.isfinite(columns).all() or (tangent is not None and not np.isfinite(tangent).all()):
        raise ValueError("molecular column arithmetic exceeds finite float64 support")
    return columns if tangent is None else (columns, tangent)


def validate_layer_interfaces(edges: NDArray[np.float64]) -> None:
    """Validate finite increasing atmospheric interfaces for material integration.

    Args:
        edges: One-dimensional bottom-to-top interfaces in a common metre datum.

    Raises:
        ValueError: Fewer than two, nonfinite or nonincreasing interfaces are supplied.
    """
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("finite increasing atmospheric height edges required")


type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class GasLayerColumns:
    """Own one species' molecular columns and centroids on finite cells.

    Attributes:
        support: Immutable horizontal cells, frame, vertical interfaces and terrain.
        species: Nonempty species identifier matched to supplied spectroscopy.
        columns_molecules_m2: Nonnegative pixel-mean molecular columns (*pixels,layer),
            molecules/m2 per full horizontal cell area.
        centroid_altitude_m: Mean absolute altitude on the same axes, metres in the
            support's datum; zero for empty layers. Amount and centroid arrays are
            owned immutable float64 snapshots. They retain only the first two mass
            moments, independently of any later within-cell reconstruction.
    """

    support: LayeredCells
    species: str
    columns_molecules_m2: FloatArray
    centroid_altitude_m: FloatArray

    def __post_init__(self) -> None:
        """Snapshot gas amounts and validate terrain-clipped centroid support.

        Raises:
            ValueError: Species, amount axes, finite values or centroid support fail.
        """
        if not self.species:
            raise ValueError("nonempty gas species required")
        for name in ("columns_molecules_m2", "centroid_altitude_m"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            owned = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, owned)
        amount, altitude = self.columns_molecules_m2, self.centroid_altitude_m
        if (
            amount.shape != self.support.shape
            or altitude.shape != amount.shape
            or not np.isfinite(amount).all()
            or np.any(amount < 0)
            or not np.isfinite(altitude).all()
        ):
            raise ValueError("finite matching gas columns and centroids required")
        edges, terrain = self.support.height_edges_m, self.support.surface_altitude_m
        lower, upper = np.maximum(edges[:-1], terrain[..., None]), edges[1:]
        tolerance = (
            64 * np.finfo(float).eps * np.maximum(1, np.maximum(np.abs(lower), np.abs(upper)))
        )
        if np.any((amount == 0) & (altitude != 0)) or np.any(
            (amount > 0)
            & ((lower >= upper) | (altitude < lower - tolerance) | (altitude > upper + tolerance))
        ):
            raise ValueError("gas centroids must lie inside their clipped layer support")


def gas_layer_columns(
    mass: LayerMassMoments, *, species: str, molar_mass_kg_mol: float
) -> GasLayerColumns:
    """Convert finite-cell mass moments to species-labelled molecular columns.

    Args:
        mass: Mass columns and first altitude moments on one complete support.
        species: Nonempty spectroscopy species identifier, for example CH4.
        molar_mass_kg_mol: Explicit positive finite species molar mass, kg/mol.

    Returns:
        Immutable molecular amounts and absolute centroids retaining the exact
        same support owner. Conversion uses 6.02214076e23 molecules/mol. Empty
        centroids stay zero; excluded mass is not renormalized into the domain.

    Raises:
        ValueError: Molar mass, species or resulting finite gas amounts are invalid.
    """
    if not np.isfinite(molar_mass_kg_mol) or molar_mass_kg_mol <= 0:
        raise ValueError("positive finite molar mass required")
    centroid = np.divide(
        mass.height_moments_kg_m,
        mass.columns_kg_m2,
        out=np.zeros_like(mass.columns_kg_m2),
        where=mass.columns_kg_m2 > 0,
    )
    return GasLayerColumns(
        mass.support, species, mass.columns_kg_m2 / molar_mass_kg_mol * AVOGADRO_MOL1, centroid
    )
