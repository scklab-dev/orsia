"""Retain fixed layer cross sections with explicit conditions and spectral resampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import LayerAtmosphere
from orsia_radiative.optics.cross_sections import cross_section_optical_depth
from orsia_radiative.spectral.grid import (
    SpectralGrid,
    convert_spectral_grid,
    require_wavenumber_grid,
)
from orsia_radiative.spectral.interpolation import spectral_interpolation_weights
from orsia_radiative.spectroscopy.components import AbsorptionComponent, _gas


def table_absorption(
    atmosphere: LayerAtmosphere, spectral: SpectralGrid, inputs: LayerCrossSections
) -> AbsorptionComponent:
    """Resample fixed cross sections and apply current molecular columns.

    Args:
        atmosphere: Layer conditions and molecular amounts.
        spectral: Target wavenumber grid required by the current optical component.
        inputs: Species table with explicit interfaces and evaluation conditions.

    Returns:
        Dimensionless absorption component with (layer,spectral) axes.

    Raises:
        ValueError: Spectral support, exact layer conditions or physical values disagree.
    """
    return evaluate_table_absorption(atmosphere, resample_layer_cross_sections(spectral, inputs))


def resample_layer_cross_sections(
    spectral: SpectralGrid, inputs: LayerCrossSections
) -> LayerCrossSections[np.float64]:
    """Interpolate a fixed table in its source coordinate without extrapolation.

    Reciprocal target coordinates are converted for the interpolation and the
    output spectral axis is reversed. Cross sections are point values in m2/molecule;
    no spectral-density Jacobian applies. Exact source knots, including singleton
    tables, remain exact. The caller retains the original table for other grids.

    Args:
        spectral: Increasing wavelength-nm or wavenumber-cm-1 target samples.
        inputs: Immutable species table in either supported spectral coordinate.

    Returns:
        The same physical table concept with float64 values on the target grid,
        independent immutable conditions/interfaces and no retained source-table owner.
        Known source coverage is retained when the target omits a domain; an explicit
        target domain may narrow that coverage but cannot widen it.

    Raises:
        ValueError: A target requires extrapolation or an interpolation stencil crosses a gap.
    """
    reverse = spectral.coordinate != inputs.grid.coordinate
    source_in_target = convert_spectral_grid(inputs.grid) if reverse else inputs.grid
    queries = spectral
    if reverse:
        # An exactly selected reciprocal source knot denotes that original knot,
        # even when two floating-point reciprocal operations would move it by one ULP.
        reciprocal = source_in_target
        indices = np.searchsorted(reciprocal.points, spectral.points)
        clipped = np.minimum(indices, reciprocal.points.size - 1)
        exact = (indices < reciprocal.points.size) & (reciprocal.points[clipped] == spectral.points)
        mapped = 1e7 / spectral.points
        mapped[exact] = inputs.grid.points[::-1][clipped[exact]]
        queries = SpectralGrid(inputs.grid.coordinate, mapped[::-1])
    left, right, wl, wr = spectral_interpolation_weights(inputs.grid, queries)
    source = inputs.cross_sections_m2_per_molecule
    rows = wl * source[:, left] + wr * source[:, right]
    if reverse:
        rows = rows[:, ::-1]
    domain = source_in_target.domain
    if domain is not None:
        if spectral.domain is None:
            spectral = SpectralGrid(spectral.coordinate, spectral.points, domain)
        elif any(
            not any(a <= low <= high <= b for a, b in domain.intervals)
            for low, high in spectral.domain.intervals
        ):
            raise ValueError("requested coverage exceeds the known source domain")
    return LayerCrossSections(
        inputs.species,
        inputs.isotopologue,
        inputs.kind,
        spectral,
        inputs.temperature_k,
        inputs.pressure_pa,
        rows,
        height_edges_m=inputs.height_edges_m,
        line_wing_convention=inputs.line_wing_convention,
        includes_water_continuum=inputs.includes_water_continuum,
    )


def evaluate_table_absorption(
    atmosphere: LayerAtmosphere, table: LayerCrossSections
) -> AbsorptionComponent:
    """Contract fixed cross sections with current gas columns on identical layers.

    Args:
        atmosphere: Molecular amounts and current exact layer conditions.
        table: Immutable table on a wavenumber grid, with declared evaluation T/p.

    Returns:
        Fresh dimensionless (layer,spectral) absorption component.

    Raises:
        ValueError: Interfaces, temperatures, pressures, species or optical depths fail.
    """
    require_wavenumber_grid(table.grid)
    amounts, _ = _gas(atmosphere, table.species)
    if (
        not np.array_equal(table.height_edges_m, atmosphere.height_boundaries_m)
        or not np.array_equal(table.temperature_k, atmosphere.temperature_k)
        or not np.array_equal(table.pressure_pa, atmosphere.pressure_pa)
    ):
        raise ValueError(
            "cross-section layer interfaces and conditions must match atmosphere exactly"
        )
    rows = cross_section_optical_depth(
        np.asarray(table.cross_sections_m2_per_molecule, dtype=np.float64),
        np.asarray(amounts, dtype=np.float64),
    )
    return AbsorptionComponent(
        table.grid,
        table.species,
        table.isotopologue,
        table.kind,
        rows,
        line_wing_convention=table.line_wing_convention,
        includes_water_continuum=table.includes_water_continuum,
    )


@dataclass(frozen=True, eq=False)
class LayerCrossSections[Scalar: np.floating | np.integer]:
    """Own fixed species absorption cross sections on declared layers and samples.

    Values keep their supplied numerical dtype. Conditions and coefficient buffers
    are independent immutable snapshots; a frozen wrapper alone would not protect
    their consistency. One spectral sample is valid for evaluated tables. Resampling
    beyond an exact singleton knot requires additional source samples.

    Attributes:
        species: Nonempty absorber identity.
        isotopologue: Required identity for complete cross sections; optional for continuum.
        kind: Complete cross section or continuum excluding line contributions.
        grid: Increasing nm or cm-1 samples and optional independent coverage.
        temperature_k: Exact positive evaluation temperatures (layer,), kelvin.
        pressure_pa: Exact positive evaluation pressures (layer,), pascals.
        cross_sections_m2_per_molecule: Nonnegative fixed values (layer,spectral), m2/molecule.
        height_edges_m: Increasing physical interfaces (layer+1,), metres in the
            caller-declared altitude datum. No index-only or fabricated support is inferred.
        line_wing_convention: Optional explicit local-line cutoff/pedestal convention.
        includes_water_continuum: Whether a supplied H2O cross section includes continuum.
            None leaves this unknown. Pairing with separate water continuum requires False
            and a matching line convention; no inclusion is inferred from the type name.
    """

    species: str
    isotopologue: str | None
    kind: Literal["cross_section", "continuum_excluding_lines"]
    grid: SpectralGrid
    temperature_k: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    cross_sections_m2_per_molecule: NDArray[Scalar]
    height_edges_m: NDArray[np.float64] = field(kw_only=True)
    line_wing_convention: Literal["unbounded", "truncated", "mt_ckd_local_25"] | None = None
    includes_water_continuum: bool | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        """Validate physical domains and freeze the retained numerical values.

        Raises:
            ValueError: Identity, layer support, dimensions or physical values are invalid.
        """
        if not self.species or self.kind not in ("cross_section", "continuum_excluding_lines"):
            raise ValueError("cross sections need species and supported kind")
        if self.kind == "cross_section" and not self.isotopologue:
            raise ValueError("cross sections require isotopologue identity")
        if self.line_wing_convention not in (None, "unbounded", "truncated", "mt_ckd_local_25"):
            raise ValueError("unsupported line-wing convention")
        if (
            self.temperature_k.ndim != 1
            or self.temperature_k.size == 0
            or self.pressure_pa.shape != self.temperature_k.shape
        ):
            raise ValueError("tabulated conditions need matching nonempty shape (layers,)")
        edges = self.height_edges_m
        if (
            edges.shape != (self.temperature_k.size + 1,)
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("cross-section interfaces must be increasing finite layer edges")
        if self.cross_sections_m2_per_molecule.shape != (
            self.temperature_k.size,
            self.grid.points.size,
        ):
            raise ValueError("cross sections must have shape (layers, spectral points)")
        if (
            not np.isfinite(self.temperature_k).all()
            or not np.isfinite(self.pressure_pa).all()
            or np.any(self.temperature_k <= 0)
            or np.any(self.pressure_pa <= 0)
        ):
            raise ValueError("tabulated layer conditions must be finite and positive")
        if not np.isfinite(self.cross_sections_m2_per_molecule).all() or np.any(
            self.cross_sections_m2_per_molecule < 0
        ):
            raise ValueError("cross section must be finite and nonnegative")
        for name in (
            "height_edges_m",
            "temperature_k",
            "pressure_pa",
            "cross_sections_m2_per_molecule",
        ):
            values = getattr(self, name)
            owned = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
            object.__setattr__(self, name, owned)

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Return borrowed cm-1 samples when this table uses that coordinate.

        Raises:
            ValueError: The table is sampled in wavelength; use its grid explicitly.
        """
        require_wavenumber_grid(self.grid)
        return self.grid.points
