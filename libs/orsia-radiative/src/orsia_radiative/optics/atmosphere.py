"""Atmospheric optical composition and explicit support compatibility checks.

No scene-wide optical state, stream quadrature, delta scaling, source term or
radiance is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.optics.absorption import Absorption
from orsia_radiative.optics.scattering import AerosolComponent, RayleighComponent
from orsia_radiative.scene import RadiativeScene
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid


def compose_atmospheric_optics(
    atmosphere: Atmosphere,
    source: SolarSource,
    *,
    absorption: Absorption | None = None,
    rayleigh: RayleighComponent | None = None,
    aerosol: AerosolComponent | None = None,
) -> AtmosphericOptics:
    """Compose atmospheric optics from total absorption and explicit scattering roles.

    None means an absent role, never implicitly generated atmosphere optics.
    Derive support from source.irradiance, sharing its immutable coordinate array,
    not its illumination, direction or identity. Share optical records without copying.
    Surface and view belong to transfer, not atmospheric optics. The caller must
    recompute contributions when their physical dependencies change. Freshness
    and provenance are not stored in the value; retain an evaluation atmosphere
    separately to check reuse with ``validate_atmospheric_optics``.

    Args:
        atmosphere: Layer support used when checking supplied optical terms.
        source: Solar source whose irradiance grid fixes spectral support.
        absorption: Optional summed molecular absorption on that support.
        rayleigh: Optional Rayleigh scattering optical depths and phase information.
        aerosol: Optional aerosol extinction and scattering on the same layers/grid.

    Returns:
        Optical roles sharing their supplied records and the source's spectral grid.

    Raises:
        ValueError: Any supplied role differs in layer or spectral support.
    """
    return atmospheric_optics(
        atmosphere,
        source.irradiance.grid,
        absorption=absorption,
        rayleigh=rayleigh,
        aerosol=aerosol,
    )


def atmospheric_optics(
    atmosphere: Atmosphere,
    grid: SpectralGrid,
    *,
    absorption: Absorption | None = None,
    rayleigh: RayleighComponent | None = None,
    aerosol: AerosolComponent | None = None,
) -> AtmosphericOptics:
    """Compose solver-neutral optics on explicit sampling without a solar source.

    Exact grid/layer support and ownership follow
    ``compose_atmospheric_optics``. No role is generated or resampled. Retain
    immutable evaluation support, snapshotting mutable caller coordinates.

    Args:
        atmosphere: Layer support for the supplied optical roles.
        grid: Explicit evaluation wavenumber grid in cm-1.
        absorption: Optional summed molecular absorption on that grid.
        rayleigh: Optional Rayleigh scattering on the atmospheric layers.
        aerosol: Optional aerosol scattering and extinction on the same support.

    Returns:
        Atmospheric optical roles on the declared spectral and layer support.

    Raises:
        ValueError: A supplied role has incompatible layers, grid or optical depths.
    """
    for role, component in (
        ("absorption", absorption),
        ("scattering", rayleigh),
        ("scattering", aerosol),
    ):
        if component is not None and len(component.optical_depth) != len(atmosphere.pressure_pa):
            raise ValueError(f"{role} axis and layer order must match the atmosphere/source")
    return AtmosphericOptics(grid, absorption, rayleigh, aerosol)


def validate_atmospheric_optics(
    scene: RadiativeScene,
    source: SolarSource,
    optics: AtmosphericOptics,
    *,
    reference_atmosphere: Atmosphere | None = None,
) -> None:
    """Check original support and an optional explicit atmospheric reference.

    Equal-valued grids are compatible; equal shapes do not prove freshness.
    Changing illumination or direction on the same grid leaves optics reusable.
    Without ``reference_atmosphere``, only numerical support is checked.

    Args:
        scene: Surface grid and atmospheric layer support to verify.
        source: Solar irradiance whose grid must match the evaluated optics.
        optics: Optical terms intended for use in the scene.
        reference_atmosphere: Optional retained atmosphere required to be the scene's
            identical atmosphere object.

    Raises:
        ValueError: Spectral grids or layer support differ, or the reference is stale.
    """
    if not source.irradiance.grid.same_points(optics.grid):
        raise ValueError("source and optical spectral axes must agree exactly")
    validate_scene_optics(scene, optics, reference_atmosphere=reference_atmosphere)


def validate_scene_optics(
    scene: RadiativeScene,
    optics: AtmosphericOptics,
    *,
    reference_atmosphere: Atmosphere | None = None,
) -> None:
    """Check support; optionally check a separately retained atmosphere by identity.

    Without a reference, the caller is responsible for physical freshness.
    Equal layer counts and grids do not prove that optics describe this scene.

    Args:
        scene: Surface spectral grid and atmospheric layer support to check.
        optics: Evaluated optical roles to be used with the scene.
        reference_atmosphere: Optional evaluation atmosphere for identity comparison.

    Raises:
        ValueError: Grids/layers disagree or the reference atmosphere is a different object.
    """
    if not scene.surface.grid.same_points(optics.grid):
        raise ValueError("atmospheric optics and scene spectral axes must agree exactly")
    for component in (optics.absorption, optics.rayleigh, optics.aerosol):
        if component is not None and len(component.optical_depth) != len(
            scene.atmosphere.pressure_pa
        ):
            raise ValueError("optical layer order must match the scene atmosphere")
    if reference_atmosphere is not None and scene.atmosphere is not reference_atmosphere:
        raise ValueError("atmospheric optics are stale for the scene atmosphere")


def validate_optical_inputs(
    atmosphere: Atmosphere,
    source: SolarSource,
    absorption: Absorption | None = None,
    *,
    rayleigh: RayleighComponent | None = None,
    aerosol: AerosolComponent | None = None,
) -> None:
    """Check optical roles against solar sampling and atmospheric layer support.

    Args:
        atmosphere: Atmospheric layer support for all optical terms.
        source: Irradiance providing the reference spectral grid.
        absorption: Optional summed molecular absorption component.
        rayleigh: Optional Rayleigh scattering component.
        aerosol: Optional aerosol scattering and extinction component.

    Raises:
        ValueError: A role has inconsistent grid/layers or total optical depth overflows.
    """
    _validate_materialised_inputs(
        len(atmosphere.pressure_pa), source.irradiance.grid, absorption, rayleigh, aerosol
    )


@dataclass(frozen=True, eq=False)
class AtmosphericOptics:
    """Optical state of one atmosphere on evaluated layer/spectral support.

    This caller-owned value shares optical records; keep their values stable during calls.
    Construction validates optical support and finite extinction, including direct
    construction or dataclass replacement.
    ``wavenumbers_cm1`` retains original evaluation support, sharing the source's
    immutable grid when composed through the public function. It does not select
    a second grid; direct construction must declare the actual evaluation support.
    It is independent of illumination and the selected transfer solver, retaining
    no solar source, extinction work buffer, species collection or hidden cache.
    Rayleigh and aerosol are optional explicit roles, not a generic collection.
    No evaluation atmosphere or contributor provenance is stored. Retain these
    separately; transfer checks numerical support, not physical freshness.

    Attributes:
        grid: Wavenumber grid shared by the materialised optical terms.
        absorption: Optional molecular absorption on this layer/spectral support.
        rayleigh: Optional scalar Rayleigh scattering component.
        aerosol: Optional aerosol extinction and scattering component.
    """

    grid: SpectralGrid
    absorption: Absorption | None
    rayleigh: RayleighComponent | None
    aerosol: AerosolComponent | None
    __hash__ = None

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        _validate_materialised_inputs(
            None,
            self.grid,
            self.absorption,
            self.rayleigh,
            self.aerosol,
        )

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        """Expose the atmospheric optics grid's sampling coordinates.

        Returns:
            Borrowed read-only (points,) wavenumbers in cm-1.
        """
        return self.grid.points


def _validate_materialised_inputs(
    n: int | None,
    grid: SpectralGrid,
    absorption: Absorption | None,
    rayleigh: RayleighComponent | None,
    aerosol: AerosolComponent | None,
) -> None:
    scattering = tuple(component for component in (rayleigh, aerosol) if component is not None)
    require_wavenumber_grid(grid)
    wavenumbers_cm1 = grid.points
    components = tuple(
        component for component in (absorption, *scattering) if component is not None
    )
    if n is None:
        n = len(components[0].optical_depth) if components else 0
    if absorption is not None:
        if not absorption.grid.same_points(grid) or len(absorption.optical_depth) != n:
            raise ValueError("absorption axis and layer order must match the atmosphere/source")
    for constituent in scattering:
        if not constituent.grid.same_points(grid) or len(constituent.optical_depth) != n:
            raise ValueError("scattering axis and layer order must match the atmosphere/source")
    # Keep one spectral row at a time: validation must not duplicate the full cube.
    with np.errstate(over="ignore", invalid="ignore"):
        for layer in range(n):
            total = np.zeros(wavenumbers_cm1.size, dtype=np.float64)
            if absorption is not None:
                total += absorption.optical_depth[layer]
            for constituent in scattering:
                total += constituent.optical_depth[layer]
            if not np.all(np.isfinite(total)):
                raise ValueError("summed optical depth overflow")
