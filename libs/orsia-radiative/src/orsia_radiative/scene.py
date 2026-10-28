"""Atmosphere, surface and observation geometry independent of active sources."""

from __future__ import annotations

from dataclasses import dataclass

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.geometry import ObservationGeometry
from orsia_radiative.surface import SurfaceState


@dataclass(frozen=True, eq=False)
class RadiativeScene:
    """Physical inputs and view geometry, independent of active radiation sources.

    Reflectance does not select illumination. Transfer operations consume the
    explicit sources supplied by the caller.

    Attributes:
        atmosphere: Materialised layer temperatures, pressures and gas columns.
        surface: Opaque Lambertian reflectance or thermal boundary state.
        geometry: Local outgoing observation direction at the surface.
    """

    atmosphere: Atmosphere
    surface: SurfaceState
    geometry: ObservationGeometry
    __hash__ = None
