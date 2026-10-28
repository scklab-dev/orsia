"""Local outgoing observation geometry, without geodetic transformations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ObservationGeometry:
    """Local view zenith from the outward surface normal, in degrees.

    The view ray is outgoing photon propagation toward the observer, not the
    instrument's downward look vector. It points away from the surface. Absolute
    view azimuth is not stored: SolarSource supplies an incoming-to-view relative
    azimuth in this horizontally homogeneous geometry. No refraction, curvature,
    terrain orientation or horizontal inhomogeneity is represented.

    Attributes:
        view_zenith_deg: Outgoing photon zenith in degrees from the upward local normal.
    """

    view_zenith_deg: float

    def __post_init__(self) -> None:
        if not isfinite(self.view_zenith_deg) or not 0 <= self.view_zenith_deg < 90:
            raise ValueError("view zenith must be in [0, 90) degrees")
