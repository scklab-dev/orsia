"""Sampled radiance on explicit time and bounded angular-coordinate support."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import Radiance, SpectralGrid
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True, eq=False)
class RadianceField[T: np.floating | np.integer = np.floating | np.integer]:
    """Unpolarised point radiance (time,u,v,spectral), W m-2 sr-1 per (cm-1).

    Time/u/v axes each have at least two increasing nodes and bound interpolation
    support. Dimensionless u/v axes must match the supplied acquisition's angular
    coordinate convention and frame: direction cosines for rectangular_acquisition,
    bounded local paraxial slopes for along_track_acquisition. In both conventions,
    the entire u/v support rectangle must satisfy u**2 + v**2 < 1. Times must share
    the acquisition's epoch. Spectral support must be a single declared interval
    whose endpoints equal the sampled cm-1 endpoints; spectral nodes are used directly.
    Numerical values are writable copies; the immutable grid may be shared.

    Attributes:
        grid: Spectral sampling of radiance values in cm-1.
        times_s: Increasing caller-epoch time knots in seconds.
        u: Increasing dimensionless first direction-coordinate knots.
        v: Increasing dimensionless second direction-coordinate knots.
        values: Owned (time,u,v,spectral) unpolarised radiance samples.
    """

    grid: SpectralGrid
    times_s: NDArray[T]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            self.grid.coordinate != "wavenumber_cm1"
            or self.grid.points.size < 2
            or self.grid.domain is None
            or self.grid.domain.intervals != ((self.grid.points[0], self.grid.points[-1]),)
        ):
            raise ValueError("field needs one explicit fully sampled cm-1 interval")
        for name in ("times_s", "u", "v", "values"):
            array = np.copy(getattr(self, name))
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, array)
        for axis in (self.times_s, self.u, self.v):
            if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError("scene axes need at least two increasing points")
        if max(self.u**2) + max(self.v**2) >= 1:
            raise ValueError("scene direction patch must lie inside the forward unit disk")
        if self.values.shape != (
            len(self.times_s),
            len(self.u),
            len(self.v),
            len(self.grid.points),
        ):
            raise ValueError("radiance shape must be (time,u,v,spectral)")
        if np.any(self.values < 0):
            raise ValueError("radiance must be nonnegative")


def uniform_radiance_field(
    radiance: Radiance,
    *,
    time_bounds_s: tuple[float, float],
    u_bounds: tuple[float, float],
    v_bounds: tuple[float, float],
    spectral_bounds_cm1: tuple[float, float],
) -> RadianceField:
    """Explicitly extend one directional prediction uniformly on supplied support.

    The caller asserts a uniform-scene approximation on the supplied angular/time
    support. Angular bounds must use the acquisition's coordinate convention and
    frame, and time bounds must use its epoch, as specified by RadianceField.
    Any existing declared spectral support must be the same single interval,
    with exactly sampled endpoints. Interpolation stays within supplied support.

    Args:
        radiance: Directional spectral radiance at each sampled wavenumber.
        time_bounds_s: Increasing supported time interval in caller-epoch seconds.
        u_bounds: Increasing first-direction support in the acquisition's convention.
        v_bounds: Increasing second-direction support in the same frame.
        spectral_bounds_cm1: Exactly sampled spectral interval endpoints in cm-1.

    Returns:
        Owned field with uniform (time,u,v,spectral) radiance samples.

    Raises:
        ValueError: Spectral support conflicts or requested axes/patch are invalid.
    """
    from orsia_radiative import SpectralDomain

    domain = SpectralDomain("wavenumber_cm1", (spectral_bounds_cm1,))
    if radiance.grid.domain is not None and radiance.grid.domain != domain:
        raise ValueError("cannot override declared radiance spectral support")
    return RadianceField(
        SpectralGrid("wavenumber_cm1", radiance.grid.points, domain),
        np.array(time_bounds_s),
        np.array(u_bounds),
        np.array(v_bounds),
        np.broadcast_to(radiance.values, (2, 2, 2, radiance.values.size)),
    )


def sample_radiance(
    field: RadianceField,
    times_s: NDArray[np.float64],
    directions_uv: NDArray[np.float64],
    spectral_slice: slice,
) -> NDArray[np.float64]:
    """Trilinearly interpolate scene nodes, rejecting all spatial/time extrapolation.

    directions_uv must share the field's angular-coordinate convention and frame;
    times_s must share its epoch. Queries must lie within declared time/u/v support.

    Args:
        field: Bounded sampled radiance with time and two angular-coordinate axes.
        times_s: (nodes,) query times in the field's epoch, in seconds.
        directions_uv: (nodes,2) query coordinates in the field's angular convention.
        spectral_slice: Selected contiguous samples along the wavenumber axis.

    Returns:
        Fresh (nodes,requested spectral samples) interpolated radiance in W m-2 sr-1 per cm-1.

    Raises:
        ValueError: Time or direction samples have incompatible shapes or values.
    """
    if times_s.ndim != 1 or directions_uv.shape != (times_s.size, 2):
        raise ValueError("sample coordinates need (nodes,) times and (nodes,2) directions")
    points = np.column_stack((times_s, directions_uv))
    if not np.all(np.isfinite(points)):
        raise ValueError("sample coordinates must be finite")
    interpolator = RegularGridInterpolator(
        (field.times_s, field.u, field.v),
        field.values[..., spectral_slice],
        method="linear",
        bounds_error=True,
    )
    return interpolator(points)
