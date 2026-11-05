"""Intersect rays with WGS84 or spheres and project points onto a fixed local chart."""

from collections.abc import Callable
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from .constants import WGS84_SEMIAXES_M
from .frames import require_time_support, require_utc_epoch, transform_rays


def sphere_intersections(
    origins_m: NDArray[np.float64], directions: NDArray[np.float64], *, radius_m: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Find the nearest forward intersection with a sphere in the rays' frame.

    Args:
        origins_m: Finite (...,3) ray origins in metres.
        directions: Matching unit (...,3) vectors pointing toward the scene.
        radius_m: Finite positive radius in metres.

    Returns:
        Owned writable hit mask (...) and points (...,3) in metres. Misses contain
        NaN coordinates. Tangencies hit; surface origins hit at distance zero;
        inside origins use the forward exit. Inputs remain unchanged.

    Raises:
        ValueError: Radius, ray shapes, finite values or unit directions are invalid.
    """
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("sphere radius must be finite and positive")
    _validate_rays(origins_m, directions)
    projection = np.sum(origins_m * directions, axis=-1)
    c = np.sum(origins_m * origins_m, axis=-1) - radius_m**2
    return _quadratic_hits(origins_m, directions, np.sum(directions**2, axis=-1), projection, c)


def ellipsoid_intersections(
    origins_m: NDArray[np.float64],
    directions: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Find forward intersections with WGS84 at ellipsoidal height zero.

    Args:
        origins_m: Finite (...,3) Earth-centred ITRF x/y/z origins in metres.
        directions: Matching unit (...,3) ITRF directions toward the scene.

    Returns:
        Owned writable Boolean hit mask (...) and float64 ITRF points (...,3)
        in metres. The nearest nonnegative distance is selected; surface origins
        hit at zero and inside origins use the forward exit. Misses have NaN
        coordinates. Tangencies hit when the float64 discriminant is nonnegative;
        no horizon snapping is applied. Inputs remain unchanged. The fixed
        reference ellipsoid excludes terrain and atmospheric ray bending.

    Raises:
        ValueError: Ray shapes, finite values or unit directions are invalid.
    """
    axes = np.asarray(WGS84_SEMIAXES_M)
    _validate_rays(origins_m, directions)
    origin = origins_m / axes
    direction = directions / axes
    a = np.sum(direction**2, axis=-1)
    b = np.sum(origin * direction, axis=-1)
    c = np.sum(origin**2, axis=-1) - 1
    return _quadratic_hits(origins_m, directions, a, b, c)


def surface_visibility(
    observers_m: NDArray[np.float64],
    targets_m: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Identify unoccluded WGS84 height-zero targets from exterior observers.

    Args:
        observers_m: Finite (...,3) Earth-centred ITRF x/y/z positions outside
            WGS84, in metres.
        targets_m: Matching ITRF points on WGS84 at ellipsoidal height zero,
            in metres; terrain is excluded.

    Returns:
        Owned Boolean (...) mask. A tangent line is visible. Target residuals must
        satisfy abs(sum((target/axes)**2)-1) <= 1e-10. Inputs remain unchanged.

    Raises:
        ValueError: Shapes, finiteness, surface membership or exterior support fail.
    """
    axes = np.asarray(WGS84_SEMIAXES_M)
    if (
        observers_m.ndim < 1
        or observers_m.shape[-1] != 3
        or targets_m.shape != observers_m.shape
        or not np.isfinite(observers_m).all()
        or not np.isfinite(targets_m).all()
        or np.any(np.sum((observers_m / axes) ** 2, axis=-1) <= 1)
        or not np.allclose(np.sum((targets_m / axes) ** 2, axis=-1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("visibility needs exterior observers and matching surface targets")
    return np.array(
        np.sum((observers_m - targets_m) * (targets_m / axes**2), axis=-1) >= 0, copy=True
    )


def project_local_east_north(
    points_itrf_m: NDArray[np.float64], *, origin_itrf_m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project ITRF displacements onto a fixed WGS84 east/north tangent plane.

    The explicit height-zero origin fixes the ellipsoid-gradient normal and the
    orthonormal east/north axes. The orthographic chart discards normal displacement;
    coordinates are projected metre displacements, rather than geodesic distances.

    Args:
        points_itrf_m: Borrowed (...,3) Earth-centred ITRF x/y/z points in metres.
            Each triplet must be finite or all NaN for a missed ray. Points may
            lie off the ellipsoid. Partial nonfinite triplets are invalid.
        origin_itrf_m: Finite (3,) ITRF height-zero chart origin in metres. The
            dimensionless residual abs(sum((origin/axes)**2)-1) must be <= 1e-10.
            An exact pole (x=y=0) is invalid because east is undefined.

    Returns:
        Owned writable float64 (...,2) coordinates ordered east, north in metres.
        All-NaN triplets remain all-NaN pairs. Scalar and empty leading axes are
        preserved. Input arrays remain unchanged.

    Raises:
        ValueError: Shapes, point finiteness, origin surface membership or polar
            chart support are invalid.
    """
    axes = _local_east_north_axes(origin_itrf_m)
    if (
        points_itrf_m.ndim < 1
        or points_itrf_m.shape[-1] != 3
        or not np.all(
            np.isfinite(points_itrf_m).all(axis=-1) | np.isnan(points_itrf_m).all(axis=-1)
        )
    ):
        raise ValueError("points must be (...,3) finite triplets or all-NaN misses")
    return (points_itrf_m - origin_itrf_m) @ axes


def wgs84_surface_projection(
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    chart_origin_itrf_m: NDArray[np.float64],
) -> Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.bool_]],
]:
    """Build a dated GCRS-ray projection onto one explicit WGS84 metre chart.

    Construction snapshots the chart origin and validates local support without
    loading SatKit. Evaluation composes transform_rays, ellipsoid_intersections
    and project_local_east_north. Use the same origin for compared geometries.
    Callers certify continuous visibility and motion/angular support bounds;
    sampled intersections alone supply no such certificate.

    Args:
        epoch: Timezone-aware UTC reference instant with zero offset.
        validity_s: Finite closed accepted interval in physical elapsed SI seconds.
        chart_origin_itrf_m: Finite nonpolar (3,) ITRF chart origin in metres on
            height-zero WGS84, with dimensionless surface residual <= 1e-10.

    Returns:
        Plain callable surface(origins_gcrs_m, directions_gcrs, times_s) returning
        (xy_m, hits). Matching finite (...,3) GCRS origins and unit directions use
        exactly (...) times. Owned writable outputs have shapes (...,2) and (...),
        ordered east/north metres and Boolean hits. The nearest forward WGS84
        intersection is used; misses return NaN pairs and false hits. Surface
        origins hit at zero; inside origins use the forward exit. Inputs remain
        unchanged. Terrain, geoid heights and ray bending are excluded. Evaluation
        lazily uses SatKit's normal public auxiliaries and process-global caches;
        callers serialize native operations and auxiliary refresh.

    Raises:
        ValueError: Construction has invalid epoch, interval or chart origin, or
            evaluation has invalid ray shapes, finite values, norms or time support.
        ImportError: Evaluation cannot load the required SatKit installation.
        RuntimeError: Evaluation lacks EOP coverage or the backend operation fails.
    """
    require_utc_epoch(epoch)
    require_time_support(np.empty(0), validity_s)
    bounds = np.asarray(validity_s, dtype=np.float64)
    validity = (float(bounds[0]), float(bounds[1]))
    origin = np.array(chart_origin_itrf_m, dtype=np.float64, copy=True)
    _local_east_north_axes(origin)

    def surface(
        origins_gcrs_m: NDArray[np.float64],
        directions_gcrs: NDArray[np.float64],
        times_s: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        origins, directions = transform_rays(
            times_s, origins_gcrs_m, directions_gcrs, epoch=epoch, validity_s=validity
        )
        hits, points = ellipsoid_intersections(origins, directions)
        return project_local_east_north(points, origin_itrf_m=origin), hits

    return surface


def wgs84_surface_projection_from_ray(
    *,
    origin_gcrs_m: NDArray[np.float64],
    direction_gcrs: NDArray[np.float64],
    time_s: float,
    epoch: datetime,
    validity_s: tuple[float, float],
) -> Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.bool_]],
]:
    """Build a fixed east/north chart at one dated GCRS ray's WGS84 intercept.

    Construction transforms the reference ray to ITRF, selects its nearest forward
    WGS84 intersection, and delegates to wgs84_surface_projection. Reuse the returned
    callback for simulation and estimated geometry to keep their chart fixed.

    Args:
        origin_gcrs_m: Borrowed finite (3,) GCRS x/y/z ray origin in metres.
        direction_gcrs: Borrowed finite (3,) GCRS scene-directed unit vector,
            with absolute norm tolerance 1e-10.
        time_s: Finite scalar physical elapsed SI seconds from epoch, within validity_s.
        epoch: Timezone-aware UTC reference instant with zero offset.
        validity_s: Finite closed accepted interval in physical elapsed SI seconds.

    Returns:
        Plain surface(origins_gcrs_m, directions_gcrs, times_s) callback with the
        wgs84_surface_projection contract: matching (...,3) rays and (...) times
        produce owned writable float64 (...,2) east/north metres and Boolean (...)
        hits. Missed evaluation rays have NaN coordinates and false hits. The chart
        and interval are snapshots; reference inputs remain unchanged and are not
        retained. Continuous visibility and support bounds remain caller duties.
        Construction and evaluation use the required SatKit backend and its normal
        public auxiliaries and process-global caches; callers serialize native
        operations and auxiliary refresh. Terrain and ray bending are excluded.

    Raises:
        ValueError: Reference ray shapes, finite values, unit norm, epoch or time
            support are invalid, the reference ray misses WGS84, or its intercept
            is an exact pole. Callback validation follows wgs84_surface_projection.
        ImportError: Construction or evaluation cannot load the required SatKit installation.
        RuntimeError: Required EOP coverage or backend evaluation fails.
    """
    time = np.asarray(time_s, dtype=np.float64)
    if origin_gcrs_m.shape != (3,) or direction_gcrs.shape != (3,) or time.shape != ():
        raise ValueError("reference ray needs (3,) origin/direction and scalar time_s")
    origin, direction = transform_rays(
        time, origin_gcrs_m, direction_gcrs, epoch=epoch, validity_s=validity_s
    )
    hit, point = ellipsoid_intersections(origin, direction)
    if not hit:
        raise ValueError("reference ray misses WGS84")
    return wgs84_surface_projection(epoch=epoch, validity_s=validity_s, chart_origin_itrf_m=point)


def _local_east_north_axes(origin: NDArray[np.float64]) -> NDArray[np.float64]:
    axes = np.asarray(WGS84_SEMIAXES_M)
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or abs(np.sum((origin / axes) ** 2) - 1) > 1e-10
    ):
        raise ValueError("chart origin must be finite (3,) on height-zero WGS84")
    equatorial_radius = np.hypot(origin[0], origin[1])
    if equatorial_radius == 0:
        raise ValueError("east is undefined at an exact pole chart origin")
    normal = origin / axes**2
    normal /= np.linalg.norm(normal)
    east = np.array([-origin[1] / equatorial_radius, origin[0] / equatorial_radius, 0.0])
    north = np.cross(normal, east)
    return np.column_stack((east, north))


def _validate_rays(origins: NDArray[np.float64], directions: NDArray[np.float64]) -> None:
    if (
        origins.ndim < 1
        or origins.shape[-1] != 3
        or directions.shape != origins.shape
        or not np.isfinite(origins).all()
        or not np.isfinite(directions).all()
        or not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("rays need matching finite (...,3) origins and unit directions")


def _quadratic_hits(
    origins: NDArray[np.float64],
    directions: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    discriminant = b**2 - a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    q = -b - np.copysign(root, b)
    first = q / a
    second = np.divide(c, q, out=np.zeros_like(c), where=q != 0)
    near, far = np.minimum(first, second), np.maximum(first, second)
    distance = np.where(near >= 0, near, far)
    hit = (discriminant >= 0) & (distance >= 0)
    points = np.where(hit[..., None], origins + distance[..., None] * directions, np.nan)
    return np.array(hit, copy=True), points
