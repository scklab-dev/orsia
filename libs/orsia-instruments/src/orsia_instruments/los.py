"""Sample scheduled GCRS sight lines and fully supported spectral radiance."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from orsia_geometry.frames import require_time_support, require_utc_epoch
from orsia_geometry.los import AttitudeLaw, Trajectory, compute_lines_of_sight
from orsia_radiative import SpectralGrid

from .acquisition import Acquisition, ReadSchedule
from .quadrature import interval_quadrature


@dataclass(frozen=True, eq=False)
class LineOfSightSamples:
    """GCRS sight lines with sampled boundaries of the timed sensor FOV.

    The acquisition owns read identities, complete time/FOV support, sensor-frame
    direction cosines, virtual image centroids and quadrature weights. Boundary
    rays cover the four FOV corners at every temporal node and at both exposure
    endpoints. The scene provider certifies continuous support between sampled
    times, including the full angular domain.
    Ray and support arrays are independent writable float64 copies. The shared
    acquisition retains immutable optical node arrays and an immutable read schedule.
    Keep writable arrays unchanged during use.

    Attributes:
        acquisition: Read schedule and sensor-frame quadrature; u/v are direction
            cosines around +instrument-z, with weights in sr s and image in metres.
        epoch: Timezone-aware UTC instant from which times are elapsed SI seconds.
        origins_gcrs_m: (reads,nodes,3) GCRS optical origins in metres.
        directions_gcrs: (reads,nodes,3) unit vectors from instrument toward scene.
        support_times_s: (reads,time_nodes+2) temporal nodes and interval endpoints.
        support_origins_gcrs_m: (reads,time_nodes+2,4,3) optical origins at FOV corners.
        support_directions_gcrs: (reads,time_nodes+2,4,3) corner-ray unit vectors.
    """

    acquisition: Acquisition[np.float64]
    epoch: datetime
    origins_gcrs_m: NDArray[np.float64]
    directions_gcrs: NDArray[np.float64]
    support_times_s: NDArray[np.float64]
    support_origins_gcrs_m: NDArray[np.float64]
    support_directions_gcrs: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_utc_epoch(self.epoch)
        n, q = self.acquisition.times_s.shape
        if self.support_times_s.ndim != 2 or self.support_times_s.shape[0] != n:
            raise ValueError("support times must have shape (reads, temporal samples)")
        b = self.support_times_s.shape[1]
        for name, shape in (
            ("origins_gcrs_m", (n, q, 3)),
            ("directions_gcrs", (n, q, 3)),
            ("support_origins_gcrs_m", (n, b, 4, 3)),
            ("support_directions_gcrs", (n, b, 4, 3)),
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            if "directions" in name and not np.allclose(
                np.linalg.norm(array, axis=-1), 1, atol=1e-10, rtol=0
            ):
                raise ValueError(f"{name} must contain unit vectors")
            object.__setattr__(self, name, array)
        times = np.array(self.support_times_s, dtype=np.float64, copy=True)
        if (
            b < 3
            or not np.isfinite(times).all()
            or np.any(times[:, 0] != self.acquisition.exposure_bounds_s[:, 0])
            or np.any(times[:, -1] != self.acquisition.exposure_bounds_s[:, 1])
            or np.any(np.diff(times, axis=1) < 0)
        ):
            raise ValueError("support times must span ordered exposure endpoints")
        object.__setattr__(self, "support_times_s", times)


# The scene receives borrowed (read,node,3) origins and unit directions,
# (read,node) times and (spectral,) cm^-1 points, and returns (read,node,spectral).
type LineOfSightRadiance = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]
# The provider attests continuous full-support coverage, including spectral edges,
# exposure endpoints and transformed FOV boundaries, per physical read.
type RadianceCoverage = Callable[[LineOfSightSamples, SpectralGrid], NDArray[np.bool_]]


def sample_lines_of_sight(
    reads: ReadSchedule[np.float64],
    *,
    pixel_bounds_m: NDArray[np.float64],
    fov_bounds_uv: NDArray[np.float64],
    epoch: datetime,
    validity_s: tuple[float, float],
    trajectory: Trajectory,
    platform_attitude: AttitudeLaw,
    instrument_attitude: AttitudeLaw,
    mount_offset_p_m: NDArray[np.float64],
    focal_length_m: float,
    time_order: int = 3,
    angular_order: int = 5,
    entrance_bounds_m: NDArray[np.float64] | None = None,
) -> LineOfSightSamples:
    """Sample a read schedule into optical image points and dated GCRS rays.

    Time is elapsed SI seconds since a timezone-aware UTC epoch, including across
    leap seconds. The trajectory supplies GCRS position and velocity. Independent
    platform-to-GCRS and instrument-to-platform attitude laws define pointing. Sensor directions
    are (u,v,sqrt(1-u²-v²)), pointing along +I-z toward the scene. The virtual
    image coordinate on the forward focal plane is f*(u,v)/sqrt(1-u²-v²).
    Time and angular Gauss weights give dt du dv / sqrt(1-u²-v²) in sr s;
    the optical integration applies throughput and pixel acceptance separately.
    Per-read FOV rectangles are
    (umin,umax,vmin,vmax), wholly inside the forward unit disk. Reads may overlap.
    With entrance_bounds_m, quadrature instead samples a finite focal-plane
    rectangle in x/y. Its solid-angle Jacobian is f/(f**2+x**2+y**2)**1.5.
    FOV rectangles must enclose the entrance domain and define conservative
    boundary support; detector rectangles remain independent output apertures.
    Inputs and callback results are borrowed and remain unchanged. Returned ray and
    support arrays own independent writable storage; acquisition nodes and read timing
    are immutable snapshots. The adapter owns continuous interpolation, time/frame
    conversion and continuous support within the supplied validity interval.

    Args:
        reads: Ordered physical identities and positive intervals in elapsed SI seconds.
        pixel_bounds_m: (reads,4) detector rectangles (xmin,xmax,ymin,ymax) in metres.
        fov_bounds_uv: (reads,4) full sensor-frame direction-cosine rectangles.
        epoch: Timezone-aware UTC reference instant with zero UTC offset.
        validity_s: Finite closed interval common to all three callbacks; must
            contain every complete exposure, including its endpoints.
        trajectory: Pure callback of elapsed seconds returning finite (times,3)
            GCRS positions in metres and velocities in metres per second.
        platform_attitude: Pure callback returning proper (times,3,3)
            platform-to-GCRS column-vector rotations.
        instrument_attitude: Independent pure callback returning proper (times,3,3)
            instrument-to-platform rotations, including a moving mount or scan.
        mount_offset_p_m: (3,) platform-frame offset to the optical origin, metres.
        focal_length_m: Positive virtual forward focal-plane focal length in metres.
        time_order: Positive Gauss order on each exposure interval.
        angular_order: Positive Gauss order along each FOV or entrance coordinate.
        entrance_bounds_m: Optional finite ordered (reads,4) entrance rectangles,
            metres. None selects direction-cosine quadrature. Arrays are borrowed.

    Returns:
        GCRS sight lines, sensor image coordinates and complete input supports.

    Raises:
        ValueError: Invalid supports, rotations, state outputs or focal length.
    """
    require_utc_epoch(epoch)
    try:
        require_time_support(
            np.asarray(reads.exposure_bounds_s, dtype=np.float64).ravel(), validity_s
        )
    except ValueError as error:
        raise ValueError(
            "complete exposures must lie in the finite closed validity interval"
        ) from error
    n = len(reads.sample_ids)
    bounds = np.asarray(fov_bounds_uv, dtype=np.float64)
    if (
        bounds.shape != (n, 4)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, [1, 3]] <= bounds[:, [0, 2]])
        or np.any(np.max(bounds[:, :2] ** 2, axis=1) + np.max(bounds[:, 2:] ** 2, axis=1) >= 1)
    ):
        raise ValueError("FOV bounds must be ordered rectangles inside the unit disk")
    if entrance_bounds_m is not None and (
        entrance_bounds_m.shape != (n, 4)
        or not np.isfinite(entrance_bounds_m).all()
        or np.any(entrance_bounds_m[:, [1, 3]] <= entrance_bounds_m[:, [0, 2]])
    ):
        raise ValueError("entrance bounds must be finite ordered (reads,4) rectangles")
    offset = np.asarray(mount_offset_p_m, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("mount offset must be finite (3,) metres")
    if not np.isfinite(focal_length_m) or focal_length_m <= 0:
        raise ValueError("focal length must be finite and positive")
    if entrance_bounds_m is not None:
        entrance_directions = focal_plane_direction_bounds(entrance_bounds_m, focal_length_m)
        if np.any(entrance_directions[:, [0, 2]] < bounds[:, [0, 2]]) or np.any(
            entrance_directions[:, [1, 3]] > bounds[:, [1, 3]]
        ):
            raise ValueError("FOV bounds must enclose the complete entrance rectangle")
    if not isinstance(time_order, int) or not isinstance(angular_order, int):
        raise ValueError("quadrature orders must be positive integers")
    times, directions, images, weights = [], [], [], []
    support_times, origins, directions_gcrs, support_origins, support_directions = (
        [],
        [],
        [],
        [],
        [],
    )
    for i, (start, end) in enumerate(reads.exposure_bounds_s):
        t, wt = interval_quadrature(float(start), float(end), time_order)
        u, wu = interval_quadrature(float(bounds[i, 0]), float(bounds[i, 1]), angular_order)
        v, wv = interval_quadrature(float(bounds[i, 2]), float(bounds[i, 3]), angular_order)
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
        angular_weights = (wu[:, None] * wv).ravel()
        if entrance_bounds_m is None:
            sensor = _unit_sensor_directions(uv)
            angular_weights /= sensor[:, 2]
        else:
            entrance = entrance_bounds_m[i]
            x, wx = interval_quadrature(float(entrance[0]), float(entrance[1]), angular_order)
            y, wy = interval_quadrature(float(entrance[2]), float(entrance[3]), angular_order)
            image = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
            sensor = np.column_stack((image, np.full(len(image), focal_length_m)))
            norm = np.linalg.norm(sensor, axis=1)
            sensor /= norm[:, None]
            uv = sensor[:, :2]
            angular_weights = (wx[:, None] * wy).ravel() * focal_length_m / norm**3
        node_t = np.repeat(t, len(uv))
        # One adapter call includes the exposure endpoints and every temporal node.
        edge_t = np.concatenate(([start], t, [end]))
        corner_uv = np.array(
            [[bounds[i, a], bounds[i, b]] for a, b in ((0, 2), (0, 3), (1, 2), (1, 3))]
        )
        corners = _unit_sensor_directions(corner_uv)
        # Evaluate each callback once on all temporal nodes and exposure endpoints.
        edge_origins, edge_ray = compute_lines_of_sight(
            edge_t,
            epoch=epoch,
            validity_s=validity_s,
            trajectory=trajectory,
            platform_attitude=platform_attitude,
            instrument_attitude=instrument_attitude,
            directions_i=np.concatenate((sensor, corners)),
            mount_offset_p_m=offset,
        )
        node_origins = edge_origins[1:-1, : len(sensor)]
        ray = edge_ray[1:-1, : len(sensor)]
        edge_origins = edge_origins[:, len(sensor) :]
        edge_ray = edge_ray[:, len(sensor) :]
        times.append(node_t)
        directions.append(np.tile(uv, (time_order, 1)))
        images.append(np.tile(focal_length_m * uv / sensor[:, 2, None], (time_order, 1)))
        weights.append(np.tile(angular_weights, time_order) * np.repeat(wt, len(uv)))
        origins.append(node_origins.reshape(-1, 3))
        directions_gcrs.append(ray.reshape(-1, 3))
        support_times.append(edge_t)
        support_origins.append(edge_origins)
        support_directions.append(edge_ray)
    acquisition = Acquisition(
        ReadSchedule(
            reads.sample_ids,
            reads.pixel_ids,
            reads.setting_ids,
            reads.port_ids,
            reads.exposure_bounds_s,
        ),
        np.array(times),
        np.array(directions),
        np.array(images),
        np.array(weights),
        pixel_bounds_m,
        bounds,
    )
    return LineOfSightSamples(
        acquisition,
        epoch,
        np.array(origins),
        np.array(directions_gcrs),
        np.array(support_times),
        np.array(support_origins),
        np.array(support_directions),
    )


def focal_plane_direction_bounds(
    bounds_m: NDArray[np.float64], focal_length_m: float
) -> NDArray[np.float64]:
    """Enclose forward focal-plane rectangles in direction-cosine rectangles.

    Args:
        bounds_m: Borrowed finite ordered (rectangles,4) xmin,xmax,ymin,ymax, metres.
        focal_length_m: Positive focal length, metres.

    Returns:
        Owned (rectangles,4) umin,umax,vmin,vmax bounds including axis extrema.

    Raises:
        ValueError: Rectangles or focal length have invalid physical support.
    """
    bounds = bounds_m
    if (
        bounds.ndim != 2
        or bounds.shape[1:] != (4,)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, [1, 3]] <= bounds[:, [0, 2]])
        or not np.isfinite(focal_length_m)
        or focal_length_m <= 0
    ):
        raise ValueError("finite ordered rectangles and positive focal length required")
    x = np.column_stack((bounds[:, 0], bounds[:, 1], np.clip(0, bounds[:, 0], bounds[:, 1])))
    y = np.column_stack((bounds[:, 2], bounds[:, 3], np.clip(0, bounds[:, 2], bounds[:, 3])))
    denominator = np.sqrt(focal_length_m**2 + x[:, :, None] ** 2 + y[:, None, :] ** 2)
    u, v = x[:, :, None] / denominator, y[:, None, :] / denominator
    return np.column_stack(
        (u.min(axis=(1, 2)), u.max(axis=(1, 2)), v.min(axis=(1, 2)), v.max(axis=(1, 2)))
    )


def sample_line_of_sight_radiance(
    rays: LineOfSightSamples,
    grid: SpectralGrid,
    scene: LineOfSightRadiance,
    coverage: RadianceCoverage,
) -> NDArray[np.float64]:
    """Sample fully supported sight-line radiance per cm^-1.

    The independent coverage callback receives the complete read intervals and
    sensor FOV bounds through rays.acquisition, plus transformed sampled FOV
    corners and their times. It must assert coverage for *every point* of each
    continuous time/FOV support and the entire spectral interval, including
    points between quadrature nodes and sampled corners. Continuous motion and scene domain
    certification belong to the provider. Returned radiance is owned writable
    float64 storage; callbacks borrow their inputs and must be pure. Times are
    elapsed SI seconds from rays.epoch; bind scene time conversion to that UTC
    instant. The coverage callback receives the epoch with the sampled rays.
    Radiance uses the observation-ray travel direction (instrument toward scene);
    provider adapters convert incoming-photon or local RadianceField conventions.

    Args:
        rays: Read-level GCRS rays and complete input time/FOV supports.
        grid: Caller-selected one-interval wavenumber grid in cm^-1.
        scene: Provider returning unpolarized W m^-2 sr^-1 per (cm^-1) radiance
            on (read,node,spectral) from origins, directions, times and points.
        coverage: Provider attestation of full continuous time/FOV and spectral
            support as a boolean (reads,) mask, using its own scene domain policy.

    Returns:
        Owned (reads,nodes,spectral) finite nonnegative spectral radiance.

    Raises:
        ValueError: Incomplete declared support, invalid grid or radiance values.
    """
    points = grid.points
    if (
        grid.coordinate != "wavenumber_cm1"
        or grid.domain is None
        or len(points) < 2
        or grid.domain.intervals != ((points[0], points[-1]),)
        or np.any(points <= 0)
        or np.any(np.diff(points) <= 0)
    ):
        raise ValueError("sight-line radiance requires one fully sampled positive cm^-1 interval")
    supported = np.asarray(coverage(rays, grid))
    if supported.shape != (len(rays.acquisition.sample_ids),) or supported.dtype != np.bool_:
        raise ValueError("scene coverage must return a boolean (reads,) mask")
    if not np.all(supported):
        raise ValueError("entire line-of-sight support must lie within scene coverage")
    values = np.array(
        scene(rays.origins_gcrs_m, rays.directions_gcrs, rays.acquisition.times_s, points),
        dtype=np.float64,
        copy=True,
    )
    if (
        values.shape != (*rays.acquisition.times_s.shape, len(points))
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("scene radiance must be finite nonnegative (reads,nodes,spectral)")
    return values


def _unit_sensor_directions(uv: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((uv, np.sqrt(1 - np.sum(uv**2, axis=-1))))
