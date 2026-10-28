"""Sample scheduled sensor sight lines in an Earth frame for radiance integration."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .acquisition import Acquisition, ReadSchedule
from .quadrature import interval_quadrature

# The adapter supplies a position (times,3) and a proper P-to-E rotation (times,3,3).
type PlatformState = Callable[
    [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]


@dataclass(frozen=True, eq=False)
class LineOfSightSamples:
    """Earth-frame sight lines with sampled boundaries of the timed sensor FOV.

    The acquisition owns read identities, complete time/FOV support, sensor-frame
    direction cosines, virtual image centroids and quadrature weights. Boundary
    rays cover the four FOV corners at every temporal node and at both exposure
    endpoints. The scene provider certifies continuous support between sampled
    times, including the full angular domain.
    Numeric arrays are independent writable float64 copies; the acquisition is
    shared and its arrays are mutable. Keep these inputs unchanged during use.

    Attributes:
        acquisition: Read schedule and sensor-frame quadrature; u/v are direction
            cosines around +instrument-z, with weights in sr s and image in metres.
        origins_e_m: (reads,nodes,3) Earth-frame optical origins in metres.
        directions_e: (reads,nodes,3) unit vectors from instrument toward scene.
        support_times_s: (reads,time_nodes+2) temporal nodes and interval endpoints.
        support_origins_e_m: (reads,time_nodes+2,4,3) optical origins at FOV corners.
        support_directions_e: (reads,time_nodes+2,4,3) corner-ray unit vectors.
    """

    acquisition: Acquisition[np.float64]
    origins_e_m: NDArray[np.float64]
    directions_e: NDArray[np.float64]
    support_times_s: NDArray[np.float64]
    support_origins_e_m: NDArray[np.float64]
    support_directions_e: NDArray[np.float64]

    def __post_init__(self) -> None:
        n, q = self.acquisition.times_s.shape
        if self.support_times_s.ndim != 2 or self.support_times_s.shape[0] != n:
            raise ValueError("support times must have shape (reads, temporal samples)")
        b = self.support_times_s.shape[1]
        for name, shape in (
            ("origins_e_m", (n, q, 3)),
            ("directions_e", (n, q, 3)),
            ("support_origins_e_m", (n, b, 4, 3)),
            ("support_directions_e", (n, b, 4, 3)),
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


def sample_lines_of_sight(
    reads: ReadSchedule[np.float64],
    *,
    pixel_bounds_m: NDArray[np.float64],
    fov_bounds_uv: NDArray[np.float64],
    platform_state: PlatformState,
    mount_offset_p_m: NDArray[np.float64],
    mount_rotation_p_i: NDArray[np.float64],
    focal_length_m: float,
    time_order: int = 3,
    angular_order: int = 5,
) -> LineOfSightSamples:
    """Sample a read schedule into optical image points and dated Earth-frame rays.

    Input time is continuous seconds since a caller-named TAI reference epoch;
    the platform adapter supplies Earth-fixed Cartesian positions and P-to-E
    attitudes at every requested time. The mount maps I to P. Sensor directions
    are (u,v,sqrt(1-u²-v²)), pointing along +I-z toward the scene. The virtual
    image coordinate on the forward focal plane is f*(u,v)/sqrt(1-u²-v²).
    Time and angular Gauss weights give dt du dv / sqrt(1-u²-v²) in sr s;
    the optical integration applies throughput and pixel acceptance separately.
    Per-read FOV rectangles are
    (umin,umax,vmin,vmax), wholly inside the forward unit disk. Reads may overlap.
    Inputs and callback results are borrowed and remain unchanged; returned arrays
    own independent writable storage. The adapter owns continuous interpolation,
    time/frame conversion and valid-domain enforcement.

    Args:
        reads: Ordered physical identities and positive exposure intervals in TAI seconds.
        pixel_bounds_m: (reads,4) detector rectangles (xmin,xmax,ymin,ymax) in metres.
        fov_bounds_uv: (reads,4) full sensor-frame direction-cosine rectangles.
        platform_state: Pure time-array adapter returning positions (times,3) metres
            and proper rotation matrices (times,3,3) from platform to Earth frame.
        mount_offset_p_m: (3,) platform-frame offset to the optical origin, metres.
        mount_rotation_p_i: (3,3) proper rotation from instrument to platform frame.
        focal_length_m: Positive virtual forward focal-plane focal length in metres.
        time_order: Positive Gauss order on each exposure interval.
        angular_order: Positive Gauss order along each FOV coordinate.

    Returns:
        Earth-frame sight lines, sensor image coordinates and complete input supports.

    Raises:
        ValueError: Invalid supports, rotations, state outputs or focal length.
    """
    n = len(reads.sample_ids)
    bounds = np.asarray(fov_bounds_uv, dtype=np.float64)
    if (
        bounds.shape != (n, 4)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, [1, 3]] <= bounds[:, [0, 2]])
        or np.any(np.max(bounds[:, :2] ** 2, axis=1) + np.max(bounds[:, 2:] ** 2, axis=1) >= 1)
    ):
        raise ValueError("FOV bounds must be ordered rectangles inside the unit disk")
    offset = np.asarray(mount_offset_p_m, dtype=np.float64)
    rotation = np.asarray(mount_rotation_p_i, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("mount offset must be finite (3,) metres")
    _proper_rotation(rotation, ())
    if not np.isfinite(focal_length_m) or focal_length_m <= 0:
        raise ValueError("focal length must be finite and positive")
    if not isinstance(time_order, int) or not isinstance(angular_order, int):
        raise ValueError("quadrature orders must be positive integers")
    times, directions, images, weights = [], [], [], []
    support_times, origins, directions_e, support_origins, support_directions = [], [], [], [], []
    for i, (start, end) in enumerate(reads.exposure_bounds_s):
        t, wt = interval_quadrature(float(start), float(end), time_order)
        u, wu = interval_quadrature(float(bounds[i, 0]), float(bounds[i, 1]), angular_order)
        v, wv = interval_quadrature(float(bounds[i, 2]), float(bounds[i, 3]), angular_order)
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
        sensor = _unit_sensor_directions(uv)
        node_t = np.repeat(t, len(uv))
        # One adapter call includes the exposure endpoints and every temporal node.
        edge_t = np.concatenate(([start], t, [end]))
        position, attitude = platform_state(edge_t)
        position = np.asarray(position, dtype=np.float64)
        attitude = np.asarray(attitude, dtype=np.float64)
        if position.shape != (len(edge_t), 3) or not np.isfinite(position).all():
            raise ValueError("platform positions must be finite (times,3) metres")
        _proper_rotation(attitude, (len(edge_t),))
        centre = position + np.einsum("tij,j->ti", attitude, offset)
        ray = np.einsum("tij,jk,ak->tai", attitude[1:-1], rotation, sensor)
        corner_uv = np.array(
            [[bounds[i, a], bounds[i, b]] for a, b in ((0, 2), (0, 3), (1, 2), (1, 3))]
        )
        corners = _unit_sensor_directions(corner_uv)
        edge_ray = np.einsum("tij,jk,ak->tai", attitude, rotation, corners)
        times.append(node_t)
        directions.append(np.tile(uv, (time_order, 1)))
        images.append(np.tile(focal_length_m * uv / sensor[:, 2, None], (time_order, 1)))
        weights.append(
            np.tile((wu[:, None] * wv).ravel() / sensor[:, 2], time_order) * np.repeat(wt, len(uv))
        )
        origins.append(np.repeat(centre[1:-1], len(uv), axis=0))
        directions_e.append(ray.reshape(-1, 3))
        support_times.append(edge_t)
        support_origins.append(np.broadcast_to(centre[:, None], (len(edge_t), 4, 3)))
        support_directions.append(edge_ray)
    acquisition = Acquisition(
        reads.sample_ids,
        reads.pixel_ids,
        reads.setting_ids,
        reads.port_ids,
        reads.exposure_bounds_s,
        np.array(times),
        np.array(directions),
        np.array(images),
        np.array(weights),
        pixel_bounds_m,
        bounds,
    )
    return LineOfSightSamples(
        acquisition,
        np.array(origins),
        np.array(directions_e),
        np.array(support_times),
        np.array(support_origins),
        np.array(support_directions),
    )


def sphere_intersections(
    rays: LineOfSightSamples, *, radius_m: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Intersect forward Earth-frame rays with a sphere centred at frame origin.

    Args:
        rays: Dated Earth-frame ray origins in metres and unit directions.
        radius_m: Finite positive sphere radius in Earth-frame metres.

    Returns:
        Hit mask (reads,nodes) and owned points (reads,nodes,3) in metres;
        missed rays have NaN point coordinates. Surface choices stay optional.

    Raises:
        ValueError: Radius is nonfinite or nonpositive.
    """
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("sphere radius must be finite and positive")
    origin, direction = rays.origins_e_m, rays.directions_e
    projection = np.sum(origin * direction, axis=-1)
    c = np.sum(origin * origin, axis=-1) - radius_m**2
    discriminant = projection**2 - c
    root = np.sqrt(np.maximum(discriminant, 0))
    near, far = -projection - root, -projection + root
    distance = np.where(near >= 0, near, far)
    hit = (discriminant >= 0) & (distance >= 0)
    points = np.where(hit[..., None], origin + distance[..., None] * direction, np.nan)
    return hit, points


def _unit_sensor_directions(uv: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((uv, np.sqrt(1 - np.sum(uv**2, axis=-1))))


def _proper_rotation(rotation: NDArray[np.float64], leading: tuple[int, ...]) -> None:
    if rotation.shape != (*leading, 3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be finite with (...,3,3) shape")
    identity = np.eye(3)
    if not np.allclose(
        rotation @ np.swapaxes(rotation, -1, -2), identity, atol=1e-10, rtol=0
    ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0):
        raise ValueError("rotation must be proper and orthonormal")
