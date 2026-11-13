"""Build sampled physical acquisitions and coordinate-measure quadrature."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from orsia_instruments.acquisition.schedule import LinearTrajectory, ReadSchedule


@dataclass(frozen=True, eq=False)
class Acquisition[T: np.floating | np.integer = np.floating | np.integer]:
    """Physical read identities and sampled time, scene direction and detector support.

    N reads, Q quadrature nodes per read. Times are seconds in a caller-chosen
    epoch; directions (u,v) are dimensionless forward-hemisphere coordinates.
    rectangular_acquisition uses direction cosines; along_track_acquisition uses
    bounded local tangent slopes and its explicit paraxial angular measure.
    ``weights_sr_s`` contain only dOmega dt, not optical acceptance or pixel area.
    ``image_xy_m`` are supplied image centroids before architecture-specific mapping.
    Pixel bounds are (xmin,xmax,ymin,ymax), not centres or pitch. Direction support
    bounds (umin,umax,vmin,vmax), shape (N,4), enclose each read's entire sky patch,
    not merely the quadrature nodes; they enable whole-support coverage checks.
    All arrays are
    independent immutable snapshots preserving input dtype. The caller owns
    ray/pointing interpretation.

    Attributes:
        reads: Shared immutable identities, pixels, settings, ports and exposure intervals.
        sample_ids: Derived unique ordered read identities from reads.
        pixel_ids: Derived pixel identities from reads.
        setting_ids: Derived optical setting identities from reads.
        port_ids: Derived physical port identities from reads.
        exposure_bounds_s: Derived (reads,2) intervals from reads, in seconds.
        times_s: (reads,nodes) quadrature times in caller epoch seconds.
        directions_uv: (reads,nodes,2) dimensionless scene directions.
        image_xy_m: (reads,nodes,2) focal-plane centroids in metres.
        weights_sr_s: (reads,nodes) solid-angle-times-time quadrature weights.
        pixel_bounds_m: (reads,4) detector pixel rectangles in metres.
        direction_bounds_uv: (reads,4) whole-patch scene direction support.
    """

    reads: ReadSchedule[T]
    times_s: NDArray[T]
    directions_uv: NDArray[T]
    image_xy_m: NDArray[T]
    weights_sr_s: NDArray[T]
    pixel_bounds_m: NDArray[T]
    direction_bounds_uv: NDArray[T]

    def __post_init__(self) -> None:
        n = len(self.reads.sample_ids)
        for name in (
            "times_s",
            "directions_uv",
            "image_xy_m",
            "weights_sr_s",
            "pixel_bounds_m",
            "direction_bounds_uv",
        ):
            array = np.copy(getattr(self, name))
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            )
        if self.times_s.ndim != 2 or self.times_s.shape[0] != n or self.times_s.shape[1] == 0:
            raise ValueError("times_s must have shape (reads, nodes) with nonempty nodes")
        shape = self.times_s.shape
        if (
            self.exposure_bounds_s.shape != (n, 2)
            or self.directions_uv.shape != (*shape, 2)
            or self.image_xy_m.shape != (*shape, 2)
            or self.weights_sr_s.shape != shape
            or self.pixel_bounds_m.shape != (n, 4)
            or self.direction_bounds_uv.shape != (n, 4)
        ):
            raise ValueError("acquisition array shapes disagree")
        if np.any(np.diff(self.exposure_bounds_s, axis=1) <= 0):
            raise ValueError("exposures must have positive duration")
        if np.any(
            (self.times_s < self.exposure_bounds_s[:, :1])
            | (self.times_s > self.exposure_bounds_s[:, 1:])
        ):
            raise ValueError("quadrature times lie outside exposure support")
        if np.any(np.sum(self.directions_uv**2, axis=-1) >= 1):
            raise ValueError("directions must be inside the forward unit disk")
        bounds = self.direction_bounds_uv
        if np.any(bounds[:, [1, 3]] <= bounds[:, [0, 2]]) or np.any(
            np.max(bounds[:, :2] ** 2, axis=1) + np.max(bounds[:, 2:] ** 2, axis=1) >= 1
        ):
            raise ValueError("direction support must be an ordered patch inside the unit disk")
        if np.any(self.directions_uv < bounds[:, None, [0, 2]]) or np.any(
            self.directions_uv > bounds[:, None, [1, 3]]
        ):
            raise ValueError("quadrature directions lie outside declared support")
        if np.any(self.weights_sr_s < 0) or np.any(self.weights_sr_s.sum(axis=1) <= 0):
            raise ValueError("quadrature weights must be nonnegative with positive support")
        if np.any(self.pixel_bounds_m[:, [1, 3]] <= self.pixel_bounds_m[:, [0, 2]]):
            raise ValueError("pixels must have positive active area")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return the shared read schedule's sample_ids."""
        return self.reads.sample_ids

    @property
    def pixel_ids(self) -> tuple[str, ...]:
        """Return the shared read schedule's pixel_ids."""
        return self.reads.pixel_ids

    @property
    def setting_ids(self) -> tuple[str, ...]:
        """Return the shared read schedule's setting_ids."""
        return self.reads.setting_ids

    @property
    def port_ids(self) -> tuple[str, ...]:
        """Return the shared read schedule's port_ids."""
        return self.reads.port_ids

    @property
    def exposure_bounds_s(self) -> NDArray[T]:
        """Return the shared read schedule's exposure_bounds_s."""
        return self.reads.exposure_bounds_s


def read_schedule[T: np.floating | np.integer](
    acquisition: Acquisition[T] | ReadSchedule[T],
) -> ReadSchedule[T]:
    """Return the immutable identity/timing schedule without reconstructing geometry.

    Args:
        acquisition: Full sampled acquisition or geometry-free physical schedule.

    Returns:
        The original ReadSchedule or the acquisition's shared immutable schedule.
    """
    return acquisition if isinstance(acquisition, ReadSchedule) else acquisition.reads


def rectangular_acquisition(
    *,
    reads: ReadSchedule[np.float64],
    pixel_bounds_m: NDArray[np.float64],
    direction_bounds_uv: tuple[float, float, float, float],
    pointing: Callable[[NDArray[np.float64]], NDArray[np.floating | np.integer]],
    focal_length_m: float,
    time_order: int = 3,
    angular_order: int = 5,
) -> Acquisition:
    """Sample a common rectangular sky patch, with explicit supplied pointing history.

    The bounded paraxial image map is f*(direction-pointing), not orbit/geolocation.
    Solid-angle weights use the exact forward-disk measure du dv/sqrt(1-u²-v²).
    ``pointing(times)`` returns (len(times),2) direction-cosine offsets. The patch
    must lie wholly inside the unit disk; it is caller-selected accepted scene
    support, with no inferred zero radiance outside it. Refine time/angular orders.

    Args:
        reads: Immutable ordered identities and finite positive exposure intervals in seconds.
        pixel_bounds_m: (reads,4) detector xmin,xmax,ymin,ymax in metres.
        direction_bounds_uv: (umin,umax,vmin,vmax) accepted sky-direction patch.
        pointing: Time-array callable yielding (nodes,2) direction-cosine offsets.
        focal_length_m: Positive image-plane focal length in metres.
        time_order: Gauss quadrature order across each read's exposure.
        angular_order: Gauss order along each sky-patch direction axis.

    Returns:
        Acquisition with per-read sampled times, directions and image positions.

    Raises:
        ValueError: Patch, focal length, pointing result or quadrature inputs are invalid.
    """
    if not np.isfinite(focal_length_m) or focal_length_m <= 0:
        raise ValueError("focal length must be finite and positive")
    u0, u1, v0, v1 = direction_bounds_uv
    if max(u0 * u0, u1 * u1) + max(v0 * v0, v1 * v1) >= 1:
        raise ValueError("direction patch must lie inside the forward unit disk")
    u, wu = interval_quadrature(u0, u1, angular_order)
    v, wv = interval_quadrature(v0, v1, angular_order)
    uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
    angular_weights = (wu[:, None] * wv).ravel() / np.sqrt(1 - (uv**2).sum(axis=1))
    times, directions, images, weights = [], [], [], []
    for low, high in reads.exposure_bounds_s:
        t, wt = interval_quadrature(low, high, time_order)
        offsets = pointing(t)
        if offsets.shape != (time_order, 2):
            raise ValueError("pointing must return (time nodes,2)")
        times.append(np.repeat(t, uv.shape[0]))
        directions.append(np.tile(uv, (time_order, 1)))
        images.append((focal_length_m * (uv[None, :, :] - offsets[:, None, :])).reshape(-1, 2))
        weights.append((wt[:, None] * angular_weights).ravel())
    return Acquisition(
        reads,
        np.array(times),
        np.array(directions),
        np.array(images),
        np.array(weights),
        pixel_bounds_m,
        np.tile(direction_bounds_uv, (len(reads.sample_ids), 1)),
    )


def along_track_acquisition(
    reads: ReadSchedule[np.float64],
    trajectory: LinearTrajectory,
    *,
    target_xy_m: NDArray[np.float64],
    pixel_bounds_m: NDArray[np.float64],
    along_edges_m: NDArray[np.float64],
    cross_edges_m: NDArray[np.float64],
    focal_length_m: float,
    pointing_law: Callable[[int, NDArray[np.float64]], NDArray[np.float64]],
    support_frame: str = "ground",
    cross_offset_m: NDArray[np.float64] | None = None,
    time_order: int = 3,
    angular_order: int = 5,
    cross_order: int | None = None,
) -> Acquisition[np.float64]:
    """Sample physical reads over a ground-fixed patch or platform-following slit.

    The platform ground projection is x=speed*(t-time_origin); its cross origin is
    chosen per target column. Exposure centres come from reads; the along offset
    at each centre is target_x-platform_x(centre). Cross offsets are optional
    (reads,) metres from that column origin. Ground support has target-relative
    along edges; platform support has edges relative to the moving platform
    projection plus the centre along offset. Cross edges remain target-relative.
    Directions store the bounded paraxial slopes ray_xy/height in Acquisition's
    uv slots. Image centroids are focal_length*(ray_uv-pointing_uv), with the
    pointing law evaluated per read at temporal Gauss nodes. An explicit zero
    local slope retains the original image mapping. The declared patch encloses
    the *entire* swept exposure, including interval endpoints, inside the unit disk.
    Weights deliberately use dOmega dt=dx dy dt/height**2, independently of the
    forward-disk measure in rectangular_acquisition. This local convention and
    its physical accuracy are caller-selected; returned node arrays are independent
    immutable snapshots. The read schedule is shared. Inputs remain unmodified;
    record validation occurs on each call.

    Args:
        reads: Ordered physical identities and exposure intervals in seconds.
        trajectory: Platform height, along-track speed and time origin.
        target_xy_m: (reads,2) target positions on the local ground plane in metres.
        pixel_bounds_m: (reads,4) focal-plane detector bounds in metres.
        along_edges_m: Increasing piecewise along support edges in ground metres.
        cross_edges_m: Increasing target-relative cross support edges in metres.
        focal_length_m: Positive detector image-plane focal length in metres.
        pointing_law: (read_index, temporal_times_s) to (time_nodes,2) finite
            dimensionless local slopes; input time nodes are borrowed.
        support_frame: 'ground' for target-fixed or 'platform' for moving along support.
        cross_offset_m: Optional (reads,) local cross ground offsets in metres.
        time_order: Positive Gauss order along each exposure interval.
        angular_order: Positive Gauss order along the ground along axis.
        cross_order: Optional distinct positive Gauss order on the cross axis.

    Returns:
        Acquisition sharing the immutable read schedule, with immutable sampled
        directions, image positions and physical dOmega dt weights for each read.

    Raises:
        ValueError: Geometry, node orders or bounded patch is invalid.
    """
    sample_ids = reads.sample_ids
    height_m, speed_m_s = trajectory.height_m, trajectory.speed_m_s
    time_origin_s = trajectory.time_origin_s
    n = len(sample_ids)
    parameters = (height_m, speed_m_s, focal_length_m)
    if not np.isfinite(parameters).all() or min(parameters) <= 0:
        raise ValueError("height, speed and focal length must be finite and positive")
    if not np.isfinite(time_origin_s):
        raise ValueError("time origin must be finite")
    if support_frame not in ("ground", "platform"):
        raise ValueError("support_frame must be ground or platform")
    if target_xy_m.shape != (n, 2) or not np.isfinite(target_xy_m).all():
        raise ValueError("targets must be finite (reads,2)")
    if cross_offset_m is None:
        cross_offset_m = np.zeros(n)
    if cross_offset_m.shape != (n,) or not np.isfinite(cross_offset_m).all():
        raise ValueError("cross offsets must be finite (reads,)")
    if any(
        not isinstance(order, (int, np.integer)) or order < 1
        for order in (
            time_order,
            angular_order,
            cross_order if cross_order is not None else angular_order,
        )
    ):
        raise ValueError("integration orders must be positive integers")
    exposures = reads.exposure_bounds_s
    centre_s = exposures.mean(axis=1)
    centre_along_offset_m = target_xy_m[:, 0] - speed_m_s * (centre_s - time_origin_s)
    x, wx = _piecewise_quadrature(along_edges_m, angular_order)
    y, wy = _piecewise_quadrature(
        cross_edges_m, angular_order if cross_order is None else cross_order
    )
    times, directions, images, weights, supports = [], [], [], [], []
    for read, (low, high) in enumerate(exposures):
        t, wt = interval_quadrature(low, high, time_order)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        ray_x = np.broadcast_to(xx, (len(t), *xx.shape)).copy()
        if support_frame == "ground":
            ray_x += (target_xy_m[read, 0] - speed_m_s * (t - time_origin_s))[:, None, None]
            limits = (
                target_xy_m[read, 0] + along_edges_m[0] - speed_m_s * (high - time_origin_s),
                target_xy_m[read, 0] + along_edges_m[-1] - speed_m_s * (low - time_origin_s),
            )
        else:
            ray_x += centre_along_offset_m[read]
            limits = tuple(centre_along_offset_m[read] + along_edges_m[[0, -1]])
        ray_y = np.broadcast_to(yy + cross_offset_m[read], ray_x.shape)
        uv = np.column_stack((ray_x.ravel(), ray_y.ravel())) / height_m
        pointing = np.asarray(pointing_law(read, t))
        if pointing.shape != (len(t), 2) or not np.isfinite(pointing).all():
            raise ValueError("pointing_law must return finite (time nodes,2) local slopes")
        times.append(np.repeat(t, xx.size))
        directions.append(uv)
        images.append(
            focal_length_m * (uv.reshape(len(t), xx.size, 2) - pointing[:, None]).reshape(-1, 2)
        )
        weights.append(
            (wt[:, None, None] * wx[None, :, None] * wy[None, None, :] / height_m**2).ravel()
        )
        supports.append(
            np.array((*limits, *(cross_offset_m[read] + cross_edges_m[[0, -1]]))) / height_m
        )
    return Acquisition(
        reads,
        np.array(times),
        np.array(directions),
        np.array(images),
        np.array(weights),
        pixel_bounds_m,
        np.array(supports),
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GroundScrollPointing:
    """Follow a ground target at a specified along-track ground rate.

    At each read centre the gaze's along coordinate equals the target's along
    coordinate. Zero ground rate holds a fixed target; a rate equal to platform
    speed keeps a constant local along slope. Each target column defines the
    platform cross origin, so the cross slope depends only on cross_offset_m.
    Attribute bindings are frozen. Input arrays are independently copied, but
    their buffers remain writable; callers must keep them stable during calls.
    Calls borrow times and return fresh writable slope arrays without caches.

    Attributes:
        trajectory: Shared frozen constant-height, constant-speed trajectory.
        target_xy_m: Owned (reads,2) along/cross ground coordinates in metres.
        centre_times_s: Owned (reads,) centres in the trajectory's epoch seconds.
        ground_rate_m_s: Signed along-track ground gaze speed in metres per second.
        cross_offset_m: Owned (reads,) local cross ground gaze offsets in metres.
    """

    trajectory: LinearTrajectory
    target_xy_m: NDArray[np.float64]
    centre_times_s: NDArray[np.float64]
    ground_rate_m_s: float
    cross_offset_m: NDArray[np.float64]

    def __init__(
        self,
        trajectory: LinearTrajectory,
        target_xy_m: NDArray[np.float64],
        centre_times_s: NDArray[np.float64],
        *,
        ground_rate_m_s: float,
        cross_offset_m: NDArray[np.float64] | None = None,
    ) -> None:
        """Snapshot the ground gaze configuration for repeated read/time calls.

        Args:
            trajectory: Positive-height, positive-speed platform trajectory.
            target_xy_m: (reads,2) along/cross target coordinates at centres, metres.
            centre_times_s: (reads,) centres in the trajectory's epoch seconds.
            ground_rate_m_s: Finite signed along-track gaze speed, metres per second.
            cross_offset_m: Optional (reads,) local cross offsets in metres; defaults
                to an owned zero array.

        Raises:
            ValueError: Geometry, target, times or ground rate are invalid.
        """
        n = len(target_xy_m)
        cross = np.zeros(n) if cross_offset_m is None else cross_offset_m
        if (
            target_xy_m.shape != (n, 2)
            or centre_times_s.shape != (n,)
            or cross.shape != (n,)
            or not np.isfinite(target_xy_m).all()
            or not np.isfinite(centre_times_s).all()
            or not np.isfinite(cross).all()
            or not np.isfinite(ground_rate_m_s)
            or not np.isfinite(
                (trajectory.height_m, trajectory.speed_m_s, trajectory.time_origin_s)
            ).all()
            or trajectory.height_m <= 0
            or trajectory.speed_m_s <= 0
        ):
            raise ValueError("ground pointing requires finite aligned coordinates, times and rates")
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "target_xy_m", target_xy_m.copy())
        object.__setattr__(self, "centre_times_s", centre_times_s.copy())
        object.__setattr__(self, "ground_rate_m_s", ground_rate_m_s)
        object.__setattr__(self, "cross_offset_m", cross.copy())

    @classmethod
    def fixed(
        cls,
        trajectory: LinearTrajectory,
        target_xy_m: NDArray[np.float64],
        centre_times_s: NDArray[np.float64],
        *,
        cross_offset_m: NDArray[np.float64] | None = None,
    ) -> GroundScrollPointing:
        """Keep each read's gaze on its fixed ground target throughout exposure.

        Args:
            trajectory: Positive-height, positive-speed platform trajectory.
            target_xy_m: (reads,2) along/cross ground target coordinates in metres.
            centre_times_s: (reads,) centres in the trajectory's epoch seconds.
            cross_offset_m: Optional (reads,) local cross offsets in metres; defaults
                to an owned zero array.

        Returns:
            Callable pointing object with zero ground rate and copied input arrays.

        Raises:
            ValueError: Geometry, target or time arrays are invalid.
        """
        return cls(
            trajectory,
            target_xy_m,
            centre_times_s,
            ground_rate_m_s=0.0,
            cross_offset_m=cross_offset_m,
        )

    def __call__(self, read_index: int, times_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate local along/cross slopes for one read at the requested times.

        Args:
            read_index: Read index with NumPy indexing semantics.
            times_s: Borrowed (time nodes,) times in the trajectory's epoch seconds.

        Returns:
            Fresh writable (time nodes,2) dimensionless along/cross slope array.

        Raises:
            IndexError: The read index is outside the stored read range.
        """
        along = (
            self.target_xy_m[read_index, 0]
            + self.ground_rate_m_s * (times_s - self.centre_times_s[read_index])
            - self.trajectory.speed_m_s * (times_s - self.trajectory.time_origin_s)
        ) / self.trajectory.height_m
        return np.column_stack(
            (
                along,
                np.full(len(times_s), self.cross_offset_m[read_index] / self.trajectory.height_m),
            )
        )


def _piecewise_quadrature(
    edges: NDArray[np.float64], order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("integration edges must be finite and strictly increasing")
    pairs = [
        interval_quadrature(low, high, order)
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]
    return np.concatenate([pair[0] for pair in pairs]), np.concatenate([pair[1] for pair in pairs])


def interval_quadrature(
    low: float, high: float, order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fresh writable nodes and measure weights on a finite positive-width interval.

    Nodes use the units of the bounds; weights integrate that coordinate measure,
    not a normalized probability or physical response. No input is mutated.

    Args:
        low: Finite lower bound in the integrated coordinate's unit.
        high: Finite upper bound strictly above low, in the same unit.
        order: Positive number of Gauss-Legendre nodes.

    Returns:
        Fresh writable (nodes, weights) arrays for the interval measure.

    Raises:
        ValueError: Bounds are invalid or quadrature order is nonpositive.
    """
    if not np.isfinite([low, high]).all() or high <= low or order < 1:
        raise ValueError("quadrature needs finite ordered bounds and positive order")
    x, w = roots_legendre(order)
    return low + (x + 1) * ((high - low) / 2), w * ((high - low) / 2)


def validate_quadrature_order(value: int) -> None:
    """Validate a Gauss-Legendre order before constructing interval nodes.

    Args:
        value: Positive integer number of quadrature nodes.

    Raises:
        ValueError: Order is zero or negative.
    """
    if value < 1:
        raise ValueError("quadrature order must be positive")
