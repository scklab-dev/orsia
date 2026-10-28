"""Explicit physical reads and sky-patch/local-scan acquisition quadrature."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quadrature import interval_quadrature


@dataclass(frozen=True, eq=False)
class ReadSchedule[T: np.floating | np.integer = np.floating | np.integer]:
    """Ordered sample/pixel/setting/port identities and exposure intervals in seconds.

    No optical geometry or quadrature is implied. Intervals are finite, positive
    duration (reads,2) writable copies; sample IDs are unique within the set.
    Other IDs may repeat. All IDs are bookkeeping, never physical parameters.

    Attributes:
        sample_ids: Unique ordered identifiers, one per physical read.
        pixel_ids: Ordered focal-plane pixel identifiers for each read.
        setting_ids: Ordered instrument-setting identifiers for each read.
        port_ids: Ordered optical output-port identifiers for each read.
        exposure_bounds_s: Owned (reads,2) start/end times in caller epoch seconds.
    """

    sample_ids: tuple[str, ...]
    pixel_ids: tuple[str, ...]
    setting_ids: tuple[str, ...]
    port_ids: tuple[str, ...]
    exposure_bounds_s: NDArray[T]

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids"):
            values = tuple(getattr(self, name))
            if len(values) != n or not all(values):
                raise ValueError("read identities must be nonempty and match reads")
            object.__setattr__(self, name, values)
        if not n or len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must be nonempty and unique")
        bounds = np.copy(self.exposure_bounds_s)
        if not np.isfinite(bounds).all():
            raise ValueError("exposure bounds must be finite")
        if bounds.shape != (n, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("exposure bounds must be (reads,2) with positive duration")
        object.__setattr__(self, "exposure_bounds_s", bounds)


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
    independent writable copies preserving input dtype. The caller owns ray/pointing interpretation.

    Attributes:
        sample_ids: Unique ordered identifiers, one per read.
        pixel_ids: Pixel identity for each read.
        setting_ids: Instrument setting identity for each read.
        port_ids: Optical port identity for each read.
        exposure_bounds_s: (reads,2) ordered temporal support in seconds.
        times_s: (reads,nodes) quadrature times in caller epoch seconds.
        directions_uv: (reads,nodes,2) dimensionless scene directions.
        image_xy_m: (reads,nodes,2) focal-plane centroids in metres.
        weights_sr_s: (reads,nodes) solid-angle-times-time quadrature weights.
        pixel_bounds_m: (reads,4) detector pixel rectangles in metres.
        direction_bounds_uv: (reads,4) whole-patch scene direction support.
    """

    sample_ids: tuple[str, ...]
    pixel_ids: tuple[str, ...]
    setting_ids: tuple[str, ...]
    port_ids: tuple[str, ...]
    exposure_bounds_s: NDArray[T]
    times_s: NDArray[T]
    directions_uv: NDArray[T]
    image_xy_m: NDArray[T]
    weights_sr_s: NDArray[T]
    pixel_bounds_m: NDArray[T]
    direction_bounds_uv: NDArray[T]

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids"):
            values = tuple(getattr(self, name))
            if len(values) != n or not all(values):
                raise ValueError("read identities must be nonempty and match reads")
            object.__setattr__(self, name, values)
        if n == 0 or len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must be nonempty and unique")
        for name in (
            "exposure_bounds_s",
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
            object.__setattr__(self, name, array)
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


@dataclass(frozen=True, kw_only=True)
class LinearTrajectory:
    """Constant positive height/speed over a local along-track tangent plane.

    Height is metres, speed is metres per second, and time_origin_s is seconds
    in the caller's epoch at along-track x=0. The cross-track platform origin is
    chosen per target column to express local offsets around that column. The
    separate pointing law specifies the gaze.

    Attributes:
        height_m: Constant positive altitude in metres above the local ground plane.
        speed_m_s: Constant positive along-track speed in metres per second.
        time_origin_s: Caller-epoch time when along-track position equals zero.
    """

    height_m: float
    speed_m_s: float
    time_origin_s: float = 0


def read_schedule[T: np.floating | np.integer](
    acquisition: Acquisition[T] | ReadSchedule[T],
) -> ReadSchedule[T]:
    """Retain only read identities/timing; never fabricate optical geometry.

    A ReadSchedule input is returned unchanged; an Acquisition produces a fresh
    independent timing copy. Existing acquisition-bearing outputs are unchanged.

    Args:
        acquisition: Full optical acquisition or geometry-free physical read schedule.

    Returns:
        Original ReadSchedule instance or a new schedule with copied exposure bounds.
    """
    if isinstance(acquisition, ReadSchedule):
        return acquisition
    return ReadSchedule(
        acquisition.sample_ids,
        acquisition.pixel_ids,
        acquisition.setting_ids,
        acquisition.port_ids,
        acquisition.exposure_bounds_s,
    )


def along_track_exposure_bounds(
    target_xy_m: NDArray[np.float64],
    ground_offsets_m: NDArray[np.float64],
    exposure_s: float | NDArray[np.float64],
    trajectory: LinearTrajectory,
) -> NDArray[np.float64]:
    """Return fresh (read, start/end) seconds for a local constant-speed scan.

    Targets and scheduling offsets have (read, along/cross) axes in metres;
    exposure_s is a scalar or one positive duration per read. Positive along
    offsets schedule earlier exposures; pointing is specified separately.
    Inputs are borrowed; the result owns its storage.

    Args:
        target_xy_m: (reads,2) along/cross target positions in ground metres.
        ground_offsets_m: Ground offsets in metres; the along component schedules
            exposure centres and the cross component participates in validation.
        exposure_s: Positive scalar or (reads,) exposure durations in seconds.
        trajectory: Constant-height, constant-speed local trajectory and time origin.

    Returns:
        Fresh writable float64 (reads,2) exposure start/end bounds in seconds.

    Raises:
        ValueError: Target, offset or duration support is invalid.
    """
    if (
        target_xy_m.ndim != 2
        or target_xy_m.shape[1] != 2
        or ground_offsets_m.shape != target_xy_m.shape
        or not np.isfinite(target_xy_m).all()
        or not np.isfinite(ground_offsets_m).all()
    ):
        raise ValueError("targets and ground offsets must be finite (reads,2)")
    return _along_track_exposures(target_xy_m[:, 0], ground_offsets_m[:, 0], exposure_s, trajectory)


def along_track_read_schedule(
    *,
    sample_ids: tuple[str, ...],
    pixel_ids: tuple[str, ...],
    setting_ids: tuple[str, ...],
    port_ids: tuple[str, ...],
    target_along_m: NDArray[np.float64],
    ground_along_offsets_m: NDArray[np.float64],
    exposure_s: float | NDArray[np.float64],
    trajectory: LinearTrajectory,
) -> ReadSchedule[np.float64]:
    """Derive physical scan reads without sampling optical geometry.

    The two one-dimensional arrays contain a target position and an along-track
    scheduling offset in metres per ordered read. Cross-track position is separate
    from timing for a constant-speed along-track trajectory. The shared scan timing
    rule gives centre = time_origin + (target - offset) / speed; exposure_s is
    a positive scalar or (reads,) durations in seconds. Inputs are borrowed;
    returned exposure intervals are an independent writable float64 snapshot.
    IDs follow the supplied order and ReadSchedule validates their uniqueness.

    Args:
        sample_ids: Unique read identities in requested order.
        pixel_ids: Pixel identity associated with each read.
        setting_ids: Optical setting identity for each read.
        port_ids: Optical port identity for each read.
        target_along_m: (reads,) ground target positions in along-track metres.
        ground_along_offsets_m: (reads,) target-to-platform along offsets at
            exposure centres in metres; these schedule read times.
        exposure_s: Positive scalar or (reads,) durations in seconds.
        trajectory: Constant-speed scan geometry and time origin.

    Returns:
        Geometry-free physical reads with owned exposure intervals in seconds.

    Raises:
        ValueError: Identifiers, positions or durations fail read support checks.
    """
    n = len(sample_ids)
    if target_along_m.shape != (n,) or ground_along_offsets_m.shape != (n,):
        raise ValueError("target and ground along positions must be (reads,)")
    return ReadSchedule(
        sample_ids,
        pixel_ids,
        setting_ids,
        port_ids,
        _along_track_exposures(target_along_m, ground_along_offsets_m, exposure_s, trajectory),
    )


def along_track_strip_templates(
    *,
    sample_ids: tuple[str, ...],
    pixel_ids: tuple[str, ...],
    setting_ids: tuple[str, ...],
    port_ids: tuple[str, ...],
    detector_x_m: NDArray[np.float64],
    ground_along_offsets_m: NDArray[np.float64],
    active_width_m: float,
    exposure_s: float | NDArray[np.float64],
    trajectory: LinearTrajectory,
) -> tuple[ReadSchedule[np.float64], NDArray[np.float64]]:
    """Construct timed origin-centred strip reads and square detector bounds.

    Detector x and along-track scheduling offsets have one entry per physical read,
    in metres. Each detector rectangle has x centre detector_x_m and y centre 0;
    its active x/y width is active_width_m. Targets are (0,0) on the local ground
    plane. exposure_s is a positive scalar or one duration per read in seconds.
    The caller supplies every read identity, stage/port assignment and offset;
    sequential and simultaneous read orders are both supported. Returned read
    timing and detector bounds are independent writable float64 arrays.

    Args:
        sample_ids: Unique sample names, one per strip read.
        pixel_ids: Detector pixel identities aligned with samples.
        setting_ids: Optical setting identities aligned with samples.
        port_ids: Port identities aligned with samples.
        detector_x_m: Per-read detector x centres in metres.
        ground_along_offsets_m: Per-read target-to-platform along offsets at
            exposure centres in metres.
        active_width_m: Positive square-pixel active x/y width in metres.
        exposure_s: Positive scalar or per-read exposure durations in seconds.
        trajectory: Platform motion and epoch used to time the strip reads.

    Returns:
        Ordered physical reads and (reads,4) detector rectangles in metres.

    Raises:
        ValueError: Read identities, dimensions, positions or durations are invalid.
    """
    count = len(sample_ids)
    if (
        not count
        or any(len(ids) != count or not all(ids) for ids in (pixel_ids, setting_ids, port_ids))
        or len(set(sample_ids)) != count
        or detector_x_m.shape != (count,)
        or ground_along_offsets_m.shape != (count,)
        or not np.isfinite(detector_x_m).all()
        or not np.isfinite(ground_along_offsets_m).all()
        or not np.isfinite(active_width_m)
        or active_width_m <= 0
    ):
        raise ValueError("strip templates require aligned identities and finite detector geometry")
    durations = np.asarray(exposure_s, dtype=np.float64)
    if durations.shape == ():
        durations = np.full(count, durations)
    if durations.shape != (count,) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("strip exposure_s must be positive scalar or (reads,)")
    bounds = np.column_stack(
        (
            detector_x_m - active_width_m / 2,
            detector_x_m + active_width_m / 2,
            np.full(count, -active_width_m / 2),
            np.full(count, active_width_m / 2),
        )
    )
    reads = along_track_read_schedule(
        sample_ids=sample_ids,
        pixel_ids=pixel_ids,
        setting_ids=setting_ids,
        port_ids=port_ids,
        target_along_m=np.zeros(count),
        ground_along_offsets_m=ground_along_offsets_m,
        exposure_s=durations,
        trajectory=trajectory,
    )
    return reads, bounds


def raster_exposure_bounds(
    templates: ReadSchedule,
    target_along_m: NDArray[np.float64],
    ground_along_offsets_m: NDArray[np.float64],
    trajectory: LinearTrajectory,
) -> NDArray[np.float64]:
    """Return fresh (row,template,start/end) local-scan exposure seconds.

    target_along_m contains row centres in ground metres; offsets locate exposure
    centres for each template in ground metres. Every cross-track column shares
    this timing. Durations come from the supplied physical-read templates; their
    original exposure centres are replaced by the supplied row positions.
    Inputs are borrowed and the result is an independent writable float64 array.

    Args:
        templates: Ordered physical reads whose exposure durations are reused.
        target_along_m: (rows,) ground along-track row centres in metres.
        ground_along_offsets_m: Per-template target-to-platform along offsets at
            exposure centres in metres.
        trajectory: Scan speed and time origin converting positions into seconds.

    Returns:
        Fresh writable (rows,templates,2) start/end seconds for each row/template.

    Raises:
        ValueError: Missing rows or template offsets, or invalid timing inputs.
    """
    count = len(templates.sample_ids)
    if (
        target_along_m.ndim != 1
        or not len(target_along_m)
        or ground_along_offsets_m.shape != (count,)
    ):
        raise ValueError("raster exposure bounds require rows and one offset per template")
    shape = (count, len(target_along_m))
    durations = np.diff(templates.exposure_bounds_s, axis=1).ravel()
    bounds = _along_track_exposures(
        np.broadcast_to(target_along_m[None, :], shape).ravel(),
        np.broadcast_to(ground_along_offsets_m[:, None], shape).ravel(),
        np.broadcast_to(durations[:, None], shape).ravel(),
        trajectory,
    )
    return bounds.reshape(*shape, 2).transpose(1, 0, 2).copy()


def along_track_raster_reads(
    templates: ReadSchedule,
    target_along_m: NDArray[np.float64],
    ground_along_offsets_m: NDArray[np.float64],
    columns: int,
    trajectory: LinearTrajectory,
    *,
    sample_id: Callable[[int, int, int], str],
    pixel_id: Callable[[int, int, int], str],
) -> ReadSchedule[np.float64]:
    """Expand ordered scan templates onto an along-row/cross-column raster.

    Output order is (template,row,column), flattened C-order. target_along_m
    lists actual target row positions in metres; ground_along_offsets_m provides
    one timing offset per template in metres. Columns share along-track timing,
    with positive offsets scheduling earlier exposures. Template durations,
    settings and ports are repeated exactly. Callbacks name each native physical
    read and pixel; callers retain row, column and simultaneous-exposure noise
    group identities separately. Inputs are borrowed; intervals are independent
    writable float64 snapshots. Reuse with translated K requires stationary
    scene and optics and complete spatial halos on the application lattice.

    Args:
        templates: Physical reads providing duration, setting and port per template.
        target_along_m: Ordered ground along-track row centres in metres.
        ground_along_offsets_m: One target-to-platform along offset per template
            at exposure centre, in metres.
        columns: Positive number of cross-track columns at each row.
        trajectory: Constant-speed scan trajectory and time origin.
        sample_id: Callback mapping (template,row,column) to a unique read name.
        pixel_id: Callback mapping (template,row,column) to a detector pixel name.

    Returns:
        Flattened physical reads in (template,row,column) order with new timing arrays.

    Raises:
        ValueError: Rows, columns, offsets or generated identities are invalid.
    """
    count = len(templates.sample_ids)
    if (
        target_along_m.ndim != 1
        or not len(target_along_m)
        or ground_along_offsets_m.shape != (count,)
        or columns < 1
    ):
        raise ValueError("raster requires target rows, one offset per template and columns")
    shape = (count, len(target_along_m), columns)
    duration = np.diff(templates.exposure_bounds_s, axis=1).ravel()
    return along_track_read_schedule(
        sample_ids=tuple(sample_id(r, i, j) for r, i, j in np.ndindex(shape)),
        pixel_ids=tuple(pixel_id(r, i, j) for r, i, j in np.ndindex(shape)),
        setting_ids=tuple(templates.setting_ids[r] for r, i, j in np.ndindex(shape)),
        port_ids=tuple(templates.port_ids[r] for r, i, j in np.ndindex(shape)),
        target_along_m=np.broadcast_to(target_along_m[None, :, None], shape).ravel(),
        ground_along_offsets_m=np.broadcast_to(
            ground_along_offsets_m[:, None, None], shape
        ).ravel(),
        exposure_s=np.broadcast_to(duration[:, None, None], shape).ravel(),
        trajectory=trajectory,
    )


def rectangular_acquisition(
    *,
    sample_ids: tuple[str, ...],
    pixel_ids: tuple[str, ...],
    setting_ids: tuple[str, ...],
    port_ids: tuple[str, ...],
    exposure_bounds_s: NDArray[np.float64],
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
        sample_ids: Per-read sample identifiers for the selected exposure intervals.
        pixel_ids: Per-read detector pixel identifiers.
        setting_ids: Per-read optical setting identifiers.
        port_ids: Per-read optical port identifiers.
        exposure_bounds_s: (reads,2) start/end intervals in caller-epoch seconds.
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
    for low, high in exposure_bounds_s:
        t, wt = interval_quadrature(low, high, time_order)
        offsets = pointing(t)
        if offsets.shape != (time_order, 2):
            raise ValueError("pointing must return (time nodes,2)")
        times.append(np.repeat(t, uv.shape[0]))
        directions.append(np.tile(uv, (time_order, 1)))
        images.append((focal_length_m * (uv[None, :, :] - offsets[:, None, :])).reshape(-1, 2))
        weights.append((wt[:, None] * angular_weights).ravel())
    return Acquisition(
        sample_ids,
        pixel_ids,
        setting_ids,
        port_ids,
        exposure_bounds_s,
        np.array(times),
        np.array(directions),
        np.array(images),
        np.array(weights),
        pixel_bounds_m,
        np.tile(direction_bounds_uv, (len(sample_ids), 1)),
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
    its physical accuracy are caller-selected; all returned arrays own storage.
    Inputs are borrowed and unmodified; record validation occurs on each call.

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
        Acquisition with owned exposure intervals, sampled directions, image
        positions and physical dOmega dt weights for each read.

    Raises:
        ValueError: Geometry, node orders or bounded patch is invalid.
    """
    sample_ids, pixel_ids = reads.sample_ids, reads.pixel_ids
    setting_ids, port_ids = reads.setting_ids, reads.port_ids
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
        sample_ids,
        pixel_ids,
        setting_ids,
        port_ids,
        exposures,
        np.array(times),
        np.array(directions),
        np.array(images),
        np.array(weights),
        pixel_bounds_m,
        np.array(supports),
    )


linear_scan_acquisition = along_track_acquisition


def ground_scroll_pointing(
    trajectory: LinearTrajectory,
    target_xy_m: NDArray[np.float64],
    centre_times_s: NDArray[np.float64],
    *,
    ground_rate_m_s: float,
    cross_offset_m: NDArray[np.float64] | None = None,
) -> Callable[[int, NDArray[np.float64]], NDArray[np.float64]]:
    """Follow a ground target translating along track at a specified ground rate.

    At each read centre the gaze crosses target_xy_m. Its ground along coordinate
    advances at ground_rate_m_s; zero holds a fixed ground target, rates between
    zero and platform speed scroll slower than the platform, and equal rates keep
    a constant local along slope. The platform cross origin is each target column.
    Input arrays are copied into the closure; returned slope arrays own storage.

    Args:
        trajectory: Constant-height, constant-speed platform trajectory.
        target_xy_m: (reads,2) ground target coordinates at read centres in metres.
        centre_times_s: (reads,) read centres in the trajectory's epoch seconds.
        ground_rate_m_s: Signed along-track ground gaze speed in metres per second.
        cross_offset_m: Optional (reads,) cross ground gaze offsets in metres.

    Returns:
        Callable returning (time nodes,2) local dimensionless slopes for each read.

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
    target, centres, cross = target_xy_m.copy(), centre_times_s.copy(), cross.copy()

    def pointing(read_index: int, times_s: NDArray[np.float64]) -> NDArray[np.float64]:
        along = (
            target[read_index, 0]
            + ground_rate_m_s * (times_s - centres[read_index])
            - trajectory.speed_m_s * (times_s - trajectory.time_origin_s)
        ) / trajectory.height_m
        return np.column_stack(
            (along, np.full(len(times_s), cross[read_index] / trajectory.height_m))
        )

    return pointing


def ground_fixed_pointing(
    trajectory: LinearTrajectory,
    target_xy_m: NDArray[np.float64],
    centre_times_s: NDArray[np.float64],
    *,
    cross_offset_m: NDArray[np.float64] | None = None,
) -> Callable[[int, NDArray[np.float64]], NDArray[np.float64]]:
    """Keep each read's gaze on its fixed ground target throughout exposure.

    Args:
        trajectory: Constant-height, constant-speed platform trajectory.
        target_xy_m: (reads,2) ground target coordinates in metres.
        centre_times_s: (reads,) read centres in caller epoch seconds.
        cross_offset_m: Optional (reads,) cross ground gaze offsets in metres.

    Returns:
        Read/time callback with (time nodes,2) dimensionless local slopes.

    Raises:
        ValueError: Geometry, target or time arrays are invalid.
    """
    return ground_scroll_pointing(
        trajectory, target_xy_m, centre_times_s, ground_rate_m_s=0.0, cross_offset_m=cross_offset_m
    )


def _along_track_exposures(
    target_along_m: NDArray[np.float64],
    ground_along_offsets_m: NDArray[np.float64],
    exposure_s: float | NDArray[np.float64],
    trajectory: LinearTrajectory,
) -> NDArray[np.float64]:
    if (
        not np.isfinite((trajectory.height_m, trajectory.speed_m_s)).all()
        or (trajectory.height_m <= 0 or trajectory.speed_m_s <= 0)
        or not np.isfinite(trajectory.time_origin_s)
    ):
        raise ValueError("height, speed and time origin must be finite with positive height/speed")
    if not np.isfinite(target_along_m).all() or not np.isfinite(ground_along_offsets_m).all():
        raise ValueError("target and ground along positions must be finite")
    duration = np.broadcast_to(exposure_s, (len(target_along_m),))
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("exposure durations must be finite and positive")
    centres = (
        trajectory.time_origin_s + (target_along_m - ground_along_offsets_m) / trajectory.speed_m_s
    )
    bounds = np.column_stack((centres - duration / 2, centres + duration / 2))
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("exposure bounds must be finite with positive duration")
    return bounds


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
