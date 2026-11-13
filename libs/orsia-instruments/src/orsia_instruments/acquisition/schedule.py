"""Describe native read chronology, raster identities and local scan timing."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class ReadSchedule[T: np.floating | np.integer = np.floating | np.integer]:
    """Ordered sample/pixel/setting/port identities and exposure intervals in seconds.

    No optical geometry or quadrature is implied. Intervals are finite, positive
    duration (reads,2) immutable snapshots; sample IDs are unique within the set.
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
        object.__setattr__(
            self,
            "exposure_bounds_s",
            np.frombuffer(bounds.tobytes(), dtype=bounds.dtype).reshape(bounds.shape),
        )


@dataclass(frozen=True, eq=False)
class RasterReadSchedule:
    """Physical reads mapped to their original complete native raster.

    Selection retains original identities, native indices and full native shape.
    Buffers are independent immutable snapshots. Dense reshape requires complete
    coverage in C order; partial or permuted selections use a read axis.

    Attributes:
        reads: Immutable ordered read identities and exposure intervals.
        native_shape: Full (frames,rows,columns) dimensions, including unselected pixels.
        native_indices: Owned immutable int64 (reads,3) frame,row,column indices.
    """

    reads: ReadSchedule
    native_shape: tuple[int, int, int]
    native_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        shape = tuple(self.native_shape)
        indices = np.copy(self.native_indices)
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError("native shape must have three positive dimensions")
        if (
            indices.dtype != np.int64
            or indices.shape != (len(self.reads.sample_ids), 3)
            or np.any(indices < 0)
            or np.any(indices >= shape)
            or len(np.unique(indices, axis=0)) != len(indices)
        ):
            raise ValueError("native indices must be unique in-range int64 (reads,3)")
        object.__setattr__(self, "native_shape", shape)
        object.__setattr__(
            self,
            "native_indices",
            np.frombuffer(indices.tobytes(), dtype=np.int64).reshape(indices.shape),
        )

    @property
    def complete(self) -> bool:
        """Return whether reads cover the full native raster in C order."""
        if len(self.native_indices) != np.prod(self.native_shape):
            return False
        flat = np.ravel_multi_index(self.native_indices.T, self.native_shape)
        return bool(np.array_equal(flat, np.arange(len(flat))))

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return dense dimensions for complete rasters, otherwise a read axis."""
        return self.native_shape if self.complete else (len(self.reads.sample_ids),)


def raster_read_schedule(
    exposure_bounds_s: NDArray[np.floating | np.integer],
    shape: tuple[int, int],
    *,
    row_offsets_s: NDArray[np.float64] | None = None,
    setting_ids: tuple[str, ...] | None = None,
    port: str = "direct",
) -> RasterReadSchedule:
    """Build one immutable frame/row/column identity and timing correspondence.

    Args:
        exposure_bounds_s: Finite positive (frames,2) intervals in caller epoch seconds.
        shape: Positive native detector (rows,columns) dimensions.
        row_offsets_s: Optional finite (rows,) rolling offsets in seconds.
        setting_ids: One setting per frame; defaults to decimal frame indices.
        port: Nonempty optical output-port identity shared by reads.

    Returns:
        Complete native raster schedule in frame/row/column C order.

    Raises:
        ValueError: Shape, identities or timing is invalid.
    """
    exposures = np.copy(exposure_bounds_s)
    if exposures.ndim != 2 or exposures.shape[1:] != (2,) or not len(exposures):
        raise ValueError("exposure bounds must have nonempty (frames,2) shape")
    if len(shape) != 2 or any(n < 1 for n in shape):
        raise ValueError("detector shape must have two positive dimensions")
    settings = tuple(map(str, range(len(exposures)))) if setting_ids is None else setting_ids
    offsets = np.zeros(shape[0]) if row_offsets_s is None else row_offsets_s
    if (
        len(settings) != len(exposures)
        or offsets.shape != (shape[0],)
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("settings and row offsets must match frames and rows")
    native_shape = (len(exposures), *shape)
    native = np.indices(native_shape, dtype=np.int64).reshape(3, -1).T
    frame, row, column = native.T
    reads = ReadSchedule(
        tuple(f"s{f}-p{r * shape[1] + c}" for f, r, c in native),
        tuple(f"r{r}-c{c}" for r, c in zip(row, column, strict=True)),
        tuple(settings[f] for f in frame),
        (port,) * len(frame),
        exposures[frame] + offsets[row, None],
    )
    return RasterReadSchedule(reads, native_shape, native)


def select_read_schedule(reads: ReadSchedule, indices: NDArray[np.int64]) -> ReadSchedule:
    """Select ordered reads with original identities and timing.

    Args:
        reads: Immutable physical schedule to select.
        indices: Nonempty unique in-range int64 read positions in requested order.

    Returns:
        Independent immutable selected schedule.

    Raises:
        ValueError: Indices are empty, repeated, malformed or outside the schedule.
    """
    if (
        indices.dtype != np.int64
        or indices.ndim != 1
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(reads.sample_ids))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("read indices must be nonempty unique in-range int64")
    return ReadSchedule(
        *(
            tuple(getattr(reads, name)[i] for i in indices)
            for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids")
        ),
        reads.exposure_bounds_s[indices],
    )


def select_raster_reads(
    schedule: RasterReadSchedule, indices: NDArray[np.int64]
) -> RasterReadSchedule:
    """Select raster reads while preserving native shape and physical identity.

    Args:
        schedule: Original immutable raster correspondence.
        indices: Nonempty unique in-range int64 read positions in requested order.

    Returns:
        Selected immutable correspondence with original native indices.

    Raises:
        ValueError: Selection fails the physical read index contract.
    """
    return RasterReadSchedule(
        select_read_schedule(schedule.reads, indices),
        schedule.native_shape,
        schedule.native_indices[indices],
    )


def check_raster_alignment(first: RasterReadSchedule, second: RasterReadSchedule) -> None:
    """Require exact native read correspondence before combining measurements.

    Args:
        first: Reference native raster identities, order and exposure timing.
        second: Supplied measurement or response schedule in the same physical order.

    Raises:
        ValueError: Native shapes, indices, identities or exposure timing disagree.
    """
    if (
        first.native_shape != second.native_shape
        or not np.array_equal(first.native_indices, second.native_indices)
        or any(
            getattr(first.reads, name) != getattr(second.reads, name)
            for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids")
        )
    ):
        raise ValueError("native raster read identities and order must agree")
    if not np.array_equal(first.reads.exposure_bounds_s, second.reads.exposure_bounds_s):
        raise ValueError("native exposure timing must agree")


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
    returned exposure intervals are an independent immutable float64 snapshot.
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
    timing is an independent immutable float64 snapshot; detector bounds are
    independent writable float64 arrays.

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
    immutable float64 snapshots. Reuse with translated K requires stationary
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
