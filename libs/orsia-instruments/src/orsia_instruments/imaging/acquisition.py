"""Exposure intervals, pixel geometry and dense image acquisition."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition, rectangular_acquisition

type Pointing = Callable[[NDArray[np.float64]], NDArray[np.floating | np.integer]]


@dataclass(frozen=True, eq=False)
class ExposureSequence[T: np.floating | np.integer = np.floating | np.integer]:
    """Exposure intervals (seconds), row delay and pointing on a declared sky patch.

    Bounds are writable copies (exposures,2); each row r shifts them by r*row_delay_s.
    Pointing is a borrowed pure function returning (time nodes,2) direction-cosine
    offsets, not a trajectory solver. Direction bounds are umin,umax,vmin,vmax.
    Exposures need not be chronological or disjoint; no detector memory is modeled.

    Attributes:
        bounds_s: Owned (exposures,2) intervals in caller-epoch seconds.
        direction_bounds_uv: Whole-patch (umin,umax,vmin,vmax) dimensionless support.
        pointing: Borrowed time-array callable returning (nodes,2) direction offsets.
        row_delay_s: Time shift per detector row in seconds.
    """

    bounds_s: NDArray[T]
    direction_bounds_uv: tuple[float, float, float, float]
    pointing: Pointing
    row_delay_s: float = 0

    def __post_init__(self) -> None:
        bounds = np.copy(self.bounds_s)
        if not np.isfinite(bounds).all():
            raise ValueError("exposure bounds must be finite")
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 2
            or not len(bounds)
            or np.any(bounds[:, 1] <= bounds[:, 0])
        ):
            raise ValueError("exposure bounds must be nonempty (exposures,2) ordered intervals")
        patch = np.copy(self.direction_bounds_uv)
        if not np.isfinite(patch).all():
            raise ValueError("direction bounds must be finite")
        if (
            patch.shape != (4,)
            or np.any(patch[[1, 3]] <= patch[[0, 2]])
            or (max(patch[:2] ** 2) + max(patch[2:] ** 2) >= 1)
        ):
            raise ValueError("direction bounds must be an ordered patch inside the unit disk")
        if not np.isfinite(self.row_delay_s):
            raise ValueError("row delay must be finite")
        object.__setattr__(self, "bounds_s", bounds)
        object.__setattr__(self, "direction_bounds_uv", tuple(float(x) for x in patch))


def regular_pixel_bounds(
    shape: tuple[int, int],
    *,
    pitch_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    active_size_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer]
    | None = None,
    centre_xy_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = (0, 0),
) -> NDArray[np.float64]:
    """Fresh writable (rows,columns,4) bounds in xmin,xmax,ymin,ymax order.

    Pitch and active size are scalars or (x,y) pairs, not (row,column) pairs.
    Columns increase x; rows increase y. centre_xy_m is the centre of the whole
    array. Active dimensions default to pitch and cannot exceed it. Gaps are lost
    collecting area, not a second fill-factor multiplier. Irregular arrays may
    supply explicit finite rectangular bounds directly to the simulation API.

    Args:
        shape: Positive (rows,columns) detector dimensions.
        pitch_m: Scalar or (x,y) pixel-centre spacing in metres.
        active_size_m: Optional scalar or (x,y) active rectangle dimensions;
            defaults to pitch when omitted.
        centre_xy_m: (x,y) centre of the entire detector array in metres.

    Returns:
        Fresh writable float64 (rows,columns,4) active pixel bounds in metres.

    Raises:
        ValueError: Dimensions, spacing, active size or centre are invalid.
    """
    rows, columns = shape
    if rows < 1 or columns < 1:
        raise ValueError("focal-plane dimensions must be positive")
    pitch = np.asarray(pitch_m, dtype=np.float64)
    if not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pixel pitch must be finite and positive")
    active = pitch if active_size_m is None else np.asarray(active_size_m, dtype=np.float64)
    if active_size_m is not None and (not np.isfinite(active).all() or np.any(active <= 0)):
        raise ValueError("active size must be finite and positive")
    centre = np.copy(centre_xy_m)
    if not np.isfinite(centre).all():
        raise ValueError("pixel centre must be finite")
    if pitch.shape not in ((), (2,)) or active.shape not in ((), (2,)) or centre.shape != (2,):
        raise ValueError("pitch/active size must be scalar or (x,y); centre must be (x,y)")
    pitch = np.broadcast_to(pitch, (2,))
    active = np.broadcast_to(active, (2,))
    if np.any(active > pitch):
        raise ValueError("active dimensions cannot exceed pixel pitch")
    x = centre[0] + (np.arange(columns) - (columns - 1) / 2) * pitch[0]
    y = centre[1] + (np.arange(rows) - (rows - 1) / 2) * pitch[1]
    result = np.empty((rows, columns, 4))
    result[..., 0] = x - active[0] / 2
    result[..., 1] = x + active[0] / 2
    result[..., 2] = y[:, None] - active[1] / 2
    result[..., 3] = y[:, None] + active[1] / 2
    return result


def exposure_sequence(
    *,
    start_s: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    duration_s: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    direction_bounds_uv: tuple[float, float, float, float],
    pointing_uv: Sequence[float] | NDArray[np.floating | np.integer] | Pointing = (0, 0),
    row_delay_s: float = 0,
) -> ExposureSequence:
    """Describe one/many exposures without labels, callbacks or zero arrays.

    Start and duration are scalars or mutually broadcastable 1-D exposure vectors.
    Duration is strictly positive. Pointing is a constant (u,v) pair or a pure
    time callback. A scalar rolling row delay is optional; for irregular timing
    use the advanced acquisition API. No temporal extrapolation is introduced.

    Args:
        start_s: Scalar or vector of exposure starts in caller-epoch seconds.
        duration_s: Positive scalar or broadcastable vector of durations in seconds.
        direction_bounds_uv: (umin,umax,vmin,vmax) accepted sky patch.
        pointing_uv: Constant (u,v) offset or borrowed time-dependent pointing callback.
        row_delay_s: Additional start/end time shift per detector row, in seconds.

    Returns:
        Sequence with owned exposure intervals and the selected pointing rule.

    Raises:
        ValueError: Timing, pointing, row delay or sky patch is invalid.
    """
    durations = np.asarray(duration_s, dtype=np.float64)
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("duration must be finite and positive")
    starts, durations = np.broadcast_arrays(np.copy(start_s), durations)
    if starts.ndim > 1 or not starts.size:
        raise ValueError("start and duration must be scalars or nonempty exposure vectors")
    bounds = np.column_stack((starts.ravel(), (starts + durations).ravel()))
    if callable(pointing_uv):
        pointing = pointing_uv
    else:
        constant = np.copy(pointing_uv)
        if not np.isfinite(constant).all():
            raise ValueError("constant pointing must be finite")
        if constant.shape != (2,):
            raise ValueError("constant pointing must be a (u,v) pair")

        def pointing(times: NDArray[np.float64]) -> NDArray[np.floating | np.integer]:
            return np.broadcast_to(constant, (len(times), 2))

    return ExposureSequence(bounds, direction_bounds_uv, pointing, row_delay_s)


def image_acquisition(
    *,
    exposure_bounds_s: NDArray[np.floating | np.integer],
    setting_ids: tuple[str, ...],
    pixel_bounds_m: NDArray[np.float64],
    row_offsets_s: NDArray[np.float64],
    direction_bounds_uv: tuple[float, float, float, float],
    pointing: Pointing,
    focal_length_m: float,
    time_order: int = 3,
    angular_order: int = 5,
) -> Acquisition:
    """Construct one physical acquisition in (setting,row,column) raster order.

    ``exposure_bounds_s`` has shape (settings,2), in caller epoch seconds;
    ``row_offsets_s`` has shape (rows,) and shifts each setting's exposure interval
    for that row (rolling timing). ``pixel_bounds_m`` has shape (rows,columns,4)
    in metres; its last axis is xmin,xmax,ymin,ymax. All pixels view the supplied
    common sky patch in (u,v). The callback supplies time-dependent pointing and
    may include measured jitter; it must be pure and return
    (time nodes,2) offsets in that same frame. No trajectory is inferred.

    IDs are ``s{index}-p{row*columns+column}``, pixel IDs ``r{row}-c{column}``,
    settings retain their supplied IDs and ports are ``direct``. The returned
    acquisition owns its arrays; the pointing callback is borrowed during evaluation.

    Args:
        exposure_bounds_s: (settings,2) start/end times in caller-epoch seconds.
        setting_ids: One optical setting name per exposure interval.
        pixel_bounds_m: (rows,columns,4) xmin,xmax,ymin,ymax active bounds in metres.
        row_offsets_s: (rows,) rolling time offsets in seconds.
        direction_bounds_uv: (umin,umax,vmin,vmax) common accepted sky patch.
        pointing: Borrowed time-array callable yielding (time nodes,2) offsets.
        focal_length_m: Image-plane focal length in metres.
        time_order: Gauss order across each exposure interval.
        angular_order: Gauss order along each sky-patch coordinate.

    Returns:
        Independently owned acquisition in (setting,row,column) read order.

    Raises:
        ValueError: Input shape, finite support, geometry or quadrature is invalid.
    """
    exposures = exposure_bounds_s
    pixels = pixel_bounds_m
    offsets = row_offsets_s
    if (
        exposures.ndim != 2
        or exposures.shape[1] != 2
        or exposures.shape[0] == 0
        or pixels.ndim != 3
        or pixels.shape[2] != 4
        or pixels.shape[0] == 0
        or pixels.shape[1] == 0
        or offsets.shape != (pixels.shape[0],)
        or len(setting_ids) != exposures.shape[0]
        or not all(setting_ids)
    ):
        raise ValueError("expected (settings,2) exposures, (rows,columns,4) pixels and row offsets")
    if (
        not np.all(np.isfinite(exposures))
        or not np.all(np.isfinite(pixels))
        or not np.all(np.isfinite(offsets))
        or np.any(np.diff(exposures, axis=1) <= 0)
        or np.any(pixels[..., [1, 3]] <= pixels[..., [0, 2]])
    ):
        raise ValueError("image exposures/pixel bounds/row offsets need finite ordered support")
    if not np.isfinite(focal_length_m) or focal_length_m <= 0:
        raise ValueError("focal length must be finite and positive")
    settings_count, rows, columns = len(setting_ids), *pixels.shape[:2]
    setting, row, column = np.indices((settings_count, rows, columns)).reshape(3, -1)
    bounds = exposures[setting] + offsets[row, None]
    return rectangular_acquisition(
        sample_ids=tuple(
            f"s{s}-p{r * columns + col}" for s, r, col in zip(setting, row, column, strict=True)
        ),
        pixel_ids=tuple(f"r{r}-c{col}" for r, col in zip(row, column, strict=True)),
        setting_ids=tuple(setting_ids[s] for s in setting),
        port_ids=("direct",) * len(setting),
        exposure_bounds_s=bounds,
        pixel_bounds_m=pixels[row, column],
        direction_bounds_uv=direction_bounds_uv,
        pointing=pointing,
        focal_length_m=focal_length_m,
        time_order=time_order,
        angular_order=angular_order,
    )
