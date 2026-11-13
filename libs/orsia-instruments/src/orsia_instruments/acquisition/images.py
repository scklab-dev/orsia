"""Exposure intervals, pixel geometry and dense image acquisition."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_instruments.acquisition.quadrature import Acquisition, rectangular_acquisition
from orsia_instruments.acquisition.schedule import RasterReadSchedule

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
    schedule: RasterReadSchedule,
    pixel_bounds_m: NDArray[np.float64],
    direction_bounds_uv: tuple[float, float, float, float],
    pointing: Pointing,
    focal_length_m: float,
    time_order: int = 3,
    angular_order: int = 5,
) -> Acquisition:
    """Sample a common sky patch using retained native read correspondence.

    The schedule supplies immutable identities and actual rolling exposure intervals.
    Pixel rectangles are selected using its original native row/column indices, so
    partial/permuted selections retain the same physical pixels. The paraxial image
    map is f*(direction-pointing); pointing supplies (time_nodes,2) offsets.

    Args:
        schedule: Complete or selected raster schedule in caller-epoch seconds.
        pixel_bounds_m: Full native (rows,columns,4) active rectangles in metres.
        direction_bounds_uv: Common accepted (umin,umax,vmin,vmax) sky patch.
        pointing: Pure time callback returning dimensionless direction-cosine offsets.
        focal_length_m: Finite positive image-plane focal length in metres.
        time_order: Positive Gauss order across each exposure.
        angular_order: Positive Gauss order on each direction coordinate.

    Returns:
        Immutable sampled acquisition referencing schedule.reads in its original order.

    Raises:
        ValueError: Native shape, pixel support, pointing or quadrature is invalid.
    """
    pixels = pixel_bounds_m
    if pixels.shape != (*schedule.native_shape[1:], 4):
        raise ValueError("pixel geometry must match native detector shape")
    _, row, column = schedule.native_indices.T
    return rectangular_acquisition(
        reads=schedule.reads,
        pixel_bounds_m=pixels[row, column],
        direction_bounds_uv=direction_bounds_uv,
        pointing=pointing,
        focal_length_m=focal_length_m,
        time_order=time_order,
        angular_order=angular_order,
    )
