"""Fit and evaluate a two-axis Matérn-3/2 temporal pointing process.

Budgets, chronology and RNG streams belong to the caller. Matrices use
position-row, position-column, velocity-row, velocity-column state order.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]
type ArrayInput = (
    float
    | np.floating
    | NDArray[np.floating | np.integer | np.bool_]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
)


def pointing_covariance_parameters(
    intervals_s: ArrayInput, rms_urad: ArrayInput, axis_correlation: float = 0.0
) -> tuple[float, NDArray[np.float64]]:
    """Fit source Matérn-3/2 decay and stationary two-axis covariance from two RPEs.

    Durations are seconds, equal-axis RMS values microradians. Returns decay 1/s
    and (2,2) covariance radians squared. Invalid constraint ratios raise ValueError.

    Args:
        intervals_s: Two increasing positive RPE integration durations, seconds.
        rms_urad: Positive equal-axis RPE RMS at each duration, microradians.
        axis_correlation: Row/column stationary correlation in [-1,1].

    Returns:
        Matérn-3/2 decay rate in 1/s and owned two-axis covariance in radians squared.

    Raises:
        ValueError: Durations, RMS values, correlation or smooth-process ratio fails.
    """
    intervals = np.asarray(intervals_s, float)
    rms = np.asarray(rms_urad, float)
    if (
        intervals.shape != (2,)
        or rms.shape != (2,)
        or np.any(intervals <= 0)
        or np.any(rms <= 0)
        or intervals[0] >= intervals[1]
        or not -1 <= axis_correlation <= 1
    ):
        raise ValueError("two increasing durations and positive RPEs required")
    target = (rms[0] / rms[1]) ** 2
    if not (intervals[0] / intervals[1]) ** 2 < target < 1:
        raise ValueError("inconsistent smooth-pointing variance ratio")
    lower, upper = 1e-5 / intervals[1], 1e5 / intervals[0]
    for _ in range(160):
        midpoint = np.sqrt(lower * upper)
        ratio = _rpe_fraction(midpoint * intervals[0]) / _rpe_fraction(midpoint * intervals[1])
        if ratio < target:
            lower = midpoint
        else:
            upper = midpoint
    rate = np.sqrt(lower * upper)
    variance = (rms[1] * 1e-6) ** 2 / _rpe_fraction(rate * intervals[1])
    return float(rate), variance * np.array([[1.0, axis_correlation], [axis_correlation, 1.0]])


def coadd_temporal_covariance(
    rows: int, ranges: Sequence[NDArray[np.int64]], period: float, rate: float
) -> NDArray[np.float64]:
    """Return coadd-averaged temporal correlations in line-major band order.

    Args:
        rows: Number of chronological detector lines.
        ranges: Zero-based per-band exposure-index vectors, including guard gaps.
        period: Row cadence in seconds.
        rate: Matérn-3/2 decay in 1/s.

    Returns:
        Owned correlation matrix in line-major band order, (rows*bands,rows*bands).
    """
    bands = len(ranges)
    result = np.empty((rows * bands, rows * bands))
    cache = {}
    for r in range(rows):
        for s in range(rows):
            lag = r - s
            if lag not in cache:
                block = np.empty((bands, bands))
                for b, left in enumerate(ranges):
                    for c, right in enumerate(ranges):
                        x = rate * period * abs(lag + left[:, None] - right[None, :])
                        block[b, c] = np.mean((1 + x) * np.exp(-x))
                cache[lag] = block
            result[r * bands : (r + 1) * bands, s * bands : (s + 1) * bands] = cache[lag]
    return result


def smooth_pointing_state_space(
    rate: float, angular: ArrayInput, period: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return shared source Matérn-3/2 stationary, transition and innovation matrices.

    Rate is positive 1/s, angular is two-axis covariance rad2 and period is seconds.
    State order is row/column position followed by row/column velocity. The owned
    4-by-4 matrices support Gaussian acquisition and alternative-law validation.
    Nonfinite inputs, invalid dimensions or nonpositive timing raise ValueError.

    Args:
        rate: Positive Matérn-3/2 decay, 1/s.
        angular: Finite two-axis stationary covariance, radians squared.
        period: Positive sample cadence, seconds.

    Returns:
        Owned 4-by-4 stationary covariance, transition and innovation covariance.
        State is row/column angular position followed by row/column angular velocity.

    Raises:
        ValueError: Rate, timing, covariance shape or finiteness fails.
    """
    angular = np.asarray(angular, float)
    if (
        angular.shape != (2, 2)
        or not np.isfinite(angular).all()
        or not np.isfinite(rate + period)
        or min(rate, period) <= 0
    ):
        raise ValueError("finite two-axis covariance and positive pointing timing required")
    dt = period
    stationary = np.zeros((4, 4))
    stationary[:2, :2] = angular
    stationary[2:, 2:] = rate**2 * angular
    identity = np.eye(2)
    decay = np.exp(-rate * dt)
    transition = decay * np.block(
        [
            [(1 + rate * dt) * identity, dt * identity],
            [-(rate**2) * dt * identity, (1 - rate * dt) * identity],
        ]
    )
    innovation = stationary - transition @ stationary @ transition.T

    return stationary, transition, innovation


def _rpe_fraction(x: float) -> float:
    if x < 1e-3:
        return x * x / 12 - x**3 / 30 + x**4 / 120 - x**5 / 630
    return max(1 - 2 * (2 * x - 3 + (x + 3) * np.exp(-x)) / x**2, 0)
