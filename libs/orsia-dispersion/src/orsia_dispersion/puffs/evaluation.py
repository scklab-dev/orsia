"""Evaluate Gaussian mass densities and analytic layer and straight-path integrals."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import log_ndtr, logsumexp

from orsia_dispersion.puffs.kernels import (
    clip_kernel_interval,
    compact_kernel_parameters,
    puff_kernels,
)
from orsia_dispersion.puffs.population import GaussianPuffs, validate_boundary
from orsia_dispersion.transport import Boundary, FloatArray

_INTERVAL_NODES, _INTERVAL_WEIGHTS = np.polynomial.legendre.leggauss(8)
_INTERVAL_NODES.setflags(write=False)
_INTERVAL_WEIGHTS.setflags(write=False)


def puff_concentration(
    state: GaussianPuffs,
    points_m: FloatArray,
    *,
    boundary: Boundary,
    batch_size: int = 4096,
    kernel_cutoff_radius: float | None = None,
) -> FloatArray:
    """Evaluate excess concentration with normalized Gaussian or compact kernels.

    Args:
        state: Stable snapshot in a fixed ENU chart.
        points_m: Finite ENU points (...,3), metres; reflecting points require z>=0.
        boundary: 'free' or separable 'reflecting'.
        batch_size: Positive maximum number of points in each temporary batch.
        kernel_cutoff_radius: Optional positive ellipsoidal cutoff in inflated
            Mahalanobis units. Normalization and inflation preserve stored moments.

    Returns:
        Fresh writable float64 concentrations (...), kg/m3. Puff batches have size one.

    Raises:
        ValueError: State, points, boundary or batch size is invalid.
    """
    compact_kernel_parameters(kernel_cutoff_radius)
    points = _points(state, points_m, 3, boundary, batch_size)
    result = np.zeros(points.shape[:-1])
    flat = points.reshape(-1, 3)
    for start in range(0, len(flat), batch_size):
        stop = start + batch_size
        for centre, chol, log_mass in puff_kernels(state, boundary, kernel_cutoff_radius):
            whitened = np.asarray(
                solve_triangular(chol, (flat[start:stop] - centre).T, lower=True)
            ).T
            squared = np.sum(whitened**2, axis=1)
            density = np.exp(
                log_mass - 1.5 * np.log(2 * np.pi) - np.log(np.diag(chol)).sum() - 0.5 * squared
            )
            if kernel_cutoff_radius is not None:
                density[squared > kernel_cutoff_radius * kernel_cutoff_radius] = 0.0
            result.reshape(-1)[start:stop] += density
    return result


def puff_layer_columns(
    state: GaussianPuffs,
    horizontal_points_m: FloatArray,
    height_edges_m: FloatArray,
    *,
    boundary: Boundary,
    batch_size: int = 4096,
    kernel_cutoff_radius: float | None = None,
) -> FloatArray:
    """Integrate vertical mass columns on caller-selected finite height support.

    Args:
        state: Stable snapshot; tilted free-space covariance is supported.
        horizontal_points_m: Finite east/north positions (...,2), metres.
        height_edges_m: Strictly increasing finite edges (l+1,), metres, at least two.
            Reflecting support starts at or above zero. Finite top excludes its tail.
        boundary: 'free' or separable 'reflecting'.
        batch_size: Positive temporary point batch limit; puffs are accumulated serially.
        kernel_cutoff_radius: Optional positive moment-preserving ellipsoidal cutoff.

    Returns:
        Fresh excess columns (...,l), kg/m2, ordered by increasing height.

    Raises:
        ValueError: Inputs violate shape, finiteness, boundary or support conditions.
    """
    compact_kernel_parameters(kernel_cutoff_radius)
    points = _points(state, horizontal_points_m, 2, boundary, batch_size)
    edges = np.asarray(height_edges_m, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("height edges must be a finite vector with at least two edges")
    if np.any(np.diff(edges) <= 0) or (boundary == "reflecting" and edges[0] < 0):
        raise ValueError("height edges must increase within the supported half-space")
    result = np.zeros((*points.shape[:-1], len(edges) - 1))
    flat = points.reshape(-1, 2)
    output = result.reshape(-1, len(edges) - 1)
    for start in range(0, len(flat), batch_size):
        stop = start + batch_size
        for centre, chol, log_mass in puff_kernels(state, boundary, kernel_cutoff_radius):
            horizontal = np.asarray(
                solve_triangular(chol[:2, :2], (flat[start:stop] - centre[:2]).T, lower=True)
            ).T
            conditional_mean = centre[2] + horizontal @ chol[2, :2]
            normalized_edges = (edges[None, :] - conditional_mean[:, None]) / chol[2, 2]
            lo, hi = clip_kernel_interval(
                normalized_edges[:, :-1],
                normalized_edges[:, 1:],
                np.sum(horizontal**2, axis=1)[:, None],
                kernel_cutoff_radius,
            )
            probability = normal_interval_log_probability(lo, hi)
            marginal = (
                log_mass
                - np.log(2 * np.pi)
                - np.log(np.diag(chol)[:2]).sum()
                - 0.5 * np.sum(horizontal**2, axis=1)
            )
            output[start:stop] += np.exp(marginal[:, None] + probability)
    return result


def puff_path_columns(
    state: GaussianPuffs,
    starts_m: FloatArray,
    ends_m: FloatArray,
    *,
    boundary: Boundary,
    batch_size: int = 4096,
    kernel_cutoff_radius: float | None = None,
) -> FloatArray:
    """Integrate excess mass along finite straight segments with geometric length once.

    Args:
        state: Stable snapshot in the same ENU frame as endpoints.
        starts_m: Finite segment start points (...,3), metres.
        ends_m: Finite end points of the same shape; zero-length segments allowed.
        boundary: 'free' or 'reflecting'; reflecting endpoints require z>=0.
        batch_size: Positive maximum temporary segment count per batch.
        kernel_cutoff_radius: Optional positive moment-preserving ellipsoidal cutoff.

    Returns:
        Fresh columns (...), kg/m2; zero-length segments return exactly zero.

    Raises:
        ValueError: Endpoint shapes, support, state or batch size are invalid.
    """
    compact_kernel_parameters(kernel_cutoff_radius)
    starts = _points(state, starts_m, 3, boundary, batch_size)
    ends = _points(state, ends_m, 3, boundary, batch_size)
    if starts.shape != ends.shape:
        raise ValueError("paired endpoints must have identical shapes")
    result = np.zeros(starts.shape[:-1])
    first, last = starts.reshape(-1, 3), ends.reshape(-1, 3)
    for start in range(0, len(first), batch_size):
        stop = start + batch_size
        displacement = last[start:stop] - first[start:stop]
        length = np.linalg.norm(displacement, axis=1)
        keep = length > 0
        for centre, chol, log_mass in puff_kernels(state, boundary, kernel_cutoff_radius):
            direction = displacement[keep] / length[keep, None]
            v = np.asarray(solve_triangular(chol, direction.T, lower=True)).T
            speed = np.linalg.norm(v, axis=1)
            unit = v / speed[:, None]
            a = np.asarray(
                solve_triangular(chol, (first[start:stop][keep] - centre).T, lower=True)
            ).T
            b = np.asarray(
                solve_triangular(chol, (last[start:stop][keep] - centre).T, lower=True)
            ).T
            lo = np.sum(a * unit, axis=1)
            hi = np.sum(b * unit, axis=1)
            perpendicular = a - lo[:, None] * unit
            lo, hi = clip_kernel_interval(
                lo, hi, np.sum(perpendicular**2, axis=1), kernel_cutoff_radius
            )
            log_column = (
                log_mass
                - np.log(2 * np.pi)
                - np.log(np.diag(chol)).sum()
                - np.log(speed)
                - 0.5 * np.sum(perpendicular**2, axis=1)
                + normal_interval_log_probability(lo, hi)
            )
            batch = result.reshape(-1)[start:stop]
            batch[keep] += np.exp(log_column)
    return result


def normal_interval_log_probability(lower: FloatArray, upper: FloatArray) -> FloatArray:
    """Compute stable log probabilities for finite standard-normal intervals.

    Args:
        lower: Lower standardized endpoints, any shape.
        upper: Upper endpoints of the same shape, greater than or equal to lower.
            Internal callers establish finite ordered support before this operation.

    Returns:
        Log probabilities with the input shape. Equal endpoints give negative infinity.
        Tail subtraction uses log CDFs; narrow intervals use positive Gaussian quadrature.
    """
    lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
    # Use the negative tail in both hemispheres to avoid subtraction of CDFs near one.
    left = np.where(lo > 0, -hi, lo)
    right = np.where(lo > 0, -lo, hi)
    log_left, log_right = log_ndtr(left), log_ndtr(right)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = log_right + np.log(-np.expm1(log_left - log_right))
    width = hi - lo
    middle = lo + width / 2
    narrow = (width > 0) & (width * (1 + np.abs(middle)) < 0.1)
    if np.any(narrow):
        nodes, weights = _INTERVAL_NODES, _INTERVAL_WEIGHTS
        values = middle[narrow, None] + width[narrow, None] / 2 * nodes
        result[narrow] = (
            np.log(width[narrow] / 2)
            - 0.5 * np.log(2 * np.pi)
            + logsumexp(-0.5 * values**2 + np.log(weights), axis=-1)
        )
    return result


def _points(
    state: GaussianPuffs, points: FloatArray, dimensions: int, boundary: Boundary, batch_size: int
) -> FloatArray:
    validate_boundary(state, boundary)
    value = np.asarray(points, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != dimensions or not np.all(np.isfinite(value)):
        raise ValueError(f"points must be finite with last dimension {dimensions}")
    if dimensions == 3 and boundary == "reflecting" and np.any(value[..., 2] < 0):
        raise ValueError("reflecting evaluation requires z>=0")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return value
