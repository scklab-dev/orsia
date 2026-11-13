"""Evaluate height fields, intersect bounded rays and split horizontal layer paths.

All coordinates share a caller-selected Cartesian metre frame and vertical datum.
The affine surface z = height + slope_x*x + slope_y*y has a reversible x/y chart.
Height fields exclude overhangs; generic intersection requires continuous heights.
These operations exclude refraction and changes of coordinate frame or datum.
Inputs are borrowed without mutation; all outputs own writable storage.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from orsia_geometry.rays import validate_rays


def surface_points(
    xy_m: NDArray[np.float64],
    *,
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a height field and its upward unit normal at each chart point.

    Args:
        xy_m: Finite (...,2) Cartesian horizontal coordinates in metres.
        height: Callable mapping (...,2) coordinates to (...) heights in metres.
        gradient: Callable mapping the same coordinates to (...,2) dimensionless
            derivatives (dz/dx,dz/dy) of height. Callers ensure consistency with
            height, including interpolation and boundary conventions.

    Returns:
        Fresh writable float64 (...,3) points and matching upward unit normals.
        Empty batches are supported. Callbacks borrow inputs without mutation;
        their state must remain stable and their outputs remain unchanged.

    Raises:
        ValueError: Coordinates or callback outputs have invalid shapes or
            nonfinite values. Callback exceptions propagate unchanged.
    """
    if xy_m.ndim < 1 or xy_m.shape[-1] != 2 or not np.isfinite(xy_m).all():
        raise ValueError("surface chart requires finite (...,2) coordinates")
    heights = _height_values(xy_m, height)
    normals = _surface_normals(xy_m, gradient)
    return np.concatenate((xy_m, heights[..., None]), axis=-1), normals


def intersect_terrain(
    origins_m: NDArray[np.float64],
    directions: NDArray[np.float64],
    *,
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    horizontal_bounds_m: tuple[float, float, float, float],
    distance_nodes_m: NDArray[np.float64],
    distance_tolerance_m: float,
    height_tolerance_m: float,
    minimum_incidence_cosine: float,
    max_iterations: int = 64,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Find the first sampled contact candidate on bounded scene-directed rays.

    The residual is ray height minus terrain height. Search increasing supplied
    distances for the first residual <= height_tolerance_m. A residual within
    +/- height_tolerance_m accepts that sample as the candidate. Otherwise bisect
    the preceding positive-to-negative bracket. Only this candidate is tested
    for incidence; rejecting it ends the search for that ray.

    Finite samples cannot certify hidden crossings or tangencies. Bisection may
    select any root within a bracket containing multiple roots. A first physical
    intersection claim requires caller evidence that preceding intervals contain
    no contacts and the selected bracket contains one root. A miss establishes
    only that this bounded sampling policy found no supported candidate.

    Args:
        origins_m: Finite (...,3) ray origins in the terrain Cartesian metre frame.
        directions: Matching unit vectors directed toward the scene.
        height: Continuous height callable, (...,2) metres to (...) metres.
        gradient: Matching (...,2) derivatives (dz/dx,dz/dy), dimensionless.
            Callers ensure consistency with height. Callbacks borrow inputs
            without mutation; keep callable state stable throughout evaluation.
        horizontal_bounds_m: Closed finite (xmin,xmax,ymin,ymax) domain in metres.
            Every complete requested ray interval must lie inside this rectangle.
            Domain violations raise before height or gradient evaluation.
        distance_nodes_m: Finite strictly increasing (nodes,) forward metre
            distances, at least two, starting at zero. Shared across all rays.
            The last node bounds the search; spacing controls contact detection.
        distance_tolerance_m: Positive finite maximum final bisection bracket
            width in metres. An exact zero residual can terminate earlier.
        height_tolerance_m: Positive finite vertical residual tolerance in metres,
            also used to recognize sampled contacts and surface origins.
        minimum_incidence_cosine: Positive lower bound <=1 on -normal dot direction.
            Grazing or exiting candidates return misses without seeking later roots.
        max_iterations: Positive maximum bisection iterations per candidate.

    Returns:
        Fresh writable Boolean (...) hits and float64 (...,3) metre points on
        the rays; misses contain NaNs. Origins below terrain by more than the
        height tolerance miss. Supported surface origins hit at zero. Inputs
        and callback outputs remain unchanged; empty batches are supported.

    Raises:
        ValueError: Rays, controls, domain or callback outputs are invalid, or
            refinement fails to meet both tolerances within max_iterations.
            Callback exceptions propagate unchanged.
    """
    validate_rays(origins_m, directions)
    xmin, xmax, ymin, ymax = horizontal_bounds_m
    if (
        not np.isfinite(horizontal_bounds_m).all()
        or xmin >= xmax
        or ymin >= ymax
        or distance_nodes_m.ndim != 1
        or len(distance_nodes_m) < 2
        or not np.isfinite(distance_nodes_m).all()
        or distance_nodes_m[0] != 0
        or np.any(np.diff(distance_nodes_m) <= 0)
        or not np.isfinite(distance_tolerance_m)
        or distance_tolerance_m <= 0
        or not np.isfinite(height_tolerance_m)
        or height_tolerance_m <= 0
        or not np.isfinite(minimum_incidence_cosine)
        or not 0 < minimum_incidence_cosine <= 1
        or max_iterations < 1
    ):
        raise ValueError("finite domain, increasing distance nodes and positive controls required")
    with np.errstate(over="ignore", invalid="ignore"):
        end = origins_m + distance_nodes_m[-1] * directions
    for points in (origins_m, end):
        if (
            not np.isfinite(points).all()
            or np.any((points[..., 0] < xmin) | (points[..., 0] > xmax))
            or np.any((points[..., 1] < ymin) | (points[..., 1] > ymax))
        ):
            raise ValueError("complete ray search interval must lie inside terrain domain")
    hits = np.zeros(origins_m.shape[:-1], dtype=np.bool_)
    points = np.full(origins_m.shape, np.nan)
    for i, (origin, direction) in enumerate(
        zip(origins_m.reshape(-1, 3), directions.reshape(-1, 3), strict=True)
    ):
        samples = origin + distance_nodes_m[:, None] * direction
        residual = samples[:, 2] - _height_values(samples[:, :2], height)
        if not np.isfinite(residual).all():
            raise ValueError("terrain residual exceeds finite support")
        if residual[0] < -height_tolerance_m:
            continue
        candidates = np.flatnonzero(residual <= height_tolerance_m)
        if not len(candidates):
            continue
        index = int(candidates[0])
        distance = float(distance_nodes_m[index])
        if residual[index] < -height_tolerance_m:
            distance = _bisect_contact(
                origin,
                direction,
                height,
                float(distance_nodes_m[index - 1]),
                distance,
                distance_tolerance_m,
                height_tolerance_m,
                max_iterations,
            )
        point = origin + distance * direction
        normal = _surface_normals(point[None, :2], gradient)[0]
        if -normal @ direction >= minimum_incidence_cosine:
            hits.flat[i] = True
            points.reshape(-1, 3)[i] = point
    return hits, points


def affine_surface_points(
    xy_m: NDArray[np.float64], *, height_m: float, slopes: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover points and the upward unit normal of an affine height field.

    Args:
        xy_m: Finite (...,2) horizontal metre coordinates.
        height_m: Finite surface height at chart origin, metres.
        slopes: Finite (2,) dimensionless derivatives dz/dx and dz/dy.

    Returns:
        Fresh (...,3) points in metres and (3,) upward unit normal.

    Raises:
        ValueError: Coordinates or terrain parameters are invalid.
    """
    normal = _normal(height_m, slopes)
    if xy_m.ndim < 1 or xy_m.shape[-1] != 2 or not np.isfinite(xy_m).all():
        raise ValueError("surface chart requires finite (...,2) coordinates")
    return np.concatenate((xy_m, (height_m + xy_m @ slopes)[..., None]), axis=-1), normal


def intersect_affine_terrain(
    origins_m: NDArray[np.float64],
    directions: NDArray[np.float64],
    *,
    height_m: float,
    slopes: NDArray[np.float64],
    minimum_incidence_cosine: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Intersect exterior scene-directed rays with a single-valued affine terrain.

    Args:
        origins_m: Finite (...,3) origins in the terrain Cartesian metre frame.
        directions: Matching unit vectors directed toward the surface.
        height_m: Finite height at x=y=0, metres.
        slopes: Finite (2,) dimensionless height derivatives.
        minimum_incidence_cosine: Positive lower bound <=1 on -normal dot direction.
            Grazing, upward, below-surface and backward rays return misses.

    Returns:
        Fresh Boolean (...) hits and (...,3) metre points; misses contain NaNs.
        A surface origin with a supported downward direction hits at zero.

    Raises:
        ValueError: Rays, terrain parameters or incidence bound are invalid.
    """
    normal = _normal(height_m, slopes)
    validate_rays(origins_m, directions)
    if not np.isfinite(minimum_incidence_cosine) or not 0 < minimum_incidence_cosine <= 1:
        raise ValueError("minimum incidence cosine must lie in (0,1]")
    altitude = (origins_m[..., 2] - height_m - origins_m[..., :2] @ slopes) * normal[2]
    incidence = -(directions @ normal)
    hits = (altitude >= 0) & (incidence >= minimum_incidence_cosine)
    distance = np.divide(altitude, incidence, out=np.zeros_like(altitude), where=hits)
    return hits.copy(), np.where(
        hits[..., None], origins_m + distance[..., None] * directions, np.nan
    )


def segment_layer_paths(
    surface_points_m: NDArray[np.float64],
    outward_directions: NDArray[np.float64],
    height_edges_m: NDArray[np.float64],
    *,
    minimum_upward_cosine: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Split outward atmospheric rays into actual full or partial horizontal layers.

    Call separately for surface-to-sun and surface-to-sensor directions. Incoming
    solar photons traverse the first path in reverse. Sensors must be above the
    model top. Terrain visibility and horizontal field support remain caller duties.

    Args:
        surface_points_m: Finite (paths,3) starts in metres, within [bottom,top).
        outward_directions: Matching unit vectors pointing out of the atmosphere.
        height_edges_m: Finite strictly increasing (layers+1,) vertical interfaces.
        minimum_upward_cosine: Positive minimum vertical direction component <=1.

    Returns:
        Fresh start/end (segments,3) metre arrays, offsets (paths+1,) and source
        layer indices (segments,). Offsets begin at zero and end at segments.
        Empty path batches are supported. Every retained segment has positive length.

    Raises:
        ValueError: Layer support, rays or direction threshold are invalid.
    """
    validate_rays(surface_points_m, outward_directions)
    if surface_points_m.ndim != 2:
        raise ValueError("path starts must have shape (paths,3)")
    if (
        height_edges_m.ndim != 1
        or len(height_edges_m) < 2
        or not np.isfinite(height_edges_m).all()
        or np.any(np.diff(height_edges_m) <= 0)
        or not np.isfinite(minimum_upward_cosine)
        or not 0 < minimum_upward_cosine <= 1
    ):
        raise ValueError("increasing height edges and positive upward threshold required")
    if (
        np.any(surface_points_m[:, 2] < height_edges_m[0])
        or np.any(surface_points_m[:, 2] >= height_edges_m[-1])
        or np.any(outward_directions[:, 2] < minimum_upward_cosine)
    ):
        raise ValueError("paths must start within atmospheric support and point upward")
    starts, ends, layers = [], [], []
    offsets = [0]
    for point, direction in zip(surface_points_m, outward_directions, strict=True):
        first = int(np.searchsorted(height_edges_m, point[2], side="right") - 1)
        heights = np.concatenate(([point[2]], height_edges_m[first + 1 :]))
        nodes = point + ((heights - point[2]) / direction[2])[:, None] * direction
        starts.extend(nodes[:-1])
        ends.extend(nodes[1:])
        layers.extend(range(first, len(height_edges_m) - 1))
        offsets.append(len(starts))
    return (
        np.array(starts, dtype=np.float64).reshape(-1, 3),
        np.array(ends, dtype=np.float64).reshape(-1, 3),
        np.array(offsets, dtype=np.int64),
        np.array(layers, dtype=np.int64),
    )


def _height_values(
    xy_m: NDArray[np.float64],
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    values = np.asarray(height(xy_m), dtype=np.float64)
    if values.shape != xy_m.shape[:-1] or not np.isfinite(values).all():
        raise ValueError("height callable must return finite (...) heights")
    return values


def _surface_normals(
    xy_m: NDArray[np.float64],
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    slopes = np.asarray(gradient(xy_m), dtype=np.float64)
    if slopes.shape != xy_m.shape or not np.isfinite(slopes).all():
        raise ValueError("gradient callable must return finite (...,2) derivatives")
    normals = np.concatenate((-slopes, np.ones((*xy_m.shape[:-1], 1))), axis=-1)
    normals /= np.max(np.abs(normals), axis=-1, keepdims=True)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def _bisect_contact(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lower: float,
    upper: float,
    distance_tolerance: float,
    height_tolerance: float,
    max_iterations: int,
) -> float:
    """Refine a continuous positive-to-negative residual bracket.

    Returns:
        Ray distance meeting both tolerances, or an exact sampled zero.

    Raises:
        ValueError: The residual is nonfinite or refinement does not converge.
    """
    for _ in range(max_iterations):
        middle = lower + (upper - lower) / 2
        point = origin + middle * direction
        residual = float(point[2] - _height_values(point[None, :2], height)[0])
        if not np.isfinite(residual):
            raise ValueError("terrain residual exceeds finite support")
        if residual == 0 or (
            upper - lower <= distance_tolerance and abs(residual) <= height_tolerance
        ):
            return middle
        if residual > 0:
            lower = middle
        else:
            upper = middle
    raise ValueError("terrain intersection refinement did not converge within max_iterations")


def _normal(height_m: float, slopes: NDArray[np.float64]) -> NDArray[np.float64]:
    if slopes.shape != (2,) or not np.isfinite(slopes).all() or not np.isfinite(height_m):
        raise ValueError("affine terrain requires finite height and two slopes")
    normal = np.array([-slopes[0], -slopes[1], 1.0])
    return normal / np.linalg.norm(normal)
