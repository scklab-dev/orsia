"""Evaluate mass fields at points, in columns and on finite pixel/layer supports."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from orsia_geometry import LayeredCells
from scipy.integrate import quad_vec
from scipy.linalg import solve_triangular

from orsia_dispersion.forcing import FloatArray
from orsia_dispersion.particles import ParticleMassField
from orsia_dispersion.puffs.evaluation import (
    normal_interval_log_probability,
    puff_concentration,
    puff_path_columns,
)
from orsia_dispersion.puffs.kernels import clip_kernel_interval, puff_kernels
from orsia_dispersion.puffs.population import validate_boundary
from orsia_dispersion.puffs.simulation import GaussianPuffField
from orsia_dispersion.sampling import MassField
from orsia_dispersion.stationary import StationaryGaussianPlume, stationary_profile

_MOMENT_NODES, _MOMENT_WEIGHTS = np.polynomial.legendre.leggauss(8)
_MOMENT_NODES.setflags(write=False)
_MOMENT_WEIGHTS.setflags(write=False)


@dataclass(frozen=True)
class LayerMassMoments:
    """Own mass and its first altitude moment on one pixel/layer support.

    Attributes:
        support: Complete immutable horizontal cells, frame, interfaces and terrain.
        columns_kg_m2: Nonnegative pixel-mean layer columns (*pixels, layers), kg/m2.
        height_moments_kg_m: Integral of altitude times mass per pixel area, kg/m,
            on the same axes and support. Empty layers have zero moment.
            Altitudes share the support's declared datum. Construction owns immutable
            float64 snapshots. No mass outside the support is renormalized inward.
    """

    support: LayeredCells
    columns_kg_m2: FloatArray
    height_moments_kg_m: FloatArray

    def __post_init__(self) -> None:
        """Snapshot and validate amounts, moments and their geometric support.

        Raises:
            ValueError: Shapes, finiteness, positivity or moment support is invalid.
        """
        for name in ("columns_kg_m2", "height_moments_kg_m"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            owned = np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)
            object.__setattr__(self, name, owned)
        m, j, edges, terrain = (
            self.columns_kg_m2,
            self.height_moments_kg_m,
            self.support.height_edges_m,
            self.support.surface_altitude_m,
        )
        _support(edges, terrain)
        if (
            m.shape != (*terrain.shape, len(edges) - 1)
            or j.shape != m.shape
            or not np.isfinite(m).all()
            or not np.isfinite(j).all()
            or np.any(m < 0)
        ):
            raise ValueError("finite matching layer columns and moments required")
        lower = np.maximum(edges[:-1], terrain[..., None])
        upper = edges[1:]
        tolerance = (
            64 * np.finfo(float).eps * np.maximum(1, np.maximum(np.abs(lower), np.abs(upper)))
        )
        centroid = np.divide(j, m, out=np.zeros_like(j), where=m > 0)
        if np.any((m == 0) & (j != 0)) or np.any(
            (m > 0)
            & ((lower >= upper) | (centroid < lower - tolerance) | (centroid > upper + tolerance))
        ):
            raise ValueError("mass moments must lie inside their clipped layer support")


def concentration(field: MassField, points_m: FloatArray, *, batch_size: int = 4096) -> FloatArray:
    """Evaluate either mass-field representation at spatial points.

    Args:
        field: Stationary or puff field, with its physical boundary fixed.
        points_m: Finite local east/north/altitude coordinates (...,3), m.
            Points below a reflecting plane are outside its physical domain.
        batch_size: Positive temporary point batch size for puffs.

    Returns:
        Fresh concentrations (...) in kg/m3, using the field's kernel support.

    Raises:
        ValueError: Coordinates, batch size or reflecting support are invalid.
        TypeError: Raw particle measures have no point concentration; integrate pixels instead.
    """
    if isinstance(field, ParticleMassField):
        raise TypeError("raw particles have no point concentration; use pixel_layer_moments")
    if isinstance(field, GaussianPuffField):
        return puff_concentration(
            field.state,
            points_m,
            boundary=field.boundary,
            batch_size=batch_size,
            kernel_cutoff_radius=field.kernel_cutoff_radius,
        )
    points = _points(points_m, 3)
    if batch_size < 1 or np.any(points[..., 2] < field.ground_altitude_m):
        raise ValueError("positive batch size and points above the reflection plane required")
    centre, sy, sz, amount = stationary_profile(field, points[..., 0])
    y = (points[..., 1] - field.source_north_m) / sy
    z = points[..., 2]
    return (
        amount
        * _pdf(y)
        / sy
        * (_pdf((z - centre) / sz) + _pdf((z - (2 * field.ground_altitude_m - centre)) / sz))
        / sz
    )


def path_columns(
    field: MassField,
    starts_m: FloatArray,
    ends_m: FloatArray,
    *,
    batch_size: int = 4096,
) -> FloatArray:
    """Integrate a supported continuous mass field along straight segments.

    Args:
        field: Gaussian or compact puff field retaining its boundary and cutoff.
            Stationary fields and raw particle measures have no declared path law.
        starts_m: Finite segment starts (...,3), metres in the field's local chart.
        ends_m: Matching segment ends, metres. Zero-length segments are valid.
        batch_size: Positive maximum temporary segment batch size.

    Returns:
        Fresh mass columns (...), kg/m2, including geometric length once and the
        supplied field's reflection and compact-support conventions.

    Raises:
        TypeError: The field has no supported straight-path representation.
        ValueError: Endpoints, boundary support, field state or batch size fail.
    """
    if not isinstance(field, GaussianPuffField):
        raise TypeError(
            "straight paths require a GaussianPuffField; "
            "stationary and particle paths are unsupported"
        )
    return puff_path_columns(
        field.state,
        starts_m,
        ends_m,
        boundary=field.boundary,
        kernel_cutoff_radius=field.kernel_cutoff_radius,
        batch_size=batch_size,
    )


def layer_columns(
    field: MassField,
    horizontal_points_m: FloatArray,
    height_edges_m: FloatArray,
    *,
    surface_altitude_m: FloatArray | None = None,
    batch_size: int = 4096,
) -> FloatArray:
    """Integrate finite vertical columns at horizontal points for either field.

    Args:
        field: Physical mass field with a fixed boundary.
        horizontal_points_m: Finite east/north coordinates (...,2), m.
        height_edges_m: Increasing finite interfaces (layers+1,), m in the field chart.
        surface_altitude_m: Optional terrain (...) in that chart, constant per point.
            Omission starts support at the first layer edge.
        batch_size: Positive maximum temporary point batch size.

    Returns:
        Fresh mass columns (...,layers), kg/m2. Terrain and reflection clip support;
        finite-top and below-surface mass remain excluded, without renormalization.

    Raises:
        ValueError: Coordinate shapes, support or field state are invalid.
        TypeError: Raw particles require finite-area pixel integration.
    """
    if isinstance(field, ParticleMassField):
        raise TypeError("raw particles have no point columns; use pixel_layer_moments")
    points = _points(horizontal_points_m, 2)
    edges = np.asarray(height_edges_m, dtype=np.float64)
    terrain = _terrain(surface_altitude_m, points.shape[:-1], edges)
    _support(edges, terrain)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result = np.zeros((*terrain.shape, len(edges) - 1))
    flat = points.reshape(-1, 2)
    for start in range(0, len(flat), batch_size):
        stop = start + batch_size
        lower, upper = _bounds(field, edges, terrain.reshape(-1)[start:stop])
        result.reshape(-1, len(edges) - 1)[start:stop] = _point_moments(
            field, flat[start:stop], lower, upper
        )[0]
    return result


def pixel_layer_moments(
    field: MassField,
    support: LayeredCells,
    *,
    rtol: float = 1e-7,
    atol_kg_m2: float = 1e-12,
    max_intervals: int = 512,
    batch_size: int = 128,
) -> LayerMassMoments:
    """Integrate pixel-mean mass and altitude moments on the same clipped support.

    Args:
        field: Stationary, puff or discrete particle field in the caller's local chart.
        support: Immutable local rectangular cells, altitude interfaces and terrain.
            The caller establishes that field coordinates use this frame and datum.
            Terrain clipping preserves the selected field's reflection physics.
        rtol: Positive relative quadrature tolerance in each integration block's max norm.
        atol_kg_m2: Absolute column tolerance at least float64's smallest normal
            positive value (about 2.23e-308 kg/m2). First moments are scaled
            by the maximum absolute interface altitude (at least 1 m) for error control.
        max_intervals: Positive adaptive subdivision budget for each integral.
        batch_size: Positive pixel block size for stationary and separable kernels.

    Returns:
        Owned read-only columns and first moments on (*pixels,layers). Vertical
        Gaussian integrals and axis-aligned puff rectangles are analytic. Other
        horizontal integrals use adaptive quadrature, with Gaussian-scale breakpoints.
        Continuous-field columns below float64's smallest normal value are zeroed
        with their moments, because their quotient loses normal relative precision.
        Tolerances are local numerical estimates, not physical or global error bounds.
        Particles deposit exactly into half-open pixels and layers [lower,upper),
        including the terrain and excluding the top. Overlapping particle pixels
        raise ValueError to prevent duplicate deposition; touching pixels are valid.
        Particle sampling error is independent of the quadrature tolerances.

    Raises:
        ValueError: Coordinates, support, field state or numerical parameters are invalid.
        RuntimeError: A horizontal integral fails to reach its requested tolerance.
    """
    lo, hi = support.horizontal.lower_xy_m, support.horizontal.upper_xy_m
    edges, terrain = support.height_edges_m, support.surface_altitude_m
    if (
        not np.isfinite([rtol, atol_kg_m2]).all()
        or rtol <= 0
        or atol_kg_m2 < np.finfo(float).tiny
        or max_intervals < 1
        or batch_size < 1
    ):
        raise ValueError("positive finite tolerances and integration budgets required")
    if isinstance(field, GaussianPuffField):
        validate_boundary(field.state, field.boundary)
    result = np.zeros((2, terrain.size, len(edges) - 1))
    scale = max(1.0, float(np.max(np.abs(edges))))
    lower, upper = _bounds(field, edges, terrain.reshape(-1))
    first, last = lo.reshape(-1, 2), hi.reshape(-1, 2)
    if isinstance(field, ParticleMassField):
        result = _particle_pixels(field, first, last, lower, upper, batch_size)
        return LayerMassMoments(
            support,
            result[0].reshape(support.shape),
            result[1].reshape(support.shape),
        )
    for start in range(0, len(first), batch_size):
        stop = start + batch_size
        a, b = first[start:stop], last[start:stop]
        zlo, zhi = lower[start:stop], upper[start:stop]
        if isinstance(field, StationaryGaussianPlume):
            result[:, start:stop] = _stationary_pixels(
                field, a, b, zlo, zhi, scale, rtol, atol_kg_m2, max_intervals
            )
        else:
            result[:, start:stop] = _puff_pixels(
                field, a, b, zlo, zhi, scale, rtol, atol_kg_m2, max_intervals
            )
    result[:, result[0] < np.finfo(float).tiny] = 0.0
    return LayerMassMoments(
        support,
        result[0].reshape(support.shape),
        result[1].reshape(support.shape),
    )


def _pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def _normal_moments(
    mean: FloatArray,
    sigma: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    probability = np.exp(normal_interval_log_probability(a, b))
    # Stable difference of endpoint densities, including same-tail narrow layers.
    la, lb = -0.5 * a**2, -0.5 * b**2
    delta = (
        np.sign(la - lb)
        * np.exp(np.maximum(la, lb))
        * (-np.expm1(-np.abs(la - lb)))
        / np.sqrt(2 * np.pi)
    )
    centroid = mean + sigma * np.divide(
        delta, probability, out=np.zeros_like(delta), where=probability > 0
    )
    # Avoid cancellation in the first moment of very thin layers. The same
    # positive eight-point rule is used for narrow interval probabilities.
    width = b - a
    narrow = (width > 0) & (width * (1 + np.abs((a + b) / 2)) < 0.1)
    if np.any(narrow):
        nodes, weights = _MOMENT_NODES, _MOMENT_WEIGHTS
        z = (
            np.broadcast_to(lower, a.shape)[narrow, None]
            + (np.broadcast_to(upper, a.shape) - lower)[narrow, None] * (nodes + 1) / 2
        )
        standardized = a[narrow, None] + width[narrow, None] * (nodes + 1) / 2
        log_weight = -0.5 * standardized**2
        w = weights * np.exp(log_weight - log_weight.max(axis=-1, keepdims=True))
        centroid[narrow] = np.sum(w * z, axis=-1) / w.sum(axis=-1)
    centroid = np.clip(centroid, lower, upper)
    return probability, probability * centroid


def _points(points: FloatArray, dimensions: int) -> FloatArray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != dimensions or not np.isfinite(value).all():
        raise ValueError(f"finite coordinates with last dimension {dimensions} required")
    return value


def _terrain(terrain: FloatArray | None, shape: tuple[int, ...], edges: FloatArray) -> FloatArray:
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("at least two height edges required")
    value = np.full(shape, edges[0]) if terrain is None else np.asarray(terrain, dtype=np.float64)
    if value.shape != shape:
        raise ValueError("terrain shape must match pixel/point axes")
    return value


def _support(edges: FloatArray, terrain: FloatArray) -> None:
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
        or not np.isfinite(terrain).all()
    ):
        raise ValueError("finite terrain and strictly increasing finite interfaces required")


def _bounds(
    field: MassField, edges: FloatArray, terrain: FloatArray
) -> tuple[FloatArray, FloatArray]:
    floor = (
        field.ground_altitude_m
        if isinstance(field, StationaryGaussianPlume)
        else 0.0
        if field.boundary == "reflecting"
        else -np.inf
    )
    upper = np.broadcast_to(edges[1:], (*terrain.shape, len(edges) - 1))
    lower = np.minimum(np.maximum(np.maximum(edges[:-1], terrain[..., None]), floor), upper)
    return lower, upper


def _point_moments(
    field: StationaryGaussianPlume | GaussianPuffField,
    xy: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    result = np.zeros((2, *lower.shape))
    if isinstance(field, StationaryGaussianPlume):
        centre, sy, sz, amount = stationary_profile(field, xy[:, 0])
        marginal = amount * _pdf((xy[:, 1] - field.source_north_m) / sy) / sy
        for z in (centre, 2 * field.ground_altitude_m - centre):
            p, j = _normal_moments(z[:, None], sz[:, None], lower, upper)
            result += np.stack((p, j)) * marginal[None, :, None]
    else:
        validate_boundary(field.state, field.boundary)
        for centre, chol, log_mass in puff_kernels(
            field.state, field.boundary, field.kernel_cutoff_radius
        ):
            horizontal = np.asarray(
                solve_triangular(chol[:2, :2], (xy - centre[:2]).T, lower=True)
            ).T
            mean = centre[2] + horizontal @ chol[2, :2]
            marginal = (
                np.exp(log_mass)
                * np.exp(-0.5 * np.sum(horizontal**2, axis=1))
                / (2 * np.pi * chol[0, 0] * chol[1, 1])
            )
            p, j = _kernel_line_moments(
                mean[:, None],
                np.asarray(chol[2, 2]),
                lower,
                upper,
                np.sum(horizontal**2, axis=1)[:, None],
                field.kernel_cutoff_radius,
            )
            result += np.stack((p, j)) * marginal[None, :, None]
    return result


def _kernel_line_moments(
    mean: FloatArray,
    sigma: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    perpendicular_squared: FloatArray,
    radius: float | None,
) -> tuple[FloatArray, FloatArray]:
    if radius is None:
        return _normal_moments(mean, sigma, lower, upper)
    a, b = clip_kernel_interval(
        (lower - mean) / sigma, (upper - mean) / sigma, perpendicular_squared, radius
    )
    return _normal_moments(mean, sigma, mean + sigma * a, mean + sigma * b)


def _particle_pixels(
    field: ParticleMassField,
    lo: FloatArray,
    hi: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    batch_size: int,
) -> FloatArray:
    # Validate the partition independently of where particles happen to lie.
    for i in range(len(lo)):
        overlaps = np.all(np.minimum(hi[i], hi[i + 1 :]) > np.maximum(lo[i], lo[i + 1 :]), axis=1)
        if np.any(overlaps):
            raise ValueError("particle pixel interiors must not overlap")
    result = np.zeros((2, *lower.shape))
    points, mass = field.state.positions_m, field.state.masses_kg
    area = np.prod(hi - lo, axis=1)
    for start in range(0, len(lo), batch_size):
        for i in range(start, min(start + batch_size, len(lo))):
            inside = np.all((points[:, :2] >= lo[i]) & (points[:, :2] < hi[i]), axis=1)
            z, m = points[inside, 2], mass[inside]
            # Accumulate each particle once, including exact lower faces.
            layer = np.searchsorted(upper[i], z, side="right")
            valid = layer < len(upper[i])
            indices = np.minimum(layer, len(upper[i]) - 1)
            valid &= z >= lower[i, indices]
            result[0, i] = np.bincount(layer[valid], weights=m[valid], minlength=len(upper[i]))
            result[1, i] = np.bincount(
                layer[valid], weights=m[valid] * z[valid], minlength=len(upper[i])
            )
    result /= area[None, :, None]
    return result


def _quadrature(
    function: Callable[[float], FloatArray],
    lower: float,
    upper: float,
    rtol: float,
    atol: float,
    limit: int,
    points: list[float] | None = None,
    circle_radius: float | None = None,
) -> FloatArray:
    if circle_radius is not None and circle_radius < 40:
        # Sine coordinates regularize square-root support edges in compact kernels.
        original = function
        radius = circle_radius

        def angular(theta: float) -> FloatArray:
            return original(float(radius * np.sin(theta))) * (radius * np.cos(theta))

        function = angular
        lower, upper = float(np.arcsin(lower / radius)), float(np.arcsin(upper / radius))
        if points is not None:
            points = [float(np.arcsin(x / radius)) for x in points]
    value, error, info = quad_vec(
        function,
        lower,
        upper,
        epsrel=rtol,
        epsabs=atol,
        norm="max",
        limit=limit,
        points=points,
        cache_size=8_000_000,
        workers=1,
        full_output=True,
    )
    if not info.success or not np.isfinite(value).all():
        raise RuntimeError(
            f"pixel integration failed: status={info.status}, estimated error={error}"
        )
    return np.asarray(value, dtype=np.float64)


def _stationary_pixels(
    field: StationaryGaussianPlume,
    lo: FloatArray,
    hi: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    scale: float,
    rtol: float,
    atol: float,
    limit: int,
) -> FloatArray:
    result = np.zeros((2, *lower.shape))
    a = np.maximum(lo[:, 0], field.source_east_m)
    b = np.maximum(hi[:, 0], a)
    middle = np.clip(field.source_east_m + field.source_regularization_m, a, b)
    area = np.prod(hi - lo, axis=1)
    for start, stop in ((a, middle), (middle, b)):
        if not np.any(stop > start):
            continue

        def integrand(t: float, start: FloatArray = start, stop: FloatArray = stop) -> FloatArray:
            centre, sy, sz, amount = stationary_profile(field, start + t * (stop - start))
            cross = np.exp(
                normal_interval_log_probability(
                    (lo[:, 1] - field.source_north_m) / sy, (hi[:, 1] - field.source_north_m) / sy
                )
            )
            weight = amount * cross * (stop - start) / area
            out = np.zeros((2, *lower.shape))
            for z in (centre, 2 * field.ground_altitude_m - centre):
                p, j = _normal_moments(z[:, None], sz[:, None], lower, upper)
                out[0] += weight[:, None] * p
                out[1] += weight[:, None] * j / scale
            return out

        result += _quadrature(integrand, 0.0, 1.0, rtol, atol / 2, limit)
    result[1] *= scale
    return result


def _breaks(lower: float, upper: float, extra: FloatArray | None = None) -> list[float]:
    values = np.array([-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    if extra is not None:
        values = np.r_[values, extra]
    return [float(x) for x in np.unique(values) if lower < x < upper]


def _crossings(edges: FloatArray, mean: float, slope: float, sigma: float) -> FloatArray:
    if slope == 0:
        return np.empty(0)
    # Seed both faces and shoulders of a narrow conditional Gaussian. Without
    # these breaks a nearly singular tilted kernel can fall between every node.
    shoulders = np.array([-8.0, -2.0, 0.0, 2.0, 8.0]) * sigma
    return ((edges[:, None] - mean + shoulders) / slope).ravel()


def _puff_pixels(
    field: GaussianPuffField,
    lo: FloatArray,
    hi: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    scale: float,
    rtol: float,
    atol: float,
    limit: int,
) -> FloatArray:
    result = np.zeros((2, *lower.shape))
    area = np.prod(hi - lo, axis=1)
    count = max(1, len(field.state.masses_kg) * (2 if field.boundary == "reflecting" else 1))
    radius = field.kernel_cutoff_radius
    for centre, chol, log_mass in puff_kernels(field.state, field.boundary, radius):
        mass = np.exp(log_mass)
        # Diagonal covariance gives exact rectangle probabilities and vertical moments.
        if radius is None and np.count_nonzero(np.tril(chol, -1)) == 0:
            horizontal = (
                np.exp(
                    normal_interval_log_probability(
                        (lo - centre[:2]) / np.diag(chol)[:2],
                        (hi - centre[:2]) / np.diag(chol)[:2],
                    )
                ).prod(axis=1)
                * mass
                / area
            )
            p, j = _normal_moments(np.asarray(centre[2]), np.asarray(chol[2, 2]), lower, upper)
            result += np.stack((p, j)) * horizontal[None, :, None]
            continue
        for i in range(len(lo)):
            xa, xb = (lo[i, 0] - centre[0]) / chol[0, 0], (hi[i, 0] - centre[0]) / chol[0, 0]
            if radius is not None:
                xa, xb = max(xa, -radius), min(xb, radius)
            if xb <= xa:
                continue
            amplitude = mass / area[i]
            absolute = atol / count / max(amplitude, np.finfo(float).tiny)

            def along_x(
                x: float,
                centre: FloatArray = centre,
                chol: FloatArray = chol,
                i: int = i,
                absolute: float = float(absolute),
            ) -> FloatArray:
                mean_y = centre[1] + chol[1, 0] * x
                ya, yb = (lo[i, 1] - mean_y) / chol[1, 1], (hi[i, 1] - mean_y) / chol[1, 1]
                if radius is not None:
                    width = np.sqrt(max(radius * radius - x * x, 0.0))
                    ya, yb = max(ya, -width), min(yb, width)
                if yb <= ya:
                    return np.zeros((2, lower.shape[1]))

                def along_y(y: float) -> FloatArray:
                    z = centre[2] + chol[2, 0] * x + chol[2, 1] * y
                    p, j = _kernel_line_moments(
                        np.asarray(z),
                        np.asarray(chol[2, 2]),
                        lower[i],
                        upper[i],
                        np.asarray(x * x + y * y),
                        radius,
                    )
                    return np.stack((p, j / scale)) * _pdf(np.asarray(y))

                if radius is None and chol[2, 1] == 0:
                    p, j = _normal_moments(
                        np.asarray(centre[2] + chol[2, 0] * x),
                        np.asarray(chol[2, 2]),
                        lower[i],
                        upper[i],
                    )
                    integrated = (
                        np.stack((p, j / scale))
                        * np.exp(
                            normal_interval_log_probability(np.asarray([ya]), np.asarray([yb]))
                        )[0]
                    )
                else:
                    transitions = _crossings(
                        np.r_[lower[i], upper[i]],
                        centre[2] + chol[2, 0] * x,
                        chol[2, 1],
                        chol[2, 2],
                    )
                    if radius is not None:
                        transitions = np.r_[
                            transitions,
                            _ellipse_crossings(
                                np.r_[lower[i], upper[i]],
                                centre[2] + chol[2, 0] * x,
                                chol[2, 1],
                                chol[2, 2],
                                float(np.sqrt(max(radius * radius - x * x, 0.0))),
                            ),
                        ]
                    integrated = _quadrature(
                        along_y,
                        float(ya),
                        float(yb),
                        rtol / 4,
                        absolute / 4,
                        limit,
                        _breaks(float(ya), float(yb), transitions),
                        float(np.sqrt(max(radius * radius - x * x, 0.0)))
                        if radius is not None
                        else None,
                    )
                return integrated * _pdf(np.asarray(x))

            transitions = np.r_[
                _crossings(np.array([lo[i, 1], hi[i, 1]]), centre[1], chol[1, 0], chol[1, 1]),
                _crossings(
                    np.r_[lower[i], upper[i]],
                    centre[2],
                    chol[2, 0],
                    np.hypot(chol[2, 1], chol[2, 2]),
                ),
            ]
            if radius is not None:
                transitions = np.r_[
                    transitions,
                    _ellipse_crossings(
                        np.array([lo[i, 1], hi[i, 1]]),
                        centre[1],
                        chol[1, 0],
                        chol[1, 1],
                        radius,
                    ),
                    _ellipse_crossings(
                        np.r_[lower[i], upper[i]],
                        centre[2],
                        chol[2, 0],
                        float(np.hypot(chol[2, 1], chol[2, 2])),
                        radius,
                    ),
                ]
            value = _quadrature(
                along_x,
                float(xa),
                float(xb),
                rtol / 4,
                absolute / 4,
                limit,
                _breaks(float(xa), float(xb), transitions),
                radius,
            )
            value[1] *= scale
            result[:, i] += amplitude * value
    return result


def _ellipse_crossings(
    edges: FloatArray, mean: float, slope: float, sigma: float, radius: float
) -> FloatArray:
    # Seed exact support intersections so quadrature sees thin clipped caps.
    distance = edges - mean
    variance = slope * slope + sigma * sigma
    discriminant = radius * radius * variance - distance * distance
    keep = discriminant >= 0
    middle = distance[keep] * slope / variance
    half = sigma * np.sqrt(discriminant[keep]) / variance
    return np.r_[middle - half, middle + half]
