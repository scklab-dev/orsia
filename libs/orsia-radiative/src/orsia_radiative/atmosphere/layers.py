"""Convert vertical coordinates and redistribute pressure-layer quantities.

Inputs are borrowed for each call. Calculations return fresh writable float64
arrays; validators inspect the supplied arrays in place and return None.
"""

from math import isfinite

import numpy as np
from numpy.typing import NDArray


def geometric_height(
    geopotential_m2_s2: NDArray[np.float64],
    *,
    reference_gravity_m_s2: float,
    earth_radius_m: float,
) -> NDArray[np.float64]:
    """Convert geopotential [m2/s2] to spherical geometric height [m].

    H=geopotential/reference_gravity is geopotential height, not geometric height;
    z=R*H/(R-H) assumes inverse-square gravity on a spherical Earth. Caller chooses
    gravity, radius and vertical datum; this is not a geoid/terrain correction.

    Args:
        geopotential_m2_s2: Finite geopotential values in square metres per second squared.
        reference_gravity_m_s2: Positive reference acceleration for geopotential height.
        earth_radius_m: Positive radius in metres used by the spherical conversion.

    Returns:
        Fresh geometric heights in metres with the input shape.

    Raises:
        ValueError: Input is nonfinite, gravity or radius is invalid, or a
            geopotential height reaches the specified Earth radius.
    """
    validate_vertical_values(geopotential_m2_s2, None, "geopotential")
    if (
        not isfinite(reference_gravity_m_s2)
        or reference_gravity_m_s2 <= 0
        or not isfinite(earth_radius_m)
        or earth_radius_m <= 0
    ):
        raise ValueError("positive finite gravity and Earth radius required")
    height = geopotential_m2_s2 / reference_gravity_m_s2
    if np.any(height >= earth_radius_m):
        raise ValueError("geopotential height must be below Earth radius")
    return earth_radius_m * height / (earth_radius_m - height)


def pressure_level_layer_means(
    pressure_pa: NDArray[np.float64],
    values: NDArray[np.float64],
    pressure_boundaries_pa: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate a point profile linear in pressure into pressure-mass layer means.

    Point knots and interfaces are strictly decreasing [Pa]. Values have one
    finite value per knot (K, kg/kg, etc.). Target interfaces must be bracketed by
    knots, including at the surface: no below-terrain or top extrapolation. This
    is a caller-selected reconstruction, not native model-layer averages.

    Args:
        pressure_pa: Bottom-to-top decreasing point-knot pressures in Pa.
        values: Finite physical values at those pressure knots.
        pressure_boundaries_pa: Target bottom-to-top layer interfaces in Pa,
            bracketed by the point knots.

    Returns:
        Fresh pressure-mass means (target_layers,) in the input value unit.

    Raises:
        ValueError: Pressure support, order or profile shape is invalid.
    """
    validate_pressure_boundaries(pressure_pa, "pressure knots")
    validate_pressure_boundaries(pressure_boundaries_pa, "pressure boundaries")
    validate_vertical_values(values, len(pressure_pa), "point profile")
    if pressure_boundaries_pa[0] > pressure_pa[0] or pressure_boundaries_pa[-1] < pressure_pa[-1]:
        raise ValueError("layer interfaces are outside pressure-knot coverage")
    means = np.empty(len(pressure_boundaries_pa) - 1)
    for i, (bottom, top) in enumerate(
        zip(pressure_boundaries_pa[:-1], pressure_boundaries_pa[1:], strict=True)
    ):
        points = np.sort(
            np.concatenate(
                ([top, bottom], pressure_pa[(pressure_pa > top) & (pressure_pa < bottom)])
            )
        )
        sampled = np.interp(points, pressure_pa[::-1], values[::-1])
        means[i] = np.sum(np.diff(points) * (sampled[:-1] + sampled[1:]) / 2) / (bottom - top)
    return means


def remap_layer_columns(
    source_pressure_boundaries_pa: NDArray[np.float64],
    columns: NDArray[np.float64],
    target_pressure_boundaries_pa: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Conservatively remap extensive layer quantities by pressure overlap.

    Both grids are bottom to top [Pa]. columns has shape (source_layer,) and
    contains nonnegative extensive amounts (molecules/m2, kg/m2, etc.). Assume
    uniform amount per Pa within each source layer. Target support must be fully
    covered: no extrapolation, missing layers or below-terrain padding. Identical
    outer boundaries conserve the integrated amount to floating-point precision;
    a narrower target intentionally retains only its overlapping column.

    Args:
        source_pressure_boundaries_pa: Decreasing source interfaces in Pa.
        columns: Finite nonnegative source-layer amounts (source_layers,).
        target_pressure_boundaries_pa: Decreasing target interfaces within the
            source pressure coverage, in Pa.

    Returns:
        Fresh target-layer amounts (target_layers,) in the input column unit.

    Raises:
        ValueError: Pressure boundaries, target coverage or source columns
            violate the required shape and physical constraints.
    """
    weights = _pressure_layer_overlap(source_pressure_boundaries_pa, target_pressure_boundaries_pa)
    validate_vertical_values(columns, weights.shape[1], "layer columns")
    if np.any(columns < 0):
        raise ValueError("layer columns must be nonnegative")
    return weights @ (columns / -np.diff(source_pressure_boundaries_pa))


def validate_pressure_boundaries(values: NDArray[np.float64], name: str) -> None:
    """Require finite, nonnegative, strictly decreasing pressure interfaces [Pa].

    The one-dimensional vector must contain at least two interfaces. ``name``
    identifies the coordinate in ValueError messages; values remain caller-owned.

    Args:
        values: Pressure interface vector in Pa, bottom to top.
        name: Name identifying the pressure coordinate in errors.

    Raises:
        ValueError: Input is short, nonfinite, negative or out of order.
    """
    validate_vertical_values(values, None, name)
    if len(values) < 2 or np.any(values < 0) or np.any(np.diff(values) >= 0):
        raise ValueError(f"{name} must have at least two nonnegative decreasing boundaries")


def validate_vertical_values(values: NDArray[np.float64], size: int | None, name: str) -> None:
    """Require a finite vector on vertical support, optionally of the given size.

    Units follow the supplied quantity (e.g. Pa, K or kg/kg); ``name`` identifies
    it in ValueError messages. Values remain caller-owned with their original dtype.

    Args:
        values: One-dimensional physical profile values.
        size: Expected vector length, or None for any length.
        name: Quantity name identifying shape or finiteness errors.

    Raises:
        ValueError: Values are nonfinite or fail the vector shape requirement.
    """
    if (
        values.ndim != 1
        or (size is not None and len(values) != size)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} must be a finite vector with the declared shape")


def _pressure_layer_overlap(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return intersecting pressure widths [Pa], shape (target_layer, source_layer).

    Shared internally by column remapping and hydrostatic construction.
    Both interface vectors decrease bottom to top and allow zero pressure at
    the top. Target support must lie within source support. The fresh writable
    array contains nonnegative widths; dividing each column by its source-layer
    width gives conservative redistribution fractions.

    Returns:
        Fresh writable pressure overlap widths in Pa, shape (target_layer, source_layer).

    Raises:
        ValueError: Pressure boundaries are invalid or target support exceeds source coverage.
    """
    validate_pressure_boundaries(source, "source pressure")
    validate_pressure_boundaries(target, "target pressure")
    if target[0] > source[0] or target[-1] < source[-1]:
        raise ValueError("target pressure support is outside the source coverage")
    return np.maximum(
        0.0,
        np.minimum(target[:-1, None], source[None, :-1])
        - np.maximum(target[1:, None], source[None, 1:]),
    )
