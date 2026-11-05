"""Describe sampled physical band means and ground-registered radiance cubes.

Spectral weights include quadrature and sum to one. They describe a positive
physical response on supplied cm^-1 nodes. Value arrays own writable copies;
canonical spectral coordinates remain immutable.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .spectral.grid import SpectralGrid, require_wavenumber_grid


@dataclass(frozen=True, eq=False)
class SpectralBand:
    """A positive normalized discrete spectral response for a radiance mean.

    Attributes:
        name: Nonempty descriptive band identity.
        grid: Shared canonical positive cm^-1 grid with immutable spectral nodes.
        weights: Owned nonnegative (spectral,) weights including quadrature,
            summing to one; the mean is sum(radiance_per_cm1 * weights).
    """

    name: str
    grid: SpectralGrid
    weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        points = self.grid.points
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if (
            not self.name
            or len(points) < 2
            or weights.shape != points.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, rtol=1e-12, atol=0)
        ):
            raise ValueError("band requires positive ordered nodes and normalized positive weights")
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, eq=False)
class RadianceCube:
    """Band radiance means on a rectangular ground coordinate grid.

    Coordinates use the caller's declared surface chart, with axes (y,x,band).
    Ground coordinates alone do not identify a terrestrial datum or projection.
    Each valid value uses W m^-2 sr^-1 per (cm^-1). Unsupported elements are NaN.
    Arrays own writable storage; callers keep bands and arrays stable during use.

    Attributes:
        x_m: Owned increasing (x,) ground cell centres in metres.
        y_m: Owned increasing (y,) ground cell centres in metres.
        bands: Ordered achieved physical responses, one per last-axis element.
        values: Owned (y,x,band) radiance means, NaN where invalid.
        coverage: Owned (y,x,band) nonnegative accumulated spatial weight.
        valid: Owned Boolean (y,x,band) support and quality mask.
    """

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    bands: tuple[SpectralBand, ...]
    values: NDArray[np.float64]
    coverage: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m"):
            axis = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if (
                axis.ndim != 1
                or not len(axis)
                or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)
            ):
                raise ValueError("cube axes must be nonempty finite increasing ground centres")
            object.__setattr__(self, name, axis)
        shape = (len(self.y_m), len(self.x_m), len(self.bands))
        values = np.array(self.values, dtype=np.float64, copy=True)
        coverage = np.array(self.coverage, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)
        if (
            not self.bands
            or values.shape != shape
            or coverage.shape != shape
            or valid.shape != shape
            or not np.isfinite(coverage).all()
            or np.any(coverage < 0)
            or np.any(valid & ((coverage <= 0) | ~np.isfinite(values)))
            or np.any(~valid & ~np.isnan(values))
        ):
            raise ValueError("cube shapes, coverage, validity and NaN support disagree")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "valid", valid)


def spectral_band(
    name: str, grid: SpectralGrid, response_density: NDArray[np.float64]
) -> SpectralBand:
    """Normalize a sampled response density into trapezoidal band-mean weights.

    A declared domain must be connected to keep trapezoids inside its coverage.
    With domain=None, the weights define a discrete response on the supplied
    nodes; physical coverage remains unknown.

    Args:
        name: Descriptive band identity.
        grid: Canonical grid carrying positive increasing cm^-1 nodes and an
            optional connected physical domain.
        response_density: Nonnegative physical energy or photon response density
            on the same nodes; the supplied law determines the interpretation.

    Returns:
        Independent band with normalized quadrature-inclusive weights.

    Raises:
        ValueError: Nodes or density are invalid, the declared domain is disconnected,
            or integrated response is zero.
    """
    require_wavenumber_grid(grid)
    if grid.domain is not None and len(grid.domain.intervals) != 1:
        raise ValueError("trapezoidal band weights require a connected spectral domain")
    points = grid.points
    density = response_density
    if points.ndim != 1 or len(points) < 2 or density.shape != points.shape:
        raise ValueError("band density and nodes must share a vector of at least two samples")
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("physical band density must be finite and nonnegative")
    weights = np.zeros(len(points))
    weights[:-1] += np.diff(points) / 2
    weights[1:] += np.diff(points) / 2
    weights *= density
    mass = weights.sum()
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("band must have positive finite integrated response")
    return SpectralBand(name, grid, weights / mass)
