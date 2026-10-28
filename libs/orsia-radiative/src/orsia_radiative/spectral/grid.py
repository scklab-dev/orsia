"""Keep spectral sample coordinates distinct from declared physical coverage.

Use spectral grids to carry positive, ordered sampling points with an optional
independent coverage domain; convert coordinate systems explicitly when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SpectralCoordinate = Literal["wavenumber_cm1", "wavelength_nm"]


@dataclass(frozen=True)
class SpectralDomain:
    """Declare physical spectral coverage independently of sampling points.

    Intervals are nonempty, finite, ordered and separated (no overlap or touching).
    A singleton interval is permitted. No coverage is inferred from sample extrema.

    Attributes:
        coordinate: Wavenumber in cm-1 or wavelength in nm for all intervals.
        intervals: Ordered, separated closed (low, high) bounds in that coordinate.
    """

    coordinate: SpectralCoordinate
    intervals: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        _validate_coordinate(self.coordinate)
        intervals = tuple((low, high) for low, high in self.intervals)
        if not intervals:
            raise ValueError("spectral domain requires at least one interval")
        previous = 0.0
        for low, high in intervals:
            if not isfinite(low) or not isfinite(high) or low <= previous or high < low:
                raise ValueError(
                    "spectral intervals must be finite, positive, ordered and separated"
                )
            previous = high
        object.__setattr__(self, "intervals", intervals)

    def contains(self, points: NDArray[np.float64]) -> bool:
        """Test whether a sampled axis fits inside declared physical coverage.

        Args:
            points: Spectral coordinates expressed in this domain's declared unit.

        Returns:
            True when every supplied point lies in one closed support interval.
        """
        bounds = np.asarray(self.intervals, dtype=np.float64)
        indices = np.searchsorted(bounds[:, 0], points, side="right") - 1
        return bool(np.all((indices >= 0) & (points <= bounds[np.maximum(indices, 0), 1])))


@dataclass(frozen=True, eq=False)
class SpectralGrid:
    """Carry spectral samples and optional declared physical coverage together.

    ``domain=None`` means physical coverage is unknown, not the min/max envelope.
    Points are an immutable bytes-backed snapshot; already safe contiguous storage
    may be shared. Neither caller buffers nor mutable array owners are retained.
    Exact sampling compatibility ignores domain extent: differently covered inputs
    may have been evaluated at the same points. Full record equality includes it.

    Attributes:
        coordinate: Wavenumber in cm-1 or wavelength in nm for the samples.
        points: Strictly increasing, positive one-dimensional float64 samples.
        domain: Optional independent declaration of physical coverage at these points.
    """

    coordinate: SpectralCoordinate
    points: NDArray[np.float64]
    domain: SpectralDomain | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        _validate_coordinate(self.coordinate)
        points = _point_snapshot(self.points)
        if self.domain is not None:
            if self.domain.coordinate != self.coordinate:
                raise ValueError("spectral domain and grid coordinate/unit must agree")
            if not self.domain.contains(points):
                raise ValueError("spectral points lie outside the declared domain")
        object.__setattr__(self, "points", points)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SpectralGrid)
            and self.domain == other.domain
            and self.same_points(other)
        )

    def same_points(self, other: SpectralGrid) -> bool:
        """Check exact sampling compatibility independently of declared coverage.

        Args:
            other: Spectral grid to compare with this coordinate and point axis.

        Returns:
            True when coordinate/unit and all point values match exactly.
        """
        return self is other or (
            self.coordinate == other.coordinate and np.array_equal(self.points, other.points)
        )


def wavenumber_grid(
    values: NDArray[np.float64] | SpectralGrid,
    *,
    domain: SpectralDomain | None = None,
) -> SpectralGrid:
    """Adapt legacy cm-1 points or retain a canonical grid, without guessing support.

    An explicit domain validates physical coverage. A wavelength grid is rejected;
    use ``convert_spectral_grid`` and a separate density conversion explicitly.

    Args:
        values: Legacy increasing cm-1 points or an existing wavenumber grid.
        domain: Optional physical coverage declared for a newly created grid.

    Returns:
        Existing canonical grid or a new immutable grid using the supplied points.

    Raises:
        ValueError: Values are invalid, use wavelength units or conflict with the domain.
    """
    if isinstance(values, SpectralGrid):
        require_wavenumber_grid(values)
        if domain is not None and domain != values.domain:
            raise ValueError("conflicting spectral domain declarations")
        return values
    return SpectralGrid("wavenumber_cm1", values, domain)


def require_wavenumber_grid(grid: SpectralGrid) -> None:
    """Check that a grid uses the cm-1 sampling required by radiative kernels.

    Args:
        grid: Spectral grid whose coordinate must be wavenumber_cm1.

    Raises:
        ValueError: The supplied grid is sampled in wavelength units.
    """
    if grid.coordinate != "wavenumber_cm1":
        raise ValueError("radiative sampling requires wavenumber_cm1; convert explicitly")


def convert_spectral_grid(grid: SpectralGrid) -> SpectralGrid:
    """Convert nm <-> cm-1 points and declared intervals, reversing their order.

    Return a new immutable grid; do not transform values, apply a density Jacobian,
    interpolate or integrate. Unknown physical support remains unknown.

    Args:
        grid: Wavelength-nm or wavenumber-cm-1 samples and optional domain.

    Returns:
        New reciprocal-coordinate grid with reversed points and interval order.
    """
    coordinate: SpectralCoordinate = (
        "wavelength_nm" if grid.coordinate == "wavenumber_cm1" else "wavenumber_cm1"
    )
    domain = None
    if grid.domain is not None:
        domain = SpectralDomain(
            coordinate,
            tuple((1e7 / high, 1e7 / low) for low, high in reversed(grid.domain.intervals)),
        )
    return SpectralGrid(coordinate, 1e7 / grid.points[::-1], domain)


def _validate_coordinate(coordinate: SpectralCoordinate) -> None:
    if coordinate not in ("wavenumber_cm1", "wavelength_nm"):
        raise ValueError("spectral coordinate must declare wavenumber_cm1 or wavelength_nm")


def _validate_points(values: NDArray[np.float64]) -> None:
    if (
        values.ndim != 1
        or values.dtype != np.float64
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(
            "wavenumbers_cm1 / spectral points must be finite, positive "
            "and strictly increasing float64"
        )


def _point_snapshot(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _validate_points(values)
    owner = values
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if values.flags.c_contiguous and isinstance(owner.base, bytes):
        return values
    return np.frombuffer(values.tobytes(), dtype=np.float64)
