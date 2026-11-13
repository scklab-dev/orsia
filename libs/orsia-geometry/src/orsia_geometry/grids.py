"""Retain explicit point and finite-cell supports in declared local coordinates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_geometry.frames import LocalCartesianFrame, same_local_frame

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class PointLattice:
    """Own increasing x/y point axes independently of any pixel area.

    Attributes:
        x_m: Finite strictly increasing float64 sample positions (x,), metres.
        y_m: Finite strictly increasing float64 sample positions (y,), metres.
        frame: Local chart shared by these coordinates. Values on this lattice use
            (y,x,...) axes. A single point per axis is permitted. Numerical arrays
            are owned immutable snapshots; no cell boundaries are implied.
    """

    x_m: FloatArray
    y_m: FloatArray
    frame: LocalCartesianFrame

    def __post_init__(self) -> None:
        """Snapshot and validate both point axes.

        Raises:
            ValueError: Coordinates are empty, nonfinite, nonincreasing or not float64.
        """
        for name in ("x_m", "y_m"):
            values = getattr(self, name)
            if values.ndim != 1 or not values.size or np.any(np.diff(values) <= 0):
                raise ValueError("point axes must be nonempty and strictly increasing")
            object.__setattr__(self, name, _snapshot(values))

    @property
    def shape(self) -> tuple[int, int]:
        """Return the (y,x) value axes."""
        return self.y_m.size, self.x_m.size

    @property
    def xy_m(self) -> FloatArray:
        """Return fresh (y,x,2) point coordinates in (x,y) order, metres."""
        x, y = np.meshgrid(self.x_m, self.y_m)
        return np.stack((x, y), axis=-1)

    @classmethod
    def regular(
        cls,
        shape_yx: tuple[int, int],
        spacing_xy_m: tuple[float, float],
        first_xy_m: tuple[float, float],
        frame: LocalCartesianFrame,
    ) -> PointLattice:
        """Construct a regular lattice with explicit first point and axis spacing.

        Args:
            shape_yx: Positive counts in (y,x) array order.
            spacing_xy_m: Positive finite spacing in (x,y) coordinate order, metres.
            first_xy_m: Finite first sample coordinates in (x,y) order, metres.
            frame: Local chart for the sample coordinates.

        Returns:
            Immutable point axes; no cell measure or value reconstruction is inferred.

        Raises:
            ValueError: Counts, spacing or resulting finite coordinates are invalid.
        """
        if (
            min(shape_yx) < 1
            or not np.isfinite(spacing_xy_m).all()
            or min(spacing_xy_m) <= 0
            or not np.isfinite(first_xy_m).all()
        ):
            raise ValueError("positive counts/spacing and finite first point required")
        return cls(
            first_xy_m[0] + np.arange(shape_yx[1], dtype=float) * spacing_xy_m[0],
            first_xy_m[1] + np.arange(shape_yx[0], dtype=float) * spacing_xy_m[1],
            frame,
        )

    def matches(self, other: PointLattice) -> bool:
        """Compare exact coordinates on an admitted common frame.

        Args:
            other: Lattice to compare without tolerance or implicit alignment.

        Returns:
            True only for equal coordinate axes and a matching declared frame.
        """
        return bool(
            same_local_frame(self.frame, other.frame)
            and np.array_equal(self.x_m, other.x_m)
            and np.array_equal(self.y_m, other.y_m)
        )


@dataclass(frozen=True, eq=False)
class RectangularCells:
    """Own finite axis-aligned rectangles in a declared Cartesian chart.

    Attributes:
        lower_xy_m: Finite float64 lower corners (*pixels,2), in (x,y) order, metres.
        upper_xy_m: Matching strictly larger upper corners, metres.
        frame: Local chart for all rectangles. Arrays are immutable snapshots.
            Arbitrary rectangle collections are allowed, including overlaps;
            a conservative deposition operation can impose disjoint interiors.
    """

    lower_xy_m: FloatArray
    upper_xy_m: FloatArray
    frame: LocalCartesianFrame

    def __post_init__(self) -> None:
        """Snapshot positive finite rectangles.

        Raises:
            ValueError: Coordinate shapes, values, dtype or positive cell extents fail.
        """
        lo, hi = _snapshot(self.lower_xy_m), _snapshot(self.upper_xy_m)
        if lo.ndim < 1 or lo.shape[-1] != 2 or hi.shape != lo.shape or np.any(hi <= lo):
            raise ValueError("matching ordered (...,2) rectangular corners required")
        areas = np.prod(hi - lo, axis=-1)
        if not np.isfinite(hi - lo).all() or not np.isfinite(areas).all() or np.any(areas <= 0):
            raise ValueError("cell extents and areas must be finite and positive")
        object.__setattr__(self, "lower_xy_m", lo)
        object.__setattr__(self, "upper_xy_m", hi)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the retained pixel axes independently of coordinate order."""
        return self.lower_xy_m.shape[:-1]

    @property
    def area_m2(self) -> FloatArray:
        """Return fresh full rectangle areas (*pixels,), square metres."""
        return np.prod(self.upper_xy_m - self.lower_xy_m, axis=-1)

    @property
    def centre_xy_m(self) -> FloatArray:
        """Return fresh geometric centres (*pixels,2), in (x,y) order, metres."""
        return self.lower_xy_m + (self.upper_xy_m - self.lower_xy_m) / 2

    @classmethod
    def around_points(
        cls, points: PointLattice, size_xy_m: tuple[float, float]
    ) -> RectangularCells:
        """Declare finite rectangles centred at lattice points with explicit sizes.

        This constructs geometry only; it does not convert point values into cell means.

        Args:
            points: Declared centre lattice with (y,x) sample axes.
            size_xy_m: Positive finite cell widths in (x,y) order, metres.

        Returns:
            Immutable cell corners with (y,x,2) axes on the same frame.

        Raises:
            ValueError: Widths or resulting coordinates are invalid.
        """
        if not np.isfinite(size_xy_m).all() or min(size_xy_m) <= 0:
            raise ValueError("positive finite rectangle widths required")
        half = np.asarray(size_xy_m) / 2
        centres = points.xy_m
        return cls(centres - half, centres + half, points.frame)

    def matches(self, other: RectangularCells) -> bool:
        """Compare exact cell bounds on an admitted common frame.

        Args:
            other: Cell collection to compare without reshaping or alignment.

        Returns:
            True only for equal ordered corners and a matching declared frame.
        """
        return bool(
            same_local_frame(self.frame, other.frame)
            and np.array_equal(self.lower_xy_m, other.lower_xy_m)
            and np.array_equal(self.upper_xy_m, other.upper_xy_m)
        )


@dataclass(frozen=True, eq=False)
class LayeredCells:
    """Retain horizontal cells and terrain-clipped vertical intervals together.

    Attributes:
        horizontal: Immutable rectangle support, with the altitude datum in its frame.
        height_edges_m: Increasing finite float64 interfaces (layer+1,), metres.
        surface_altitude_m: Finite float64 piecewise-constant terrain (*pixels,), metres.
            Altitudes share the horizontal frame's datum. Arrays are immutable
            snapshots. Terrain above the model top leaves zero retained volume.
    """

    horizontal: RectangularCells
    height_edges_m: FloatArray
    surface_altitude_m: FloatArray

    def __post_init__(self) -> None:
        """Snapshot interfaces and terrain after checking their shared support.

        Raises:
            ValueError: Interfaces, terrain shape, finite values or dtype are invalid.
        """
        edges, terrain = _snapshot(self.height_edges_m), _snapshot(self.surface_altitude_m)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("at least two strictly increasing height interfaces required")
        if terrain.shape != self.horizontal.shape:
            raise ValueError("terrain shape must match horizontal cells")
        object.__setattr__(self, "height_edges_m", edges)
        object.__setattr__(self, "surface_altitude_m", terrain)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return (*pixels,layer) axes for amounts on this support."""
        return (*self.horizontal.shape, self.height_edges_m.size - 1)

    @property
    def volume_m3(self) -> FloatArray:
        """Return fresh terrain-clipped cell volumes (*pixels,layer), cubic metres."""
        lower = np.maximum(self.height_edges_m[:-1], self.surface_altitude_m[..., None])
        thickness = np.maximum(self.height_edges_m[1:] - lower, 0)
        return self.horizontal.area_m2[..., None] * thickness

    def matches(self, other: LayeredCells) -> bool:
        """Compare the complete horizontal and vertical support exactly.

        Args:
            other: Layered support to compare without remapping.

        Returns:
            True for matching cells/frame, equal interfaces and equal terrain.
        """
        return bool(
            self.horizontal.matches(other.horizontal)
            and np.array_equal(self.height_edges_m, other.height_edges_m)
            and np.array_equal(self.surface_altitude_m, other.surface_altitude_m)
        )


def cell_edges(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Snapshot a finite strictly increasing cell-edge vector.

    Args:
        values: One-dimensional edges in one caller-declared coordinate unit.

    Returns:
        Independent writable edges with at least one cell.

    Raises:
        ValueError: Edges are nonfinite, unordered or have fewer than two entries.
    """
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("cell edges must be finite strictly increasing vectors")
    return np.copy(values)


def cell_indices(edges: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Assign samples to half-open cells with an inclusive final upper edge.

    Args:
        edges: Validated increasing cell-edge vector.
        values: Sample coordinates with arbitrary shape, in the same units.

    Returns:
        Same-shaped indices; -1 and len(edges)-1 identify exterior samples.
    """
    indices = np.searchsorted(edges, values, side="right") - 1
    return np.where(values == edges[-1], len(edges) - 2, indices)


def resample_point_field(
    values: FloatArray,
    source: PointLattice,
    target: PointLattice,
    *,
    boundary: Literal["reject", "nearest"],
    source_valid: NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Bilinearly evaluate supplied numerical fields on another explicit point lattice.

    Args:
        values: Finite (y,x,...) samples in any consistent units, including signed
            coefficient or derivative fields. All trailing axes are retained.
        source: Input point coordinates; singleton axes are allowed.
        target: Output point coordinates on the same declared local frame.
        boundary: Reject outside points or explicitly extend nearest edge values.
            No implicit grid-centre alignment, reprojection or unit conversion occurs.
        source_valid: Optional Boolean (y,x) support; omission admits all source points.
            All four selected bounded endpoints must be valid, even at zero weight.

    Returns:
        Fresh float64 values on target (y,x,...) and a Boolean target support mask.
        Invalid results are zero. Nearest continuation certifies only its selected
        numerical extension, not additional physical coverage. No value clipping occurs.

    Raises:
        ValueError: Shapes, fields, frames, intervals or selected boundary fail.
    """
    if not same_local_frame(source.frame, target.frame):
        raise ValueError("point resampling requires an explicitly shared local frame")
    if values.ndim < 2 or values.shape[:2] != source.shape or not np.isfinite(values).all():
        raise ValueError("finite fields must match the source point axes")
    valid = np.ones(source.shape, dtype=bool) if source_valid is None else source_valid
    if (
        valid.shape != source.shape
        or valid.dtype != np.bool_
        or boundary not in ("reject", "nearest")
    ):
        raise ValueError("Boolean source support and explicit reject/nearest boundary required")
    ix, jx, fx = _interpolation_brackets(source.x_m, target.x_m, boundary)
    iy, jy, fy = _interpolation_brackets(source.y_m, target.y_m, boundary)
    output = np.zeros((*target.shape, *values.shape[2:]), dtype=np.float64)
    supported = np.ones(target.shape, dtype=bool)
    extra = (1,) * (values.ndim - 2)
    for y, wy in ((iy, 1 - fy), (jy, fy)):
        for x, wx in ((ix, 1 - fx), (jx, fx)):
            selected = (y[:, None], x[None, :])
            supported &= valid[selected]
            weights = (wy[:, None] * wx[None, :]).reshape((*target.shape, *extra))
            output += values[selected] * weights
    if not np.isfinite(output).all():
        raise ValueError("point-field interpolation arithmetic overflow")
    output[~supported] = 0
    return output, supported


def _interpolation_brackets(axis, points, boundary):
    with np.errstate(over="ignore"):
        intervals = np.diff(axis)
    if not np.isfinite(intervals).all():
        raise ValueError("interpolation intervals must be finite")
    if boundary == "reject" and np.any((points < axis[0]) | (points > axis[-1])):
        raise ValueError("target points lie outside the source interpolation support")
    points = np.clip(points, axis[0], axis[-1])
    left = np.minimum(np.searchsorted(axis, points, side="right") - 1, axis.size - 1)
    right = np.minimum(left + 1, axis.size - 1)
    fraction = np.divide(
        points - axis[left],
        axis[right] - axis[left],
        out=np.zeros(points.shape),
        where=right != left,
    )
    return left, right, fraction


def _snapshot(values: FloatArray) -> FloatArray:
    if values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError("support coordinates must be finite float64 arrays")
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
