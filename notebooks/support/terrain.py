"""Bind application-owned cubic terrain interpolation to height and gradient queries.

This SciPy adapter stays outside Geometry's NumPy-only terrain algorithms. One
interpolant supplies both operations and their closed Cartesian metre domain.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True, eq=False)
class SplineTerrain:
    """Represent a cubic height surface on a fixed rectangular metre grid.

    Attributes:
        x_m: Independent read-only increasing (nx,) x coordinates, metres.
        y_m: Independent read-only increasing (ny,) y coordinates, metres.
        heights_m: Independent read-only (nx,ny) heights, metres, in (x,y) order.
        horizontal_bounds_m: Closed (xmin,xmax,ymin,ymax) support, metres.

    Each axis needs at least four samples. Height and slopes come from the same
    cubic interpolant. Queries never extrapolate. Concurrent queries borrow the
    fixed interpolator without mutation; scientific native-thread safety remains
    unqualified. The surface supplies no full-ray visibility certificate.
    """

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    heights_m: NDArray[np.float64]
    _interpolator: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and snapshot the grid before preparing its cubic interpolant."""
        for axis in (self.x_m, self.y_m):
            if (
                axis.ndim != 1
                or len(axis) < 4
                or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)
            ):
                raise ValueError("cubic terrain requires finite increasing axes with four samples")
        if (
            self.heights_m.shape != (len(self.x_m), len(self.y_m))
            or not np.isfinite(self.heights_m).all()
        ):
            raise ValueError("terrain heights must be finite with shape (nx,ny)")
        for name in ("x_m", "y_m", "heights_m"):
            values = getattr(self, name)
            copied = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
            object.__setattr__(self, name, copied)
        object.__setattr__(
            self,
            "_interpolator",
            RegularGridInterpolator(
                (self.x_m, self.y_m),
                self.heights_m,
                method="cubic",
                bounds_error=True,
                solver_args={"rtol": 1e-13, "atol": 1e-13},
            ),
        )

    @property
    def horizontal_bounds_m(self) -> tuple[float, float, float, float]:
        """Return the closed rectangular support in Cartesian metres."""
        return float(self.x_m[0]), float(self.x_m[-1]), float(self.y_m[0]), float(self.y_m[-1])

    def height(self, xy_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return fresh writable (...) metre heights at finite (...,2) coordinates.

        Scalar (2,) coordinates and empty batches are supported. Invalid shapes,
        nonfinite coordinates and queries outside the closed domain raise ValueError.
        """
        self._require_domain(xy_m)
        return self._interpolator(xy_m).reshape(xy_m.shape[:-1]).copy()

    def gradient(self, xy_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return fresh writable (...,2) dimensionless (dz/dx,dz/dy) derivatives.

        Queries use the same finite (...,2) coordinates and closed support as
        height. Invalid coordinates raise ValueError. Boundary derivatives use
        the same cubic interpolant as interior heights.
        """
        self._require_domain(xy_m)
        return np.stack(
            [
                self._interpolator(xy_m, nu=order).reshape(xy_m.shape[:-1])
                for order in ((1, 0), (0, 1))
            ],
            axis=-1,
        )

    def _require_domain(self, xy_m: NDArray[np.float64]) -> None:
        if xy_m.ndim < 1 or xy_m.shape[-1] != 2 or not np.isfinite(xy_m).all():
            raise ValueError("terrain queries require finite (...,2) coordinates")
        xmin, xmax, ymin, ymax = self.horizontal_bounds_m
        if np.any((xy_m[..., 0] < xmin) | (xy_m[..., 0] > xmax)) or np.any(
            (xy_m[..., 1] < ymin) | (xy_m[..., 1] > ymax)
        ):
            raise ValueError("outside spline terrain support")
