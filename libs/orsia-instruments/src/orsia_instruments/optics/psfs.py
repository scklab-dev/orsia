"""Spatial intensity densities in m^-2 with declared physical mass.

Gaussian and Airy models have unit whole-plane mass; prepared tables retain
their finite-support mass. Callback normalization belongs to the caller. Inputs
remain unchanged, and throughput belongs outside the spatial density."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.special import j1

type PointSpreadFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]
]


@dataclass(frozen=True, eq=False, init=False)
class GaussianPSF:
    """Declared normalized diagonal Gaussian intensity model.

    Attributes:
        sigma_m: Owned immutable scalar or (x,y) positive widths in metres.
    """

    sigma_m: NDArray[np.float64]

    def __init__(
        self,
        sigma_m: float
        | np.floating
        | np.integer
        | Sequence[float]
        | NDArray[np.floating | np.integer],
    ) -> None:
        """Snapshot positive scalar or (x,y) Gaussian widths in metres.

        Args:
            sigma_m: Finite positive scalar or (x,y) standard deviations in metres.

        Raises:
            ValueError: Widths have invalid shape, sign or finiteness.
        """
        sigma = _psf_sigma(sigma_m)
        object.__setattr__(
            self, "sigma_m", np.frombuffer(sigma.tobytes(), dtype=np.float64).reshape(sigma.shape)
        )


@dataclass(frozen=True, eq=False, init=False)
class TabulatedPSF:
    """Prepared bilinear spatial density with retained finite-support physical mass.

    Values outside the table are zero. Mass below unity remains lost. The exact
    bilinear integral uses tensor trapezoids. Preparation validates and snapshots
    axes and density once; pixel quadrature still requires refinement.

    Attributes:
        x_offsets_m: Owned immutable strictly increasing x offsets in metres.
        y_offsets_m: Owned immutable strictly increasing y offsets in metres.
        density_per_m2: Owned immutable nonnegative (x,y) density in inverse square metres.
        mass: Finite table integral in [0,1], with roundoff tolerance 1e-10.
        interpolator: Prepared zero-extrapolating bilinear interpolator over immutable buffers.
    """

    x_offsets_m: NDArray[np.float64]
    y_offsets_m: NDArray[np.float64]
    density_per_m2: NDArray[np.float64]
    mass: float = field(init=False)
    interpolator: RegularGridInterpolator = field(init=False, repr=False)

    def __init__(
        self,
        x_offsets_m: Sequence[float] | NDArray[np.floating | np.integer],
        y_offsets_m: Sequence[float] | NDArray[np.floating | np.integer],
        density_per_m2: Sequence[Sequence[float]] | NDArray[np.floating | np.integer],
    ) -> None:
        """Prepare immutable axes, density, finite-support mass and interpolation once.

        Args:
            x_offsets_m: Strictly increasing x offsets in metres, at least two nodes.
            y_offsets_m: Strictly increasing y offsets in metres, at least two nodes.
            density_per_m2: Finite nonnegative density (x,y) in inverse square metres.

        Raises:
            ValueError: Axes, density shape, finiteness or physical mass is invalid.
        """
        x, y = _axis(x_offsets_m), _axis(y_offsets_m)
        density = np.array(density_per_m2, dtype=np.float64, copy=True)
        if (
            density.shape != (len(x), len(y))
            or not np.isfinite(density).all()
            or np.any(density < 0)
        ):
            raise ValueError("PSF table density must be finite nonnegative (x,y)")
        mass = float(np.trapezoid(np.trapezoid(density, x=y, axis=1), x=x))
        if not np.isfinite(mass) or mass < 0 or mass > 1 + 1e-10:
            raise ValueError("PSF table must have finite mass at most one")
        arrays = []
        for name, value in (("x_offsets_m", x), ("y_offsets_m", y), ("density_per_m2", density)):
            snapshot = np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)
            object.__setattr__(self, name, snapshot)
            arrays.append(snapshot)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(
            self,
            "interpolator",
            RegularGridInterpolator(
                (arrays[0], arrays[1]),
                arrays[2],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            ),
        )


type SpatialDensity = GaussianPSF | TabulatedPSF | PointSpreadFunction


def gaussian_psf(
    offset_xy_m: NDArray[np.float64],
    wavelengths_m: NDArray[np.float64],
    image_xy_m: NDArray[np.float64],
    exposure_index: int,
    *,
    sigma_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
) -> NDArray[np.float64]:
    """Normalized diagonal Gaussian density in m^-2, not a pixel fraction.

    Bind positive scalar or (x,y) sigma_m using functools.partial. Offsets are
    detector position minus mapped centroid, shape (nodes,wavelengths,2), in m.
    wavelengths_m is (wavelengths,) in m, in any order; image_xy_m is incident
    field position (nodes,2) in m, before any dispersive map. exposure_index is
    the zero-based global exposure index. This law ignores field/exposure and
    wavelength values; custom laws may use them. Sigma is not FWHM.

    Args:
        offset_xy_m: Detector position minus mapped centroid, (nodes, wavelengths, 2).
        wavelengths_m: Positive wavelengths in metres, (wavelengths,).
        image_xy_m: Incident image-plane positions in metres, (nodes, 2).
        exposure_index: Zero-based exposure identifier; density is invariant across exposures.
        sigma_m: Positive scalar width or (x, y) widths in metres.

    Returns:
        Gaussian intensity density in m^-2, (nodes, wavelengths).

    Raises:
        ValueError: If coordinates, wavelengths or widths have invalid support.
    """
    offsets = _coordinates(offset_xy_m, wavelengths_m, image_xy_m)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("PSF sigma must be finite and positive")
    if sigma.shape not in ((), (2,)):
        raise ValueError("PSF sigma must be scalar or an (x,y) pair")
    sigma = np.broadcast_to(sigma, (2,))
    z = offsets / sigma
    return np.exp(-0.5 * np.sum(z * z, axis=-1)) / (2 * np.pi * np.prod(sigma))


def airy_psf(
    offset_xy_m: NDArray[np.float64],
    wavelengths_m: NDArray[np.float64],
    image_xy_m: NDArray[np.float64],
    exposure_index: int,
    *,
    f_number: float,
    wavelength_max_m: float | None = None,
) -> NDArray[np.float64]:
    """Evaluate a whole-plane normalized circular-pupil Airy intensity density.

    The density is pi/(4*(wavelength*f_number)^2) * (2*J1(z)/z)^2,
    where z=pi*radius/(wavelength*f_number). The removable origin is exact.
    Field position and exposure do not change this ideal unobscured pupil.

    Args:
        offset_xy_m: Detector-minus-centroid offsets (nodes,wavelengths,2), metres.
        wavelengths_m: Positive (wavelengths,) wavelengths in metres.
        image_xy_m: Finite incident positions (nodes,2), metres.
        exposure_index: Physical exposure index, unused by this stationary law.
        f_number: Positive focal length divided by circular pupil diameter.
        wavelength_max_m: Optional positive upper support limit in metres.

    Returns:
        Finite nonnegative (nodes,wavelengths) density in m^-2.

    Raises:
        ValueError: Coordinates, f-number or wavelength support are invalid.
    """
    offsets = _coordinates(offset_xy_m, wavelengths_m, image_xy_m)
    if not np.isfinite(f_number) or f_number <= 0:
        raise ValueError("Airy f-number must be finite and positive")
    if wavelength_max_m is not None and (
        not np.isfinite(wavelength_max_m)
        or wavelength_max_m <= 0
        or np.any(wavelengths_m > wavelength_max_m)
    ):
        raise ValueError("Airy wavelengths exceed declared maximum support")
    scale = wavelengths_m * f_number
    z = np.pi * np.linalg.norm(offsets, axis=-1) / scale
    amplitude = np.ones_like(z)
    np.divide(2 * j1(z), z, out=amplitude, where=z != 0)
    return np.pi / (4 * scale**2) * amplitude**2


def _coordinates(
    offset_xy_m: NDArray[np.float64],
    wavelengths_m: NDArray[np.float64],
    image_xy_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    offsets = offset_xy_m
    wave = wavelengths_m
    if not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError("wavelength must be finite and positive")
    image = image_xy_m
    if (
        wave.ndim != 1
        or image.ndim != 2
        or image.shape[1] != 2
        or offsets.shape != (len(image), wave.size, 2)
        or not np.isfinite(offsets).all()
        or not np.isfinite(image).all()
    ):
        raise ValueError(
            "PSF needs finite offsets (nodes,wavelengths,2), wavelengths (K,), field (nodes,2)"
        )
    return offsets


def _axis(values: Sequence[float] | NDArray[np.floating | np.integer]) -> NDArray[np.float64]:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError(
            "PSF table axes must be finite strictly increasing vectors with at least two nodes"
        )
    return axis


def _psf_sigma(
    value: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
) -> NDArray[np.float64]:
    sigma = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("PSF sigma must be finite and positive")
    if sigma.shape not in ((), (2,)):
        raise ValueError("PSF sigma must be scalar or an (x,y) pair")
    return sigma
