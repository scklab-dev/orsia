"""Spatial intensity densities in m^-2, normalized on the whole image plane.

Inputs and callbacks are borrowed without mutation; no normalization of arbitrary
callbacks is inferred from point samples. Throughput belongs outside the PSF."""

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

type PointSpreadFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]
]


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


def tabulated_psf(
    offset_xy_m: NDArray[np.float64],
    wavelengths_m: NDArray[np.float64],
    image_xy_m: NDArray[np.float64],
    exposure_index: int,
    *,
    x_offsets_m: Sequence[float] | NDArray[np.floating | np.integer],
    y_offsets_m: Sequence[float] | NDArray[np.floating | np.integer],
    density_per_m2: Sequence[Sequence[float]] | NDArray[np.floating | np.integer],
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Bilinear density on a rectangular offset grid, zero outside its support.

    The callback coordinates/context follow gaussian_psf. Bind x_offsets_m, y_offsets_m and
    density_per_m2 with partial: axes are finite strictly increasing vectors
    (at least two nodes each), density is finite nonnegative (len(x),len(y))
    in m^-2. Values at support boundaries are included; beyond them density is
    zero, never extrapolated. Field, exposure and wavelength do not change this
    table. Inputs/captured arrays remain caller-owned and must stay stable.

    The tensor trapezoidal integral is exact for this bilinear interpolant.
    By default it must be unity (relative tolerance 1e-10). normalize=True
    explicitly divides by its positive finite integral without mutating the
    table. This normalizes the declared finite-support model, not unknown lost
    optical tails. It does not renormalize detector truncation. Repeated calls
    validate the table again; pixel quadrature still needs convergence checks.

    Args:
        offset_xy_m: Detector offsets in metres, (nodes, wavelengths, 2).
        wavelengths_m: Positive wavelength vector in metres.
        image_xy_m: Incident field positions in metres, (nodes, 2).
        exposure_index: Zero-based exposure identifier; density is invariant across exposures.
        x_offsets_m: Strictly increasing x offsets from the centroid in metres.
        y_offsets_m: Strictly increasing y offsets from the centroid in metres.
        density_per_m2: Nonnegative density with shape (len(x_offsets_m), len(y_offsets_m)).
        normalize: Whether to divide by the positive finite whole-table integral.

    Returns:
        Interpolated intensity density in m^-2, (nodes, wavelengths).

    Raises:
        ValueError: If coordinate support or table normalization is invalid.
    """
    offsets = _coordinates(offset_xy_m, wavelengths_m, image_xy_m)
    x = _axis(x_offsets_m)
    y = _axis(y_offsets_m)
    density = np.asarray(density_per_m2, dtype=np.float64)
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("PSF table density must be finite and in [0, inf]")
    if density.shape != (len(x), len(y)):
        raise ValueError("PSF table must have shape (len(x_offsets_m),len(y_offsets_m))")
    mass = float(np.trapezoid(np.trapezoid(density, x=y, axis=1), x=x))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("PSF table must have a positive finite integral")
    if not normalize and not np.isclose(mass, 1, rtol=1e-10, atol=0):
        raise ValueError("PSF table integral must be one; use normalize=True explicitly")
    interpolator = RegularGridInterpolator(
        (x, y), density, method="linear", bounds_error=False, fill_value=0.0
    )
    result = interpolator(offsets)
    return result / mass if normalize else result


def select_psf(
    sigma: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer]
    | None,
    psf: PointSpreadFunction | None,
) -> tuple[NDArray[np.float64] | None, PointSpreadFunction | None]:
    """Select exactly one spatial law without changing Gaussian pixel integration.

    Positive scalar/(x,y) sigma in metres is snapshotted. An exact
    partial(gaussian_psf, sigma_m=...) binding receives the same treatment;
    other PSF callbacks remain borrowed for quadrature and return m^-2 density.
    The pair is (copied Gaussian sigma, None) or (None, borrowed callable).

    Args:
        sigma: Scalar or (x, y) Gaussian width in metres, or None for a callback.
        psf: Borrowed spatial density callback, or None for direct Gaussian integration.

    Returns:
        A copied width paired with None for analytic integration, or None paired
        with a borrowed callback for numerical integration.

    Raises:
        ValueError: If both spatial laws are supplied, neither is supplied, or a
            selected Gaussian width is nonpositive or has the wrong shape.
    """
    if (sigma is None) == (psf is None):
        raise ValueError("supply exactly one of psf and psf_sigma_m")
    if sigma is not None:
        return _psf_sigma(sigma), None
    # Only a known, unmodified catalogue binding has an exact Gaussian integral.
    # Wrappers and arbitrary callable attributes never opt into this shortcut.
    if (
        type(psf) is partial
        and psf.func is gaussian_psf
        and not psf.args
        and set(psf.keywords) == {"sigma_m"}
    ):
        return _psf_sigma(psf.keywords["sigma_m"]), None
    return None, psf


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
