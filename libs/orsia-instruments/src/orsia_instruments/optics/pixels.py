"""Finite-pixel acceptance for Gaussian, tabulated and supplied spatial responses."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr

from ..quadrature import interval_quadrature
from .psfs import PointSpreadFunction


def gaussian_pixel_fraction(
    bounds_m: Sequence[float] | NDArray[np.floating | np.integer],
    centres_m: NDArray[np.float64],
    sigma_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
) -> NDArray[np.float64]:
    """Integrate a normalized diagonal 2-D Gaussian over one finite rectangular pixel.

    centres_m has shape (...,2); sigma is a scalar or broadcastable (...,2).
    Bounds are (xmin,xmax,ymin,ymax). Missing tails remain lost, not renormalised.
    Correlated/tilted PSFs and diffraction are outside this bounded profile.

    Args:
        bounds_m: Pixel x/y lower and upper edges in metres, (xmin, xmax, ymin, ymax).
        centres_m: Gaussian centroids in metres, shape (..., 2).
        sigma_m: Positive scalar or broadcastable (x, y) Gaussian widths in metres.

    Returns:
        Accepted fraction for each centroid, with shape centres_m.shape[:-1].

    Raises:
        ValueError: If pixel bounds, centroids or widths have invalid support.
    """
    bounds, centres, sigma = _profile_inputs(bounds_m, centres_m, sigma_m)
    lo = (bounds[[0, 2]] - centres) / sigma
    hi = (bounds[[1, 3]] - centres) / sigma
    # Use the survival side for positive-tail intervals to avoid 1-1 cancellation.
    mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    return np.prod(mass, axis=-1)


def modulated_pixel_response(
    wavelength_m: NDArray[np.float64],
    bounds_m: Sequence[float] | NDArray[np.floating | np.integer],
    image_xy_m: NDArray[np.float64],
    *,
    sigma_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    transmission: Callable[[float, float, NDArray[np.float64]], NDArray[np.float64]],
    pixel_order: int = 8,
) -> NDArray[np.float64]:
    """Integrate a joint local intensity law over a finite pixel and Gaussian relay.

    transmission(x_m,y_m,wavelengths_m) returns values broadcastable to (nodes,K).
    The caller closes over ray angles, time, setting, port or mapped filter/cavity
    coordinates. The profile is wavelength independent here; dispersive_response
    supplies a wavelength-dependent centroid instead. Refinement is mandatory
    for narrow images/local fringes; no quadrature accuracy is inferred from order.

    Args:
        wavelength_m: Positive wavelength vector in metres, (K,).
        bounds_m: Pixel edges (xmin, xmax, ymin, ymax) in metres.
        image_xy_m: Image centroids in metres, (nodes, 2).
        sigma_m: Positive Gaussian widths in metres, broadcastable to (nodes, 2).
        transmission: Local bounded transmission at (x_m, y_m, wavelength_m),
            broadcastable to (nodes, K).
        pixel_order: Quadrature order per pixel axis; refine for narrow features.

    Returns:
        Accepted fractions per node and wavelength, shape (nodes, K).

    Raises:
        ValueError: If geometry, wavelengths, widths or sampled transmission are invalid,
            or the estimated fraction exceeds unity beyond tolerance.
    """
    if not np.isfinite(wavelength_m).all() or np.any(wavelength_m <= 0):
        raise ValueError("wavelength must be finite and positive")
    bounds, centres, sigma = _profile_inputs(bounds_m, image_xy_m, sigma_m)
    if centres.ndim != 2:
        raise ValueError("image centroids must have shape (nodes,2)")
    sigma = np.broadcast_to(sigma, centres.shape)
    x, wx = interval_quadrature(bounds[0], bounds[1], pixel_order)
    y, wy = interval_quadrature(bounds[2], bounds[3], pixel_order)
    result = np.zeros((len(centres), len(wavelength_m)))
    for qx, dx in zip(x, wx, strict=True):
        for qy, dy in zip(y, wy, strict=True):
            z = (np.array([qx, qy]) - centres) / sigma
            density = np.exp(-0.5 * (z * z).sum(axis=-1)) / (2 * np.pi * np.prod(sigma, axis=-1))
            local = transmission(float(qx), float(qy), wavelength_m)
            if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
                raise ValueError("local response must be finite and in [0, 1]")
            result += (dx * dy * density[:, None]) * np.broadcast_to(local, result.shape)
    if np.any(result > 1 + 1e-10):
        raise ValueError("pixel quadrature is under-resolved: accepted fraction exceeds one")
    return np.minimum(result, 1)


def tabulated_pixel_response(
    wavelength_m: NDArray[np.float64],
    bounds_m: Sequence[float] | NDArray[np.floating | np.integer],
    image_xy_m: NDArray[np.float64],
    x_offsets_m: Sequence[float] | NDArray[np.floating | np.integer],
    y_offsets_m: Sequence[float] | NDArray[np.floating | np.integer],
    density_per_m2: Sequence[Sequence[float]] | NDArray[np.floating | np.integer],
    *,
    transmission: Callable[[float, float, NDArray[np.float64]], NDArray[np.float64]],
    pixel_order: int = 12,
) -> NDArray[np.float64]:
    """Integrate a supplied non-Gaussian spatial density on one finite pixel.

    The density is tabulated with axes (x_offset,y_offset), shape (Nx,Ny),
    in m^-2, relative to ``image_xy_m`` with shape (nodes,2) or
    (nodes,wavelengths,2). Its integral on its *entire* tabulated rectangle
    must be <=1; missing mass is lost, not renormalized. Values outside that
    rectangle are zero. The table can encode tilted/correlated profiles, but
    bilinear interpolation and pixel quadrature require explicit refinement.
    ``transmission(x,y,wavelength)`` supplies local bounded loss/order selection,
    broadcastable to (nodes,wavelengths); it is applied exactly once.

    Args:
        wavelength_m: Positive wavelengths in metres, (K,).
        bounds_m: Pixel edges (xmin, xmax, ymin, ymax) in metres.
        image_xy_m: Centroids in metres, (nodes, 2) or (nodes, K, 2).
        x_offsets_m: Increasing x offsets relative to each centroid in metres.
        y_offsets_m: Increasing y offsets relative to each centroid in metres.
        density_per_m2: Nonnegative table, (len(x_offsets_m), len(y_offsets_m)), in m^-2.
        transmission: Local bounded transmission at each pixel quadrature point.
        pixel_order: Quadrature order per pixel axis; caller checks convergence.

    Returns:
        Accepted fractions per node and wavelength, shape (nodes, K).

    Raises:
        ValueError: If table mass, support, geometry or sampled transmission is invalid,
            or quadrature exceeds unit accepted fraction beyond tolerance.
    """
    wavelength = wavelength_m
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    x = np.asarray(x_offsets_m, dtype=np.float64)
    y = np.asarray(y_offsets_m, dtype=np.float64)
    density = np.asarray(density_per_m2, dtype=np.float64)
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("spatial density must be finite and in [0, inf]")
    bounds = np.asarray(bounds_m, dtype=np.float64)
    centres = image_xy_m
    if wavelength.ndim != 1 or len(wavelength) < 1:
        raise ValueError("wavelengths must be a nonempty vector")
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) < 2
        or len(y) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
        or density.shape != (len(x), len(y))
    ):
        raise ValueError("spatial density requires ordered finite x/y axes")
    mass = np.trapezoid(np.trapezoid(density, y, axis=1), x)
    if mass > 1 + 1e-10:
        raise ValueError("spatial density has more than unit physical mass")
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or np.any(bounds[[1, 3]] <= bounds[[0, 2]])
        or centres.ndim not in (2, 3)
        or centres.shape[-1] != 2
        or not np.isfinite(centres).all()
        or (centres.ndim == 3 and centres.shape[1] != len(wavelength))
    ):
        raise ValueError("pixel or image-centroid support invalid")
    centres = np.broadcast_to(
        centres[:, None, :] if centres.ndim == 2 else centres,
        (centres.shape[0], len(wavelength), 2),
    )
    px, wx = interval_quadrature(bounds[0], bounds[1], pixel_order)
    py, wy = interval_quadrature(bounds[2], bounds[3], pixel_order)
    interpolator = RegularGridInterpolator(
        (x, y), density, method="linear", bounds_error=False, fill_value=0.0
    )
    result = np.zeros(centres.shape[:2])
    for qx, dx in zip(px, wx, strict=True):
        for qy, dy in zip(py, wy, strict=True):
            offsets = np.array([qx, qy]) - centres
            profile = interpolator(offsets)
            local = transmission(float(qx), float(qy), wavelength)
            if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
                raise ValueError("local transmission must be finite and in [0, 1]")
            result += dx * dy * profile * np.broadcast_to(local, result.shape)
    if np.any(result > 1 + 1e-10):
        raise ValueError("pixel quadrature is under-resolved: fraction exceeds one")
    return np.minimum(result, 1)


def psf_pixel_response(
    wavelengths_m: NDArray[np.float64],
    bounds_m: NDArray[np.float64],
    centres_m: NDArray[np.float64],
    image_xy_m: NDArray[np.float64],
    exposure_index: int,
    *,
    psf: PointSpreadFunction,
    transmission: Callable[[float, float, NDArray[np.float64]], NDArray[np.float64]],
    pixel_order: int,
) -> NDArray[np.float64]:
    """Joint quadrature for imaging factories with already validated geometry.

    centres_m is (nodes,wavelengths,2); image_xy_m is the unmapped (nodes,2)
    field. At each detector-plane quadrature point call psf(offsets,wave,field,
    exposure) for exact (nodes,wavelengths) densities. No input is mutated.
    transmission returns wavelength-shaped fractions. This checks sampled
    density/acceptance, not whole-plane normalization or convergence. Fraction
    overshoot no larger than 1e-10 is rounded to one, as in the Gaussian path.

    Args:
        wavelengths_m: Wavelength vector in metres, (K,).
        bounds_m: Pixel edges (xmin, xmax, ymin, ymax) in metres.
        centres_m: Mapped centroids in metres, (nodes, K, 2).
        image_xy_m: Unmapped incident positions in metres, (nodes, 2).
        exposure_index: Zero-based exposure passed to the PSF callback.
        psf: Borrowed intensity-density callback returning (nodes, K) in m^-2.
        transmission: Local pixel transmission returning fractions for (nodes, K).
        pixel_order: Quadrature order along each pixel axis.

    Returns:
        Accepted fractions for each node and wavelength, shape (nodes, K).

    Raises:
        ValueError: If sampled PSF density or transmission is invalid or the accepted
            fraction exceeds unity beyond tolerance.
    """
    x, wx = interval_quadrature(bounds_m[0], bounds_m[1], pixel_order)
    y, wy = interval_quadrature(bounds_m[2], bounds_m[3], pixel_order)
    result = np.zeros((len(image_xy_m), len(wavelengths_m)))
    for qx, dx in zip(x, wx, strict=True):
        for qy, dy in zip(y, wy, strict=True):
            offsets = np.array([qx, qy]) - centres_m
            density = psf(offsets, wavelengths_m, image_xy_m, exposure_index)
            if not np.isfinite(density).all() or np.any(density < 0):
                raise ValueError("PSF density must be finite and in [0, inf]")
            if density.shape != result.shape:
                raise ValueError("PSF density must have shape (nodes,wavelengths)")
            local = transmission(float(qx), float(qy), wavelengths_m)
            if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
                raise ValueError("transmission must be finite and in [0, 1]")
            result += dx * dy * density * local
    if not np.isfinite(result).all() or np.any(result > 1 + 1e-10):
        raise ValueError(
            "PSF pixel fraction exceeds one or is nonfinite: check normalization/quadrature"
        )
    return np.minimum(result, 1)


def _profile_inputs(
    bounds_m: Sequence[float] | NDArray[np.floating | np.integer],
    centres_m: NDArray[np.float64],
    sigma_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
):
    bounds = np.asarray(bounds_m, dtype=np.float64)
    centres = centres_m
    sigma = np.asarray(sigma_m, dtype=np.float64)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("image sigma must be finite and positive")
    if (
        bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[[1, 3]] <= bounds[[0, 2]])
    ):
        raise ValueError("pixel bounds must be finite and ordered")
    if centres.ndim < 1 or centres.shape[-1] != 2 or not np.all(np.isfinite(centres)):
        raise ValueError("centres must be finite with final axis xy")
    return bounds, centres, sigma
