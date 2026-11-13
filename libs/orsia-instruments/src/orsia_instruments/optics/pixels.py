"""Finite-pixel acceptance for Gaussian, tabulated and supplied spatial responses."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from orsia_instruments.acquisition.quadrature import interval_quadrature
from orsia_instruments.optics.psfs import GaussianPSF, SpatialDensity, TabulatedPSF, gaussian_psf


def square_pixel_etendue(
    pixel_size_m: float,
    optical_f_number: float,
    detector_f_number: float,
) -> float:
    """Calculate the retained square-pixel acceptance-cone approximation.

    Args:
        pixel_size_m: Positive active square width, metres. Using detector pitch
            asserts a full-fill square pixel; geometry is otherwise caller-owned.
        optical_f_number: Positive finite optical f-number.
        detector_f_number: Positive finite detector acceptance f-number.

    Returns:
        Finite positive accepted etendue in m2 sr, pi*size^2/(4*max(f_numbers)^2).
        The larger f-number limits acceptance. This approximation supplies no PSF,
        field geometry, throughput, QE, duration or number of reads.

    Raises:
        ValueError: Parameters or represented etendue are nonpositive or nonfinite.
    """
    values = np.array([pixel_size_m, optical_f_number, detector_f_number], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("positive finite square-pixel size and f-numbers required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.pi / 4 * (values[0] / max(values[1:])) ** 2
    if not np.isfinite(result) or result <= 0:
        raise ValueError("square-pixel etendue exceeds positive finite support")
    return float(result)


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


def pixel_acceptance(
    wavelengths_m: NDArray[np.float64],
    bounds_m: Sequence[float] | NDArray[np.floating | np.integer],
    centres_m: NDArray[np.float64],
    *,
    spatial: SpatialDensity,
    transmission: float
    | NDArray[np.float64]
    | Callable[[float, float, NDArray[np.float64]], NDArray[np.float64]] = 1.0,
    image_xy_m: NDArray[np.float64] | None = None,
    exposure_index: int = 0,
    pixel_order: int = 8,
) -> NDArray[np.float64]:
    """Integrate one spatial intensity model and transmission over a finite pixel.

    A declared Gaussian and explicitly uniform scalar/array transmission use the
    analytic backend. Callbacks and prepared tables share one quadrature loop.
    Missing table mass and detector-truncated tails remain lost. Callback whole-plane
    normalization belongs to the caller; sampled guards establish local support only.
    Refine pixel_order to establish convergence for the supplied density/transmission.

    Args:
        wavelengths_m: Nonempty positive finite (wavelengths,) vector in metres.
        bounds_m: Finite ordered (xmin,xmax,ymin,ymax) pixel edges in metres.
        centres_m: Mapped centroids (nodes,2) or (nodes,wavelengths,2), in metres.
        spatial: Declared Gaussian, prepared tabulated density or pure density callback.
        transmission: Explicit uniform fraction(s) or local (x,y,wavelengths) callback.
        image_xy_m: Incident field (nodes,2) in metres for callbacks; defaults to centroids
            for wavelength-independent centroids.
        exposure_index: Global exposure index supplied to a density callback.
        pixel_order: Positive per-axis quadrature order; caller checks refinement.

    Returns:
        Fresh finite accepted fractions (nodes,wavelengths) in [0,1].

    Raises:
        ValueError: Geometry, density, transmission or accumulated fraction is invalid.
    """
    wave = wavelengths_m
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError("wavelengths must be a nonempty finite positive vector")
    bounds = np.asarray(bounds_m, dtype=np.float64)
    centres = centres_m
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or np.any(bounds[[1, 3]] <= bounds[[0, 2]])
        or centres.ndim not in (2, 3)
        or centres.shape[-1] != 2
        or not len(centres)
        or not np.isfinite(centres).all()
        or (centres.ndim == 3 and centres.shape[1] != len(wave))
    ):
        raise ValueError("pixel or image-centroid support invalid")
    image = centres if centres.ndim == 2 else centres[:, 0]
    if image_xy_m is not None:
        image = image_xy_m
    if image.shape != (len(centres), 2) or not np.isfinite(image).all():
        raise ValueError("incident image field must be finite (nodes,2)")
    mapped = np.broadcast_to(
        centres[:, None, :] if centres.ndim == 2 else centres, (len(centres), len(wave), 2)
    )
    shape = mapped.shape[:2]
    if isinstance(spatial, GaussianPSF) and not callable(transmission):
        local = _transmission(transmission, shape)
        result = gaussian_pixel_fraction(bounds, mapped, spatial.sigma_m) * local
    else:
        x, wx = interval_quadrature(bounds[0], bounds[1], pixel_order)
        y, wy = interval_quadrature(bounds[2], bounds[3], pixel_order)
        result = np.zeros(shape)
        for qx, dx in zip(x, wx, strict=True):
            for qy, dy in zip(y, wy, strict=True):
                offsets = np.array([qx, qy]) - mapped
                if isinstance(spatial, GaussianPSF):
                    density = gaussian_psf(
                        offsets, wave, image, exposure_index, sigma_m=spatial.sigma_m
                    )
                elif isinstance(spatial, TabulatedPSF):
                    density = spatial.interpolator(offsets)
                else:
                    density = spatial(offsets, wave, image, exposure_index)
                if density.shape != shape or not np.isfinite(density).all() or np.any(density < 0):
                    raise ValueError("PSF density must be finite nonnegative (nodes,wavelengths)")
                local = (
                    transmission(float(qx), float(qy), wave)
                    if callable(transmission)
                    else transmission
                )
                result += dx * dy * density * _transmission(local, shape)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1 + 1e-10):
        raise ValueError(
            "pixel fraction exceeds one or is nonfinite: check normalization/quadrature"
        )
    return np.minimum(result, 1.0)


def _transmission(
    value: float | NDArray[np.float64], shape: tuple[int, int]
) -> NDArray[np.float64]:
    local = np.asarray(value, dtype=np.float64)
    if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
        raise ValueError("transmission must be finite and in [0, 1]")
    return np.broadcast_to(local, shape)


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
