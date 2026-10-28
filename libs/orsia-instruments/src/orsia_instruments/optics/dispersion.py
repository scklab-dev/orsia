"""Wavelength-dependent centroid maps, finite-pixel response and dispersive imaging."""

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition
from ..quadrature import validate_quadrature_order
from .pixels import gaussian_pixel_fraction, psf_pixel_response
from .psfs import PointSpreadFunction, select_psf
from .response import (
    ImagingOptics,
    SpectralParameter,
    constant_or_default,
    spectral_value,
)


def polynomial_image_map(
    image_xy_m: NDArray[np.floating | np.integer],
    wavelength_m: NDArray[np.float64],
    *,
    reference_m: float,
    dispersion: float,
    quadratic_per_m: float = 0,
    smile: float = 0,
    keystone: float = 0,
) -> NDArray[np.float64]:
    """Map incident (nodes,2) image positions to fresh (nodes,wavelengths,2) centroids.

    All positions/wavelengths are metres; wavelengths are a 1-D vector in any
    order. x=image_x + dispersion*dlambda + quadratic_per_m*dlambda**2
    + smile*image_y; y=image_y + keystone*dlambda. Coefficients are dimensionless
    except quadratic_per_m (1/m). Inputs are borrowed and never mutated.
    Geometry may preserve an acquisition's input dtype; output is fresh float64
    storage so in-place wavelength shifts cannot truncate to integer centroids.

    Args:
        image_xy_m: (nodes,2) incident detector-plane centroids in metres.
        wavelength_m: One-dimensional positive wavelength samples in metres.
        reference_m: Positive zero-dispersion wavelength in metres.
        dispersion: Linear x displacement per wavelength difference, dimensionless.
        quadratic_per_m: Quadratic x displacement coefficient in inverse metres.
        smile: Coupling of incident y displacement into x, dimensionless.
        keystone: Linear wavelength-dependent y displacement, dimensionless.

    Returns:
        Owned float64 (nodes,wavelengths,2) mapped image centroids in metres.

    Raises:
        ValueError: Wavelengths, incident positions or map parameters are invalid.
    """
    image = image_xy_m
    if not np.isfinite(wavelength_m).all() or np.any(wavelength_m <= 0):
        raise ValueError("wavelength must be finite and positive")
    if not np.isfinite(reference_m) or reference_m <= 0:
        raise ValueError("reference must be finite and positive")
    delta = wavelength_m - reference_m
    if image.ndim != 2 or image.shape[1] != 2 or not np.isfinite(image).all():
        raise ValueError("image positions must be finite with shape (nodes,2)")
    if delta.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional")
    if not np.isfinite([dispersion, quadratic_per_m, smile, keystone]).all():
        raise ValueError("dispersion parameters must be finite")
    # Acquisition snapshots preserve dtype; in-place spectral shifts need floating storage.
    centres = np.array(
        np.broadcast_to(image[:, None, :], (len(image), len(delta), 2)), dtype=np.float64
    )
    centres[..., 0] += dispersion * delta + quadratic_per_m * delta**2 + smile * centres[..., 1]
    centres[..., 1] += keystone * delta
    if not np.isfinite(centres).all():
        raise ValueError("dispersed image positions must be finite")
    return centres


def dispersive_response(
    wavelength_m: NDArray[np.float64],
    bounds_m: Sequence[float] | NDArray[np.floating | np.integer],
    image_xy_m: NDArray[np.float64],
    *,
    sigma_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    reference_m: float,
    dispersion: float,
    quadratic_per_m: float = 0,
    smile: float = 0,
    keystone: float = 0,
    transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
) -> NDArray[np.float64]:
    """Finite-pixel Gaussian redistribution, returning (scene nodes,wavelengths).

    x=image_x + D1*dlambda + D2*dlambda² + smile*image_y;
    y=image_y + keystone*dlambda. D1,smile,keystone are dimensionless; D2 is 1/m.
    Transmission includes supplied grating/slit losses, separately from width.

    Args:
        wavelength_m: Positive sampled wavelengths in metres.
        bounds_m: Detector active pixel (xmin,xmax,ymin,ymax) in metres.
        image_xy_m: (nodes,2) incident detector-plane centroids in metres.
        sigma_m: Gaussian PSF scalar or (x,y) width in detector metres.
        reference_m: Positive zero-dispersion wavelength in metres.
        dispersion: Linear wavelength-to-x displacement, dimensionless.
        quadratic_per_m: Quadratic x displacement coefficient in inverse metres.
        smile: Incident y-to-x coupling, dimensionless.
        keystone: Wavelength-to-y displacement, dimensionless.
        transmission: Broadcastable dimensionless throughput fraction in [0,1].

    Returns:
        Float64 (nodes,wavelengths) accepted fraction after pixel and throughput losses.

    Raises:
        ValueError: Dispersion, pixel/PSF geometry or transmission is invalid.
    """
    centres = polynomial_image_map(
        image_xy_m,
        wavelength_m,
        reference_m=reference_m,
        dispersion=dispersion,
        quadratic_per_m=quadratic_per_m,
        smile=smile,
        keystone=keystone,
    )
    fraction = gaussian_pixel_fraction(bounds_m, centres, sigma_m)
    efficiency = np.asarray(transmission, dtype=np.float64)
    if not np.isfinite(efficiency).all() or np.any((efficiency < 0) | (efficiency > 1)):
        raise ValueError("transmission must be finite and in [0, 1]")
    return fraction * efficiency


def dispersive_imager(
    *,
    focal_length_m: float,
    collecting_area_m2: float,
    psf_sigma_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer]
    | None = None,
    psf: PointSpreadFunction | None = None,
    pixel_order: int = 8,
    reference_m: float | None = None,
    dispersion: float | None = None,
    quadratic_per_m: float = 0,
    smile: float = 0,
    keystone: float = 0,
    image_map: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
    | None = None,
    transmission: SpectralParameter = 1,
    throughput: float = 1,
) -> ImagingOptics:
    """Finite-pixel disperser with a polynomial or supplied centroid map.

    PSF selection and callback coordinates follow filter_imager. General PSFs
    use pixel_order; Gaussian bindings retain analytic pixel integration.
    PSF field coordinates precede image_map; offsets use mapped centroids.

    Dispersion, smile and keystone are dimensionless; quadratic coefficient is
    1/m. This does not infer slit acceptance or reconstruct wavelength channels.
    Supply reference_m and dispersion for polynomial_image_map, or image_map
    with no polynomial parameters. image_map(image_xy_m,wavelengths_m) maps
    (nodes,2) incident positions and a 1-D wavelength vector to exact
    (nodes,wavelengths,2) finite centroids, all in metres. The map and a spectral
    transmission callable are borrowed, pure, stable, nonmutating and pointwise
    in wavelength. Transmission is scalar or a function returning the wavelength
    vector's exact shape with values in [0,1]. Lost PSF tails are not renormalised.

    Args:
        focal_length_m: Positive detector-plane focal length in metres.
        collecting_area_m2: Nonnegative pupil area in square metres.
        psf_sigma_m: Optional analytic Gaussian scalar or (x,y) width in metres.
        psf: Alternative borrowed point-spread density callback.
        pixel_order: Gauss order for numerical finite-pixel integration.
        reference_m: Optional positive zero-dispersion wavelength in metres.
        dispersion: Optional dimensionless linear wavelength-to-x displacement.
        quadratic_per_m: Quadratic x displacement coefficient in inverse metres.
        smile: Incident y-to-x coupling, dimensionless.
        keystone: Wavelength-to-y displacement, dimensionless.
        image_map: Alternative pure incident-position/wavelength centroid callback.
        transmission: Scalar or pure wavelength callback giving fractions in [0,1].
        throughput: Additional physical optical transmission fraction in [0,1].

    Returns:
        Imaging optics evaluating spectral centroid mapping and finite-pixel response.

    Raises:
        ValueError: PSF/map selection, geometry, quadrature or transmission is invalid.
    """
    sigma, density = select_psf(psf_sigma_m, psf)
    validate_quadrature_order(pixel_order)
    constant_transmission = constant_or_default(transmission, 1)
    if not np.isfinite(constant_transmission) or not 0 <= constant_transmission <= 1:
        raise ValueError("transmission must be finite and in [0, 1]")
    settings = {}
    if image_map is None:
        if reference_m is None or dispersion is None:
            raise ValueError("supply reference_m and dispersion, or image_map")
        if not np.isfinite(reference_m) or reference_m <= 0:
            raise ValueError("reference wavelength must be finite and positive")
        if not np.isfinite([dispersion, quadratic_per_m, smile, keystone]).all():
            raise ValueError("dispersion parameters must be finite")
        mapping = partial(
            polynomial_image_map,
            reference_m=reference_m,
            dispersion=dispersion,
            quadratic_per_m=quadratic_per_m,
            smile=smile,
            keystone=keystone,
        )
        settings = {"reference_m": reference_m}
    else:
        if (
            reference_m is not None
            or dispersion is not None
            or any(value != 0 for value in (quadratic_per_m, smile, keystone))
        ):
            raise ValueError("image_map cannot be combined with polynomial parameters")
        mapping = image_map

    def response(
        exposure: int, read: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        image = acquisition.image_xy_m[read]
        centres = mapping(image, wave)
        if centres.shape != (len(image), len(wave), 2) or not np.isfinite(centres).all():
            raise ValueError("image_map must return finite (nodes,wavelengths,2) centroids")
        efficiency = spectral_value(transmission, wave, "transmission")
        if not np.isfinite(efficiency).all() or np.any((efficiency < 0) | (efficiency > 1)):
            raise ValueError("transmission must be finite and in [0, 1]")
        if sigma is None:
            assert density is not None
            return psf_pixel_response(
                wave,
                acquisition.pixel_bounds_m[read],
                centres,
                image,
                exposure,
                psf=density,
                pixel_order=pixel_order,
                transmission=lambda x, y, w: efficiency,
            )
        return (
            gaussian_pixel_fraction(
                acquisition.pixel_bounds_m[read],
                centres,
                sigma,
            )
            * efficiency
        )

    return ImagingOptics(focal_length_m, collecting_area_m2, response, settings, throughput)
