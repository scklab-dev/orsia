"""Spectral filter laws and their uniform, spatial and LVF imaging factories.

Physical transmission is peak-normalized, never implicitly unit-area normalized."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from ..quadrature import validate_quadrature_order
from .psfs import PointSpreadFunction
from .response import (
    ImagingOptics,
    exposure_setting,
    imaging_optics,
    setting_at_exposure,
)


def gaussian_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    centre_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    width_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    peak: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1.0,
) -> NDArray[np.float64]:
    """Peak-normalised Gaussian transmission; width is wavelength sigma, not FWHM.

    Inputs broadcast explicitly. This is a local passband approximation, not a
    coating/blocking model. Increasing width at fixed peak admits more continuum.

    Args:
        wavelength_m: Positive evaluated wavelength(s) in metres.
        centre_m: Positive passband centre wavelength(s) in metres.
        width_m: Positive Gaussian wavelength standard deviation(s) in metres.
        peak: Maximum accepted intensity fraction(s) in [0,1].

    Returns:
        Broadcast float64 dimensionless transmission fractions.

    Raises:
        ValueError: Wavelength, centre, width or peak leaves physical support.
    """
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    centre = np.asarray(centre_m, dtype=np.float64)
    if not np.isfinite(centre).all() or np.any(centre <= 0):
        raise ValueError("passband centre must be finite and positive")
    width = np.asarray(width_m, dtype=np.float64)
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("passband width must be finite and positive")
    amplitude = np.asarray(peak, dtype=np.float64)
    if not np.isfinite(amplitude).all() or np.any((amplitude < 0) | (amplitude > 1)):
        raise ValueError("peak must be finite and in [0, 1]")
    return amplitude * np.exp(-0.5 * ((wavelength - centre) / width) ** 2)


def lorentzian_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    centre_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    fwhm_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    peak: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1.0,
) -> NDArray[np.float64]:
    """Peak-normalised Lorentzian transmission, with full width at half maximum.

    All wavelengths/widths are positive finite metres; inputs broadcast. No
    unit-area normalisation, automatic blocking or spectral support is inferred.

    Args:
        wavelength_m: Evaluated positive wavelength(s) in metres.
        centre_m: Positive passband centre wavelength(s) in metres.
        fwhm_m: Positive Lorentzian full width at half maximum in metres.
        peak: Dimensionless maximum accepted fraction(s) in [0,1].

    Returns:
        Broadcast float64 Lorentzian transmission fractions.

    Raises:
        ValueError: Wavelength, centre, width or peak is invalid.
    """
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    centre = np.asarray(centre_m, dtype=np.float64)
    if not np.isfinite(centre).all() or np.any(centre <= 0):
        raise ValueError("passband centre must be finite and positive")
    width = np.asarray(fwhm_m, dtype=np.float64)
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("passband FWHM must be finite and positive")
    amplitude = np.asarray(peak, dtype=np.float64)
    if not np.isfinite(amplitude).all() or np.any((amplitude < 0) | (amplitude > 1)):
        raise ValueError("peak must be finite and in [0, 1]")
    with np.errstate(over="ignore"):
        return amplitude * (1 / np.hypot(1, 2 * ((wavelength - centre) / width))) ** 2


def super_gaussian_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    centre_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    fwhm_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    order: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 2,
    peak: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1.0,
) -> NDArray[np.float64]:
    """Transmission peak*exp(-log(2)*abs(2*(wavelength-centre)/FWHM)**(2*order)).

    Inputs broadcast; wavelengths and FWHM are positive finite metres. Finite
    order >=1 need not be integral; order=1 is Gaussian with the specified FWHM,
    not the sigma convention of gaussian_filter. Large order sharpens the edges.

    Args:
        wavelength_m: Evaluated positive wavelength(s) in metres.
        centre_m: Positive passband centre wavelength(s) in metres.
        fwhm_m: Positive full width at half maximum in metres.
        order: Finite shape exponent at least one, defaulting to two.
        peak: Maximum transmitted fraction(s) in [0,1].

    Returns:
        Broadcast float64 super-Gaussian transmission fractions.

    Raises:
        ValueError: Wavelength, centre, width, shape or peak is invalid.
    """
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    centre = np.asarray(centre_m, dtype=np.float64)
    if not np.isfinite(centre).all() or np.any(centre <= 0):
        raise ValueError("passband centre must be finite and positive")
    width = np.asarray(fwhm_m, dtype=np.float64)
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("passband FWHM must be finite and positive")
    exponent = np.asarray(order, dtype=np.float64)
    if not np.isfinite(exponent).all() or np.any(exponent < 1):
        raise ValueError("super-Gaussian order must be finite and in [1, inf]")
    amplitude = np.asarray(peak, dtype=np.float64)
    if not np.isfinite(amplitude).all() or np.any((amplitude < 0) | (amplitude > 1)):
        raise ValueError("peak must be finite and in [0, 1]")
    with np.errstate(over="ignore"):
        return amplitude * np.exp(
            -np.log(2) * np.abs(2 * ((wavelength - centre) / width)) ** (2 * exponent)
        )


def rectangular_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    lower_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    upper_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    peak: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1.0,
) -> NDArray[np.float64]:
    """Ideal passband, peak on the closed interval [lower_m,upper_m], zero outside.

    Positive finite bounds in metres must be strictly ordered; inputs broadcast.
    The discontinuous edges require explicit spectral quadrature refinement.

    Args:
        wavelength_m: Positive sampled wavelength(s) in metres.
        lower_m: Inclusive lower passband edge(s) in metres.
        upper_m: Inclusive upper passband edge(s) in metres.
        peak: Dimensionless accepted fraction(s) inside the interval.

    Returns:
        Broadcast float64 transmission equal to peak on the interval and zero elsewhere.

    Raises:
        ValueError: Bounds, wavelengths or peak violate support.
    """
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    lower = np.asarray(lower_m, dtype=np.float64)
    if not np.isfinite(lower).all() or np.any(lower <= 0):
        raise ValueError("lower wavelength must be finite and positive")
    upper = np.asarray(upper_m, dtype=np.float64)
    if not np.isfinite(upper).all() or np.any(upper <= 0):
        raise ValueError("upper wavelength must be finite and positive")
    if np.any(upper <= lower):
        raise ValueError("passband bounds must be strictly ordered")
    amplitude = np.asarray(peak, dtype=np.float64)
    if not np.isfinite(amplitude).all() or np.any((amplitude < 0) | (amplitude > 1)):
        raise ValueError("peak must be finite and in [0, 1]")
    return amplitude * ((wavelength >= lower) & (wavelength <= upper))


def longpass_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    cutoff_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    transition_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    pass_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    stop_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
) -> NDArray[np.float64]:
    """Logistic edge passing longer wavelengths, with finite leakage/rejection.

    cutoff_m is the midpoint; positive transition_m is the 10-to-90 percent
    width relative to the supplied stop/pass levels, in metres. Levels obey
    0 <= stop <= pass <= 1. Inputs broadcast; this is a phenomenological law.

    Args:
        wavelength_m: Evaluated positive wavelength(s) in metres.
        cutoff_m: Midpoint of the increasing transmission edge in metres.
        transition_m: Positive 10-to-90 percent transition width in metres.
        pass_transmission: Long-wavelength accepted intensity fraction in [0,1].
        stop_transmission: Short-wavelength leakage fraction up to the pass level.

    Returns:
        Broadcast float64 logistic transmission from stop to pass level.

    Raises:
        ValueError: Wavelength, transition width or transmission levels are invalid.
    """
    edge = _edge_fraction(wavelength_m, cutoff_m, transition_m)
    passed, stopped = _edge_levels(pass_transmission, stop_transmission)
    return stopped + (passed - stopped) * edge


def shortpass_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    cutoff_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    transition_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    pass_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    stop_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
) -> NDArray[np.float64]:
    """Model a finite-width optical edge that passes shorter wavelengths.

    Args:
        wavelength_m: Evaluated wavelengths in metres, scalar or array.
        cutoff_m: Half-transmission wavelength in metres, broadcast with samples.
        transition_m: Positive 10-to-90 percent edge width in metres.
        pass_transmission: Short-wavelength accepted intensity fraction in [0,1].
        stop_transmission: Long-wavelength leakage fraction no greater than pass level.

    Returns:
        Broadcast float64 accepted fractions across the wavelength inputs.

    Raises:
        ValueError: Wavelength, width or transmission levels violate their domains.
    """
    edge = _edge_fraction(wavelength_m, cutoff_m, transition_m)
    passed, stopped = _edge_levels(pass_transmission, stop_transmission)
    return stopped + (passed - stopped) * (1 - edge)


def notch_filter(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    centre_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    fwhm_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    depth: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    pass_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    order: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 2,
) -> NDArray[np.float64]:
    """Super-Gaussian rejection: pass_transmission*(1-depth*profile), not a passband.

    depth in [0,1] is fractional rejection relative to pass_transmission; the
    centre transmits pass_transmission*(1-depth). FWHM (m) is the full width at
    half rejection depth. order follows super_gaussian_filter; inputs broadcast.

    Args:
        wavelength_m: Evaluated positive wavelength(s) in metres.
        centre_m: Centre of maximum rejection in metres.
        fwhm_m: Full width at half rejection depth in metres.
        depth: Fractional rejection at the notch centre in [0,1].
        pass_transmission: Off-notch accepted fraction in [0,1].
        order: Super-Gaussian shape exponent, at least one.

    Returns:
        Broadcast float64 transmission with the selected rejection notch.

    Raises:
        ValueError: Wavelength, width, depth, pass level or order is invalid.
    """
    rejection = super_gaussian_filter(wavelength_m, centre_m, fwhm_m, order=order)
    passed = np.asarray(pass_transmission, dtype=np.float64)
    if not np.isfinite(passed).all() or np.any((passed < 0) | (passed > 1)):
        raise ValueError("pass transmission must be finite and in [0, 1]")
    depth = np.asarray(depth, dtype=np.float64)
    if not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)):
        raise ValueError("notch depth must be finite and in [0, 1]")
    return passed * (1 - depth * rejection)


def lvf_centre(
    coordinate_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    *,
    reference_m: float,
    gradient: float,
    curvature_per_m: float = 0,
    temperature_delta_k: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
    thermal_shift_m_per_k: float = 0,
    angle_rad: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
    effective_index: float = 1.5,
) -> NDArray[np.float64]:
    """Local LVF centre: polynomial position/temperature law and effective-index shift.

    coordinate_m is displacement in the mapped filter plane; external angle is
    from air/vacuum. Gradient has units m/m; curvature 1/m. No polarization or
    angular change in peak/width is implied. Caller must bound the law's validity.

    Args:
        coordinate_m: Detector/filter-plane displacement from reference in metres.
        reference_m: Normal-incidence central wavelength at zero displacement in metres.
        gradient: Linear central-wavelength gradient in metre per metre.
        curvature_per_m: Quadratic position coefficient in inverse metres.
        temperature_delta_k: Temperature departure from reference in kelvin.
        thermal_shift_m_per_k: Linear centre shift per kelvin in metres per kelvin.
        angle_rad: External incidence angle(s) from normal in radians.
        effective_index: Positive effective coating index, at least one.

    Returns:
        Broadcast float64 central wavelengths in metres after position/angle shifts.

    Raises:
        ValueError: Angle/index or resulting wavelength leaves physical support.
    """
    coordinate = np.asarray(coordinate_m, dtype=np.float64)
    delta = np.asarray(temperature_delta_k, dtype=np.float64)
    angle = np.asarray(angle_rad, dtype=np.float64)
    if not np.isfinite(angle).all() or np.any((angle < -np.pi / 2) | (angle > np.pi / 2)):
        raise ValueError(f"external angle must be finite and in [{-np.pi / 2}, {np.pi / 2}]")
    if not np.isfinite(effective_index) or effective_index < 1:
        raise ValueError("effective index must be finite and >=1")
    centre = reference_m + gradient * coordinate + curvature_per_m * coordinate**2
    centre = centre + thermal_shift_m_per_k * delta
    shifted = np.asarray(centre * np.sqrt(1 - (np.sin(angle) / effective_index) ** 2))
    if not np.isfinite(shifted).all() or np.any(shifted <= 0):
        raise ValueError("LVF centre must be finite and positive")
    return shifted


def tabulated_transmission(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    nodes_m: Sequence[float] | NDArray[np.floating | np.integer],
    fractions: Sequence[float] | NDArray[np.floating | np.integer],
    *,
    outside: str = "error",
) -> NDArray[np.float64]:
    """Linear transmission interpolation on an increasing supplied wavelength grid.

    ``outside='zero'`` asserts complete blocking beyond tabulated support;
    ``outside='error'`` rejects it. Neither policy extrapolates transmission.
    Include leaks and every passed order in the supplied spectrum and table.
    Sampling must resolve real peaks; no interpolation accuracy is promised.

    Args:
        wavelength_m: Positive evaluation wavelength(s) in metres.
        nodes_m: Increasing positive table wavelengths in metres.
        fractions: Matching physical transmission fractions in [0,1].
        outside: "error" to reject unsupported samples or "zero" for blocking.

    Returns:
        Float64 interpolated transmission matching wavelength_m in shape.

    Raises:
        ValueError: Table, samples or out-of-support selection is invalid.
    """
    wavelengths = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
        raise ValueError("wavelength must be finite and positive")
    nodes = np.asarray(nodes_m, dtype=np.float64)
    if not np.isfinite(nodes).all() or np.any(nodes <= 0):
        raise ValueError("transmission wavelengths must be finite and positive")
    values = np.asarray(fractions, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("transmission must be finite and in [0, 1]")
    if nodes.ndim != 1 or len(nodes) < 2 or np.any(np.diff(nodes) <= 0):
        raise ValueError("transmission wavelengths must strictly increase")
    if values.shape != nodes.shape or outside not in ("error", "zero"):
        raise ValueError("transmission table shape or outside policy invalid")
    if outside == "error" and np.any((wavelengths < nodes[0]) | (wavelengths > nodes[-1])):
        raise ValueError("wavelengths exceed transmission table support")
    return np.interp(wavelengths, nodes, values, left=0, right=0)


def filter_imager(
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
    filter: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    throughput: float = 1,
    pixel_order: int = 8,
) -> ImagingOptics:
    """Image through a supplied spatially uniform, wavelength-only intensity filter.

    ``filter(wavelengths_m)`` receives a 1-D float64 wavelength vector in metres
    (not necessarily increasing) and returns an equally shaped array of finite
    dimensionless transmission fractions in [0,1]. Values are checked when the
    response is evaluated, before pixel acceptance is applied; no normalisation
    or clipping is performed. The same filter applies to every exposure.

    The callable and its captured data are borrowed, not copied into ``settings``.
    It must remain pure, stable and pointwise in wavelength, preserving its input.
    Bind parameters of ``gaussian_filter`` or ``tabulated_transmission`` with
    ``functools.partial``, or supply another spectral law. Tabulated support and
    any out-of-support policy remain the filter's responsibility.

    Supply exactly one of ``psf`` and ``psf_sigma_m``. The callback is
    psf(offset_xy_m, wavelengths_m, image_xy_m, exposure_index), returning
    finite nonnegative (nodes,wavelengths) density in m^-2, with unit integral
    on the whole plane. Offsets (nodes,wavelengths,2) are detector position
    minus mapped centroid; field positions (nodes,2) are before dispersion,
    all in metres. Exposure is the global zero-based index. Inputs/captures
    are borrowed, stable and nonmutating; normalization is caller-owned.
    No registration or inheritance is needed. Bind gaussian_psf/tabulated_psf
    with partial, or supply your own law. General PSFs use pixel_order
    quadrature; point checks do not prove normalization or convergence.

    Legacy sigma is a positive scalar or (x,y) pair in focal-plane metres.
    Both it and partial(gaussian_psf, sigma_m=...) snapshot sigma and retain
    analytic finite-pixel integration here. Wrapped Gaussians use quadrature.
    Lost tails are never renormalised. Filter transmission and additional
    optical ``throughput`` each enter once. Spatially varying filters require
    ``lvf_imager`` or ``spatial_filter_imager`` instead.

    Args:
        focal_length_m: Positive detector-plane focal length in metres.
        collecting_area_m2: Nonnegative projected collecting area in square metres.
        psf_sigma_m: Optional analytic Gaussian scalar or (x,y) width in metres.
        psf: Alternative borrowed point-spread density callback.
        filter: Pure wavelength-vector callback returning accepted fractions in [0,1].
        throughput: Additional dimensionless optical transmission in [0,1].
        pixel_order: Gauss order for numerical pixel integration of general PSFs.

    Returns:
        Imaging response evaluating filter transmission and finite-pixel acceptance.

    Raises:
        ValueError: PSF selection, geometry, throughput or callback output is invalid.
    """

    def transmission(
        exposure: int, x: float, y: float, wave: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        values = filter(wave)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("filter transmission must be finite and in [0, 1]")
        if values.shape != wave.shape:
            raise ValueError("filter transmission must have the same shape as wavelengths")
        return values

    return imaging_optics(
        focal_length_m,
        collecting_area_m2,
        psf_sigma_m,
        throughput,
        transmission,
        {},
        psf=psf,
        pixel_order=pixel_order,
    )


def spatial_filter_imager(
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
    filter: Callable[[float, float, NDArray[np.float64]], NDArray[np.float64]],
    throughput: float = 1,
    pixel_order: int = 8,
) -> ImagingOptics:
    """Image through a local detector-plane filter, integrated jointly with the PSF.

    PSF selection and callback coordinates follow filter_imager; pixel_order
    controls the joint integral for both Gaussian and supplied PSFs.

    filter(x_m,y_m,wavelengths_m) receives scalar detector positions and a 1-D
    wavelength vector in metres, returning the same shape with finite values in
    [0,1]. It is shared across exposures and borrowed, pure, stable, nonmutating
    and pointwise in wavelength. No ray/angle/time dependence or filter-plane mapping
    is inferred. Discontinuities (including mosaics) need pixel quadrature
    refinement; pixel_order alone does not establish accuracy.

    Args:
        focal_length_m: Positive focal length in detector metres.
        collecting_area_m2: Nonnegative projected pupil area in square metres.
        psf_sigma_m: Optional analytic Gaussian width in detector metres.
        psf: Alternative borrowed point-spread density callback.
        filter: Pure (x_m,y_m,wavelengths_m) accepted-fraction callback.
        throughput: Additional dimensionless optical transmission in [0,1].
        pixel_order: Gauss order for joint PSF/filter pixel integration.

    Returns:
        Imaging response integrating the spatial filter over finite active pixels.

    Raises:
        ValueError: PSF selection, quadrature or filter output is invalid.
    """
    validate_quadrature_order(pixel_order)

    def transmission(
        exposure: int, x: float, y: float, wave: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _filter_values(filter(x, y, wave), wave, "spatial filter transmission")

    return imaging_optics(
        focal_length_m,
        collecting_area_m2,
        psf_sigma_m,
        throughput,
        transmission,
        {},
        psf=psf,
        spatial=True,
        pixel_order=pixel_order,
    )


def lvf_imager(
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
    reference_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    gradient: float,
    sigma_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer]
    | None = None,
    profile: Callable[[NDArray[np.float64], float], NDArray[np.float64]] | None = None,
    axis: str = "x",
    reference_position_m: float = 0,
    curvature_per_m: float = 0,
    peak: float = 1,
    throughput: float = 1,
    pixel_order: int = 8,
) -> ImagingOptics:
    """Local LVF in the detector plane, integrated jointly with the selected PSF.

    PSF selection and callback coordinates follow filter_imager.

    Centre = reference + gradient*coordinate + curvature*coordinate**2;
    coordinate is x or y minus reference_position_m. Gradient is m/m, curvature
    is 1/m. Reference and spectral sigma can vary per exposure. Supply either
    sigma_m for the existing Gaussian law, or profile(wavelengths_m,centre_m)
    returning the wavelength vector's exact shape with transmission in [0,1].
    peak multiplies either profile once. The profile is borrowed, pure, stable,
    nonmutating and pointwise in wavelength; its centre argument is a scalar in metres.
    This convenience model assumes normal incidence and no thermal shift; use a custom response
    for ray/time-dependent coating physics or a nonidentity filter-plane map.
    Refine pixel_order as well as the spectral/time/angular quadratures.

    Args:
        focal_length_m: Positive image-plane focal length in metres.
        collecting_area_m2: Nonnegative projected collecting area in square metres.
        psf_sigma_m: Optional Gaussian PSF width in detector metres.
        psf: Alternative borrowed point-spread density callback.
        reference_m: Scalar or per-exposure central wavelength at reference position in metres.
        gradient: Filter-centre gradient in metre per metre of detector coordinate.
        sigma_m: Optional Gaussian spectral width in metres, scalar or per exposure.
        profile: Alternative pure (wavelengths_m,centre_m) transmission callback.
        axis: Detector "x" or "y" axis controlling spatial dispersion.
        reference_position_m: Zero-displacement position along that axis in metres.
        curvature_per_m: Quadratic spatial coefficient in inverse metres.
        peak: Maximum filter transmission fraction in [0,1].
        throughput: Additional dimensionless optical transmission in [0,1].
        pixel_order: Gauss order for joint spectral-filter/PSF integration.

    Returns:
        Local LVF imaging response with scalar or per-exposure settings.

    Raises:
        ValueError: Axis, spectral/PSF choice, spatial law or transmission is invalid.
    """
    reference = np.asarray(reference_m, dtype=np.float64)
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("LVF reference must be finite and positive")
    reference = exposure_setting(reference, "reference_m")
    if (sigma_m is None) == (profile is None):
        raise ValueError("supply exactly one of sigma_m and profile")
    if sigma_m is None:
        sigma = None
    else:
        sigma = np.asarray(sigma_m, dtype=np.float64)
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("spectral sigma must be finite and positive")
        sigma = exposure_setting(sigma, "sigma_m")
    if (
        axis not in ("x", "y")
        or not np.isfinite([gradient, reference_position_m, curvature_per_m]).all()
    ):
        raise ValueError("LVF needs x/y axis and finite spatial parameters")
    if not np.isfinite(peak) or not 0 <= peak <= 1:
        raise ValueError("peak must be finite and in [0, 1]")
    validate_quadrature_order(pixel_order)

    def transmission(
        exposure: int, x: float, y: float, wave: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        coordinate = (x if axis == "x" else y) - reference_position_m
        centre = lvf_centre(
            coordinate,
            reference_m=setting_at_exposure(reference, exposure),
            gradient=gradient,
            curvature_per_m=curvature_per_m,
        )
        if profile is not None:
            return peak * _filter_values(profile(wave, float(centre)), wave, "LVF profile")
        assert sigma is not None
        return gaussian_filter(wave, centre, setting_at_exposure(sigma, exposure), peak)

    return imaging_optics(
        focal_length_m,
        collecting_area_m2,
        psf_sigma_m,
        throughput,
        transmission,
        {"reference_m": reference}
        if sigma is None
        else {"reference_m": reference, "sigma_m": sigma},
        psf=psf,
        spatial=True,
        pixel_order=pixel_order,
    )


def _filter_values(
    values: NDArray[np.float64], wave: NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if values.shape != wave.shape:
        raise ValueError(f"{name} must have the same shape as wavelengths")
    return values


def _edge_fraction(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    cutoff_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    transition_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
):
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    cutoff = np.asarray(cutoff_m, dtype=np.float64)
    if not np.isfinite(cutoff).all() or np.any(cutoff <= 0):
        raise ValueError("cutoff wavelength must be finite and positive")
    width = np.asarray(transition_m, dtype=np.float64)
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("transition width must be finite and positive")
    with np.errstate(over="ignore"):
        return expit(2 * np.log(9) * ((wavelength - cutoff) / width))


def _edge_levels(
    pass_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    stop_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
):
    passed = np.asarray(pass_transmission, dtype=np.float64)
    if not np.isfinite(passed).all() or np.any((passed < 0) | (passed > 1)):
        raise ValueError("pass transmission must be finite and in [0, 1]")
    stopped = np.asarray(stop_transmission, dtype=np.float64)
    if not np.isfinite(stopped).all() or np.any((stopped < 0) | (stopped > 1)):
        raise ValueError("stop transmission must be finite and in [0, 1]")
    if np.any(stopped > passed):
        raise ValueError("stop transmission must not exceed pass transmission")
    return passed, stopped
