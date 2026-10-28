"""Local Airy cavity transmission and cavity imaging assembly."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..quadrature import validate_quadrature_order
from .psfs import PointSpreadFunction
from .response import (
    ImagingOptics,
    SpectralParameter,
    constant_or_default,
    exposure_setting,
    imaging_optics,
    setting_at_exposure,
    spatial_offset,
    spectral_value,
)


def cavity_transmission(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    spacing_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    *,
    refractive_index: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    internal_angle_rad: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
    reflectivity: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0.8,
    coating_phase_rad: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
    peak: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    order_transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
) -> NDArray[np.float64]:
    """Local Airy intensity law for equal mirrors, with explicit order transmission.

    Reflectivity is mirror intensity reflectivity in [0,1); 4R/(1-R)^2 is the
    Airy coefficient, not finesse. peak<1 is a labelled phenomenological loss,
    not an unequal-mirror solution. A spatial spacing map is local-cavity only:
    no walk-off, coherent lateral coupling or general Fizeau qualification.

    Args:
        wavelength_m: Positive wavelength values in metres.
        spacing_m: Positive local mirror spacing in metres.
        refractive_index: Positive cavity-medium refractive index.
        internal_angle_rad: Internal ray angle in radians, between -pi/2 and pi/2.
        reflectivity: Equal-mirror intensity reflectivity in [0, 1).
        coating_phase_rad: Additional round-trip coating phase in radians.
        peak: Bounded phenomenological peak-intensity multiplier.
        order_transmission: Bounded order-selection transmission supplied explicitly.

    Returns:
        Broadcast local transmitted intensity fraction from the Airy law.

    Raises:
        ValueError: If wavelength, spacing, phase or bounded physical parameters
            are invalid.
    """
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    spacing = np.asarray(spacing_m, dtype=np.float64)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must be finite and positive")
    index = np.asarray(refractive_index, dtype=np.float64)
    if not np.isfinite(index).all() or np.any(index <= 0):
        raise ValueError("refractive index must be finite and positive")
    angle = np.asarray(internal_angle_rad, dtype=np.float64)
    if not np.isfinite(angle).all() or np.any((angle < -np.pi / 2) | (angle > np.pi / 2)):
        raise ValueError(f"internal angle must be finite and in [{-np.pi / 2}, {np.pi / 2}]")
    rho = np.asarray(reflectivity, dtype=np.float64)
    if not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1)):
        raise ValueError("reflectivity must be finite and in [0, 1]")
    if np.any(rho == 1):
        raise ValueError("reflectivity must be <1")
    phase = 4 * np.pi * index * spacing * np.cos(angle) / wavelength + np.asarray(coating_phase_rad)
    if not np.all(np.isfinite(phase)):
        raise ValueError("cavity phase must be finite")
    amplitude = np.asarray(peak, dtype=np.float64)
    if not np.isfinite(amplitude).all() or np.any((amplitude < 0) | (amplitude > 1)):
        raise ValueError("peak must be finite and in [0, 1]")
    order = np.asarray(order_transmission, dtype=np.float64)
    if not np.isfinite(order).all() or np.any((order < 0) | (order > 1)):
        raise ValueError("order selection must be finite and in [0, 1]")
    return amplitude * order / (1 + 4 * rho / (1 - rho) ** 2 * np.sin(phase / 2) ** 2)


def cavity_imager(
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
    spacing_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    order_transmission: SpectralParameter,
    refractive_index: SpectralParameter = 1,
    internal_angle_rad: float = 0,
    reflectivity: SpectralParameter = 0.8,
    coating_phase_rad: SpectralParameter = 0,
    spacing_offset_m: Callable[[float, float], float] | None = None,
    pixel_order: int = 8,
    peak: float = 1,
    throughput: float = 1,
) -> ImagingOptics:
    """Local equal-mirror Airy cavity, with scalar/per-exposure base spacing in metres.

    PSF selection and callback coordinates follow filter_imager.
    Explicit order transmission is a constant or wavelength-only callable, such
    as a function evaluating tabulated_transmission against a fixed table. The
    callable's result may broadcast to the requested wavelength vector (legacy contract).
    Index (>0), reflectivity ([0,1)) and coating phase (finite radians) also accept
    wavelength-only functions, returning the wavelength vector's exact shape. The optional
    spacing_offset_m(x_m,y_m) adds a finite scalar detector-plane thickness to
    each base spacing; the total must be positive. With this law, pixel_order
    controls the joint PSF/cavity quadrature; without it Gaussian integration is analytic.
    Callbacks are borrowed, pure, stable, nonmutating and pointwise in wavelength.
    No automatic order blocking, walk-off or coherent lateral coupling is
    inferred; a wedge is only a local-cavity approximation. Parameters remain
    constant in time during each exposure.

    Args:
        focal_length_m: Imager focal length in metres.
        collecting_area_m2: Entrance collecting area in square metres.
        psf_sigma_m: Gaussian width in metres, scalar or (x, y), for analytic pixels.
        psf: Alternative borrowed spatial intensity-density callback.
        spacing_m: Positive cavity spacing in metres, scalar or per exposure.
        order_transmission: Bounded constant or wavelength-only order selection.
        refractive_index: Positive constant or wavelength-only cavity index.
        internal_angle_rad: Internal ray angle in radians.
        reflectivity: Equal-mirror intensity reflectivity, constant or wavelength law.
        coating_phase_rad: Residual phase in radians, constant or wavelength law.
        spacing_offset_m: Optional detector-position-dependent thickness in metres.
        pixel_order: Quadrature order per pixel axis for joint spatial laws.
        peak: Bounded phenomenological peak transmission.
        throughput: Additional bounded physical throughput.

    Returns:
        Imaging optics with local cavity response and base exposure spacing setting.

    Raises:
        ValueError: If cavity, spatial law or quadrature support is invalid.
    """
    spacing = np.asarray(spacing_m, dtype=np.float64)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("cavity spacing must be finite and positive")
    spacing = exposure_setting(spacing, "spacing_m")
    validate_quadrature_order(pixel_order)
    cavity_transmission(
        1.0,
        spacing,
        refractive_index=constant_or_default(refractive_index, 1),
        internal_angle_rad=internal_angle_rad,
        reflectivity=constant_or_default(reflectivity, 0.8),
        coating_phase_rad=constant_or_default(coating_phase_rad, 0),
        peak=peak,
        order_transmission=constant_or_default(order_transmission, 1),
    )

    def law(exposure: int, x: float, y: float, wave: NDArray[np.float64]) -> NDArray[np.float64]:
        return cavity_transmission(
            wave,
            setting_at_exposure(spacing, exposure)
            + spatial_offset(spacing_offset_m, x, y, "spacing offset"),
            refractive_index=spectral_value(refractive_index, wave, "refractive index"),
            internal_angle_rad=internal_angle_rad,
            reflectivity=spectral_value(reflectivity, wave, "reflectivity"),
            coating_phase_rad=spectral_value(coating_phase_rad, wave, "coating phase"),
            peak=peak,
            order_transmission=spectral_value(order_transmission, wave, "order", broadcast=True),
        )

    return imaging_optics(
        focal_length_m,
        collecting_area_m2,
        psf_sigma_m,
        throughput,
        law,
        {"spacing_m": spacing},
        psf=psf,
        spatial=spacing_offset_m is not None,
        pixel_order=pixel_order,
    )
