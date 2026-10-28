"""Nonnegative two-beam port intensities and interferometric imaging assembly."""

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


def two_beam_ports(
    wavelength_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    opd_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    *,
    phase_rad: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 0,
    visibility: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
    transmission: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Balanced nonnegative physical ports, H+ + H- = transmission, before detection.

    OPD is optical path difference (not mirror travel); phase is residual radians.
    Arrays broadcast. Scanning/static OPD maps are supplied by the acquisition
    caller. A port difference is not a physical photon-counting input.

    Args:
        wavelength_m: Positive wavelength values in metres.
        opd_m: Optical path difference in metres, broadcastable with wavelength.
        phase_rad: Residual phase in radians added to the OPD phase.
        visibility: Fringe visibility, bounded by zero and one.
        transmission: Total physical port transmission, bounded by zero and one.

    Returns:
        Plus and minus physical port fractions, each broadcast over the inputs;
        their sum is the supplied transmission.

    Raises:
        ValueError: If wavelength, phase, visibility or transmission is invalid.
    """
    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive")
    phase = 2 * np.pi * np.asarray(opd_m) / wavelength + np.asarray(phase_rad)
    if not np.all(np.isfinite(phase)):
        raise ValueError("OPD and phase must be finite")
    visibility = np.asarray(visibility, dtype=np.float64)
    if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
        raise ValueError("visibility must be finite and in [0, 1]")
    contrast = visibility * np.cos(phase)
    total = np.asarray(transmission, dtype=np.float64)
    if not np.isfinite(total).all() or np.any((total < 0) | (total > 1)):
        raise ValueError("transmission must be finite and in [0, 1]")
    return total * (1 + contrast) / 2, total * (1 - contrast) / 2


def interferometric_imager(
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
    opd_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    port: str,
    phase_rad: SpectralParameter = 0,
    visibility: SpectralParameter = 1,
    transmission: SpectralParameter = 1,
    opd_offset_m: Callable[[float, float], float] | None = None,
    pixel_order: int = 8,
    throughput: float = 1,
) -> ImagingOptics:
    """Two-beam interferometric imager for one physical '+' or '-' output port.

    Optical path difference is scalar or one value per exposure; PSF selection
    and callback coordinates follow filter_imager. Phase (rad), visibility and
    transmission can be scalars or wavelength-only functions returning the
    wavelength vector's exact shape.
    Optional opd_offset_m(x_m,y_m) adds a finite scalar detector-plane OPD to
    each exposure's base OPD. Its joint integral with the PSF uses pixel_order;
    refine for local fringes. Without a spatial law, Gaussian pixel integration is analytic.
    Callbacks are borrowed, pure, stable, nonmutating and pointwise in wavelength.
    This factory models two-beam interference, not arbitrary
    interferometer architectures.
    Simulate each port separately before differencing measured outputs. OPD (m)
    is not mirror travel; the exposure axis remains OPD samples, not reconstructed
    wavelength channels. Constant OPD during each exposure is assumed. Separate
    port calls do not model shared electronic or scene fluctuations.

    Args:
        focal_length_m: Imager focal length in metres.
        collecting_area_m2: Entrance collecting area in square metres.
        psf_sigma_m: Gaussian width in metres, scalar or (x, y), for analytic pixels.
        psf: Alternative borrowed spatial intensity-density callback.
        opd_m: Base optical path difference in metres, scalar or per exposure.
        port: Physical output port, ``"+"`` or ``"-"``.
        phase_rad: Residual phase in radians, constant or wavelength law.
        visibility: Bounded fringe visibility, constant or wavelength law.
        transmission: Bounded port-pair transmission, constant or wavelength law.
        opd_offset_m: Optional detector-position-dependent OPD offset in metres.
        pixel_order: Quadrature order per pixel axis for joint spatial laws.
        throughput: Additional bounded physical throughput applied to both ports.

    Returns:
        Imaging optics for the selected physical port, with the exposure OPD setting.

    Raises:
        ValueError: If port, spatial law, optical parameters or quadrature order
            have invalid support.
    """
    opd = exposure_setting(opd_m, "opd_m")
    if port not in ("+", "-"):
        raise ValueError("two-beam port must be '+' or '-'")
    validate_quadrature_order(pixel_order)
    two_beam_ports(
        1.0,
        opd,
        phase_rad=constant_or_default(phase_rad, 0),
        visibility=constant_or_default(visibility, 1),
        transmission=constant_or_default(transmission, 1),
    )

    def law(exposure: int, x: float, y: float, wave: NDArray[np.float64]) -> NDArray[np.float64]:
        ports = two_beam_ports(
            wave,
            setting_at_exposure(opd, exposure) + spatial_offset(opd_offset_m, x, y, "OPD offset"),
            phase_rad=spectral_value(phase_rad, wave, "phase"),
            visibility=spectral_value(visibility, wave, "visibility"),
            transmission=spectral_value(transmission, wave, "transmission"),
        )
        return ports[0 if port == "+" else 1]

    return imaging_optics(
        focal_length_m,
        collecting_area_m2,
        psf_sigma_m,
        throughput,
        law,
        {"opd_m": opd},
        port=port,
        psf=psf,
        spatial=opd_offset_m is not None,
        pixel_order=pixel_order,
    )
