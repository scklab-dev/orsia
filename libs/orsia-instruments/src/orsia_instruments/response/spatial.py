"""Apply explicit replicated-boundary optical/pixel filtering to sampled values."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type ArrayInput = NDArray[np.floating | np.integer | np.bool_] | Sequence[float]


def gaussian_sigma_from_mtf(optical_mtf: float) -> float:
    """Convert Gaussian MTF at sampling Nyquist to standard deviation in samples.

    Args:
        optical_mtf: Finite dimensionless transfer at half a cycle per sample, in (0,1].

    Returns:
        Standard deviation sqrt(-2*log(MTF))/pi in reference-sample units.

    Raises:
        ValueError: The requested MTF is outside its physical domain.
    """
    if not np.isfinite(optical_mtf) or not 0 < optical_mtf <= 1:
        raise ValueError("optical MTF must be finite in (0,1]")
    return float(np.sqrt(-2 * np.log(optical_mtf)) / np.pi)


def gaussian_pixel_transfer(
    frequency_y: FloatArray,
    frequency_x: FloatArray,
    *,
    optical_mtf: float,
    pitch_ratio_yx: tuple[float, float] | FloatArray,
    pixel_width_yx_samples: tuple[float, float] | FloatArray | None,
) -> FloatArray:
    """Evaluate the shared Gaussian/pixel transfer on a source FFT frequency grid.

    Args:
        frequency_y: Finite (y_frequency,) frequencies in cycles per source sample.
        frequency_x: Finite (x_frequency,) frequencies in cycles per source sample.
        optical_mtf: Gaussian MTF at the detector reference-pitch Nyquist.
        pitch_ratio_yx: Positive detector reference pitch divided by source spacing,
            in (y,x) order. These are dimensionless coordinate conversions.
        pixel_width_yx_samples: Optional positive active pixel widths in source-sample
            units. None selects optical blur only. Pixel transfer is area-normalized.

    Returns:
        Fresh real (y_frequency,x_frequency) transfer. Sinc lobes can be signed;
        physical value reconstruction and its clipping policy remain separate.

    Raises:
        ValueError: Frequency axes, pitch/width domains or finite transfer fail.
    """
    sigma = gaussian_sigma_from_mtf(optical_mtf)
    pitch = np.asarray(pitch_ratio_yx, dtype=np.float64)
    if (
        frequency_y.ndim != 1
        or frequency_x.ndim != 1
        or not np.isfinite(frequency_y).all()
        or not np.isfinite(frequency_x).all()
        or pitch.shape != (2,)
        or not np.isfinite(pitch).all()
        or np.any(pitch <= 0)
    ):
        raise ValueError("finite frequency axes and positive (y,x) pitch ratios required")
    if sigma == 0:
        transfer = np.ones((frequency_y.size, frequency_x.size))
    else:
        with np.errstate(over="ignore"):
            transfer = np.exp(
                -2
                * np.pi**2
                * sigma**2
                * ((frequency_y[:, None] * pitch[0]) ** 2 + (frequency_x[None, :] * pitch[1]) ** 2)
            )
    if pixel_width_yx_samples is not None:
        width = np.asarray(pixel_width_yx_samples, dtype=np.float64)
        if width.shape != (2,) or not np.isfinite(width).all() or np.any(width <= 0):
            raise ValueError("pixel widths must be positive finite (y,x) source-sample extents")
        transfer *= np.sinc(frequency_y[:, None] * width[0]) * np.sinc(
            frequency_x[None, :] * width[1]
        )
    if not np.isfinite(transfer).all():
        raise ValueError("spatial transfer arithmetic exceeds finite support")
    return transfer


def apply_spatial_response(
    values: ArrayInput,
    optical_mtf: float = 0.30,
    pixel_integration: bool = True,
    *,
    upper_bound: bool = False,
) -> FloatArray:
    """Filter finite (row,column,...) fields with replicated-padding source MTF.

    Returns owned arrays with original units and shape. Upper-bound mode retains
    only positive impulse-response weights, without renormalization. ValueError
    reports invalid MTF or nonfinite fields. Inputs remain unchanged.

    Args:
        values: Borrowed finite (row,column,...) array in any consistent unit.
        optical_mtf: Gaussian optical MTF at Nyquist in (0,1].
        pixel_integration: Include square-pixel sinc response.
        upper_bound: Retain only positive impulse weights without renormalization.

    Returns:
        Owned filtered array with original shape and units, using replicated padding.

    Raises:
        ValueError: Field dimensions/finiteness or MTF domain fails.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim < 2 or not np.isfinite(values).all() or not 0 < optical_mtf <= 1:
        raise ValueError("finite fields and MTF in (0,1] required")
    sigma = gaussian_sigma_from_mtf(optical_mtf)
    padding = max(16, int(np.ceil(8 * np.sqrt(sigma**2 + pixel_integration / 12))))
    padded = np.pad(values, [(padding, padding)] * 2 + [(0, 0)] * (values.ndim - 2), mode="edge")
    transfer = gaussian_pixel_transfer(
        np.fft.fftfreq(padded.shape[0]),
        np.fft.fftfreq(padded.shape[1]),
        optical_mtf=optical_mtf,
        pitch_ratio_yx=(1.0, 1.0),
        pixel_width_yx_samples=(1.0, 1.0) if pixel_integration else None,
    )
    if upper_bound:
        transfer = np.fft.fft2(np.maximum(np.fft.ifft2(transfer).real, 0))
    transfer = transfer.reshape(transfer.shape + (1,) * (values.ndim - 2))
    result = np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * transfer, axes=(0, 1)).real
    return result[padding:-padding, padding:-padding].copy()


def spatial_response_support(
    mask: ArrayInput,
    optical_mtf: float = 0.30,
    pixel_integration: bool = True,
    tolerance: float = 1e-6,
) -> BoolArray:
    """Return source square-neighbourhood support; outside-grid cells are ignored.

    Mask is Boolean (row,column); MTF/tolerance are dimensionless positive values.
    Inputs remain unchanged. Invalid domains raise ValueError.

    Args:
        mask: Borrowed Boolean (row,column) source support.
        optical_mtf: Gaussian optical MTF at Nyquist in (0,1].
        pixel_integration: Include the half-pixel support extension.
        tolerance: Gaussian support-tail fraction in (0,1).

    Returns:
        Owned Boolean full-neighbourhood support; outside-grid cells are ignored.

    Raises:
        ValueError: Mask dtype/shape, MTF or tolerance fails.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2 or not 0 < tolerance < 1 or not 0 < optical_mtf <= 1:
        raise ValueError("Boolean mask and valid MTF/tolerance required")
    sigma = gaussian_sigma_from_mtf(optical_mtf)
    radius = int(np.ceil(sigma * np.sqrt(-2 * np.log(tolerance)) + 0.5 * pixel_integration))
    return minimum_filter(mask, size=2 * radius + 1, mode="constant", cval=True)
