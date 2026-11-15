"""Apply optical/pixel responses to sampled fields and form displaced detector images.

Filtering uses replicated boundaries; detector image formation uses zero-extended
finite Fourier fields on explicit physical point lattices. Their support and
negative-value policies remain distinct. Dense and optional FINUFFT evaluation
share the same padded polynomial, with one per-call, single-thread native plan.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import PointLattice, same_local_frame
from scipy.ndimage import minimum_filter

if TYPE_CHECKING:
    from finufft import Plan


type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type ArrayInput = NDArray[np.floating | np.integer | np.bool_] | Sequence[float]

type SampledField = (
    NDArray[np.floating | np.integer | np.bool_] | Sequence[Sequence[Sequence[float]]]
)


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


def form_detector_images(
    values: SampledField,
    source_support: BoolArray,
    source_grid: PointLattice,
    detector_grid: PointLattice,
    displacements_m: Sequence[FloatArray],
    *,
    detector_pitch_xy_m: tuple[float, float],
    pixel_size_xy_m: tuple[float, float] | None,
    negative_policy: Literal["preserve", "clip", "reject"],
    optical_mtf: float = 0.30,
    tail_tolerance: float = 1e-6,
    fourier_backend: Literal["dense", "nufft"] = "dense",
) -> tuple[list[FloatArray], list[BoolArray]]:
    """Form detector images with Gaussian optical blur at supplied exposure offsets.

    Optional rectangular pixel averaging shares the supplied spatial response.
    Values retain their input units; detector noise and temporal integration
    remain separate operations.

    Args:
        values: Finite (source_y,source_x,channel) fields in consistent units, such
            as radiance, electron means or signed coefficient fields. Preserve units.
        source_support: Boolean mask on the source point axes.
        source_grid: Regular increasing source points with at least two per axis.
        detector_grid: Increasing output point axes on the same declared frame.
            Output axes may be nonuniform or singleton; no pixel area is inferred.
        displacements_m: One (detector_y,exposure,2) array per channel, with physical
            (x,y) metre components. The caller owns timing and any full-exposure
            coverage or temporal averaging interpretation of these samples.
        detector_pitch_xy_m: Positive reference detector pitches for the specified
            Nyquist MTF, in (x,y) metres, independent of output sample spacing.
        pixel_size_xy_m: Explicit positive active pixel widths (x,y), metres, for
            area-normalized rectangular averaging. None omits pixel integration.
        negative_policy: Preserve signed polynomial values, explicitly clip all
            negative outputs, or reject materially negative physical expectations.
            Reject requires nonnegative supported inputs and clips only outputs
            within 64*float64_epsilon*max(abs(supported channel input)) of zero.
            Clipping and rejection have no unconditional linear/transpose contract.
        optical_mtf: Gaussian optical MTF at the reference-pitch Nyquist, in (0,1].
        tail_tolerance: Gaussian support-tail fraction in (0,1). The returned mask
            follows this finite neighborhood rule, not a global polynomial-error bound.
        fourier_backend: Dense Fourier sum or optional FINUFFT type 2 with positive
            exponent, FFT mode order, tolerance 1e-12 and one native thread.

    Returns:
        Lists of fresh values and Boolean masks, each (detector_y,detector_x,exposure),
        in channel order. Unsupported neighborhoods are zero. Zero extension remains
        explicit; there is no edge renormalization, inferred alignment or second blur.

    Raises:
        ValueError: Frames, arrays, regular source support, physical controls,
            reconstruction policy or finite arithmetic are invalid.
        ImportError: FINUFFT is explicitly selected but unavailable.
    """
    values = np.asarray(values, dtype=np.float64)
    sampling = _prepare_sampling(
        values,
        source_support,
        source_grid,
        detector_grid,
        displacements_m,
        detector_pitch_xy_m=detector_pitch_xy_m,
        pixel_size_xy_m=pixel_size_xy_m,
        negative_policy=negative_policy,
        optical_mtf=optical_mtf,
        tail_tolerance=tail_tolerance,
        fourier_backend=fourier_backend,
    )
    return _sample_field(values, sampling, negative_policy)


def form_detector_images_with_tangents(
    values: FloatArray,
    tangents: FloatArray,
    source_support: BoolArray,
    source_grid: PointLattice,
    detector_grid: PointLattice,
    displacements_m: Sequence[FloatArray],
    *,
    detector_pitch_xy_m: tuple[float, float],
    pixel_size_xy_m: tuple[float, float] | None,
    optical_mtf: float = 0.30,
    tail_tolerance: float = 1e-6,
    fourier_backend: Literal["dense", "nufft"] = "dense",
) -> tuple[list[FloatArray], list[FloatArray], list[BoolArray]]:
    """Form detector images and propagate signed tangents through a fixed spatial response.

    Prepare the Gaussian/pixel response and optional native plan once per call.
    Evaluate the mean and each direction sequentially, preserving signed values.

    Args:
        values: Finite (source_y,source_x,channel) sampled fields.
        tangents: Same axes plus a final direction axis in caller parameter units.
        source_support: Boolean source-point availability, fixed for differentiation.
        source_grid: Regular source lattice in metres.
        detector_grid: Explicit target lattice on the same frame.
        displacements_m: Fixed (detector_y,exposure,2) offsets per channel in (x,y) m.
        detector_pitch_xy_m: Positive reference pitches for Nyquist MTF in metres.
        pixel_size_xy_m: Active footprint widths in metres, or None.
        optical_mtf: Fixed Gaussian MTF at reference Nyquist, in (0,1].
        tail_tolerance: Fixed finite-neighborhood support-tail fraction in (0,1).
        fourier_backend: Dense evaluation or explicitly selected one-thread FINUFFT.

    Returns:
        Channel lists of values, tangents and masks. Values/masks use (y,x,exposure);
        tangents append direction. All outputs are fresh. Signed values are preserved,
        including polynomial ringing. Clipping and changing response/support require
        separate derivative contracts and are deliberately excluded from this operation.

    Raises:
        ValueError: Tangent axes, physical grids, controls or finite values fail.
        ImportError: The explicitly selected native backend is unavailable.
    """
    if tangents.ndim != 4 or tangents.shape[:-1] != values.shape or not np.isfinite(tangents).all():
        raise ValueError("finite source/channel/direction tangent axes required")

    values = np.asarray(values, dtype=np.float64)
    sampling = _prepare_sampling(
        values,
        source_support,
        source_grid,
        detector_grid,
        displacements_m,
        detector_pitch_xy_m=detector_pitch_xy_m,
        pixel_size_xy_m=pixel_size_xy_m,
        negative_policy="preserve",
        optical_mtf=optical_mtf,
        tail_tolerance=tail_tolerance,
        fourier_backend=fourier_backend,
    )
    result, masks = _sample_field(values, sampling, "preserve")
    derivatives = [np.empty((*array.shape, tangents.shape[-1])) for array in result]
    for direction in range(tangents.shape[-1]):
        sampled, _ = _sample_field(tangents[..., direction], sampling, "preserve")
        for output, array in zip(derivatives, sampled, strict=True):
            output[..., direction] = array
    return result, derivatives, masks


@dataclass(eq=False)
class _FourierSampling:
    """Fixed spatial response and query geometry for one image-formation call.

    Attributes:
        source_grid: Borrowed regular source lattice in metres.
        detector_grid: Borrowed output lattice in the same frame.
        source_support: Borrowed Boolean source-point mask, stable during the call.
        spacing: Owned source spacings in (y,x) metres.
        radii: Owned finite support radii in source (y,x) samples.
        padding: Owned int64 zero-padding widths in source (y,x) samples.
        padded_shape: Owned int64 padded (y,x) lengths.
        fr: Owned row frequencies in cycles per source sample.
        fc: Owned column frequencies in cycles per source sample.
        transfer: Owned dimensionless (padded_y,padded_x) Fourier response.
        integral: Owned integer summed-area table of missing source points.
        displacements: Borrowed or converted (y,exposure,2) metre offsets per channel.
        plan: Optional mutable FINUFFT plan, used sequentially within this call only.
    """

    source_grid: PointLattice
    detector_grid: PointLattice
    source_support: BoolArray
    spacing: FloatArray
    radii: FloatArray
    padding: NDArray[np.int64]
    padded_shape: NDArray[np.int64]
    fr: FloatArray
    fc: FloatArray
    transfer: FloatArray
    integral: NDArray[np.int64]
    displacements: list[FloatArray]
    plan: Plan | None


def _prepare_sampling(
    values,
    source_support,
    source_grid,
    detector_grid,
    displacements_m,
    *,
    detector_pitch_xy_m,
    pixel_size_xy_m,
    negative_policy,
    optical_mtf,
    tail_tolerance,
    fourier_backend,
) -> _FourierSampling:
    """Validate fixed inputs and prepare one local response and optional native plan.

    Returns:
        Fixed response and geometry with one call-local native plan, if selected.

    Raises:
        ValueError: Source arrays, grids, displacement axes or controls are invalid.
        ImportError: The selected FINUFFT backend is unavailable.
    """
    if not same_local_frame(source_grid.frame, detector_grid.frame):
        raise ValueError("Fourier sampling requires an explicitly shared local frame")
    shape, output_shape = source_grid.shape, detector_grid.shape
    if (
        values.ndim != 3
        or values.shape[:2] != shape
        or not np.isfinite(values).all()
        or source_support.shape != shape
        or source_support.dtype != np.bool_
        or len(displacements_m) != values.shape[2]
        or negative_policy not in ("preserve", "clip", "reject")
        or fourier_backend not in ("dense", "nufft")
        or not np.isfinite(tail_tolerance)
        or not 0 < tail_tolerance < 1
    ):
        raise ValueError("valid source/support arrays, backend and explicit policies required")
    if negative_policy == "reject" and np.any(values[source_support] < 0):
        raise ValueError("physical Fourier expectation requires nonnegative supported inputs")
    spacing = np.array([_regular_spacing(source_grid.y_m), _regular_spacing(source_grid.x_m)])
    pitch = np.asarray(detector_pitch_xy_m, dtype=np.float64)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("detector reference pitches must be positive finite (x,y) metres")
    pitch_ratio = pitch[::-1] / spacing
    width = None
    if pixel_size_xy_m is not None:
        sizes = np.asarray(pixel_size_xy_m, dtype=np.float64)
        if sizes.shape != (2,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("active pixel sizes must be positive finite (x,y) metres")
        width = sizes[::-1] / spacing
    sigma = gaussian_sigma_from_mtf(optical_mtf)
    radii = np.sqrt(-2 * np.log(tail_tolerance)) * sigma * pitch_ratio
    if width is not None:
        radii += width / 2
    margins = np.ceil(radii) + 2
    if not np.isfinite(margins).all() or np.any(2 * margins >= shape):
        raise ValueError("source grid too small for optical-plus-pixel support")
    padding = margins.astype(np.int64)
    padded_shape = np.array(shape) + 2 * padding
    fr, fc = np.fft.fftfreq(padded_shape[0]), np.fft.fftfreq(padded_shape[1])
    transfer = gaussian_pixel_transfer(
        fr, fc, optical_mtf=optical_mtf, pitch_ratio_yx=pitch_ratio, pixel_width_yx_samples=width
    )
    integral = np.pad(np.cumsum(np.cumsum(~source_support, axis=0), axis=1), ((1, 0), (1, 0)))
    displacements = []
    for displacement in displacements_m:
        displacement = np.asarray(displacement, dtype=np.float64)
        if (
            displacement.ndim != 3
            or displacement.shape[0] != output_shape[0]
            or displacement.shape[2] != 2
            or not np.isfinite(displacement).all()
        ):
            raise ValueError("displacements require finite (detector_y,exposure,x_y) axes")
        displacements.append(displacement)
    plan = None
    if fourier_backend == "nufft":
        from finufft import Plan

        plan = Plan(
            2,
            tuple(int(n) for n in padded_shape),
            eps=1e-12,
            isign=1,
            nthreads=1,
            modeord=1,
            dtype="complex128",
        )
    return _FourierSampling(
        source_grid,
        detector_grid,
        source_support,
        spacing,
        radii,
        padding,
        padded_shape,
        fr,
        fc,
        transfer,
        integral,
        displacements,
        plan,
    )


def _sample_field(values: FloatArray, sampling: _FourierSampling, negative_policy):
    """Apply the prepared response to one finite field, with separate support checks.

    Returns:
        Fresh channel-ordered image and support-mask lists.

    Raises:
        ValueError: Query coordinates or reconstructed values violate finite support
            or the selected negative-value policy.
    """
    values = np.asarray(values, dtype=np.float64)
    source_grid, detector_grid = sampling.source_grid, sampling.detector_grid
    shape, output_shape = source_grid.shape, detector_grid.shape
    source_support, spacing = sampling.source_support, sampling.spacing
    radii, padding, padded_shape = sampling.radii, sampling.padding, sampling.padded_shape
    fr, fc, transfer = sampling.fr, sampling.fc, sampling.transfer
    integral, displacements, plan = sampling.integral, sampling.displacements, sampling.plan
    results, masks = [], []
    for channel, displacement in enumerate(displacements):
        field = np.where(source_support, values[..., channel], 0)
        scale = float(np.max(np.abs(field), initial=0))
        padded = np.pad(field, [(int(p), int(p)) for p in padding])
        spectrum = np.fft.fft2(padded) * transfer
        result = np.zeros((*output_shape, displacement.shape[1]))
        valid = np.zeros(result.shape, dtype=bool)
        for exposure in range(displacement.shape[1]):
            sr = (detector_grid.y_m + displacement[:, exposure, 1] - source_grid.y_m[0]) / spacing[
                0
            ]
            sc = (
                detector_grid.x_m[None, :] + displacement[:, exposure, 0, None] - source_grid.x_m[0]
            ) / spacing[1]
            if not np.isfinite(sr).all() or not np.isfinite(sc).all():
                raise ValueError("Fourier query coordinates exceed finite support")
            if plan is None:
                sampled = np.empty(output_shape)
                for row in range(output_shape[0]):
                    row_phase = np.exp(2j * np.pi * fr * (sr[row] + padding[0]))
                    column_phase = np.exp(
                        2j * np.pi * fc[:, None] * (sc[row, None, :] + padding[1])
                    )
                    sampled[row] = (row_phase @ spectrum @ column_phase).real / np.prod(
                        padded_shape
                    )
            else:
                row_angles = np.ascontiguousarray(
                    np.broadcast_to(
                        2 * np.pi * (sr[:, None] + padding[0]) / padded_shape[0], output_shape
                    ).ravel()
                )
                column_angles = np.ascontiguousarray(
                    (2 * np.pi * (sc + padding[1]) / padded_shape[1]).ravel()
                )
                plan.setpts(row_angles, column_angles)
                sampled = plan.execute(np.ascontiguousarray(spectrum)).real.reshape(
                    output_shape
                ) / np.prod(padded_shape)
            okay = _footprint_support(sr, sc, radii, shape, integral)
            result[..., exposure] = _apply_output_policy(sampled, okay, negative_policy, scale)
            valid[..., exposure] = okay
        results.append(result)
        masks.append(valid)
    return results, masks


def _regular_spacing(axis):
    if axis.size < 2:
        raise ValueError("Fourier source axes require at least two regularly spaced points")
    spacing = axis[1] - axis[0]
    if not np.isfinite(spacing) or not np.allclose(np.diff(axis), spacing, rtol=1e-12, atol=1e-12):
        raise ValueError("Fourier source axes must be regular with finite spacing")
    return float(spacing)


def _footprint_support(sr, sc, radii, shape, integral):
    okay = (
        ((sr - radii[0] >= -0.5) & (sr + radii[0] <= shape[0] - 0.5))[:, None]
        & (sc - radii[1] >= -0.5)
        & (sc + radii[1] <= shape[1] - 0.5)
    )
    rlo = np.clip(np.floor(sr - radii[0]), 0, shape[0] - 1).astype(np.int64)[:, None]
    rhi = np.clip(np.ceil(sr + radii[0]), 0, shape[0] - 1).astype(np.int64)[:, None]
    clo = np.clip(np.floor(sc - radii[1]), 0, shape[1] - 1).astype(np.int64)
    chi = np.clip(np.ceil(sc + radii[1]), 0, shape[1] - 1).astype(np.int64)
    okay &= (
        integral[rhi + 1, chi + 1]
        - integral[rlo, chi + 1]
        - integral[rhi + 1, clo]
        + integral[rlo, clo]
    ) == 0
    return okay


def _apply_output_policy(values, valid, policy, scale):
    if not np.isfinite(values).all():
        raise ValueError("Fourier reconstruction arithmetic exceeds finite support")
    if policy == "reject" and np.any(values[valid] < -64 * np.finfo(float).eps * scale):
        raise ValueError("Fourier reconstruction produced materially negative physical values")
    if policy != "preserve":
        values = np.maximum(values, 0)
    return np.where(valid, values, 0)
