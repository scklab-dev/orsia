"""Sample zero-extended finite Fourier fields on explicit physical point lattices.

Dense and optional FINUFFT evaluation use the same padded trigonometric polynomial.
Pixel averaging and Gaussian blur are selected separately from value reconstruction.
Native plans have per-call lifetime and one thread; no backend fallback changes the law.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import PointLattice, same_local_frame

from orsia_instruments.response.spatial import gaussian_pixel_transfer, gaussian_sigma_from_mtf

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type SampledField = (
    NDArray[np.floating | np.integer | np.bool_] | Sequence[Sequence[Sequence[float]]]
)


def sample_fourier_exposures(
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
    """Evaluate optical/pixel spatial response at explicitly displaced exposure samples.

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
            result[..., exposure] = _reconstruction(sampled, okay, negative_policy, scale)
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


def _reconstruction(values, valid, policy, scale):
    if not np.isfinite(values).all():
        raise ValueError("Fourier reconstruction arithmetic exceeds finite support")
    if policy == "reject" and np.any(values[valid] < -64 * np.finfo(float).eps * scale):
        raise ValueError("Fourier reconstruction produced materially negative physical values")
    if policy != "preserve":
        values = np.maximum(values, 0)
    return np.where(valid, values, 0)
