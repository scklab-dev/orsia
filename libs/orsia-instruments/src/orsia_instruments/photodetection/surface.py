"""Sample fixed-view surface radiance on explicit points and native read schedules."""

from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import LocalCartesianFrame, PointLattice, same_local_frame
from orsia_geometry.frames import require_utc_epoch
from orsia_radiative import SpectralGrid

from orsia_instruments.acquisition.schedule import select_read_schedule
from orsia_instruments.photodetection.frames import (
    ResponseCoefficientTangent,
    SurfaceRadiance,
    SurfaceRadianceTangent,
)
from orsia_instruments.photodetection.photoelectrons import integrate_photoelectron_tangents
from orsia_instruments.response.spatial import form_detector_images
from orsia_instruments.response.spectral import SpectralElectronResponse, spectral_electrons

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]


def sample_surface_electrons(
    radiance: SurfaceRadiance,
    response: SpectralElectronResponse,
    points: PointLattice,
    view_local: FloatArray,
    *,
    frame: LocalCartesianFrame,
    epoch: datetime,
    support: BoolArray,
    read_block_size: int = 16,
    spectral_block_size: int = 64,
) -> tuple[FloatArray, BoolArray]:
    """Evaluate photoelectrons under a fixed exposure-midpoint viewing approximation.

    Args:
        radiance: Pure callback (xy_m,view_local,times_s,grid) returning finite
            nonnegative (read,spectral) radiance per grid density coordinate.
        response: Physical read schedule and coefficients with full-domain quadrature.
        points: Query lattice with (y,x) order in the supplied frame.
        view_local: Upward unit (read,3) surface-to-sensor directions in frame axes.
        frame: Explicit callback frame, equal to the query lattice frame.
        epoch: UTC origin of response.reads times, also used by the callback.
        support: Boolean (y,x) coverage certificate. Unsupported points are not queried.
        read_block_size: Positive maximum reads per callback block.
        spectral_block_size: Positive maximum spectral samples per callback block.

    Returns:
        Fresh (y,x,read) electrons and Boolean support; unsupported values are zero.
        Scratch is bounded by one point/read/spectral block, excluding full outputs.
        Callbacks must be slice-invariant. Geometry and coverage certification remain
        caller-owned. This approximation omits cone and within-exposure integration.

    Raises:
        ValueError: Context, support, physical callback values or block controls fail.
    """
    values, _, valid = _sample_surface(
        radiance,
        None,
        response,
        points,
        view_local,
        frame,
        epoch,
        support,
        None,
        None,
        read_block_size,
        spectral_block_size,
    )
    return values, valid


def sample_surface_electron_tangents(
    radiance_tangent: SurfaceRadianceTangent,
    response: SpectralElectronResponse,
    points: PointLattice,
    view_local: FloatArray,
    directions: int,
    *,
    frame: LocalCartesianFrame,
    epoch: datetime,
    support: BoolArray,
    response_tangent: ResponseCoefficientTangent | None = None,
    read_block_size: int = 16,
    spectral_block_size: int = 64,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Sample physical radiance and signed total directions through one photon response.

    Args:
        radiance_tangent: Callback (xy_m,view_local,times_s,grid) returning physical
            (read,spectral) radiance and signed (read,spectral,direction) derivatives.
            Query-motion and time perturbations must be included in total derivatives.
        response: Native reads and precomputed full-domain photon coefficients.
        points: Query lattice in (y,x) order on the declared frame.
        view_local: Upward unit (read,3) directions at exposure midpoints.
        directions: Nonnegative final direction count, fixed across all blocks.
        frame: Callback frame, equal to the query lattice frame.
        epoch: UTC origin shared by read times and callback.
        support: Boolean (y,x) coverage certificate; unsupported points are not queried.
        response_tangent: Optional (read_slice,spectral_slice) callback returning
            signed (read,spectral,direction) coefficient derivatives.
        read_block_size: Positive maximum reads in scratch.
        spectral_block_size: Positive maximum spectral samples in scratch.

    Returns:
        Fresh (y,x,read) means, (y,x,read,direction) derivatives and Boolean support.
        Computes sum(K*dL+dK*L), with zero unsupported outputs. Callbacks must be
        slice-invariant; all input buffers remain borrowed and unchanged.

    Raises:
        ValueError: Context, controls, physical values or signed derivative axes fail.
    """
    if directions < 0:
        raise ValueError("nonnegative direction count required")
    values, derivative, valid = _sample_surface(
        None,
        radiance_tangent,
        response,
        points,
        view_local,
        frame,
        epoch,
        support,
        directions,
        response_tangent,
        read_block_size,
        spectral_block_size,
    )
    assert derivative is not None
    return values, derivative, valid


def sample_fourier_surface_electrons(
    radiance: SurfaceRadiance,
    response: SpectralElectronResponse,
    view_local: FloatArray,
    source_points: PointLattice,
    source_support: BoolArray,
    detector_points: PointLattice,
    displacement_xy_m: FloatArray,
    *,
    frame: LocalCartesianFrame,
    epoch: datetime,
    detector_pitch_xy_m: tuple[float, float],
    pixel_size_xy_m: tuple[float, float] | None,
    optical_mtf: float,
    negative_policy: Literal["preserve", "clip", "reject"],
    tail_tolerance: float = 1e-6,
    backend: Literal["dense", "nufft"] = "dense",
    read_block_size: int = 16,
    spectral_block_size: int = 64,
) -> tuple[FloatArray, BoolArray]:
    """Apply Fourier spatial response once to individually integrated physical read fields.

    Args:
        radiance: Physical callback on supported source points and response coordinates.
        response: Per-read photon coefficients, with complete precomputed quadrature.
        view_local: Fixed upward unit (read,3) midpoint directions in frame axes.
        frame: Callback frame, shared by source and detector points.
        epoch: UTC origin of response.reads times, shared by the callback.
        source_points: Increasing regular source lattice in (y,x) storage order.
        source_support: Boolean source-point coverage; unsupported samples are never queried.
        detector_points: Output point lattice on the same declared local frame.
        displacement_xy_m: Finite (detector_y,read,2) physical (x,y) offsets in metres.
        detector_pitch_xy_m: Explicit reference pitches for optical Nyquist MTF, metres.
        pixel_size_xy_m: Active footprint widths in metres, or None to omit pixel averaging.
        optical_mtf: Gaussian optical MTF at the declared detector Nyquist pitch.
        negative_policy: Explicit signed preservation, clipping or physical-expectation rejection.
        tail_tolerance: Selected finite Gaussian neighborhood tail fraction.
        backend: Dense polynomial evaluation or explicitly selected FINUFFT, without fallback.
        read_block_size: Positive maximum number of read fields in scratch.
        spectral_block_size: Positive spectral working-block size.

    Returns:
        Owned (detector_y,detector_x,read) photoelectrons and availability. This is
        fixed-view sampled-field reconstruction, not native cone/time quadrature.
        Spatial convolution is applied after spectral contraction; read-specific
        radiance and durations are preserved. No mixed scene/measurement owner is used.

    Raises:
        ValueError: Physical supports, schedule, numerical controls or reconstruction fail.
        ImportError: The selected native backend is unavailable.
    """
    count = len(response.reads.sample_ids)
    if type(read_block_size) is not int or read_block_size < 1:
        raise ValueError("positive integer read_block_size required")
    if (
        view_local.shape != (count, 3)
        or displacement_xy_m.shape != (detector_points.shape[0], count, 2)
        or not np.isfinite(displacement_xy_m).all()
        or source_support.shape != source_points.shape
        or source_support.dtype != bool
    ):
        raise ValueError("read displacements and Boolean source support must match physical axes")
    values = np.zeros((*detector_points.shape, count))
    supported = np.zeros(values.shape, bool)

    for first in range(0, count, read_block_size):
        last = min(first + read_block_size, count)
        reads = select_read_schedule(response.reads, np.arange(first, last, dtype=np.int64))
        block = SpectralElectronResponse(reads, response.grid, response.coefficients[first:last])
        source, _ = sample_surface_electrons(
            radiance,
            block,
            source_points,
            view_local[first:last],
            frame=frame,
            epoch=epoch,
            support=source_support,
            read_block_size=read_block_size,
            spectral_block_size=spectral_block_size,
        )
        sampled, masks = form_detector_images(
            source,
            source_support,
            source_points,
            detector_points,
            [displacement_xy_m[:, i : i + 1] for i in range(first, last)],
            detector_pitch_xy_m=detector_pitch_xy_m,
            pixel_size_xy_m=pixel_size_xy_m,
            optical_mtf=optical_mtf,
            tail_tolerance=tail_tolerance,
            negative_policy=negative_policy,
            fourier_backend=backend,
        )
        values[..., first:last] = np.concatenate(sampled, axis=2)
        supported[..., first:last] = np.concatenate(masks, axis=2)
    return values, supported


def _sample_surface(
    radiance: SurfaceRadiance | None,
    tangent: SurfaceRadianceTangent | None,
    response: SpectralElectronResponse,
    points: PointLattice,
    view: FloatArray,
    frame: LocalCartesianFrame,
    epoch: datetime,
    support: BoolArray,
    directions: int | None,
    response_tangent: ResponseCoefficientTangent | None,
    read_block: int,
    spectral_block: int,
) -> tuple[FloatArray, FloatArray | None, BoolArray]:
    require_utc_epoch(epoch)
    count = len(response.reads.sample_ids)
    if read_block < 1 or spectral_block < 1 or not same_local_frame(frame, points.frame):
        raise ValueError("positive block controls and shared local frame required")
    if (
        support.shape != points.shape
        or support.dtype != np.bool_
        or view.shape != (count, 3)
        or not np.isfinite(view).all()
        or np.any(view[:, 2] <= 0)
        or not np.allclose(np.linalg.norm(view, axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Boolean point support and upward unit read directions required")
    values = np.zeros((*points.shape, count))
    derivative = None if directions is None else np.zeros((*values.shape, directions))
    valid = np.broadcast_to(support[..., None], values.shape).copy()
    centres = points.xy_m
    times = response.reads.exposure_bounds_s.mean(axis=1)
    for start in range(0, count, read_block):
        stop = min(start + read_block, count)
        selected = slice(start, stop)
        reads = select_read_schedule(response.reads, np.arange(start, stop, dtype=np.int64))
        for first in range(0, response.grid.points.size, spectral_block):
            last = min(first + spectral_block, response.grid.points.size)
            samples = slice(first, last)
            grid = SpectralGrid(
                response.grid.coordinate, response.grid.points[samples], response.grid.domain
            )
            block = SpectralElectronResponse(reads, grid, response.coefficients[selected, samples])
            seed = None if response_tangent is None else response_tangent(selected, samples)
            for pixel in np.ndindex(points.shape):
                if not support[pixel]:
                    continue
                xy = np.broadcast_to(centres[pixel], (stop - start, 2))
                if tangent is None:
                    assert radiance is not None
                    light = radiance(xy, view[selected], times[selected], grid)
                    values[pixel][selected] += spectral_electrons(block, light).total
                else:
                    light, dlight = tangent(xy, view[selected], times[selected], grid)
                    if dlight.shape != (stop - start, last - first, directions):
                        raise ValueError("radiance tangent must match read/spectral/direction axes")
                    value, delta = integrate_photoelectron_tangents(
                        block.coefficients, light, dlight, seed
                    )
                    values[pixel][selected] += value
                    assert derivative is not None
                    derivative[pixel][selected] += delta
    if not np.isfinite(values).all() or (
        derivative is not None and not np.isfinite(derivative).all()
    ):
        raise ValueError("sampled electrons exceed finite numerical domain")
    return values, derivative, valid
