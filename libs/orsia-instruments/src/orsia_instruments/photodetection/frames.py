"""Integrate physical surface radiance and tangents through native photon response."""

from collections.abc import Callable, Iterator
from dataclasses import fields

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid
from orsia_radiative.spectral.grid import require_wavenumber_grid

from orsia_instruments.acquisition.quadrature import Acquisition
from orsia_instruments.acquisition.schedule import select_read_schedule
from orsia_instruments.acquisition.surface import SurfaceAcquisitionGrid
from orsia_instruments.detector import check_detector_maps
from orsia_instruments.optics.camera import FilterCamera, FocalPlaneImager, camera_read_bands
from orsia_instruments.optics.spectrometer import DispersiveSpectrometer
from orsia_instruments.photodetection.charge import ExpectedElectrons
from orsia_instruments.photodetection.coefficients import (
    photon_conversion_weights,
    sample_optical_acceptance,
    weighted_photoelectron_response,
)
from orsia_instruments.photodetection.images import ImagePhotoelectrons
from orsia_instruments.response.bands import NativeBandResponse

type SurfaceRadiance = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], SpectralGrid],
    NDArray[np.float64],
]


type SurfaceRadianceCoverage = Callable[[SurfaceAcquisitionGrid, SpectralGrid], NDArray[np.bool_]]


type SurfaceRadianceTangent = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], SpectralGrid],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]


type ResponseCoefficientTangent = Callable[[slice, slice], NDArray[np.float64]]


def frame_response(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    acquisition_grid: SurfaceAcquisitionGrid,
    spectral_grid: SpectralGrid,
    *,
    read_batch_size: int = 64,
    spectral_block_size: int | None = None,
) -> NativeBandResponse:
    """Calculate supplied spectral/radiometric characterization on native reads.

    Use an independently estimated instrument and detector for calibration.
    Joint finite-pixel/PSF/transmission coefficients retain all spectral peaks
    in the declared support. Spatial weights are flat-spectrum marginals.

    Args:
        camera: Detector-owned physical or independently estimated instrument.
        acquisition_grid: Instrument-selected sensor quadrature and native identities.
        spectral_grid: Increasing cm-1 samples with explicit endpoint-sampled coverage.
        read_batch_size: Positive maximum reads per dense coefficient batch.
        spectral_block_size: Optional positive spectral slice size; None retains the full grid.

    Returns:
        Owned native spectral coefficients and normalized spatial marginal.

    Raises:
        ValueError: Instrument, detector, quadrature or spectral support is invalid.
    """
    require_wavenumber_grid(spectral_grid)
    _, response = _frame_integrals(
        camera, acquisition_grid, spectral_grid, None, read_batch_size, spectral_block_size
    )
    return response


def frame_photoelectrons(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    acquisition_grid: SurfaceAcquisitionGrid,
    spectral_grid: SpectralGrid,
    radiance: SurfaceRadiance,
    coverage: SurfaceRadianceCoverage,
    *,
    read_batch_size: int = 64,
    spectral_block_size: int | None = None,
) -> tuple[ImagePhotoelectrons, NativeBandResponse]:
    """Integrate supported physical surface radiance before detector readout.

    Args:
        camera: Physical instrument owning detector QE and optical response.
        acquisition_grid: Native read/node geometry and complete support declarations.
        spectral_grid: Endpoint-sampled declared cm-1 integration support.
        radiance: Pure SpectralGrid-aware physical query callback.
        coverage: Boolean full-read support certificate, including epoch/frame agreement.
        read_batch_size: Positive maximum reads per coefficient/radiance block.
        spectral_block_size: Optional positive spectral width; full-domain weights are retained.

    Returns:
        Owned source-labelled native photoelectrons and their achieved spectral/spatial
        response marginals. Noise, dark current, electronics and calibration are separate.

    Raises:
        ValueError: Coverage, physical inputs, callback values or block controls fail.
    """
    require_wavenumber_grid(spectral_grid)
    _require_coverage(acquisition_grid, spectral_grid, coverage)
    counts, response = _frame_integrals(
        camera, acquisition_grid, spectral_grid, radiance, read_batch_size, spectral_block_size
    )
    signal = ExpectedElectrons(response.reads, {"scene": counts})
    charge = ImagePhotoelectrons(signal, acquisition_grid.schedule, camera.optics.settings)
    return charge, response


def frame_electron_tangents(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    acquisition_grid: SurfaceAcquisitionGrid,
    spectral_grid: SpectralGrid,
    radiance_tangent: SurfaceRadianceTangent,
    coverage: SurfaceRadianceCoverage,
    *,
    response_tangent: ResponseCoefficientTangent | None = None,
    source_name: str = "scene",
    read_batch_size: int = 64,
    spectral_block_size: int | None = None,
) -> tuple[ExpectedElectrons, NDArray[np.float64]]:
    """Contract physical radiance and response directions into native electron means.

    Args:
        camera: Physical camera/imager/spectrometer owning QE and optical response.
        acquisition_grid: Native read/node queries with fixed frame and time conventions.
        spectral_grid: Full declared spectral domain in nm or cm-1.
        radiance_tangent: Pure callback on flat physical queries and a SpectralGrid.
            Returns finite nonnegative (query,spectral) radiance and signed
            (query,spectral,direction) total radiance directions. Direction order is fixed.
        coverage: Boolean full-read support certificate, checked before evaluation.
        response_tangent: Optional pure callback (read_slice,spectral_slice) returning
            signed (read,node,spectral,direction) photon-coefficient seeds in the same
            direction order, including any quadrature/physical factors exactly once.
        source_name: Nonempty source label for expected photoelectrons.
        read_batch_size: Positive maximum reads in coefficient and radiance scratch.
        spectral_block_size: Optional positive spectral slice size; full-domain weights
            are prepared once. Optical/radiance callbacks must be slice-invariant.

    Returns:
        Owned source-resolved ExpectedElectrons and signed (read,direction) electron
        derivatives, preserving native read order. Computes sum(K*dL + dK*L).
        Zero directions are valid. Detector realization and calibration are separate.
        Geometry/time perturbations require total dL, including query-motion terms;
        a response-coefficient perturbation alone does not supply those terms.

    Raises:
        ValueError: Coverage, callback shapes, direction counts or finite values fail.
    """
    if not source_name:
        raise ValueError("source_name must be nonempty")
    grid = acquisition_grid
    _require_coverage(grid, spectral_grid, coverage)
    n, q = grid.rays.acquisition.times_s.shape
    counts = np.zeros(n)
    derivatives = None
    for selected, samples, coefficients in frame_response_coefficients(
        camera,
        grid,
        spectral_grid,
        batch_size=read_batch_size,
        spectral_block_size=spectral_block_size,
    ):
        spectral = SpectralGrid(
            spectral_grid.coordinate, spectral_grid.points[samples], spectral_grid.domain
        )
        values, seeds = radiance_tangent(
            grid.xy_m[selected].reshape(-1, 2),
            grid.view_local[selected].reshape(-1, 3),
            grid.rays.acquisition.times_s[selected].reshape(-1),
            spectral,
        )
        shape = (coefficients.shape[0] * q, coefficients.shape[2])
        _require_radiance(values, shape)
        if seeds.ndim != 3 or seeds.shape[:2] != shape or not np.isfinite(seeds).all():
            raise ValueError("radiance tangents must be finite (query,spectral,direction)")
        if derivatives is None:
            derivatives = np.zeros((n, seeds.shape[-1]))
        if seeds.shape[-1] != derivatives.shape[1]:
            raise ValueError("tangent direction count must remain constant across blocks")
        values = values.reshape(coefficients.shape)
        counts[selected] += np.einsum("rqk,rqk->r", coefficients, values)
        derivatives[selected] += np.einsum(
            "rqk,rqkp->rp", coefficients, seeds.reshape((*coefficients.shape, seeds.shape[-1]))
        )
        if response_tangent is not None:
            response_seed = response_tangent(selected, samples)
            if (
                response_seed.shape != (*coefficients.shape, seeds.shape[-1])
                or not np.isfinite(response_seed).all()
            ):
                raise ValueError("response tangents must match read/node/spectral/direction axes")
            derivatives[selected] += np.einsum("rqkp,rqk->rp", response_seed, values)
    if derivatives is None or not np.isfinite(derivatives).all():
        raise ValueError("nonempty finite electron tangent result required")
    return ExpectedElectrons(grid.schedule.reads, {source_name: counts}), derivatives


def frame_response_coefficients(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    grid: SurfaceAcquisitionGrid,
    spectral: SpectralGrid,
    *,
    batch_size: int = 64,
    spectral_block_size: int | None = None,
) -> Iterator[tuple[slice, slice, NDArray[np.float64]]]:
    """Yield the shared joint photon response in bounded native-read batches.

    Each coefficient includes time/solid-angle/spectral quadrature, photon
    conversion, collecting area, throughput and detector QE exactly once.
    Arrays own writable storage; optical callbacks and input grids remain borrowed.
    This generator owns coefficient construction for simulation and inversion.

    Args:
        camera: Detector-owned camera, local spectral imager or finite-slit instrument.
        grid: Matching native sensor quadrature and physical support.
        spectral: Positive nm or cm-1 grid with endpoint-sampled declared coverage intervals.
        batch_size: Positive maximum number of reads in each returned batch.
        spectral_block_size: Optional positive maximum spectral samples; None uses the full grid.

    Yields:
        Native read slice, spectral slice and (batch,node,block_spectral) electrons per
        unit spectral radiance. Full-domain photon weights are prepared once before slicing.

    Raises:
        ValueError: Instrument, schedule, detector, support or coefficients are invalid.
    """
    if batch_size < 1 or grid.image_shape[1:] != camera.pixel_bounds_m.shape[:2]:
        raise ValueError("positive batch size and camera/grid detector agreement required")
    spectral_size = spectral.points.size if spectral_block_size is None else spectral_block_size
    if spectral_size < 1:
        raise ValueError("spectral_block_size must be positive")
    photon_weights = photon_conversion_weights(spectral)
    wavelengths = (
        0.01 / spectral.points
        if spectral.coordinate == "wavenumber_cm1"
        else spectral.points * 1e-9
    )
    detector = camera.detector
    check_detector_maps(detector, grid.image_shape[1:], include_qe=True)
    acquisition = grid.rays.acquisition
    n, q = acquisition.times_s.shape
    reads = grid.schedule.reads
    if any(
        getattr(acquisition.reads, name) != getattr(reads, name)
        for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids")
    ) or not np.array_equal(acquisition.reads.exposure_bounds_s, reads.exposure_bounds_s):
        raise ValueError("surface acquisition and native schedule identities/timing disagree")
    if n != len(grid.native_indices) or grid.xy_m.shape != (n, q, 2):
        raise ValueError("surface grid and native raster disagree")
    if any(port != camera.optics.port for port in reads.port_ids):
        raise ValueError("physical optical port and native read identities disagree")
    expected_indices = grid.schedule.native_indices
    if isinstance(camera, DispersiveSpectrometer):
        expected_bands = tuple(
            camera.pixel_bands[r * grid.image_shape[2] + c] for _, r, c in expected_indices
        )
        if grid.psf_support_radius_m != 0 or not np.array_equal(
            grid.entrance_bounds_m, camera.slit_bounds_m
        ):
            raise ValueError("spectrometer entrance slit differs from acquisition")
    else:
        expected_bands = camera_read_bands(camera, expected_indices)
        if (
            grid.psf_support_radius_m != camera.support_radius_m
            or grid.entrance_bounds_m is not None
        ):
            raise ValueError("camera PSF support differs from acquisition")
    if grid.focal_length_m != camera.optics.focal_length_m or grid.read_bands != expected_bands:
        raise ValueError("instrument geometry or native mapping differs from acquisition")
    if not np.array_equal(
        acquisition.pixel_bounds_m,
        camera.pixel_bounds_m[grid.native_indices[:, 1], grid.native_indices[:, 2]],
    ):
        raise ValueError("instrument active pixels differ from the selected acquisition")
    qe = np.broadcast_to(detector.quantum_efficiency, grid.image_shape[1:])
    qe = qe[grid.native_indices[:, 1], grid.native_indices[:, 2]][:, None]
    for value in camera.optics.settings.values():
        setting = np.asarray(value)
        if setting.ndim == 1 and len(setting) != grid.image_shape[0]:
            raise ValueError("optical setting vectors must match native exposure count")
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        selected = slice(start, stop)
        current = Acquisition(
            **{
                field.name: (
                    select_read_schedule(acquisition.reads, np.arange(start, stop, dtype=np.int64))
                    if field.name == "reads"
                    else getattr(acquisition, field.name)[selected]
                )
                for field in fields(acquisition)
            }
        )

        def response(local_read, wavelengths, _current, start=start):
            read = start + local_read
            return camera.optics.response(
                int(grid.native_indices[read, 0]), read, wavelengths, acquisition
            )

        throughput = np.asarray(camera.optics.throughput)
        if throughput.ndim:
            throughput = np.broadcast_to(throughput, (n, spectral.points.size))[selected]
        for first in range(0, spectral.points.size, spectral_size):
            spectral_slice = slice(first, min(first + spectral_size, spectral.points.size))
            acceptance = sample_optical_acceptance(current, response, wavelengths[spectral_slice])
            coefficients = weighted_photoelectron_response(
                acceptance,
                photon_weights[spectral_slice],
                collecting_area_m2=camera.optics.collecting_area_m2,
                throughput=throughput if throughput.ndim == 0 else throughput[:, spectral_slice],
                quantum_efficiency=qe[selected],
            )
            yield selected, spectral_slice, coefficients


def _frame_integrals(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    grid: SurfaceAcquisitionGrid,
    spectral: SpectralGrid,
    radiance: SurfaceRadiance | None,
    batch_size: int,
    spectral_block_size: int | None,
) -> tuple[NDArray[np.float64], NativeBandResponse]:
    acquisition = grid.rays.acquisition
    n, q = acquisition.times_s.shape
    counts = np.zeros(n)
    spectral_coefficients = np.empty((n, len(spectral.points)))
    marginal = np.zeros((n, q))
    for selected, spectral_slice, coefficients in frame_response_coefficients(
        camera, grid, spectral, batch_size=batch_size, spectral_block_size=spectral_block_size
    ):
        spectral_coefficients[selected, spectral_slice] = coefficients.sum(axis=1)
        marginal[selected] += coefficients.sum(axis=2)
        if radiance is not None:
            values = radiance(
                grid.xy_m[selected].reshape(-1, 2),
                grid.view_local[selected].reshape(-1, 3),
                acquisition.times_s[selected].reshape(-1),
                SpectralGrid(spectral.coordinate, spectral.points[spectral_slice], spectral.domain),
            )
            _require_radiance(values, (coefficients.shape[0] * q, coefficients.shape[2]))
            counts[selected] += np.einsum(
                "nqk,nqk->n", values.reshape(coefficients.shape), coefficients
            )
    mass = marginal.sum(axis=1, keepdims=True)
    spatial_weights = np.divide(marginal, mass, out=np.zeros_like(marginal), where=mass > 0)
    return counts, NativeBandResponse(
        grid.schedule,
        spectral,
        grid.read_bands,
        spectral_coefficients,
        spatial_weights,
        acquisition.times_s,
        acquisition.directions_uv,
    )


def _require_coverage(grid, spectral, coverage):
    supported = coverage(grid, spectral)
    n = len(grid.schedule.reads.sample_ids)
    if supported.shape != (n,) or supported.dtype != np.bool_ or not supported.all():
        raise ValueError(
            "radiance must certify complete continuous acquisition and spectral support"
        )


def _require_radiance(values, shape):
    if values.shape != shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("radiance must be finite nonnegative (N,K)")
