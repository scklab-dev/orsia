"""Integrate supplied surface radiance into native camera or slit-spectrometer frames.

Radiance callbacks receive instrument-selected coordinates. Numerical integration
is chunked over physical reads; random sampling follows complete charge assembly.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid

from ..acquisition import Acquisition, ReadSchedule, select_read_schedule
from ..charge import ExpectedElectrons
from ..detector import check_detector_maps
from ..optics.camera import FilterCamera, FocalPlaneImager, camera_read_bands
from ..optics.spectrometer import DispersiveSpectrometer
from ..photoelectrons import photoelectron_response, sample_optical_acceptance
from .geolocation import SurfaceAcquisitionGrid
from .measurements import ImageMeasurements
from .photodetection import ImagePhotoelectrons
from .readout import image_readout
from .response.band import NativeBandResponse

type SurfaceRadiance = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]
type SurfaceRadianceCoverage = Callable[[SurfaceAcquisitionGrid, SpectralGrid], NDArray[np.bool_]]


@dataclass(frozen=True, eq=False)
class FrameMeasurements:
    """Native detector image measurements and their physical spectral sensitivity.

    Arrays in images and response own writable storage. The schedule is shared
    with response; keep all buffers stable during calibration or reconstruction.
    Simulation truth coordinates are deliberately separate caller-owned inputs.

    Attributes:
        images: Expected photoelectrons in dense (frame,row,column) shape or a selected
            (reads,) axis; optional DN/flags add a realisation axis. The record retains
            optical settings and immutable timing through its raster schedule.
        reads: Native read identities and complete exposure intervals in retained read order.
        response: Achieved per-read photon sensitivity and sampled spatial weights.
    """

    images: ImageMeasurements
    response: NativeBandResponse

    @property
    def reads(self) -> ReadSchedule:
        """Return the measurement's retained physical schedule."""
        return self.images.schedule.reads


def frame_response(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    acquisition_grid: SurfaceAcquisitionGrid,
    spectral_grid: SpectralGrid,
    *,
    read_batch_size: int = 64,
) -> NativeBandResponse:
    """Calculate supplied spectral/radiometric characterization on native reads.

    Use an independently estimated instrument and detector for calibration.
    Joint finite-pixel/PSF/transmission coefficients retain all spectral peaks
    in the declared support. Spatial weights are flat-spectrum marginals.

    Args:
        camera: Detector-owned physical or independently estimated instrument.
        acquisition_grid: Instrument-selected sensor quadrature and native identities.
        spectral_grid: Connected positive increasing wavenumber nodes, cm^-1.
        read_batch_size: Positive maximum reads per dense coefficient batch.

    Returns:
        Owned native spectral coefficients and normalized spatial marginal.

    Raises:
        ValueError: Instrument, detector, quadrature or spectral support is invalid.
    """
    _, response = _frame_integrals(camera, acquisition_grid, spectral_grid, None, read_batch_size)
    return response


def simulate_frames(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    acquisition_grid: SurfaceAcquisitionGrid,
    spectral_grid: SpectralGrid,
    radiance: SurfaceRadiance,
    coverage: SurfaceRadianceCoverage,
    *,
    rng: np.random.Generator | None = None,
    realisations: int = 1,
    read_batch_size: int = 64,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
) -> FrameMeasurements:
    """Simulate native camera or slit-spectrometer electrons and optional noisy DN.

    The coverage callback certifies every continuous surface/time/direction and
    spectral support before radiance evaluation. Radiance borrows (reads,nodes,2)
    surface metre coordinates, GCRS instrument-to-scene unit directions,
    elapsed SI seconds, and cm^-1 spectral nodes. It returns same-batch
    (reads,nodes,spectral) W m^-2 sr^-1 per (cm^-1). Direction-dependent callbacks
    own conversion to local viewing coordinates and their declared support.
    Callback outputs are borrowed without mutation. RNG sampling occurs once in
    native raster order after deterministic chunked integration, independent of
    batch size. The supplied grid is the simulation's true geometry. The attached
    detector supplies QE once during integration and electronics during readout.

    Args:
        camera: Physical FilterCamera or DispersiveSpectrometer. Slit models use
            entrance-plane rays, then wavelength-dependent finite detector acceptance;
            detector columns never define independent undispersed ground sight lines.
        acquisition_grid: True instrument-selected integration coordinates and ray support.
        spectral_grid: Connected positive increasing cm^-1 nodes covering the declared model.
        radiance: Pure stable surface/direction/time/spectral radiance callback.
        coverage: Boolean (reads,) full-support attestation, including spectral edges.
        rng: Optional caller-owned generator advanced by noise sampling; None returns means only.
        realisations: Positive number of independent noisy DN realisations.
        read_batch_size: Positive maximum dense integration batch size in physical reads.
        row_noise_electrons: Shared within-row noise standard deviation, electrons.
        column_noise_electrons: Shared within-column noise standard deviation, electrons.
        exposure_noise_electrons: Shared within-frame noise standard deviation, electrons.

    Returns:
        Owned native images, physical read schedule and achieved photon responses.

    Raises:
        ValueError: Support attestation, radiance, optical response or readout inputs are invalid.
    """
    supported = coverage(acquisition_grid, spectral_grid)
    n = len(acquisition_grid.rays.acquisition.sample_ids)
    if supported.shape != (n,) or supported.dtype != np.bool_ or not supported.all():
        raise ValueError(
            "radiance must certify complete continuous acquisition and spectral support"
        )
    counts, response = _frame_integrals(
        camera, acquisition_grid, spectral_grid, radiance, read_batch_size
    )
    signal = ExpectedElectrons(response.reads, {"scene": counts})
    charge = ImagePhotoelectrons(signal, acquisition_grid.schedule, camera.optics.settings)
    images = image_readout(
        charge,
        detector=camera.detector,
        rng=rng,
        realisations=realisations,
        row_noise_electrons=row_noise_electrons,
        column_noise_electrons=column_noise_electrons,
        exposure_noise_electrons=exposure_noise_electrons,
    )
    return FrameMeasurements(images, response)


def frame_response_coefficients(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    grid: SurfaceAcquisitionGrid,
    spectral: SpectralGrid,
    *,
    batch_size: int = 64,
) -> Iterator[tuple[slice, NDArray[np.float64]]]:
    """Yield the shared joint photon response in bounded native-read batches.

    Each coefficient includes time/solid-angle/spectral quadrature, photon
    conversion, collecting area, throughput and detector QE exactly once.
    Arrays own writable storage; optical callbacks and input grids remain borrowed.
    This generator owns coefficient construction for simulation and inversion.

    Args:
        camera: Detector-owned camera, local spectral imager or finite-slit instrument.
        grid: Matching native sensor quadrature and physical support.
        spectral: Connected positive wavenumber grid in cm^-1.
        batch_size: Positive maximum number of reads in each returned batch.

    Yields:
        Native read slice and (batch,node,spectral) electrons per unit spectral
        radiance W m^-2 sr^-1 per (cm^-1).

    Raises:
        ValueError: Instrument, schedule, detector, support or coefficients are invalid.
    """
    if batch_size < 1 or grid.image_shape[1:] != camera.pixel_bounds_m.shape[:2]:
        raise ValueError("positive batch size and camera/grid detector agreement required")
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

        acceptance = sample_optical_acceptance(current, response, 0.01 / spectral.points)
        coefficients = photoelectron_response(
            spectral,
            acceptance,
            collecting_area_m2=camera.optics.collecting_area_m2,
            throughput=camera.optics.throughput,
            quantum_efficiency=qe[selected],
        )
        yield selected, coefficients


def _frame_integrals(
    camera: FilterCamera | FocalPlaneImager | DispersiveSpectrometer,
    grid: SurfaceAcquisitionGrid,
    spectral: SpectralGrid,
    radiance: SurfaceRadiance | None,
    batch_size: int,
) -> tuple[NDArray[np.float64], NativeBandResponse]:
    acquisition = grid.rays.acquisition
    n, q = acquisition.times_s.shape
    counts = np.zeros(n)
    spectral_coefficients = np.empty((n, len(spectral.points)))
    spatial_weights = np.empty((n, q))
    for selected, coefficients in frame_response_coefficients(
        camera, grid, spectral, batch_size=batch_size
    ):
        spectral_coefficients[selected] = coefficients.sum(axis=1)
        marginal = coefficients.sum(axis=2)
        mass = marginal.sum(axis=1, keepdims=True)
        spatial_weights[selected] = np.divide(
            marginal, mass, out=np.zeros_like(marginal), where=mass > 0
        )
        if radiance is not None:
            values = radiance(
                grid.xy_m[selected],
                grid.rays.directions_gcrs[selected],
                acquisition.times_s[selected],
                spectral.points,
            )
            if (
                values.shape != coefficients.shape
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError("radiance must be finite nonnegative (reads,nodes,spectral)")
            counts[selected] = np.einsum("nqk,nqk->n", values, coefficients)
    return counts, NativeBandResponse(
        grid.schedule,
        spectral,
        grid.read_bands,
        spectral_coefficients,
        spatial_weights,
        acquisition.times_s,
        acquisition.directions_uv,
    )
