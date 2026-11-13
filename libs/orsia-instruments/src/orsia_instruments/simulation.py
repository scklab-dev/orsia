"""Compose physical photon integration with detector readout."""

from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid

from orsia_instruments.acquisition.images import ExposureSequence
from orsia_instruments.acquisition.surface import SurfaceAcquisitionGrid
from orsia_instruments.detector import (
    DetectorProperties,
    check_detector_maps,
    detector_operating_points,
    sample_readout,
)
from orsia_instruments.observation.measurements import FrameMeasurements, ImageMeasurements
from orsia_instruments.optics.camera import FilterCamera, FocalPlaneImager
from orsia_instruments.optics.response import ImagingOptics
from orsia_instruments.optics.spectrometer import DispersiveSpectrometer
from orsia_instruments.photodetection.frames import (
    SurfaceRadiance,
    SurfaceRadianceCoverage,
    frame_photoelectrons,
)
from orsia_instruments.photodetection.images import (
    ImagePhotoelectrons,
    ImageQuantumEfficiency,
    ImageScene,
    image_photoelectrons,
)


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
    spectral_block_size: int | None = None,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
) -> FrameMeasurements:
    """Simulate native camera or slit-spectrometer electrons and optional noisy DN.

    The coverage callback certifies every continuous surface/time/direction and
    spectral support before radiance evaluation. Radiance borrows flat (N,2)
    surface metre coordinates, (N,3) chart-local surface-to-sensor unit directions,
    (N,) elapsed SI seconds, and a SpectralGrid with K samples. It returns (N,K)
    W m^-2 sr^-1 per declared coordinate. N flattens reads then nodes in C order. Geometry
    preparation owns frame/sign conversion; this integrator restores read/node axes.
    Callback outputs are borrowed without mutation. RNG sampling occurs once in
    native raster order after deterministic integration. Spectral partitioning can
    change rounding of the means and therefore exact random draws. The supplied grid
    is the simulation's true geometry. Detector QE enters once during integration;
    electronics enter during readout.

    Args:
        camera: Physical FilterCamera or DispersiveSpectrometer. Slit models use
            entrance-plane rays, then wavelength-dependent finite detector acceptance;
            detector columns never define independent undispersed ground sight lines.
        acquisition_grid: True instrument-selected integration coordinates and ray support.
        spectral_grid: Increasing cm-1 samples covering the endpoint-sampled declared intervals.
        radiance: Pure stable surface/direction/time/spectral radiance callback.
        coverage: Boolean (reads,) full-support attestation, including spectral edges.
        rng: Optional caller-owned generator advanced by noise sampling; None returns means only.
        realisations: Positive number of independent noisy DN realisations.
        read_batch_size: Positive maximum dense integration batch size in physical reads.
        spectral_block_size: Optional positive spectral slice size.
            Full-domain quadrature is retained.
        row_noise_electrons: Shared within-row noise standard deviation, electrons.
        column_noise_electrons: Shared within-column noise standard deviation, electrons.
        exposure_noise_electrons: Shared within-frame noise standard deviation, electrons.

    Returns:
        Owned native images, physical read schedule and achieved photon responses.

    Raises:
        ValueError: Support attestation, radiance, optical response or readout inputs are invalid.
    """
    charge, response = frame_photoelectrons(
        camera,
        acquisition_grid,
        spectral_grid,
        radiance,
        coverage,
        read_batch_size=read_batch_size,
        spectral_block_size=spectral_block_size,
    )
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


def image_readout(
    charge: ImagePhotoelectrons,
    *,
    detector: DetectorProperties,
    rng: np.random.Generator | None = None,
    realisations: int = 1,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
) -> ImageMeasurements:
    """Return image or selected-read expectations and optional conditional noisy DN.

    Row and column common noise groups renew for each exposure; frame noise is
    shared by its whole exposure. QE has already entered optical charge. An
    explicit RNG controls all stochastic draws and is advanced in raster order.

    Args:
        charge: Pre-dark source-resolved charge in retained read order.
        detector: Supplied fixed electronics; QE is already included in charge.
        rng: Optional caller-owned random generator; omission returns means only.
        realisations: Positive count of independent digitised image samples.
        row_noise_electrons: Within-row common Gaussian sigma in electrons.
        column_noise_electrons: Within-column common Gaussian sigma in electrons.
        exposure_noise_electrons: Whole-exposure Gaussian sigma in electrons.

    Returns:
        Independent writable charge and optional DN/flags in schedule.value_shape,
        with a leading realisation axis on DN/flags and a shared immutable schedule.

    Raises:
        ValueError: Shape, noise width, detector map or realization count is invalid.
    """
    if realisations < 1:
        raise ValueError("realisations must be positive")
    amplitudes = np.array([row_noise_electrons, column_noise_electrons, exposure_noise_electrons])
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise ValueError("common noise must be finite and in [0, inf]")
    shape = charge.schedule.value_shape
    check_detector_maps(detector, charge.schedule.native_shape[1:], include_qe=False)
    dn = full = clipped = None
    if rng is not None:
        exposure, row, column = charge.schedule.native_indices.T
        points = detector_operating_points(detector, rows=row, columns=column)
        readings = sample_readout(
            charge.signal,
            points,
            rng,
            realisations=realisations,
            row_ids=tuple(f"{s}:{r}" for s, r in zip(exposure, row, strict=True)),
            column_ids=tuple(f"{s}:{c}" for s, c in zip(exposure, column, strict=True)),
            exposure_ids=tuple(map(str, exposure)),
            row_noise_electrons=row_noise_electrons,
            column_noise_electrons=column_noise_electrons,
            exposure_noise_electrons=exposure_noise_electrons,
        )
        size = (realisations, *shape)
        dn = np.array(readings.dn.reshape(size), copy=True)
        full = np.array(readings.full_well_saturated.reshape(size), copy=True)
        clipped = np.array(readings.adc_clipped.reshape(size), copy=True)
    return ImageMeasurements(
        charge.signal.total.reshape(shape).copy(),
        charge.schedule,
        dn,
        full,
        clipped,
        MappingProxyType({name: np.copy(value) for name, value in charge.settings.items()}),
        MappingProxyType(
            {
                name: value.reshape(shape).copy()
                for name, value in charge.signal.contributions.items()
            }
        ),
    )


def simulate_images(
    field: ImageScene,
    *,
    pixel_bounds_m: NDArray[np.float64],
    optics: ImagingOptics,
    exposures: ExposureSequence,
    detector: DetectorProperties,
    rng: np.random.Generator | None = None,
    realisations: int = 1,
    time_order: int = 3,
    angular_order: int = 5,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
    quantum_efficiency: ImageQuantumEfficiency | None = None,
    source_name: str = "scene",
) -> ImageMeasurements:
    """Simulate a full image using dense charge integration and detector readout.

    Scene callbacks receive one whole-image acquisition. Without an RNG, the
    result contains expected pre-dark electrons with DN and flags set to None.
    With an explicit RNG, counts and flags have shape (realisation,exposure,row,column).
    Arrays in the returned record own independently writable storage.

    Args:
        field: Resident radiance field or full-acquisition scene callback.
        pixel_bounds_m: (rows,columns,4) active pixel bounds in metres.
        optics: Image acceptance, pupil area, setting and throughput.
        exposures: Time bounds, pointing and optional rolling row delay.
        detector: Supplied fixed pixel QE and electronics.
        rng: Optional explicit generator for Poisson/electronic noise and DN.
        realisations: Positive number of independently sampled images.
        time_order: Gauss order over each read's exposure.
        angular_order: Gauss order over each sky direction.
        row_noise_electrons: Common within-row Gaussian sigma in electrons.
        column_noise_electrons: Common within-column Gaussian sigma in electrons.
        exposure_noise_electrons: Frame-common Gaussian sigma in electrons.
        quantum_efficiency: Optional spectral QE callback replacing detector QE.
        source_name: Label of the optical source contribution.

    Returns:
        Full-image source-labelled photoelectron means and optional noisy DN.

    Raises:
        ValueError: Invalid geometry, physical coefficients or detector properties.
    """
    charge = image_photoelectrons(
        field,
        pixel_bounds_m=pixel_bounds_m,
        optics=optics,
        exposures=exposures,
        detector=detector,
        quantum_efficiency=quantum_efficiency,
        source_name=source_name,
        time_order=time_order,
        angular_order=angular_order,
    )
    return image_readout(
        charge,
        detector=detector,
        rng=rng,
        realisations=realisations,
        row_noise_electrons=row_noise_electrons,
        column_noise_electrons=column_noise_electrons,
        exposure_noise_electrons=exposure_noise_electrons,
    )
