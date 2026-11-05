"""Integrate full image acquisitions into source-resolved photoelectron rasters."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition, RasterReadSchedule, raster_read_schedule
from ..charge import ExpectedElectrons, combine_source_paths
from ..detector import DetectorProperties, check_detector_maps
from ..optics.response import ImagingOptics
from ..photoelectrons import expected_electrons
from ..scene import RadianceField
from .acquisition import ExposureSequence, image_acquisition

type ImageIndices = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]
type ImageQuantumEfficiency = Callable[[ImageIndices, NDArray[np.float64]], NDArray[np.float64]]
type ImageScene = RadianceField | Callable[[Acquisition], RadianceField]


@dataclass(frozen=True, eq=False)
class ImagePhotoelectrons:
    """Source-resolved photoelectrons with original native raster correspondence.

    The signal stores pre-dark means on the selected read axis. Complete C-order
    rasters can be reshaped densely. Charge and settings own writable snapshots.

    Attributes:
        signal: Source-labelled physical charge for all image reads.
        schedule: Immutable physical identities, timing and native raster correspondence.
        image_shape: Derived full (frames,rows,columns) native dimensions.
        settings: Scalar or per-exposure descriptive optical settings.
    """

    signal: ExpectedElectrons
    schedule: RasterReadSchedule
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reads = self.signal.reads
        expected = self.schedule.reads
        if any(
            getattr(reads, name) != getattr(expected, name)
            for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids")
        ) or not np.array_equal(reads.exposure_bounds_s, expected.exposure_bounds_s):
            raise ValueError("image charge and raster schedule must share identities and timing")
        settings = {name: np.copy(value) for name, value in self.settings.items()}
        if any(
            not np.isfinite(value).all()
            or value.ndim > 1
            or (value.ndim == 1 and value.shape != (self.schedule.native_shape[0],))
            for value in settings.values()
        ):
            raise ValueError("settings must be finite scalars or exposure vectors")
        object.__setattr__(self, "settings", MappingProxyType(settings))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return full native raster dimensions, including unselected reads."""
        return self.schedule.native_shape


def image_photoelectrons(
    field: ImageScene,
    *,
    pixel_bounds_m: NDArray[np.float64],
    optics: ImagingOptics,
    exposures: ExposureSequence,
    detector: DetectorProperties,
    quantum_efficiency: ImageQuantumEfficiency | None = None,
    source_name: str = "scene",
    time_order: int = 3,
    angular_order: int = 5,
) -> ImagePhotoelectrons:
    """Integrate an image's complete optical path and physical QE into charge.

    The scene callback, when supplied, receives the entire acquisition once.
    Spectral QE replaces the detector map: it receives global exposure/row/column
    index vectors and every wavelength in metres, returning (reads,wavenumbers).
    All image charge and geometry are evaluated as dense arrays. Optical response
    retains a per-read callback contract, evaluated on the full spectral grid.

    Args:
        field: Resident spectral radiance or whole-acquisition scene callback.
        pixel_bounds_m: (rows,columns,4) finite active pixel bounds in metres.
        optics: Optical response, area, settings and throughput.
        exposures: Exposure intervals, pointing and rolling row timing.
        detector: Supplied fixed pixel QE and electronics; only QE enters integration.
        quantum_efficiency: Optional spectral QE callback replacing map QE.
        source_name: Name assigned to this optical path.
        time_order: Gauss order across each exposure interval.
        angular_order: Gauss order across each direction coordinate.

    Returns:
        Owned source-labelled complete image charge before dark current.

    Raises:
        ValueError: Geometry, support, QE or response fractions are invalid.
    """
    pixels = np.copy(pixel_bounds_m)
    if (
        not np.isfinite(pixels).all()
        or pixels.ndim != 3
        or pixels.shape[-1] != 4
        or not all(pixels.shape[:2])
        or np.any(pixels[..., [1, 3]] <= pixels[..., [0, 2]])
    ):
        raise ValueError("pixel bounds must be finite ordered (rows,columns,4)")
    shape = (len(exposures.bounds_s), *pixels.shape[:2])
    for value in optics.settings.values():
        setting = np.asarray(value)
        if setting.ndim == 1 and setting.shape != (shape[0],):
            raise ValueError("optical setting vectors must match exposures")
    check_detector_maps(detector, shape[1:], include_qe=quantum_efficiency is None)
    if not source_name:
        raise ValueError("source name must be nonempty")
    schedule = raster_read_schedule(
        exposures.bounds_s,
        pixels.shape[:2],
        row_offsets_s=np.arange(shape[1]) * exposures.row_delay_s,
        port=optics.port,
    )
    acquisition = image_acquisition(
        schedule=schedule,
        pixel_bounds_m=pixels,
        direction_bounds_uv=exposures.direction_bounds_uv,
        pointing=exposures.pointing,
        focal_length_m=optics.focal_length_m,
        time_order=time_order,
        angular_order=angular_order,
    )
    exposure, row, column = schedule.native_indices.T
    indices = (exposure, row, column)

    # Image factories index the exposure separately; per-read geometry stays global.
    def response(read: int, wave: NDArray[np.float64], current: Acquisition) -> NDArray[np.float64]:
        return optics.response(int(exposure[read]), read, wave, current)

    if quantum_efficiency is None:
        qe = np.broadcast_to(detector.quantum_efficiency, shape[1:])[row, column][:, None]
    else:
        qe = partial(_spectral_qe, quantum_efficiency, indices)
    scene = field(acquisition) if callable(field) else field
    signal = expected_electrons(
        scene,
        acquisition,
        response,
        collecting_area_m2=optics.collecting_area_m2,
        throughput=optics.throughput,
        quantum_efficiency=qe,
        source_name=source_name,
    )
    return ImagePhotoelectrons(signal, schedule, optics.settings)


def image_charge(
    contributions: Mapping[str, NDArray[np.float64]],
    *,
    schedule: RasterReadSchedule,
) -> ImagePhotoelectrons:
    """Attach retained raster correspondence to already integrated electron means.

    Dense input requires complete C-order coverage. Selected or permuted reads
    have shape (reads,) and retain the original native shape and identities.

    Args:
        contributions: Named nonnegative electrons in schedule.value_shape, already
            including upstream optical response and QE; dark/readout follow.
        schedule: Immutable original or selected raster correspondence.

    Returns:
        Owned source-labelled charge with the supplied immutable schedule.

    Raises:
        ValueError: Sources are empty or shapes, charges or correspondence disagree.
    """
    if not contributions:
        raise ValueError("at least one image source is required")
    if any(values.shape != schedule.value_shape for values in contributions.values()):
        raise ValueError("contributions must match complete raster or selected read-axis shape")
    return ImagePhotoelectrons(
        ExpectedElectrons(
            schedule.reads, {name: values.ravel().copy() for name, values in contributions.items()}
        ),
        schedule,
    )


def combine_image_source_paths(*paths: ImagePhotoelectrons) -> ImagePhotoelectrons:
    """Sum dense paths aligned by physical read identity and timing.

    Args:
        *paths: Complete raster paths with distinct source names and shared reads.

    Returns:
        Owned source-resolved charge retaining the first path's settings and port.

    Raises:
        ValueError: Paths are empty, misaligned or have repeated source names.
    """
    if not paths or any(
        path.image_shape != paths[0].image_shape
        or not np.array_equal(path.schedule.native_indices, paths[0].schedule.native_indices)
        for path in paths
    ):
        raise ValueError("image paths must have matching shape and port")
    return ImagePhotoelectrons(
        combine_source_paths(*(p.signal for p in paths)),
        paths[0].schedule,
        paths[0].settings,
    )


def _spectral_qe(
    law: ImageQuantumEfficiency,
    indices: ImageIndices,
    reads: NDArray[np.int64],
    wave: NDArray[np.float64],
) -> NDArray[np.float64]:
    return law((indices[0][reads], indices[1][reads], indices[2][reads]), wave)
