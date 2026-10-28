"""Integrate full image acquisitions into source-resolved photoelectron rasters."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition, ReadSchedule
from ..charge import ExpectedElectrons, combine_source_paths
from ..detector import DetectorMaps, check_detector_maps
from ..integration import expected_electrons
from ..optics.response import ImagingOptics
from ..scene import RadianceField
from .acquisition import ExposureSequence, image_acquisition

type ImageIndices = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]
type ImageQuantumEfficiency = Callable[[ImageIndices, NDArray[np.float64]], NDArray[np.float64]]
type ImageScene = RadianceField | Callable[[Acquisition], RadianceField]


@dataclass(frozen=True, eq=False)
class ImagePhotoelectrons:
    """One complete source-resolved photoelectron raster in physical read order.

    The signal stores pre-dark mean electrons for every exposure/row/column,
    flattened in C order. Arrays and settings are independently writable snapshots.

    Attributes:
        signal: Source-labelled physical charge for all image reads.
        image_shape: (exposures,rows,columns) raster dimensions.
        settings: Scalar or per-exposure descriptive optical settings.
        port: Physical output port shared by the reads.
    """

    signal: ExpectedElectrons
    image_shape: tuple[int, int, int]
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ] = field(default_factory=dict)
    port: str = "direct"

    def __post_init__(self) -> None:
        if len(self.image_shape) != 3 or any(s < 1 for s in self.image_shape):
            raise ValueError("image shape must have three positive dimensions")
        if len(self.signal.total) != np.prod(self.image_shape):
            raise ValueError("signal must cover the entire image")
        if not self.port or any(p != self.port for p in self.signal.acquisition.port_ids):
            raise ValueError("image port must agree with physical reads")
        settings = {name: np.copy(value) for name, value in self.settings.items()}
        if any(
            not np.isfinite(value).all()
            or value.ndim > 1
            or (value.ndim == 1 and value.shape != (self.image_shape[0],))
            for value in settings.values()
        ):
            raise ValueError("settings must be finite scalars or exposure vectors")
        object.__setattr__(self, "settings", MappingProxyType(settings))


def image_photoelectrons(
    field: ImageScene,
    *,
    pixel_bounds_m: NDArray[np.float64],
    optics: ImagingOptics,
    exposures: ExposureSequence,
    detector: DetectorMaps | None = None,
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
        detector: Fixed detector QE map, with uniform defaults when omitted.
        quantum_efficiency: Optional spectral QE callback replacing map QE.
        source_name: Name assigned to this optical path.
        time_order: Gauss order across each exposure interval.
        angular_order: Gauss order across each direction coordinate.

    Returns:
        Owned source-labelled complete image charge before dark current.

    Raises:
        ValueError: Geometry, support, QE or response fractions are invalid.
    """
    detector = DetectorMaps() if detector is None else detector
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
    acquisition = image_acquisition(
        exposure_bounds_s=exposures.bounds_s,
        setting_ids=tuple(map(str, range(shape[0]))),
        pixel_bounds_m=pixels,
        row_offsets_s=np.arange(shape[1]) * exposures.row_delay_s,
        direction_bounds_uv=exposures.direction_bounds_uv,
        pointing=exposures.pointing,
        focal_length_m=optics.focal_length_m,
        time_order=time_order,
        angular_order=angular_order,
    )
    exposure, row, column = np.indices(shape).reshape(3, -1)
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
    acquisition = signal.acquisition
    signal = ExpectedElectrons(
        ReadSchedule(
            acquisition.sample_ids,
            acquisition.pixel_ids,
            acquisition.setting_ids,
            (optics.port,) * len(acquisition.sample_ids),
            acquisition.exposure_bounds_s,
        ),
        signal.contributions,
    )
    return ImagePhotoelectrons(signal, shape, optics.settings, optics.port)


def image_charge(
    contributions: Mapping[str, NDArray[np.float64]],
    *,
    exposure_bounds_s: NDArray[np.floating | np.integer],
    row_delay_s: float = 0,
    port: str = "direct",
) -> ImagePhotoelectrons:
    """Attach physical reads to already integrated image electron means.

    Args:
        contributions: Named (exposure,row,column) nonnegative electron means,
            already integrated over each read's exposure and pixel, including
            upstream optical response where applicable; dark current and readout
            follow this function.
        exposure_bounds_s: (exposure,2) start/end times in seconds.
        row_delay_s: Rolling time offset per row in seconds.
        port: Physical output port shared by the image.

    Returns:
        Fresh full-image source-labelled charge, before dark current and readout.

    Raises:
        ValueError: Empty, inconsistent, nonfinite or invalid shape, time or charge.
    """
    if not contributions:
        raise ValueError("at least one image source is required")
    shape = next(iter(contributions.values())).shape
    if (
        len(shape) != 3
        or any(s < 1 for s in shape)
        or any(v.shape != shape for v in contributions.values())
    ):
        raise ValueError("contributions must share a nonempty (exposure,row,column) shape")
    bounds = np.copy(exposure_bounds_s)
    if (
        bounds.shape != (shape[0], 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] <= bounds[:, 0])
        or not np.isfinite(row_delay_s)
    ):
        raise ValueError("exposure bounds must be ordered finite (exposure,2) intervals")
    exposure, row, column = np.indices(shape).reshape(3, -1)
    reads = ReadSchedule(
        tuple(f"s{s}-p{r * shape[2] + c}" for s, r, c in zip(exposure, row, column, strict=True)),
        tuple(f"r{r}-c{c}" for r, c in zip(row, column, strict=True)),
        tuple(map(str, exposure)),
        (port,) * len(exposure),
        bounds[exposure] + row[:, None] * row_delay_s,
    )
    return ImagePhotoelectrons(
        ExpectedElectrons(
            reads, {name: np.copy(values).ravel() for name, values in contributions.items()}
        ),
        shape,
        port=port,
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
        path.image_shape != paths[0].image_shape or path.port != paths[0].port for path in paths
    ):
        raise ValueError("image paths must have matching shape and port")
    return ImagePhotoelectrons(
        combine_source_paths(*(p.signal for p in paths)),
        paths[0].image_shape,
        paths[0].settings,
        paths[0].port,
    )


def _spectral_qe(
    law: ImageQuantumEfficiency,
    indices: ImageIndices,
    reads: NDArray[np.int64],
    wave: NDArray[np.float64],
) -> NDArray[np.float64]:
    return law((indices[0][reads], indices[1][reads], indices[2][reads]), wave)
