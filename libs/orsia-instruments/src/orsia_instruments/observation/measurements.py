"""Retain native measurement rasters and registered achieved-band radiance products."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from orsia_radiative.spectral.bands import SpectralBand

from orsia_instruments.acquisition.schedule import RasterReadSchedule, ReadSchedule
from orsia_instruments.response.bands import NativeBandResponse


@dataclass(frozen=True, eq=False)
class ImageMeasurements:
    """Photoelectrons and optional digitised reads with native raster correspondence.

    Images returned by image_readout own writable independent arrays. Construction
    snapshots numerical arrays while sharing the immutable schedule.

    Attributes:
        photoelectrons: Pre-dark charge in electrons, dense native shape or (reads,) for selection.
        schedule: Immutable read identities, actual intervals and original native indices.
        dn: Optional counts (realisation,*schedule.value_shape).
        full_well_saturated: Optional Boolean full-well flags in DN shape.
        adc_clipped: Optional Boolean ADC flags in DN shape.
        settings: Descriptive scalar/per-exposure optical settings.
        contributions: Named pre-dark means in image shape, in electrons.
    """

    photoelectrons: NDArray[np.float64]
    schedule: RasterReadSchedule
    dn: NDArray[np.int64] | None
    full_well_saturated: NDArray[np.bool_] | None
    adc_clipped: NDArray[np.bool_] | None
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ]
    contributions: Mapping[str, NDArray[np.float64]] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.photoelectrons.shape != self.schedule.value_shape:
            raise ValueError("measurement shape must match dense raster or selected read axis")
        for name in ("photoelectrons", "dn", "full_well_saturated", "adc_clipped"):
            value = getattr(self, name)
            if value is not None:
                if name != "photoelectrons" and (
                    value.ndim != len(self.schedule.value_shape) + 1
                    or value.shape[1:] != self.schedule.value_shape
                ):
                    raise ValueError("readout shape must match raster correspondence")
                object.__setattr__(self, name, np.copy(value))
        object.__setattr__(
            self,
            "settings",
            MappingProxyType({name: np.copy(value) for name, value in self.settings.items()}),
        )
        object.__setattr__(
            self,
            "contributions",
            MappingProxyType({name: np.copy(value) for name, value in self.contributions.items()}),
        )


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


@dataclass(frozen=True, eq=False)
class BandRadianceCube:
    """Band radiance means on a rectangular ground coordinate grid.

    Coordinates use the caller's declared surface chart, with axes (y,x,band).
    Ground coordinates alone do not identify a terrestrial datum or projection.
    Each valid value uses W m^-2 sr^-1 per (cm^-1). Unsupported elements are NaN.
    Arrays own writable storage; callers keep bands and arrays stable during use.

    Attributes:
        x_m: Owned increasing (x,) ground cell centres in metres.
        y_m: Owned increasing (y,) ground cell centres in metres.
        bands: Ordered achieved physical responses, one per last-axis element.
        values: Owned (y,x,band) radiance means, NaN where invalid.
        coverage: Owned (y,x,band) nonnegative accumulated spatial weight.
        valid: Owned Boolean (y,x,band) support and quality mask.
    """

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    bands: tuple[SpectralBand, ...]
    values: NDArray[np.float64]
    coverage: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m"):
            axis = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if (
                axis.ndim != 1
                or not len(axis)
                or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)
            ):
                raise ValueError("cube axes must be nonempty finite increasing ground centres")
            object.__setattr__(self, name, axis)
        shape = (len(self.y_m), len(self.x_m), len(self.bands))
        values = np.array(self.values, dtype=np.float64, copy=True)
        coverage = np.array(self.coverage, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)
        if (
            not self.bands
            or values.shape != shape
            or coverage.shape != shape
            or valid.shape != shape
            or not np.isfinite(coverage).all()
            or np.any(coverage < 0)
            or np.any(valid & ((coverage <= 0) | ~np.isfinite(values)))
            or np.any(~valid & ~np.isnan(values))
        ):
            raise ValueError("cube shapes, coverage, validity and NaN support disagree")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "valid", valid)
