"""Dense source-resolved image charge and optional detector readout rasters."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..acquisition import RasterReadSchedule


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
