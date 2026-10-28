"""Dense source-resolved image charge and optional detector readout rasters."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class ImageMeasurements:
    """Complete photoelectron image and optional digitised detector realisations.

    Images returned by image_readout own writable independent arrays. Direct
    construction borrows arrays; frozen records do not freeze their buffers.

    Attributes:
        photoelectrons: Pre-dark charge (exposure,row,column) in electrons.
        exposure_bounds_s: Actual rolling intervals (exposure,row,2) in seconds.
        dn: Optional integer counts (realisation,exposure,row,column).
        full_well_saturated: Optional Boolean full-well flags in DN shape.
        adc_clipped: Optional Boolean ADC flags in DN shape.
        settings: Descriptive scalar/per-exposure optical settings.
        port: Physical output port of these images.
        contributions: Named pre-dark means in image shape, in electrons.
    """

    photoelectrons: NDArray[np.float64]
    exposure_bounds_s: NDArray[np.float64]
    dn: NDArray[np.int64] | None
    full_well_saturated: NDArray[np.bool_] | None
    adc_clipped: NDArray[np.bool_] | None
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ]
    port: str
    contributions: Mapping[str, NDArray[np.float64]] = dataclass_field(default_factory=dict)
