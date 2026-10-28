"""Compose whole-image optical integration and optional physical detector readout."""

import numpy as np
from numpy.typing import NDArray

from ..detector import DetectorMaps
from ..optics.response import ImagingOptics
from .acquisition import ExposureSequence
from .charge import ImageQuantumEfficiency, ImageScene, image_photoelectrons
from .measurements import ImageMeasurements
from .readout import image_readout


def simulate_images(
    field: ImageScene,
    *,
    pixel_bounds_m: NDArray[np.float64],
    optics: ImagingOptics,
    exposures: ExposureSequence,
    detector: DetectorMaps | None = None,
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
        detector: Pixel QE and electronics, with uniform defaults when omitted.
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
