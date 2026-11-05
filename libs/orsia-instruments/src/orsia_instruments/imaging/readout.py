"""Apply physical detector maps and optional noise to a complete charge image."""

from types import MappingProxyType

import numpy as np

from ..detector import (
    DetectorProperties,
    check_detector_maps,
    detector_operating_points,
    sample_readout,
)
from .measurements import ImageMeasurements
from .photodetection import ImagePhotoelectrons


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
