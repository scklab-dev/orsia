"""Apply physical detector maps and optional noise to a complete charge image."""

from types import MappingProxyType

import numpy as np

from ..detector import (
    DETECTOR_MAP_NAMES,
    DetectorMaps,
    DetectorOperatingPoints,
    check_detector_maps,
    sample_readout,
)
from .charge import ImagePhotoelectrons
from .measurements import ImageMeasurements


def image_readout(
    charge: ImagePhotoelectrons,
    *,
    detector: DetectorMaps | None = None,
    rng: np.random.Generator | None = None,
    realisations: int = 1,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
) -> ImageMeasurements:
    """Return full-image expectations and optional conditional noisy DN.

    Row and column common noise groups renew for each exposure; frame noise is
    shared by its whole exposure. QE has already entered optical charge. An
    explicit RNG controls all stochastic draws and is advanced in raster order.

    Args:
        charge: Complete pre-dark source-resolved image charge.
        detector: Fixed pixel maps, with uniform defaults when omitted.
        rng: Optional caller-owned random generator; omission returns means only.
        realisations: Positive count of independent digitised image samples.
        row_noise_electrons: Within-row common Gaussian sigma in electrons.
        column_noise_electrons: Within-column common Gaussian sigma in electrons.
        exposure_noise_electrons: Whole-exposure Gaussian sigma in electrons.

    Returns:
        Owned writable dense charge, timing and optional DN and flag arrays.

    Raises:
        ValueError: Shape, noise width, detector map or realization count is invalid.
    """
    if realisations < 1:
        raise ValueError("realisations must be positive")
    amplitudes = np.array([row_noise_electrons, column_noise_electrons, exposure_noise_electrons])
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise ValueError("common noise must be finite and in [0, inf]")
    detector = DetectorMaps() if detector is None else detector
    shape = charge.image_shape
    check_detector_maps(detector, shape[1:], include_qe=False)
    duration_bounds = charge.signal.acquisition.exposure_bounds_s.reshape((*shape, 2))
    if np.any(duration_bounds != duration_bounds[:, :, :1, :]):
        raise ValueError("exposure bounds must agree across columns of each row")
    dn = full = clipped = None
    if rng is not None:
        values = {
            name: np.broadcast_to(getattr(detector, name), shape[1:])[None, :, :]
            for name in DETECTOR_MAP_NAMES
            if name != "quantum_efficiency"
        }
        points = DetectorOperatingPoints(
            **{name: np.broadcast_to(value, shape).ravel() for name, value in values.items()},
            adc_bits=detector.adc_bits,
        )
        exposure, row, column = np.indices(shape).reshape(3, -1)
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
        np.asarray(duration_bounds[:, :, 0, :], dtype=np.float64).copy(),
        dn,
        full,
        clipped,
        MappingProxyType({name: np.copy(value) for name, value in charge.settings.items()}),
        charge.port,
        MappingProxyType(
            {
                name: value.reshape(shape).copy()
                for name, value in charge.signal.contributions.items()
            }
        ),
    )
