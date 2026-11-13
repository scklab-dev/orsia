"""Apply supplied linear detector and spectral calibration to native camera images."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import validate_covariance

from orsia_instruments.acquisition.schedule import (
    RasterReadSchedule,
    check_raster_alignment,
    select_read_schedule,
)
from orsia_instruments.detector import (
    DetectorProperties,
    DetectorReadings,
    check_detector_maps,
    correct_digital_readings,
    detector_operating_points,
)
from orsia_instruments.observation.measurements import ImageMeasurements
from orsia_instruments.response.bands import NativeBandResponse


@dataclass(frozen=True, eq=False)
class CalibratedFrames:
    """Native achieved-band radiance means with retained read identity and validity.

    Arrays own writable storage. Valid values can be negative after detector
    correction. Unsupported reads are NaN. Keep the shared supplied response stable.

    Attributes:
        response: Supplied native response, including identities, times and achieved bands.
        values: Owned (realisation,read) W m^-2 sr^-1 per (cm^-1) achieved-band means.
        valid: Owned Boolean mask with the same shape as values.
    """

    response: NativeBandResponse
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != len(self.response.reads.sample_ids)
            or valid.shape != values.shape
            or np.any(valid & ~np.isfinite(values))
            or np.any(~valid & ~np.isnan(values))
        ):
            raise ValueError("calibrated values require aligned validity and NaN unsupported reads")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, eq=False)
class CalibratedElectrons:
    """Electron-valued native observations after supplied electronic correction.

    Attributes:
        schedule: Immutable physical identities and original native correspondence.
        values: Owned (realisation,read) photoelectrons; invalid entries are NaN.
        valid: Owned same-shaped Boolean quality mask.
        covariance: Optional owned (read,) variances or (read,read) covariance in
            electron squared units, retaining full read order including masked reads.
            This is supplied uncertainty propagated through fixed electronic gain.
    """

    schedule: RasterReadSchedule
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    covariance: NDArray[np.float64] | None


def calibrate_frame_electrons(
    images: ImageMeasurements,
    *,
    detector: DetectorProperties,
    covariance: NDArray[np.float64] | None = None,
    mask_flagged: bool = False,
    use_expectation: bool = False,
    read_valid: NDArray[np.bool_] | None = None,
) -> CalibratedElectrons:
    """Correct supplied gain, pedestal and dark before combining physical ports.

    Each port is corrected with its own estimated detector. Negative corrected
    values are valid. Clipped reads are rejected or masked across all realizations.
    No electronic model is inferred from simulation truth. Covariance remains in
    native order and is transformed by the fixed linear gain on both axes.
    Calibration-parameter uncertainty must be included by the caller separately.

    Args:
        images: Native expected charge and optional DN/flags with retained identities.
        detector: Independently supplied fixed electronic characterization.
        covariance: Optional native variances/covariance in DN squared for DN
            correction, or electrons squared when using expected photoelectrons.
        mask_flagged: Mask reads flagged in any realization instead of rejecting them.
        use_expectation: Select expected pre-dark charge even when DN are available.
        read_valid: Optional Boolean native read mask for missing/invalid observations.

    Returns:
        Corrected charge, validity and optional propagated covariance in electrons.

    Raises:
        ValueError: Flags, quality, covariance, detector maps or correction are invalid.
    """
    schedule = images.schedule
    n = len(schedule.reads.sample_ids)
    supported = np.ones(n, dtype=np.bool_) if read_valid is None else np.copy(read_valid)
    if supported.shape != (n,) or supported.dtype != np.bool_:
        raise ValueError("read validity must be Boolean (reads,)")
    _, row, column = schedule.native_indices.T
    scale = np.ones(n)
    if images.dn is None or use_expectation:
        electrons = images.photoelectrons.reshape(1, n).copy()
    else:
        if images.full_well_saturated is None or images.adc_clipped is None:
            raise ValueError("DN calibration requires both detector flag arrays")
        check_detector_maps(detector, schedule.native_shape[1:], include_qe=False)
        full = images.full_well_saturated.reshape(-1, n)
        clipped = images.adc_clipped.reshape(-1, n)
        flagged = np.any(full | clipped, axis=0)
        if np.any(flagged & supported) and not mask_flagged:
            raise ValueError("clipped detector reads cannot be linearly corrected")
        supported &= ~flagged
        electrons = np.full((images.dn.shape[0], n), np.nan)
        scale = np.broadcast_to(detector.electrons_per_dn, schedule.native_shape[1:])[row, column]
        if supported.any():
            indices = np.flatnonzero(supported)
            points = detector_operating_points(
                detector, rows=row[supported], columns=column[supported]
            )
            readings = DetectorReadings(
                select_read_schedule(schedule.reads, indices),
                images.dn.reshape(-1, n)[:, supported],
                full[:, supported],
                clipped[:, supported],
            )
            electrons[:, supported] = correct_digital_readings(readings, points)
    valid = np.broadcast_to(supported, electrons.shape).copy()
    electrons[~valid] = np.nan
    propagated = None
    if covariance is not None:
        if covariance.shape not in ((n,), (n, n)):
            raise ValueError("covariance must follow native read order")
        propagated = validate_covariance(covariance)
        propagated *= scale**2 if propagated.ndim == 1 else scale[:, None] * scale[None, :]
    return CalibratedElectrons(schedule, electrons, valid, propagated)


def calibrate_frames(
    images: ImageMeasurements,
    response: NativeBandResponse,
    *,
    detector: DetectorProperties,
    mask_flagged: bool = False,
    use_expectation: bool = False,
) -> CalibratedFrames:
    """Apply supplied gain, pedestal, dark and physical sensitivity calibration.

    Expected photoelectrons already exclude dark/electronics. DN correction uses
    correct_digital_readings and preserves its rejection of nonlinear or flagged
    reads. With mask_flagged, reads flagged in any realization are excluded from
    correction in all realizations and retained as invalid NaN. Zero sensitivity
    also becomes invalid. Only supported unflagged operating points enter the
    nonlinearity check. Scalar response normalization yields the supplied
    achieved photon-weighted band mean. It cannot convert an arbitrary source
    spectrum into a different energy-response mean.

    Args:
        images: Native image electrons, optional DN, flags and row exposure bounds.
        response: Separately supplied per-read ideal or perturbed calibration response.
        detector: Supplied fixed linear electronics; unused for expected photoelectrons.
        mask_flagged: Exclude flagged reads before correction instead of rejecting the batch.
        use_expectation: Explicitly use expected photoelectrons even when DN are available.

    Returns:
        Owned achieved-band means and validity with native identity and supplied response.

    Raises:
        ValueError: Identity/timing shapes disagree, or detector correction is unsupported.
    """
    check_raster_alignment(images.schedule, response.schedule)
    sensitivity = response.sensitivity
    charge = calibrate_frame_electrons(
        images,
        detector=detector,
        mask_flagged=mask_flagged,
        use_expectation=use_expectation,
        read_valid=sensitivity > 0,
    )
    values = np.divide(
        charge.values, sensitivity, out=np.full_like(charge.values, np.nan), where=charge.valid
    )
    return CalibratedFrames(response, values, charge.valid)
