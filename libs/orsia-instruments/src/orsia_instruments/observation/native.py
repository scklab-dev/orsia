"""Calibrate and organize native radiances with conditional first-order uncertainty.

Logical cube indices retain band-dependent native supports. Supplied accumulation
weights and geometry are fixed, independent of the measured values. This module
performs no common-grid registration or joint spectral-spatial reconstruction."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty.projection import ProjectedCovariance
from scipy.sparse import csr_array

from orsia_instruments.acquisition.schedule import check_raster_alignment
from orsia_instruments.acquisition.surface import SurfaceAcquisitionGrid
from orsia_instruments.calibration.frames import calibrate_frame_electrons
from orsia_instruments.calibration.radiometry import (
    RadiometricObservations,
    accumulate_observations,
    calibrate_readings,
    normalized_accumulation_weights,
)
from orsia_instruments.calibration.radiometry import (
    RadiometricParameterMaps as RadiometricParameterMaps,
)
from orsia_instruments.detector import DetectorProperties
from orsia_instruments.observation.measurements import ImageMeasurements
from orsia_instruments.response.bands import NativeBandResponse


@dataclass(frozen=True, eq=False)
class CalibratedObservations:
    """One native radiance realization with retained calibration error derivatives.

    Factory arrays own writable storage; direct construction borrows its inputs.
    Keep all buffers and the shared response stable during use.

    Attributes:
        response: Shared native schedule, spectral response and spatial marginal.
        values: Float64 (reads,) achieved-band radiances, W m^-2 sr^-1 per (cm^-1).
        valid: Boolean (reads,) support; invalid values are NaN.
        read_indices: Int64 valid native indices in increasing order.
        covariance: Covariance of valid reads only, dense or source-projected.
        bias: Optional signed radiance offsets in valid-read order, separate from covariance.
        parameter_names: Names defining sensitivity columns.
        parameter_units: Units defining sensitivity columns and input parameter errors.
        parameter_sensitivity: Float64 (valid reads,parameters) radiance/parameter derivatives.
    """

    response: NativeBandResponse
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    read_indices: NDArray[np.int64]
    covariance: NDArray[np.float64] | ProjectedCovariance
    bias: NDArray[np.float64] | None
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameter_sensitivity: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class ObservationCube:
    """Logical line/sample/band radiances with native supports and error dependencies.

    Factory arrays own writable storage; geometry and response are shared. Direct
    construction borrows inputs. Keep all buffers stable during use. Equal spatial
    indices across bands need not refer to the same ground support. Native response
    margins do not define a general joint spectral-spatial kernel.

    Attributes:
        values: Float64 (line,sample,band) W m^-2 sr^-1 per (cm^-1), with NaN gaps.
        valid: Boolean mask with the values shape.
        observation_indices: Int64 valid flattened C-order indices, shared by all error arrays.
        band_names: Ordered physical channel labels.
        geometry: Shared supplied native coordinates, bounds, times, rays and read identities.
        response: Shared native spectral coefficients and spatial response marginal.
        native_weights: CSR (valid observations,native reads) fixed dimensionless
            nonnegative normalized contribution weights, preserving native column order.
        covariance: Dense or source-projected valid-observation covariance,
            in squared radiance units.
        bias: Optional signed offsets in valid-observation order.
        parameter_names: Calibration parameter names in sensitivity-column order.
        parameter_units: Units of calibration parameters.
        parameter_sensitivity: Float64 (valid observations,parameters) derivatives,
            in radiance/parameter units.
    """

    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    observation_indices: NDArray[np.int64]
    band_names: tuple[str, ...]
    geometry: SurfaceAcquisitionGrid
    response: NativeBandResponse
    native_weights: csr_array
    covariance: NDArray[np.float64] | ProjectedCovariance
    bias: NDArray[np.float64] | None
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameter_sensitivity: NDArray[np.float64]


def calibrate_observations(
    images: ImageMeasurements,
    response: NativeBandResponse,
    *,
    detector: DetectorProperties,
    read_covariance: NDArray[np.float64] | ProjectedCovariance,
    parameters: RadiometricParameterMaps | None = None,
    parameter_covariance: NDArray[np.float64] | None = None,
    read_parameter_cross_covariance: NDArray[np.float64] | Literal["zero"] | None = None,
    representation: Literal["dense", "projected"] = "projected",
    max_dense_bytes: int = 8_000_000,
    realisation: int = 0,
    read_valid: NDArray[np.bool_] | None = None,
    mask_flagged: bool = False,
    use_expectation: bool = False,
    read_bias: NDArray[np.float64] | None = None,
    parameter_bias: NDArray[np.float64] | None = None,
) -> CalibratedObservations:
    """Correct image/DN values and propagate joint read and calibration errors once.

    For v=(g*(d-o)-dark*time)/s, derivatives are A=g/s and
    B=((d-o)*dg-g*do-v*ds)/s. Dark calibration is fixed. Expected-electron
    input uses A=1/s and permits only normalization derivatives. Geometry and
    supplied spectral/spatial supports are fixed. Read covariance must exclude
    the calibration-parameter contribution supplied here to avoid double counting.
    Fixed calibration has zero parameter columns. For uncertain calibration,
    explicit 'zero' declares uncorrelated read and parameter errors. A matrix
    declaration requires dense/diagonal read input and validates a joint dense
    source before transport, including both cross terms. Its allocation is
    checked against max_dense_bytes separately from output materialization;
    the budget excludes peak process memory.

    Args:
        images: Native electrons and optional DN/flags; one realization is selected.
        response: Supplied aligned native spectral/spatial response and read schedule.
        detector: Supplied fixed nominal electronic calibration, including dark rate.
        read_covariance: Float64 native variances, dense or projected covariance, DN squared
            for DN input or electrons squared for expected charge, including invalid reads.
        parameters: Optional named physical gain, pedestal and normalization derivatives.
            Omit for fixed calibration with zero uncertain parameters.
        parameter_covariance: Parameter variances or dense covariance in map order;
            omit for fixed calibration or supply an empty float64 array.
        read_parameter_cross_covariance: Explicit 'zero' or finite float64
            (native reads,parameters) cross covariance in read units times parameter units.
            May be omitted for fixed calibration only.
        representation: 'projected' preserves source blocks; 'dense' materializes their sum.
        max_dense_bytes: Nonnegative budget per explicit dense joint/output allocation.
        realisation: Nonnegative realization index, zero for expected charge.
        read_valid: Optional Boolean native quality mask, fixed independently of noise.
        mask_flagged: Exclude reads flagged in any realization instead of rejecting them.
        use_expectation: Use pre-dark expected electrons even when DN are present.
        read_bias: Optional finite float64 signed (reads,) read offsets.
        parameter_bias: Optional finite float64 signed (parameters,) parameter offsets.

    Returns:
        One calibrated realization with valid-read covariance, bias and derivatives.

    Raises:
        ValueError: Maps, units, covariance, masks, support, representation or calibration
            disagree, or a transported bias or its sum is nonfinite.
        MemoryError: Requested joint source or dense output exceeds its allocation budget.
    """
    check_raster_alignment(images.schedule, response.schedule)
    n = len(response.reads.sample_ids)
    sensitivity = response.sensitivity
    supported = sensitivity > 0
    if read_valid is not None:
        if read_valid.dtype != np.bool_ or read_valid.shape != (n,):
            raise ValueError("read validity must be Boolean (reads,)")
        supported &= read_valid
    charge = calibrate_frame_electrons(
        images,
        detector=detector,
        mask_flagged=mask_flagged,
        use_expectation=use_expectation,
        read_valid=supported,
    )
    if not 0 <= realisation < charge.values.shape[0]:
        raise ValueError("realisation is out of range")
    gain, offset, dark = np.ones(n), np.zeros(n), np.zeros(n)
    if images.dn is None or use_expectation:
        if parameters is not None and (np.any(parameters.gain) or np.any(parameters.offset)):
            raise ValueError("expected charge supports normalization derivatives only")
        readings = charge.values[realisation]
    else:
        _, row, col = images.schedule.native_indices.T
        shape = images.schedule.native_shape[1:]
        gain = np.broadcast_to(detector.electrons_per_dn, shape)[row, col].astype(np.float64)
        offset = np.broadcast_to(detector.bias_dn, shape)[row, col].astype(np.float64)
        duration = np.diff(response.reads.exposure_bounds_s, axis=1).ravel()
        dark = np.broadcast_to(detector.dark_electrons_per_s, shape)[row, col] * duration
        readings = images.dn.reshape(-1, n)[realisation].astype(np.float64)
    numerical = calibrate_readings(
        readings,
        sensitivity=sensitivity,
        read_covariance=read_covariance,
        gain=gain,
        offset=offset,
        dark_electrons=dark,
        parameters=parameters,
        parameter_covariance=parameter_covariance,
        read_parameter_cross_covariance=read_parameter_cross_covariance,
        representation=representation,
        max_dense_bytes=max_dense_bytes,
        read_valid=charge.valid[realisation],
        read_bias=read_bias,
        parameter_bias=parameter_bias,
    )
    return CalibratedObservations(
        response,
        numerical.values,
        numerical.valid,
        numerical.indices,
        numerical.covariance,
        numerical.bias,
        numerical.parameter_names,
        numerical.parameter_units,
        numerical.parameter_sensitivity,
    )


def form_observation_cube(
    calibrated: CalibratedObservations,
    geometry: SurfaceAcquisitionGrid,
    *,
    shape: tuple[int, int, int],
    band_names: tuple[str, ...],
    native_weights: csr_array,
    response_tolerance: float = 1e-8,
    max_dense_bytes: int = 8_000_000,
) -> ObservationCube:
    """Organize or accumulate native reads with supplied fixed within-band weights.

    Rows address all logical C-order cube cells; columns address all native reads.
    Empty rows become invalid NaN cells. Every nonempty row must sum to one and
    refer only to valid reads of that band's compatible achieved response. Invalid
    contributions are rejected; callers must explicitly recalculate weights after
    mask changes. The function estimates neither weights nor registration.

    Args:
        calibrated: One native calibrated realization with valid-read errors.
        geometry: Aligned native geometry with identical read identities and sensor quadrature.
        shape: Positive logical (line,sample,band) dimensions, independent of ground coordinates.
        band_names: Unique nonempty labels in cube band order.
        native_weights: Float64 CSR (prod(shape),native reads) nonnegative normalized weights.
        response_tolerance: Finite nonnegative pairwise normalized spectral L1 tolerance per band.
        max_dense_bytes: Dense output allocation budget when the input covariance is dense.

    Returns:
        Cube retaining shared native geometry/response and owned valid-row contribution map.

    Raises:
        ValueError: Geometry, logical shape, weights, masks or spectral compatibility
            are invalid, or input or transported bias is nonfinite.
        MemoryError: Dense covariance or dense accumulation workspace exceeds its budget.
    """
    response = calibrated.response
    check_raster_alignment(geometry.schedule, response.schedule)
    rays = geometry.rays.acquisition
    if (
        rays.sample_ids != response.reads.sample_ids
        or rays.pixel_ids != response.reads.pixel_ids
        or rays.setting_ids != response.reads.setting_ids
        or rays.port_ids != response.reads.port_ids
        or not np.array_equal(rays.exposure_bounds_s, response.reads.exposure_bounds_s)
        or not np.array_equal(rays.times_s, response.node_times_s)
        or not np.array_equal(rays.directions_uv, response.node_directions_uv)
        or geometry.xy_m.shape != (*response.spatial_weights.shape, 2)
        or not np.isfinite(geometry.xy_m).all()
        or geometry.bounds_m.shape != (len(response.reads.sample_ids), 4)
        or not np.isfinite(geometry.bounds_m).all()
        or np.any(geometry.bounds_m[:, 0] > geometry.bounds_m[:, 1])
        or np.any(geometry.bounds_m[:, 2] > geometry.bounds_m[:, 3])
    ):
        raise ValueError("geometry must preserve native sensor quadrature and finite supports")
    if (
        len(shape) != 3
        or min(shape) < 1
        or shape[2] != len(band_names)
        or len(set(band_names)) != len(band_names)
        or not all(band_names)
    ):
        raise ValueError("positive line/sample/band shape and unique band names are required")
    if not np.isfinite(response_tolerance) or response_tolerance < 0 or max_dense_bytes < 0:
        raise ValueError("response tolerance and budget must be nonnegative")
    n = calibrated.values.size
    if (
        not isinstance(native_weights, csr_array)
        or native_weights.dtype != np.float64
        or native_weights.shape != (int(np.prod(shape)), n)
    ):
        raise ValueError("weights must be float64 CSR (logical cells,native reads)")
    weights = normalized_accumulation_weights(native_weights, n)
    if np.any(~calibrated.valid[weights.indices]):
        raise ValueError("weights include invalid reads; recompute normalized weights explicitly")
    row_indices = np.repeat(
        np.arange(cast(tuple[int, int], weights.shape)[0]), np.diff(weights.indptr)
    )
    names = np.array(response.read_bands)
    if np.any(names[weights.indices] != np.array(band_names)[row_indices % shape[2]]):
        raise ValueError("accumulation must stay within each declared band")
    table = response.normalized_responses
    for name in band_names:
        selected = np.unique(weights.indices[names[weights.indices] == name]).astype(
            np.int64, copy=False
        )
        rows = np.unique(response.response_rows(selected))
        # Exact deduplication preserves the pairwise contract for repeated responses.
        normalized = np.unique(table[rows], axis=0)
        for i, reference in enumerate(normalized):
            if np.any(np.sum(np.abs(normalized[i + 1 :] - reference), axis=1) > response_tolerance):
                raise ValueError("within-band achieved spectral responses are incompatible")
    numerical = accumulate_observations(
        RadiometricObservations(
            calibrated.values,
            calibrated.valid,
            calibrated.read_indices,
            calibrated.covariance,
            calibrated.bias,
            calibrated.parameter_names,
            calibrated.parameter_units,
            calibrated.parameter_sensitivity,
        ),
        weights,
        max_dense_bytes=max_dense_bytes,
    )
    return ObservationCube(
        numerical.values.reshape(shape),
        numerical.valid.reshape(shape),
        numerical.indices,
        band_names,
        geometry,
        response,
        cast(csr_array, weights[numerical.indices, :]),
        numerical.covariance,
        numerical.bias,
        calibrated.parameter_names,
        calibrated.parameter_units,
        numerical.parameter_sensitivity,
    )
