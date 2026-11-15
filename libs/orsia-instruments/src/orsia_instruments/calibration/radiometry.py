"""Calibrate and accumulate radiances independently of detector geometry.

Inputs describe one fixed-support linear calibration. Callers retain physical
read identity, spectral compatibility, geometry and clipping/nonlinearity checks.
All covariance coordinates follow the supplied read order. Shared parameter
columns describe the same error source across every read, including read blocks."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import propagate_covariance, validate_covariance
from orsia_uncertainty.projection import (
    ProjectedCovariance,
    materialize_covariance,
    transport_projected_covariance,
)
from scipy.sparse import csr_array, hstack


@dataclass(frozen=True, eq=False)
class RadiometricParameterMaps:
    """Derivatives of nominal radiometric calibration with respect to named errors.

    Arrays are borrowed and must remain stable during use. A column denotes one
    shared parameter across all reads, including separately acquired blocks.

    Attributes:
        names: Unique nonempty parameter names in covariance-column order; may be empty.
        units: Parameter units, including '1' for dimensionless errors.
        gain: Float64 (reads,parameters) derivatives of electrons per input unit.
        offset: Same-shaped derivatives of the pedestal in input units.
        normalization: Same-shaped derivatives of sensitivity in electrons per
            unit radiance. These affect scalar normalization, with fixed band shape.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    gain: NDArray[np.float64]
    offset: NDArray[np.float64]
    normalization: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class RadiometricObservations:
    """Radiance values with fixed validity and first-order error dependencies.

    Factories own writable arrays. Direct construction borrows inputs; keep all
    buffers stable during use. Covariance, bias and parameter derivatives use
    valid-coordinate order. Geometry and spectral response remain caller-owned.

    Attributes:
        values: Float64 (observations,) radiance, with NaN at invalid coordinates.
        valid: Boolean (observations,) support independent of measurement noise.
        indices: Int64 increasing valid coordinates in the full value vector.
        covariance: Dense or projected covariance of valid coordinates in radiance squared.
        bias: Optional signed radiance offsets in valid-coordinate order.
        parameter_names: Shared calibration parameter names in derivative-column order.
        parameter_units: Units of those parameter errors.
        parameter_sensitivity: Float64 (valid observations,parameters) radiance/parameter.
    """

    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    indices: NDArray[np.int64]
    covariance: NDArray[np.float64] | ProjectedCovariance
    bias: NDArray[np.float64] | None
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameter_sensitivity: NDArray[np.float64]


def calibrate_readings(
    readings: NDArray[np.float64],
    *,
    sensitivity: NDArray[np.float64],
    read_covariance: NDArray[np.float64] | ProjectedCovariance,
    gain: float | NDArray[np.float64] = 1.0,
    offset: float | NDArray[np.float64] = 0.0,
    dark_electrons: float | NDArray[np.float64] = 0.0,
    parameters: RadiometricParameterMaps | None = None,
    parameter_covariance: NDArray[np.float64] | None = None,
    read_parameter_cross_covariance: NDArray[np.float64] | Literal["zero"] | None = None,
    read_valid: NDArray[np.bool_] | None = None,
    read_bias: NDArray[np.float64] | None = None,
    parameter_bias: NDArray[np.float64] | None = None,
    representation: Literal["dense", "projected"] = "projected",
    max_dense_bytes: int = 8_000_000,
) -> RadiometricObservations:
    """Apply supplied linear calibration and transport read/parameter errors once.

    For v=(g*(d-o)-dark)/s, derivatives are A=g/s and
    B=((d-o)*dg-g*do-v*ds)/s. Negative radiances are valid. Zero sensitivity
    or an explicit false mask excludes a read. Fixed calibration omits parameters
    and their covariance, yielding zero derivative columns and no dummy source.
    Named uncertain parameters require an explicit cross-covariance declaration.
    Read covariance must exclude their contribution to avoid double counting.

    Concatenate readings and derivative rows from separate blocks in caller order
    before calling this function. Use common parameter columns and one parameter
    covariance to preserve cross-block correlations. Projected read covariance
    can already retain shared sources across these blocks. Dense read/parameter
    cross covariance requires a bounded joint dense source; projected reads with
    such cross terms must instead be supplied as one joint source externally.

    Args:
        readings: Float64 (reads,) input values; finite on supported reads.
        sensitivity: Finite nonnegative float64 (reads,) electrons per unit radiance.
        read_covariance: Float64 variances/dense covariance or projected read errors,
            in squared input units, including invalid reads in original order.
        gain: Finite positive scalar or float64 (reads,) electrons per input unit.
        offset: Finite scalar or float64 (reads,) pedestal in input units.
        dark_electrons: Finite nonnegative scalar or float64 (reads,) fixed dark charge.
        parameters: Optional derivative maps; zero parameters represent fixed calibration.
        parameter_covariance: Parameter variances or dense covariance; omit for fixed
            calibration, or supply an empty float64 (0,) or (0,0) array.
        read_parameter_cross_covariance: Explicit 'zero' or finite float64
            (reads,parameters) matrix; may be omitted for fixed calibration only.
        read_valid: Optional Boolean (reads,) quality mask fixed independently of noise.
        read_bias: Optional finite float64 signed (reads,) offsets in input units.
        parameter_bias: Optional finite float64 signed (parameters,) offsets.
        representation: 'projected' retains source blocks; 'dense' materializes their sum.
        max_dense_bytes: Nonnegative budget per dense joint-source or output allocation,
            excluding source snapshots and validation workspaces.

    Returns:
        Owned radiances, validity, covariance, signed bias and parameter derivatives.

    Raises:
        ValueError: Shapes, numerical support, maps, covariance or declarations disagree,
            projected reads accompany a matrix cross declaration for uncertain
            parameters, or transported errors overflow.
        MemoryError: A requested dense allocation exceeds its budget.
    """
    if readings.dtype != np.float64 or readings.ndim != 1 or not readings.size:
        raise ValueError("readings must be a nonempty float64 vector")
    n = readings.size
    if representation not in ("dense", "projected") or max_dense_bytes < 0:
        raise ValueError("invalid representation or dense allocation budget")
    _finite_array(sensitivity, (n,), "sensitivity")
    if np.any(sensitivity < 0):
        raise ValueError("sensitivity must be nonnegative")
    g, o, dark = (
        _read_vector(value, n, name)
        for value, name in ((gain, "gain"), (offset, "offset"), (dark_electrons, "dark charge"))
    )
    if np.any(g <= 0) or np.any(dark < 0):
        raise ValueError("gain must be positive and dark charge nonnegative")
    valid = sensitivity > 0
    if read_valid is not None:
        if read_valid.dtype != np.bool_ or read_valid.shape != (n,):
            raise ValueError("read validity must be Boolean (reads,)")
        valid &= read_valid
    indices = np.flatnonzero(valid)
    if not indices.size:
        raise ValueError("at least one supported read is required")
    if not np.isfinite(readings[valid]).all():
        raise ValueError("supported readings must be finite")
    if parameters is None:
        parameters = RadiometricParameterMaps((), (), *(np.empty((n, 0)) for _ in range(3)))
    p = len(parameters.names)
    if (
        len(set(parameters.names)) != p
        or not all(parameters.names)
        or len(parameters.units) != p
        or not all(parameters.units)
    ):
        raise ValueError("calibration parameters need unique names and explicit units")
    for derivative in (parameters.gain, parameters.offset, parameters.normalization):
        _finite_array(derivative, (n, p), "calibration derivative")
    calibration = None
    if p:
        if parameter_covariance is None:
            raise ValueError("uncertain parameters require parameter covariance")
        calibration = validate_covariance(parameter_covariance)
        if calibration.shape[0] != p:
            raise ValueError("parameter covariance dimensions must match their maps")
    elif parameter_covariance is not None:
        if parameter_covariance.shape not in ((0,), (0, 0)):
            raise ValueError("fixed calibration requires empty parameter covariance")
        _finite_array(parameter_covariance, parameter_covariance.shape, "parameter covariance")
    norm = sensitivity[valid]
    values = np.full(n, np.nan)
    values[valid] = (g[valid] * (readings[valid] - o[valid]) - dark[valid]) / norm
    _finite_array(values[valid], (indices.size,), "calibrated radiances")
    a = g[valid] / norm
    b = (
        (readings[valid] - o[valid])[:, None] * parameters.gain[valid]
        - g[valid, None] * parameters.offset[valid]
        - values[valid, None] * parameters.normalization[valid]
    ) / norm[:, None]
    _finite_array(b, (indices.size, p), "calibration sensitivity")
    read_map = csr_array((a, (np.arange(indices.size), indices)), shape=(indices.size, n))
    projected = _calibration_covariance(
        read_covariance,
        read_map,
        b,
        calibration,
        read_parameter_cross_covariance,
        max_dense_bytes,
    )
    bias = None
    if read_bias is not None or parameter_bias is not None:
        bias = np.zeros(indices.size)
        if read_bias is not None:
            _finite_array(read_bias, (n,), "read bias")
            bias += read_map @ read_bias
        if parameter_bias is not None:
            _finite_array(parameter_bias, (p,), "parameter bias")
            bias += b @ parameter_bias
        _finite_array(bias, (indices.size,), "calibrated bias")
    covariance = (
        materialize_covariance(projected, max_bytes=max_dense_bytes)
        if representation == "dense"
        else projected
    )
    return RadiometricObservations(
        values, valid, indices, covariance, bias, parameters.names, parameters.units, b
    )


def coadd_readings(
    charge: NDArray[np.float64],
    *,
    sensitivity: NDArray[np.float64],
    read_variance: NDArray[np.float64],
    read_valid: NDArray[np.bool_],
    weights: NDArray[np.float64] | None = None,
    parameters: RadiometricParameterMaps | None = None,
    parameter_covariance: NDArray[np.float64] | None = None,
    read_parameter_cross_covariance: Literal["zero"] | None = None,
    read_bias: NDArray[np.float64] | None = None,
    parameter_bias: NDArray[np.float64] | None = None,
) -> RadiometricObservations:
    """Form sensitivity-normalized disjoint groups with compact independent errors.

    Args:
        charge: Corrected electron charge (...,read). Leading axes identify disjoint
            groups, flattened in C order in the result. Supported charge is finite.
        sensitivity: Nonnegative electrons per unit radiance, broadcastable to charge.
        read_variance: Independent electron variances with the charge shape. Shared
            calibration contributions must be excluded from these variances.
        read_valid: Boolean charge-shaped support fixed independently of noise.
        weights: Optional nonnegative charge-shaped fixed participation weights.
            Defaults to one; zero sensitivity, zero weight or false support excludes a read.
        parameters: Optional (charge.size,parameter) maps in C order. Gain is the
            derivative of a unit multiplicative charge calibration, offset is an
            electron pedestal derivative, and normalization is the derivative of
            the actual supplied sensitivity in electrons per unit radiance.
        parameter_covariance: One common parameter covariance, retained as one source.
        read_parameter_cross_covariance: Explicit 'zero' for uncertain parameters.
            Correlated read noise, overlapping groups and nonzero cross terms require
            the general projected calibration path instead of diagonal compression.
        read_bias: Optional charge-shaped signed electron bias.
        parameter_bias: Optional signed bias in the named parameter coordinates.

    Returns:
        Owned radiances sum(a*Q)/sum(a*s), support and first-order errors. The first
        covariance source is output-scale independent variance; the second, when
        present, is the common parameter source. No read-sized projections persist.
        Assemble parameter derivative rows against one source when joining blocks.
        The quotient derivative includes both summed charge and sensitivity changes.

    Raises:
        ValueError: Physical domains, shapes, covariance, cross declaration or support fail.
    """
    if charge.ndim < 1 or charge.dtype != np.float64 or not charge.size:
        raise ValueError("nonempty float64 charge with a final read axis required")
    shape = charge.shape
    if read_valid.shape != shape or read_valid.dtype != np.bool_:
        raise ValueError("Boolean charge-shaped read validity required")
    _finite_array(read_variance, shape, "independent electron variance")
    if np.any(read_variance < 0):
        raise ValueError("independent variance must be nonnegative")
    norm = np.broadcast_to(sensitivity, shape)
    _finite_array(norm, shape, "sensitivity")
    a = np.ones(shape) if weights is None else weights
    _finite_array(a, shape, "participation weights")
    if np.any(norm < 0) or np.any(a < 0):
        raise ValueError("sensitivity and participation weights must be nonnegative")
    valid = read_valid & (norm > 0) & (a > 0)
    if not np.isfinite(charge[valid]).all():
        raise ValueError("supported charge must be finite")
    a = np.where(valid, a, 0.0)
    q = np.where(valid, charge, 0.0)
    numerator = np.sum(a * q, axis=-1).reshape(-1)
    denominator = np.sum(a * norm, axis=-1).reshape(-1)
    variance = np.sum(a**2 * read_variance, axis=-1).reshape(-1)
    maps = None
    if parameters is not None:
        p = len(parameters.names)
        for array in (parameters.gain, parameters.offset, parameters.normalization):
            _finite_array(array, (charge.size, p), "calibration derivative")
        dq = q[..., None] * parameters.gain.reshape((*shape, p))
        dq -= parameters.offset.reshape((*shape, p))
        ds = parameters.normalization.reshape((*shape, p))
        maps = RadiometricParameterMaps(
            parameters.names,
            parameters.units,
            np.zeros((numerator.size, p)),
            -np.sum(a[..., None] * dq, axis=-2).reshape((numerator.size, p)),
            np.sum(a[..., None] * ds, axis=-2).reshape((numerator.size, p)),
        )
    bias = None
    if read_bias is not None:
        _finite_array(read_bias, shape, "read bias")
        bias = np.sum(a * read_bias, axis=-1).reshape(-1)
    if read_parameter_cross_covariance not in (None, "zero"):
        raise ValueError("compact coaddition requires independent reads and parameters")
    return calibrate_readings(
        numerator,
        sensitivity=denominator,
        read_covariance=variance,
        parameters=maps,
        parameter_covariance=parameter_covariance,
        read_parameter_cross_covariance=read_parameter_cross_covariance,
        read_bias=bias,
        parameter_bias=parameter_bias,
    )


def accumulate_observations(
    observations: RadiometricObservations,
    weights: csr_array,
    *,
    max_dense_bytes: int = 8_000_000,
) -> RadiometricObservations:
    """Accumulate fixed normalized weights while preserving error-source correlations.

    Rows address output coordinates; columns address the full input value vector.
    Empty rows become invalid NaN outputs. Nonempty rows must sum to one and use
    valid inputs only. This operation leaves spectral and spatial compatibility
    checks to the caller. It neither estimates weights nor repairs invalid support.

    Args:
        observations: Stable radiances and valid-coordinate error arrays.
        weights: Float64 CSR (outputs,input observations) nonnegative normalized map.
        max_dense_bytes: Nonnegative per-allocation budget for dense output covariance
            and dense accumulation workspace; unused for projected output storage.

    Returns:
        Owned output values and error arrays, retaining calibration-column identity.

    Raises:
        ValueError: Weights, validity, coordinate order or error support are invalid,
            or transported values, bias or sensitivities are nonfinite.
        MemoryError: Dense accumulation workspace or output covariance exceeds its budget.
    """
    n = observations.values.size
    if (
        observations.values.ndim != 1
        or observations.values.dtype != np.float64
        or observations.valid.dtype != np.bool_
        or observations.valid.shape != (n,)
        or observations.indices.dtype != np.int64
        or not np.array_equal(observations.indices, np.flatnonzero(observations.valid))
        or np.any(~np.isnan(observations.values[~observations.valid]))
        or max_dense_bytes < 0
    ):
        raise ValueError("observation support and increasing valid coordinates must agree")
    _finite_array(
        observations.values[observations.valid], (observations.indices.size,), "radiances"
    )
    weights = normalized_accumulation_weights(weights, n)
    if np.any(~observations.valid[weights.indices]):
        raise ValueError("weights include invalid reads; recompute normalized weights explicitly")
    supported = np.diff(weights.indptr) > 0
    indices = np.flatnonzero(supported)
    retained = cast(csr_array, weights[indices, :])
    transform = cast(csr_array, retained[:, observations.indices])
    values = np.full(cast(tuple[int, int], weights.shape)[0], np.nan)
    values[indices] = transform @ observations.values[observations.indices]
    _finite_array(values[indices], (indices.size,), "accumulated radiances")
    if isinstance(observations.covariance, ProjectedCovariance):
        covariance = transport_projected_covariance(observations.covariance, transform)
    else:
        if max(indices.size**2, indices.size * observations.indices.size) * 8 > max_dense_bytes:
            raise MemoryError("dense accumulation workspace or covariance exceeds max_dense_bytes")
        covariance = propagate_covariance(observations.covariance, transform.toarray())
    bias = None
    if observations.bias is not None:
        _finite_array(observations.bias, (observations.indices.size,), "calibrated bias")
        bias = transform @ observations.bias
        _finite_array(bias, (indices.size,), "accumulated bias")
    p = len(observations.parameter_names)
    _finite_array(
        observations.parameter_sensitivity, (observations.indices.size, p), "parameter sensitivity"
    )
    sensitivity = transform @ observations.parameter_sensitivity
    _finite_array(sensitivity, (indices.size, p), "accumulated parameter sensitivity")
    return RadiometricObservations(
        values,
        supported,
        indices,
        covariance,
        bias,
        observations.parameter_names,
        observations.parameter_units,
        sensitivity,
    )


def normalized_accumulation_weights(weights: csr_array, input_size: int) -> csr_array:
    """Snapshot a fixed nonnegative accumulation map and validate normalized support.

    Args:
        weights: Float64 CSR (outputs,input_size), with optional empty rows.
        input_size: Number of input coordinates, including invalid inputs.

    Returns:
        Owned canonical CSR map with duplicate columns summed and explicit zeros removed.

    Raises:
        ValueError: Shape, dtype or entries are invalid, no row is supported, or a
            nonempty row does not sum to one within rtol=1e-12 and atol=1e-14.
    """
    if (
        not isinstance(weights, csr_array)
        or weights.dtype != np.float64
        or weights.ndim != 2
        or cast(tuple[int, int], weights.shape)[1] != input_size
        or not cast(tuple[int, int], weights.shape)[0]
    ):
        raise ValueError("weights must be float64 CSR (output coordinates,input observations)")
    result = weights.copy()
    if not np.isfinite(result.data).all() or np.any(result.data < 0):
        raise ValueError("weights must be finite and nonnegative")
    result.sum_duplicates()
    result.eliminate_zeros()
    sums = np.asarray(result.sum(axis=1)).ravel()
    supported = sums > 0
    if not supported.any() or not np.allclose(sums[supported], 1, rtol=1e-12, atol=1e-14):
        raise ValueError("nonempty accumulation rows must sum to one")
    return result


def _calibration_covariance(
    reads: NDArray[np.float64] | ProjectedCovariance,
    read_map: csr_array,
    b: NDArray[np.float64],
    calibration: NDArray[np.float64] | None,
    cross: NDArray[np.float64] | Literal["zero"] | None,
    max_dense_bytes: int,
) -> ProjectedCovariance:
    """Transport read errors and one common calibration source in supplied order.

    Returns:
        Projected valid-read covariance with retained independent or joint sources.

    Raises:
        ValueError: Source dimensions or cross-covariance declarations disagree.
        MemoryError: The joint dense source exceeds its allocation budget.
    """
    n, p = cast(tuple[int, int], read_map.shape)[1], b.shape[1]
    if isinstance(reads, ProjectedCovariance):
        if cast(tuple[int, int], reads.projections[0].shape)[0] != n:
            raise ValueError("read covariance dimensions must match their maps")
    else:
        reads = validate_covariance(reads)
        if reads.shape[0] != n:
            raise ValueError("read covariance dimensions must match their maps")
    if cross is None:
        if p:
            raise ValueError("declare cross covariance explicitly as 'zero' or a matrix")
        cross = "zero"
    if isinstance(cross, str):
        if cross != "zero":
            raise ValueError("declare cross covariance explicitly as 'zero' or a matrix")
    else:
        _finite_array(cross, (n, p), "read/parameter cross covariance")
        if not p:
            cross = "zero"
    if isinstance(cross, str):
        projected = (
            transport_projected_covariance(reads, read_map)
            if isinstance(reads, ProjectedCovariance)
            else ProjectedCovariance((read_map,), (reads,))
        )
        if calibration is not None:
            projected = ProjectedCovariance(
                (*projected.projections, b), (*projected.source_covariances, calibration)
            )
        return projected
    if isinstance(reads, ProjectedCovariance):
        raise ValueError("projected reads with cross covariance require an external joint source")
    if (n + p) ** 2 * 8 > max_dense_bytes:
        raise MemoryError("joint source covariance exceeds max_dense_bytes")
    joint = np.zeros((n + p, n + p))
    if reads.ndim == 1:
        joint[np.arange(n), np.arange(n)] = reads
    else:
        joint[:n, :n] = reads
    parameter_covariance = cast(NDArray[np.float64], calibration)
    if parameter_covariance.ndim == 1:
        joint[n + np.arange(p), n + np.arange(p)] = parameter_covariance
    else:
        joint[n:, n:] = parameter_covariance
    joint[:n, n:], joint[n:, :n] = cross, cross.T
    return ProjectedCovariance((csr_array(hstack((read_map, csr_array(b)))),), (joint,))


def _read_vector(value: float | NDArray[np.float64], n: int, name: str) -> NDArray[np.float64]:
    """Broadcast a finite scalar or validate a full physical read map.

    Returns:
        Float64 read vector, possibly a borrowed broadcast view.
    """
    result = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,))
    _finite_array(result, (n,), name)
    return result


def _finite_array(value: NDArray[np.float64], shape: tuple[int, ...], name: str) -> None:
    """Require finite float64 support for a numerical array.

    Raises:
        ValueError: Dtype, shape or finiteness differs from the declared support.
    """
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite float64 with shape {shape}")
