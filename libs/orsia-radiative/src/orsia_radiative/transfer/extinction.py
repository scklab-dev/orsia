"""Evaluate spectral extinction means and their explicit local log linearization.

All operations use supplied air masses and nonnegative averaging weights. Source
radiance carries the caller's spectral-density coordinate and units unchanged.
Read selection, band integration and fitted parameter coordinates belong to callers.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

type FloatArray = NDArray[np.float64]


def exposure_mean_radiance(
    source_radiance: FloatArray,
    optical_depth: FloatArray,
    air_masses: FloatArray,
    averaging_weights: FloatArray,
) -> FloatArray:
    """Average attenuated spectral radiance over explicitly weighted samples.

    Args:
        source_radiance: Nonnegative unattenuated spectrum (spectral,), in any
            declared radiance-density units, including surface reflection if needed.
        optical_depth: Matching nonnegative vertical extinction spectrum.
        air_masses: Positive finite dimensionless path factors (samples,).
        averaging_weights: Nonnegative finite (samples,) weights with positive sum.
            Weights are normalized internally; their overall scale cancels.

    Returns:
        Fresh spectral radiance in the source units. Passing the weighted mean air
        mass as one sample selects the distinct mean-air-mass approximation.

    Raises:
        ValueError: Spectral axes, values, air masses or averaging weights are invalid.
    """
    log_radiance, _ = _log_mean(source_radiance, optical_depth, air_masses, averaging_weights)
    return np.exp(log_radiance)


def exposure_mean_radiance_tangent(
    source_radiance: FloatArray,
    optical_depth: FloatArray,
    air_masses: FloatArray,
    averaging_weights: FloatArray,
    *,
    source_radiance_tangent: FloatArray,
    optical_depth_tangent: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Propagate source and extinction directions at fixed masses and weights.

    Args:
        source_radiance: Nonnegative unattenuated spectrum (spectral,).
        optical_depth: Matching nonnegative vertical extinction spectrum.
        air_masses: Positive dimensionless factors (samples,).
        averaging_weights: Nonnegative (samples,) averaging weights with positive sum.
        source_radiance_tangent: Finite signed (spectral,direction) source seeds.
        optical_depth_tangent: Matching signed vertical-depth seeds.

    Returns:
        Fresh mean spectrum and (spectral,direction) derivatives in source units
        per caller parameter. Source derivatives remain defined at zero source.
        Mass, weight, geometry and time perturbations are outside this operation.

    Raises:
        ValueError: Physical inputs, seed shapes or finite arithmetic are invalid.
    """
    log_transmission, effective = _log_mean(
        np.ones_like(source_radiance), optical_depth, air_masses, averaging_weights
    )
    # Validate the actual source independently, including its spectral axes.
    log_value, _ = _log_mean(source_radiance, optical_depth, air_masses, averaging_weights)
    ds, dt = source_radiance_tangent, optical_depth_tangent
    if (
        ds.ndim != 2
        or ds.shape[0] != source_radiance.size
        or dt.shape != ds.shape
        or not np.isfinite(ds).all()
        or not np.isfinite(dt).all()
    ):
        raise ValueError("matching finite (spectral,direction) tangent arrays required")
    value = np.exp(log_value)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # Retain signed logarithmic products until cancellation and final scaling.
        # Transmission alone can underflow while its product with a seed is finite.
        source_term = log_transmission[:, None] + np.log(np.abs(ds))
        depth_term = (log_value + np.log(effective))[:, None] + np.log(np.abs(dt))
        log_magnitude, sign = logsumexp(
            np.stack((source_term, depth_term)),
            b=np.stack((np.sign(ds), -np.sign(dt))),
            axis=0,
            return_sign=True,
        )
        tangent = sign * np.exp(log_magnitude)
    if not np.isfinite(tangent).all():
        raise ValueError("extinction tangent arithmetic exceeds finite support")
    return value, tangent


def linearize_exposure_mean_radiance(
    source_radiance: FloatArray,
    reference_optical_depth: FloatArray,
    air_masses: FloatArray,
    averaging_weights: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Prepare a local log-tangent approximation to an exposure-mean spectrum.

    Args:
        source_radiance: Nonnegative unattenuated spectral radiance (spectral,).
        reference_optical_depth: Matching nonnegative reference vertical extinction.
        air_masses: Positive dimensionless path factors (samples,).
        averaging_weights: Nonnegative (samples,) weights with positive sum.

    Returns:
        Exact reference radiance, effective log source and tangent air mass, each
        (spectral,). The local model is exp(log_source - mass * depth). It matches
        the reference value and first depth derivative. Off-reference values are
        approximate. A zero source gives zero radiance and negative-infinite log
        source. Numerically fully opaque samples use zero effective mass, because
        both reference radiance and its representable depth derivative are zero.

    Raises:
        ValueError: Inputs or finite log-linearization arithmetic are invalid.
    """
    log_value, effective = _log_mean(
        source_radiance, reference_optical_depth, air_masses, averaging_weights
    )
    with np.errstate(over="ignore", invalid="ignore"):
        intercept = log_value + effective * reference_optical_depth
    if np.isnan(intercept).any() or np.isposinf(intercept).any():
        raise ValueError("extinction log-linearization exceeds finite support")
    return np.exp(log_value), intercept, effective


def _log_mean(source, tau, masses, weights):
    if (
        source.ndim != 1
        or not source.size
        or tau.shape != source.shape
        or masses.ndim != 1
        or not masses.size
        or weights.shape != masses.shape
        or any(not np.isfinite(a).all() for a in (source, tau, masses, weights))
        or np.any(source < 0)
        or np.any(tau < 0)
        or np.any(masses <= 0)
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError("finite aligned spectra, positive masses and nonnegative weights required")
    admitted = weights > 0
    mass = masses[admitted]
    log_weights = np.log(weights[admitted])
    log_weights -= log_weights.max()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exponents = log_weights[:, None] - mass[:, None] * tau
        log_sum = logsumexp(exponents, axis=0)
        log_transmission = log_sum - logsumexp(log_weights)
        opaque = np.isneginf(log_sum)
        posterior = np.exp(exponents - np.where(opaque, 0, log_sum))
        effective = np.sum(posterior * mass[:, None], axis=0)
        log_value = np.log(source) + log_transmission
    return log_value, effective
