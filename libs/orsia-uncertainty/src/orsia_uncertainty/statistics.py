"""Summarize sampled probabilities and finite-interval metric sensitivities."""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from orsia_uncertainty.covariance import factor_covariance

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]


class FiniteIntervalSensitivity(TypedDict):
    """Finite-interval metric slope and dimensionless elasticity.

    Attributes:
        lower_metric: Metric at the lower sampled parameter, caller metric units.
        nominal_metric: Metric at the nominal parameter, same units.
        upper_metric: Metric at the upper sampled parameter, same units.
        ideal_metric: Optional comparison metric in the same units.
        slope: Metric change divided by parameter change, metric/parameter units.
        elasticity: Nominal parameter times slope divided by nominal metric;
            None when either nominal value is zero.
    """

    lower_metric: float
    nominal_metric: float
    upper_metric: float
    ideal_metric: float | None
    slope: float
    elasticity: float | None


def finite_interval_sensitivity(
    nominal_parameter: float,
    lower_parameter: float,
    upper_parameter: float,
    nominal_metric: float,
    lower_metric: float,
    upper_metric: float,
    ideal_metric: float | None = None,
) -> FiniteIntervalSensitivity:
    """Return the source finite-interval slope and dimensionless elasticity.

    Parameter and metric units follow supplied values. Ideal metric is optional.
    Distinct finite ordered parameter bounds are required. Zero nominal parameter
    or metric makes elasticity unavailable (None), preserving source semantics.

    Args:
        nominal_parameter: Parameter at the nominal evaluation, caller units.
        lower_parameter: Lower finite sampled parameter.
        upper_parameter: Upper finite sampled parameter, strictly above lower.
        nominal_metric: Nominal finite metric, caller units.
        lower_metric: Finite metric at lower_parameter.
        upper_metric: Finite metric at upper_parameter.
        ideal_metric: Optional comparison metric in the same units.

    Returns:
        Endpoint metrics, finite-interval slope and dimensionless elasticity. Elasticity
        is None when the nominal parameter or nominal metric is zero.

    Raises:
        ValueError: Sampled parameters/metrics are nonfinite or bounds are unordered.
    """
    if not lower_parameter < upper_parameter or not all(
        np.isfinite(x)
        for x in (
            nominal_parameter,
            lower_parameter,
            upper_parameter,
            nominal_metric,
            lower_metric,
            upper_metric,
        )
    ):
        raise ValueError("finite metrics and ordered distinct parameter bounds required")
    slope = (upper_metric - lower_metric) / (upper_parameter - lower_parameter)
    return dict(
        lower_metric=lower_metric,
        nominal_metric=nominal_metric,
        upper_metric=upper_metric,
        ideal_metric=ideal_metric,
        slope=slope,
        elasticity=None
        if nominal_parameter == 0 or nominal_metric == 0
        else nominal_parameter * slope / nominal_metric,
    )


def half_probability_crossing(
    rates: FloatArray, probability: FloatArray
) -> tuple[float | None, list[float] | None]:
    """Interpolate the first half-probability crossing and its sampled rate bracket.

    Args:
        rates: Ordered sampled amplitude/rate coordinates, caller units.
        probability: Sampled detection probabilities on the same axis.

    Returns:
        First linearly interpolated half-probability crossing and sampled coordinate
        bracket, or (None,None) when no interior first crossing is present.
    """
    crossing = np.flatnonzero(probability >= 0.5)
    if not len(crossing) or crossing[0] == 0:
        return None, None
    upper = int(crossing[0])
    lower = upper - 1
    value = float(
        rates[lower]
        + (0.5 - probability[lower])
        / (probability[upper] - probability[lower])
        * (rates[upper] - rates[lower])
    )
    return value, [float(rates[lower]), float(rates[upper])]


def wilson_interval(successes: int, count: int) -> list[float]:
    """Compute the two-sided 95 percent Wilson binomial interval.

    Args:
        successes: Integer successes between zero and count.
        count: Positive number of independent Bernoulli observations.

    Returns:
        Lower and upper probability limits clipped to [0,1].

    Raises:
        ValueError: Count or successes is outside its binomial domain.
    """
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("positive sample count and valid successes required")
    probability = successes / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    centre = (probability + z * z / (2 * count)) / denominator
    half = (
        z
        / denominator
        * np.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count))
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


class ProjectedNoiseStatistics(TypedDict):
    """Gaussian projected-noise population summary in caller target units.

    Attributes:
        sample_count: Number of independent Gaussian draws.
        analytical_sigma: Positive target RMS from the declared covariance.
        empirical_sigma: Sample RMS with ddof=1 in target units.
        sigma_ratio: Empirical-to-analytical RMS ratio.
        normalized_mean: Empirical mean divided by analytical_sigma.
        coverage_one_sigma: Fraction inside plus/minus one declared sigma.
        coverage_two_sigma: Fraction inside plus/minus two declared sigmas.
    """

    sample_count: int
    analytical_sigma: float
    empirical_sigma: float
    sigma_ratio: float
    normalized_mean: float
    coverage_one_sigma: float
    coverage_two_sigma: float


def projected_gaussian_statistics(
    weights: FloatArray,
    covariance: FloatArray,
    rng: np.random.Generator,
    *,
    sample_count: int,
) -> ProjectedNoiseStatistics:
    """Summarize explicitly sampled linear Gaussian uncertainty without acceptance policy.

    Args:
        weights: Target-unit per measurement-unit projection vector, (measurement,).
        covariance: Positive-semidefinite covariance in squared measurement units.
        rng: Explicit generator; draws use batches of at most 4096 sample rows.
        sample_count: Number of independent draws, at least two.

    Returns:
        Analytical/empirical RMS, normalized mean and one/two-sigma coverage. These
        condition on the supplied Gaussian law and establish no physical-model accuracy.

    Raises:
        ValueError: Axes, covariance, sample count or positive projected variance fail.
    """
    if weights.ndim != 1 or covariance.shape != (len(weights), len(weights)) or sample_count < 2:
        raise ValueError("aligned covariance and at least two projected samples required")
    factored = factor_covariance(covariance, method="eigh")
    factor = np.sqrt(factored.scale) * factored.root
    sigma = np.sqrt(weights @ covariance @ weights)
    if not sigma > 0:
        raise ValueError("positive projected variance required")
    # Bounded batches preserve sample population without an N-by-native-read allocation.
    draws = []
    for first in range(0, sample_count, 4096):
        normals = rng.standard_normal((min(4096, sample_count - first), factor.shape[1]))
        draws.append(normals @ (weights @ factor))
    values = np.concatenate(draws)
    empirical = values.std(ddof=1)
    normalized_mean = values.mean() / sigma

    return dict(
        sample_count=sample_count,
        analytical_sigma=float(sigma),
        empirical_sigma=float(empirical),
        sigma_ratio=float(empirical / sigma),
        normalized_mean=float(normalized_mean),
        coverage_one_sigma=float(np.mean(abs(values) <= sigma)),
        coverage_two_sigma=float(np.mean(abs(values) <= 2 * sigma)),
    )
