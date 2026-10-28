"""Propagate instrument parameter tolerances into physical detector reads.

Compare first-order native-electron budgets with Monte Carlo charge or DN
simulations using the caller's parameter law and explicit random generator.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty import (
    propagate_bias,
    propagate_covariance,
    sample_gaussian,
    validate_covariance,
)

from .charge import ExpectedElectrons
from .detector import (
    DetectorElectronics,
    DetectorOperatingPoints,
    DetectorReadings,
    electron_moments,
    sample_readout,
)


@dataclass(frozen=True, eq=False)
class NativeSignalBudget:
    """Signed bias, instrumental covariance and conditional noise for native reads.

    Array rows follow ``sample_ids``; Jacobian columns follow ``parameter_names``.
    All three components are distinct; do not fold detector noise into a parameter
    covariance unless its correlation and conditional law warrant that calculation.
    ``jacobian`` is an independent writable (reads,parameters) matrix. ``unit`` is
    electron or analogue DN. These first-order values are *not* quantised DN
    moments; conditional readout is not a Gaussian guarantee near clipping.

    Attributes:
        sample_ids: Ordered physical read identities.
        parameter_names: Ordered perturbation parameter identities.
        jacobian: (reads,parameters) local native-response derivatives.
        signed_bias: (reads,) signed mean offsets in the declared output unit.
        instrument_covariance: (reads,reads) instrument variance in squared units.
        detector_covariance: (reads,reads) conditional detector variance in squared units.
        unit: Native response unit, electron or analogue DN.
    """

    sample_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]
    jacobian: NDArray[np.float64]
    signed_bias: NDArray[np.float64]
    instrument_covariance: NDArray[np.float64]
    detector_covariance: NDArray[np.float64]
    unit: str = "electron"

    def __post_init__(self) -> None:
        object.__setattr__(self, "jacobian", np.copy(self.jacobian))
        object.__setattr__(self, "signed_bias", np.copy(self.signed_bias))
        object.__setattr__(
            self, "instrument_covariance", validate_covariance(self.instrument_covariance)
        )
        object.__setattr__(
            self, "detector_covariance", validate_covariance(self.detector_covariance)
        )
        if not np.isfinite(self.jacobian).all():
            raise ValueError("native tolerance jacobian must be finite")
        if (
            self.jacobian.shape != (len(self.sample_ids), len(self.parameter_names))
            or len(set(self.sample_ids)) != len(self.sample_ids)
            or self.signed_bias.shape != (len(self.sample_ids),)
            or not np.isfinite(self.signed_bias).all()
            or self.instrument_covariance.shape != (len(self.sample_ids), len(self.sample_ids))
            or self.detector_covariance.shape != (len(self.sample_ids), len(self.sample_ids))
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or self.unit not in ("electron", "analogue DN")
        ):
            raise ValueError("native tolerance axes must agree")


def native_difference(
    nominal: ExpectedElectrons, perturbed: ExpectedElectrons
) -> NDArray[np.float64]:
    """Signed perturbed-minus-nominal electrons with physical-read identity checks.

    Geometry/timing may change intentionally; identities and their order must not.
    Callers must rebuild all paths affected by a perturbation, not reuse stale
    pointing/scene/response samples. Source names may not silently change.

    Args:
        nominal: Reference source-resolved charge for ordered physical reads.
        perturbed: Charge from the same source names and read identities under
            a changed physical parameter.

    Returns:
        Fresh signed (reads,) difference in electrons, perturbed minus nominal.

    Raises:
        ValueError: Physical read or source identities differ between evaluations.
    """
    for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids"):
        if getattr(nominal.acquisition, name) != getattr(perturbed.acquisition, name):
            raise ValueError("perturbations must preserve physical-read identities")
    if nominal.contributions.keys() != perturbed.contributions.keys():
        raise ValueError("perturbations must preserve source identities")
    return perturbed.total - nominal.total


def central_difference(
    evaluate: Callable[[float], ExpectedElectrons],
    value: float,
    step: float,
) -> NDArray[np.float64]:
    """Deterministic native-electron derivative per parameter unit, central finite step.

    No RNG is used. Step refinement and validity near parameter boundaries remain
    explicit caller responsibilities; this is not automatic differentiation.

    Args:
        evaluate: Deterministic charge evaluator for a scalar physical parameter.
        value: Nominal parameter value in its caller-defined unit.
        step: Finite positive representable step in the same unit.

    Returns:
        Fresh signed (reads,) derivative in electrons per parameter unit.

    Raises:
        ValueError: Step is invalid or charge evaluations disagree on read/source
            identities.
    """
    if not np.isfinite([value, step]).all() or step <= 0 or value + step == value:
        raise ValueError("finite, positive, representable perturbation step required")
    return native_difference(evaluate(value - step), evaluate(value + step)) / (2 * step)


def native_sensitivities(
    evaluate: Callable[[Mapping[str, float]], ExpectedElectrons],
    parameters: Mapping[str, float],
    steps: Mapping[str, float],
) -> tuple[ExpectedElectrons, NDArray[np.float64]]:
    """Central native-electron derivatives in parameter insertion order.

    ``evaluate`` must be deterministic and recompute every affected optical,
    scene, timing, detector or geometric path at each perturbation. No RNG or
    hidden reuse. Steps have the same declared physical units as parameter keys.
    Steps and parameter names must match exactly; refine steps before relying on
    a linearised uncertainty budget. No cross-parameter finite differences.

    Args:
        evaluate: Deterministic evaluator of named physical parameter values.
        parameters: Ordered nominal parameter names and values.
        steps: Finite positive perturbation size for each matching parameter name.

    Returns:
        Nominal charge and fresh (reads,parameters) electron derivatives, with
        columns in the insertion order of ``parameters``.

    Raises:
        ValueError: Parameter/step names or sizes are invalid, or perturbed
            evaluations change physical read or source identities.
    """
    names = tuple(parameters)
    if not names or any(not name for name in names) or set(steps) != set(names):
        raise ValueError("one named step is required per named parameter")
    nominal = evaluate(parameters)
    columns = []
    for name in names:
        value, step = parameters[name], steps[name]
        if not np.isfinite([value, step]).all() or step <= 0 or value + step == value:
            raise ValueError("finite, positive representable parameter step required")
        low = evaluate({**parameters, name: value - step})
        high = evaluate({**parameters, name: value + step})
        native_difference(nominal, low)
        native_difference(nominal, high)
        columns.append(native_difference(low, high) / (2 * step))
    return nominal, np.stack(columns, axis=1)


def native_tolerance_budget(
    nominal: ExpectedElectrons,
    parameter_names: tuple[str, ...],
    jacobian: NDArray[np.float64],
    parameter_bias: NDArray[np.float64],
    parameter_covariance: NDArray[np.float64],
    detector_covariance_e2: NDArray[np.float64],
) -> NativeSignalBudget:
    """Propagate a supplied correlated parameter law to native electrons.

    Input bias (p,) and covariance (p,) variances or (p,p) matrix follow the ordered
    parameter names. Jacobian column units are electrons/parameter unit; the
    caller owns parameter ordering and unit consistency. ``detector_covariance_e2``
    describes conditional readout (for example
    ``electron_covariance``), not fixed-pattern parameter variation; it may
    include common row/column/exposure noise. No independence from an externally
    supplied parameter/noise coupling can be inferred from this API.

    Args:
        nominal: Charge defining ordered physical read identities.
        parameter_names: Unique parameter identities in Jacobian column order.
        jacobian: Finite derivatives (reads,parameters) in electrons per unit.
        parameter_bias: Signed mean offsets (parameters,) in the parameter units.
        parameter_covariance: Variances (parameters,) or dense covariance
            (parameters,parameters) in squared parameter units.
        detector_covariance_e2: Conditional covariance (reads,reads) in electrons squared.

    Returns:
        First-order native-electron budget with separate signed parameter bias,
        parameter covariance and conditional detector covariance.

    Raises:
        ValueError: Named axes, derivatives or supplied covariances are invalid.
    """
    n, p = len(nominal.acquisition.sample_ids), len(parameter_names)
    matrix = jacobian
    noise = detector_covariance_e2
    if (
        p < 1
        or len(set(parameter_names)) != p
        or any(not name for name in parameter_names)
        or matrix.shape != (n, p)
        or not np.isfinite(matrix).all()
        or parameter_bias.shape != (p,)
        or parameter_covariance.shape not in ((p,), (p, p))
        or noise.shape != (n, n)
        or not np.isfinite(noise).all()
    ):
        raise ValueError("parameter, signal and noise axes must agree")
    # Validate the conditional covariance as PSD, not merely a plausible diagonal.
    detector = validate_covariance(noise)
    bias = propagate_bias(parameter_bias, matrix)
    instrumental = propagate_covariance(parameter_covariance, matrix)
    return NativeSignalBudget(
        nominal.acquisition.sample_ids, parameter_names, matrix, bias, instrumental, detector
    )


def monte_carlo_native(
    evaluate: Callable[[Mapping[str, float]], ExpectedElectrons],
    parameters: Mapping[str, float],
    parameter_bias: NDArray[np.float64],
    parameter_covariance: NDArray[np.float64],
    rng: np.random.Generator,
    count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Empirical signed mean change and parameter-only covariance in electrons.

    Draws are Gaussian in parameter insertion order, use caller-owned RNG,
    and are compared to a deterministic nominal evaluation. Charge/readout noise
    is intentionally NOT drawn; compare the returned covariance with the budget's
    *instrument* part. Sample count >=2; the covariance uses N-1. Gaussian draws
    may leave physical parameter support, in which case the evaluator must reject.

    Args:
        evaluate: Deterministic charge evaluator for named parameter values.
        parameters: Nominal parameter names and values in insertion order.
        parameter_bias: Signed Gaussian mean offsets (parameters,).
        parameter_covariance: Gaussian variances or covariance in parameter units squared.
        rng: Caller-owned NumPy generator advanced for parameter draws.
        count: Number of draws, at least two.

    Returns:
        Empirical signed mean (reads,) and sample covariance (reads,reads) in
        electrons and electrons squared, relative to the nominal evaluation.

    Raises:
        ValueError: Count or parameter axes are invalid, covariance fails
            validation, or physical read/source identities change during evaluation.
    """
    if (
        count < 2
        or parameter_bias.shape != (len(parameters),)
        or not np.isfinite(parameter_bias).all()
        or parameter_covariance.shape
        not in ((len(parameters),), (len(parameters), len(parameters)))
    ):
        raise ValueError("Monte Carlo needs >=2 draws and matching parameter dimensions")
    nominal = evaluate(parameters)
    draws = sample_gaussian(parameter_covariance, rng, count) + parameter_bias
    keys = tuple(parameters)
    values = np.array(tuple(parameters.values()), dtype=np.float64)
    outputs = np.empty((count, len(nominal.acquisition.sample_ids)))
    for i, draw in enumerate(draws):
        trial = evaluate(dict(zip(keys, values + draw, strict=True)))
        outputs[i] = native_difference(nominal, trial)
    return outputs.mean(axis=0), np.atleast_2d(np.cov(outputs, rowvar=False))


def analogue_dn_budget(
    electron_budget: NativeSignalBudget,
    nominal: ExpectedElectrons,
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> tuple[NDArray[np.float64], NativeSignalBudget]:
    """Convert a locally unsaturated *linear* electron budget to analogue DN.

    Reject clipping at the nominal operating point and nonlinear detector maps.
    Quantisation, stochastic excursions across limits, biased nonlinear moments,
    and ADC/full-well saturation are NOT covered by the returned linear budget;
    use ``monte_carlo_dn`` to study them. Dark means remain fixed in this map.

    Args:
        electron_budget: Native first-order budget in electrons for nominal reads.
        nominal: Charge at the unsaturated operating point.
        electronics: Gain, pedestal and readout limits for these physical reads.

    Returns:
        Nominal analogue (reads,) DN and first-order budget in analogue DN,
        including gain-scaled Jacobian and distinct covariance components.

    Raises:
        ValueError: Read axes disagree, electronics are nonlinear, or the nominal
            charge/analogue DN reaches a saturation or clipping threshold.
    """
    if electron_budget.unit != "electron" or electron_budget.sample_ids != (
        nominal.acquisition.sample_ids
    ):
        raise ValueError("electron budget and nominal physical reads must agree")
    if isinstance(electronics, DetectorOperatingPoints) and np.any(electronics.nonlinearity):
        raise ValueError("nonlinear detector maps need direct DN sampling")
    charge, _ = electron_moments(nominal, electronics)
    gain = electronics.electrons_per_dn
    analogue = charge / gain + electronics.bias_dn
    if np.any(charge >= electronics.full_well_electrons) or np.any(
        (analogue <= 0) | (analogue >= 2**electronics.adc_bits - 1)
    ):
        raise ValueError("nominal charge/analogue DN must be strictly unsaturated")
    n = len(charge)
    inverse_gain = np.broadcast_to(1 / gain, (n,))
    scale = np.diag(inverse_gain)
    return analogue, NativeSignalBudget(
        electron_budget.sample_ids,
        electron_budget.parameter_names,
        scale @ electron_budget.jacobian,
        propagate_bias(electron_budget.signed_bias, scale),
        propagate_covariance(electron_budget.instrument_covariance, scale),
        propagate_covariance(electron_budget.detector_covariance, scale),
        "analogue DN",
    )


def monte_carlo_dn(
    evaluate: Callable[[Mapping[str, float]], ExpectedElectrons],
    parameters: Mapping[str, float],
    parameter_bias: NDArray[np.float64],
    parameter_covariance: NDArray[np.float64],
    electronics: DetectorElectronics | DetectorOperatingPoints,
    rng: np.random.Generator,
    count: int,
) -> DetectorReadings:
    """Draw supplied parameter law and physical *quantised* readings, with flags.

    Unlike analogue linearisation, saturation and nonlinear electronics operate
    on every draw. Conditional detector noise is resampled once per draw and is
    not added to the parameter covariance beforehand. The result holds ``count``
    realisations in memory; use bounded batches for large experiments. Read
    identities/exposure intervals must agree with nominal; RNG is caller-owned.
    No correlated group noise is inferred: use ``sample_readout`` directly for
    independently specified group IDs when that effect matters.

    Args:
        evaluate: Charge evaluator for the named physical parameter state.
        parameters: Nominal parameter names and values in insertion order.
        parameter_bias: Signed Gaussian mean offsets (parameters,).
        parameter_covariance: Gaussian variances or covariance in parameter units squared.
        electronics: Physical detector readout settings for each charge draw.
        rng: Caller-owned generator advanced by parameter and detector noise draws.
        count: Positive number of quantised detector realisations.

    Returns:
        Detector readings with (count,reads) native DN, full-well and ADC flags
        aligned to nominal physical read identities and exposure intervals.

    Raises:
        ValueError: Parameter dimensions or count are invalid, covariance fails
            validation, or sampled charge changes nominal read identity or timing.
    """
    if (
        count < 1
        or parameter_bias.shape != (len(parameters),)
        or not np.isfinite(parameter_bias).all()
        or parameter_covariance.shape
        not in ((len(parameters),), (len(parameters), len(parameters)))
    ):
        raise ValueError("DN sampling needs draws and matching parameter dimensions")
    nominal = evaluate(parameters)
    draws = sample_gaussian(parameter_covariance, rng, count) + parameter_bias
    keys = tuple(parameters)
    values = np.array(tuple(parameters.values()), dtype=np.float64)
    shape = (count, len(nominal.acquisition.sample_ids))
    dn = np.empty(shape, dtype=np.int64)
    well = np.empty(shape, dtype=np.bool_)
    adc = np.empty(shape, dtype=np.bool_)
    for i, draw in enumerate(draws):
        trial = evaluate(dict(zip(keys, values + draw, strict=True)))
        native_difference(nominal, trial)
        if not np.array_equal(
            nominal.acquisition.exposure_bounds_s, trial.acquisition.exposure_bounds_s
        ):
            raise ValueError("physical exposure intervals must match for DN sampling")
        result = sample_readout(trial, electronics, rng)
        dn[i], well[i], adc[i] = result.dn[0], result.full_well_saturated[0], result.adc_clipped[0]
    return DetectorReadings(nominal.acquisition, dn, well, adc)
