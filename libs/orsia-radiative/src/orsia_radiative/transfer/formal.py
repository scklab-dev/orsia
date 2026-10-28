"""Propagate supplied layer sources to cell centres and exit boundaries.

This module owns exponential path integration and propagation to cell centres
and exit boundaries. It selects neither optical constituents nor a transfer
approximation and does not assemble radiance records or reflect exit fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.special import exprel

from .native.propagation_native import apply as _native_apply
from .native.propagation_native import prepare as _native_prepare


@dataclass(frozen=True, eq=False)
class PropagationCoefficients[Real: np.float32 | np.float64 = np.float64]:
    """Call-owned transport weights for one fixed cell/direction grid.

    Arrays are read-only owned snapshots, not a cache or a retained scene state.
    ``mu`` is [direction]; all weights are [cell][direction], bottom to top.
    Source weights include 1/abs(mu), and centre weights also include the half
    cell measure. Solar weights, if present, integrate direct-beam attenuation.
    No coefficient depends on source amplitude or bottom boundary radiance.
    Construct with ``propagation_coefficients``; reuse only on this exact grid.
    All arrays share float32 or float64 precision, selected by depth and mu.
    Float32 halves retained storage but has reduced range and accumulated
    roundoff; this operator alone does not make the solar solver float32.

    Attributes:
        mu: (directions,) signed propagation cosines.
        transmission: (cells,directions) full-cell attenuation factors.
        centre_transmission: (cells,directions) half-cell attenuation factors.
        source_weight: (cells,directions) integrated source coefficients.
        centre_source_weight: (cells,directions) centre-reaching source coefficients.
        solar_source_weight: Optional direct-beam source coefficients per cell/direction.
        solar_centre_source_weight: Optional centre-reaching direct-beam coefficients.
    """

    mu: NDArray[Real]
    transmission: NDArray[Real]
    centre_transmission: NDArray[Real]
    source_weight: NDArray[Real]
    centre_source_weight: NDArray[Real]
    solar_source_weight: NDArray[Real] | None
    solar_centre_source_weight: NDArray[Real] | None


def integrated_exponential[Real: np.float32 | np.float64](
    a: NDArray[Real], b: NDArray[Real]
) -> NDArray[Real]:
    """Integrate exp(a + (b-a)t) over [0, 1] for nonpositive endpoint arrays.

    Inputs have equal shapes and are read-only; the output is a fresh array.
    The near-equal endpoint limit avoids division by zero and cancellation.

    Args:
        a: Nonpositive logarithmic integrand values at the interval start.
        b: Nonpositive logarithmic integrand values at the interval end, with
            the same shape as ``a``.

    Returns:
        Fresh array of dimensionless exponential integrals in the input shape.
    """
    return np.exp(np.maximum(a, b)) * exprel(-np.abs(b - a))


def propagation_coefficients[Real: np.float32 | np.float64](
    extinction: NDArray[Real],
    mu: NDArray[Real],
    *,
    solar_mu: float | None = None,
) -> PropagationCoefficients[Real]:
    """Compute cell transport coefficients once, independently of source values.

    Extinction is finite nonnegative vertical optical depth [cell], bottom to
    top; mu is finite, nonzero [direction], positive upward. Solar mu is in
    (0, 1]. Inputs are read-only. Storage is O(cells*directions), independent of
    spectral points, source count and scattering order. Float32 depth and mu
    retain float32 arithmetic and coefficient storage; any float64 geometry input
    promotes both to float64. Solar mu is rounded to that precision and must
    remain in (0, 1]. Packaged float32/float64 builds
    fuse preparation for small (<=32) or strongly repeated direction grids
    (<=1024 directions, at least half repeated). Exact signed cosines are reused
    within the call; unique large grids use full-grid NumPy/SciPy vector
    operations. SOS releases coefficients after each spectral point. Inputs
    must not be mutated during preparation.
    The source-constant limit uses -expm1(-x)/x, with the exact value one at x=0.
    No optical-depth prefix sum is needed without direct-beam forcing.

    Args:
        extinction: Nonnegative vertical optical depths (cells,), bottom to top.
        mu: Signed nonzero direction cosines (directions,), positive upward.
        solar_mu: Optional positive direct-beam cosine for solar source weights.

    Returns:
        Read-only coefficient snapshots for the supplied cell and direction grid.

    Raises:
        ValueError: Solar cosine is outside (0, 1] in the coefficient precision.
    """
    dtype = np.result_type(extinction.dtype, mu.dtype)
    if solar_mu is not None:
        solar_mu = float(dtype.type(solar_mu))
        if not 0 < solar_mu <= 1:
            raise ValueError("solar cosine must be in (0, 1] in the coefficient precision")
    extinction = np.asarray(extinction, dtype=dtype)
    mu = np.asarray(mu, dtype=dtype)
    shape = (extinction.size, mu.size)
    transmission, centre_transmission, source_weight, centre_source_weight = (
        np.empty(shape, dtype=dtype) for _ in range(4)
    )
    solar_source_weight = np.empty(shape, dtype=dtype) if solar_mu is not None else None
    solar_centre_source_weight = np.empty(shape, dtype=dtype) if solar_mu is not None else None
    if (
        dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and extinction.ndim == 1
        and mu.ndim == 1
        and (mu.size <= 32 or (mu.size <= 1024 and np.unique(mu).size <= mu.size // 2))
    ):
        _native_prepare(
            np.ascontiguousarray(extinction),
            np.ascontiguousarray(mu),
            solar_mu if solar_mu is not None else 0.0,
            transmission,
            centre_transmission,
            source_weight,
            centre_source_weight,
            solar_source_weight,
            solar_centre_source_weight,
        )
    else:
        _prepare_numpy(
            extinction,
            mu,
            solar_mu,
            transmission,
            centre_transmission,
            source_weight,
            centre_source_weight,
            solar_source_weight,
            solar_centre_source_weight,
        )
    owned_mu = mu.copy()
    for array in (
        owned_mu,
        transmission,
        centre_transmission,
        source_weight,
        centre_source_weight,
        solar_source_weight,
        solar_centre_source_weight,
    ):
        if array is not None:
            array.flags.writeable = False
    return PropagationCoefficients(
        owned_mu,
        transmission,
        centre_transmission,
        source_weight,
        centre_source_weight,
        solar_source_weight,
        solar_centre_source_weight,
    )


@overload
def apply_propagation[Real: np.float32 | np.float64](
    source_depth: NDArray[Real],
    coefficients: PropagationCoefficients[Real],
    *,
    bottom_radiance: float = 0.0,
    solar_source: bool = False,
    return_centres: Literal[True] = True,
) -> tuple[NDArray[Real], NDArray[Real]]: ...


@overload
def apply_propagation[Real: np.float32 | np.float64](
    source_depth: NDArray[Real],
    coefficients: PropagationCoefficients[Real],
    *,
    bottom_radiance: float = 0.0,
    solar_source: bool = False,
    return_centres: Literal[False],
) -> tuple[None, NDArray[Real]]: ...


@overload
def apply_propagation[Real: np.float32 | np.float64](
    source_depth: NDArray[Real],
    coefficients: PropagationCoefficients[Real],
    *,
    bottom_radiance: float = 0.0,
    solar_source: bool = False,
    return_centres: bool,
) -> tuple[NDArray[Real] | None, NDArray[Real]]: ...


def apply_propagation[Real: np.float32 | np.float64](
    source_depth: NDArray[Real],
    coefficients: PropagationCoefficients[Real],
    *,
    bottom_radiance: float = 0.0,
    solar_source: bool = False,
    return_centres: bool = True,
) -> tuple[NDArray[Real] | None, NDArray[Real]]:
    """Apply fixed coefficients to a source, retaining the cell recurrence.

    Source [cell][direction] uses the exact coefficient grid and the same units
    as ``propagate``. Sources and coefficients are read-only, outputs fresh.
    Solar forcing requires preparation with solar_mu; otherwise the source is
    cell-constant. With return_centres=False, return (None, exits) without
    allocating or calculating the centre field. No reflection is performed.
    Buffers are private to this application, so coefficients can be shared by
    concurrent applications without shared writable state. Callers must not
    mutate source or coefficient arrays during an application. Packaged builds
    use a serial compiled float32/float64 recurrence (no fast math or fused
    operations). Both outputs use the common source/coefficient dtype: float32
    only when both are float32, otherwise float64. Bottom radiance is rounded
    to the output precision. Noncontiguous inputs are copied as needed. The
    NumPy recurrence handles mixed precision and arrays outside the native
    float/double ABI.

    Args:
        source_depth: Source values (cells,directions) on the prepared grid.
        coefficients: Prepared cell and direction transport coefficients.
        bottom_radiance: Incident radiance at the lower boundary for upward rays.
        solar_source: Select direct-beam source weights from solar preparation.
        return_centres: Include the cell-centre radiances when true.

    Returns:
        Pair of optional centre radiances (cells,directions) and exit radiances
        (directions,); centres are None when ``return_centres`` is false.

    Raises:
        ValueError: Source shape differs from the prepared grid, or solar source
            weights were requested from coefficients prepared without solar mu.
    """
    weight = coefficients.source_weight
    centre_weight = coefficients.centre_source_weight
    if solar_source:
        if (
            coefficients.solar_source_weight is None
            or coefficients.solar_centre_source_weight is None
        ):
            raise ValueError("solar forcing requires coefficients prepared with solar_mu")
        weight = coefficients.solar_source_weight
        centre_weight = coefficients.solar_centre_source_weight
    if source_depth.shape != coefficients.transmission.shape:
        raise ValueError("source must match the prepared cell/direction grid")
    if (
        source_depth.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and coefficients.mu.dtype == source_depth.dtype
        and coefficients.mu.ndim == 1
        and source_depth.ndim == 2
        and source_depth.shape[1] == coefficients.mu.size
        and all(
            a.shape == source_depth.shape and a.dtype == source_depth.dtype
            for a in (
                coefficients.transmission,
                weight,
                coefficients.centre_transmission,
                centre_weight,
            )
        )
    ):
        native_dtype = source_depth.dtype
        centres = np.empty(source_depth.shape, dtype=native_dtype) if return_centres else None
        exits = np.empty(coefficients.mu.size, dtype=native_dtype)
        _native_apply(
            np.ascontiguousarray(coefficients.mu),
            np.ascontiguousarray(source_depth),
            np.ascontiguousarray(coefficients.transmission),
            np.ascontiguousarray(weight),
            np.ascontiguousarray(coefficients.centre_transmission) if return_centres else None,
            np.ascontiguousarray(centre_weight) if return_centres else None,
            bottom_radiance,
            centres,
            exits,
        )
        return centres, exits
    return _apply_numpy(
        source_depth, coefficients, weight, centre_weight, bottom_radiance, return_centres
    )


def _apply_numpy[Real: np.float32 | np.float64](
    source_depth: NDArray[Real],
    coefficients: PropagationCoefficients[Real],
    weight: NDArray[Real],
    centre_weight: NDArray[Real],
    bottom_radiance: float,
    return_centres: bool,
) -> tuple[NDArray[Real] | None, NDArray[Real]]:
    """Propagate arrays outside the native ABI with a NumPy reference recurrence.

    Returns:
        Cell-centre radiance or None, followed by outgoing radiance per direction.
    """
    dtype = np.result_type(source_depth.dtype, coefficients.transmission.dtype)
    centres = np.empty(source_depth.shape, dtype=dtype) if return_centres else None
    mu = coefficients.mu
    exit_radiance = np.zeros(mu.size, dtype=dtype)
    for upward in (False, True):
        indices = np.flatnonzero(mu > 0 if upward else mu < 0)
        if not indices.size:
            continue
        selected = _direction_selector(indices)
        # Contiguous hemispheres (as in SOS) use views. Interleaved directions
        # are gathered once here, never once per cell via Boolean indexing.
        source = source_depth[:, selected]
        transmission = coefficients.transmission[:, selected]
        source_weight = weight[:, selected]
        local_centres = None
        if centres is not None:
            local_centres = (
                centres[:, selected]
                if isinstance(selected, slice)
                else np.empty(source.shape, dtype=dtype)
            )
        half_transmission = (
            coefficients.centre_transmission[:, selected] if return_centres else None
        )
        half_weight = centre_weight[:, selected] if return_centres else None
        running = np.full(indices.size, bottom_radiance if upward else 0.0, dtype=dtype)
        scratch = np.empty_like(running)
        cells = range(source.shape[0]) if upward else range(source.shape[0] - 1, -1, -1)
        for cell in cells:
            if local_centres is not None:
                assert half_transmission is not None and half_weight is not None
                np.multiply(running, half_transmission[cell], out=local_centres[cell])
                np.multiply(source[cell], half_weight[cell], out=scratch)
                local_centres[cell] += scratch
            running *= transmission[cell]
            np.multiply(source[cell], source_weight[cell], out=scratch)
            running += scratch
        if centres is not None:
            centres[:, selected] = local_centres
        exit_radiance[selected] = running
    return centres, exit_radiance


def propagate[Real: np.float32 | np.float64](
    source_depth: NDArray[Real],
    extinction: NDArray[Real],
    mu: NDArray[Real],
    *,
    bottom_radiance: float = 0.0,
    solar_mu: float | None = None,
) -> tuple[NDArray[Real], NDArray[Real]]:
    """Transport to cell centres and exit boundaries, without reflection.

    ``source_depth`` is [cell][direction], integrated over vertical cell fraction;
    ``extinction`` is [cell], bottom to top; ``mu`` is [direction], nonzero, with
    positive values pointing upward. All inputs are read-only and finite.
    Extinction is nonnegative and ``solar_mu``, when supplied, is in (0, 1].
    With ``solar_mu`` the source is the unattenuated direct-beam source, whose
    exponential variation is integrated exactly; otherwise it is cell-constant.
    Top incidence is zero. The returned [cell][direction] centres and [direction]
    exit radiances are fresh call-owned arrays. No exit field is reflected here,
    so this operator cannot introduce an unintended second surface encounter.
    Precision follows ``propagation_coefficients`` and ``apply_propagation``; mixed
    float32/float64 array inputs promote outputs to float64.

    Args:
        source_depth: Source values (cells,directions) in radiance units.
        extinction: Vertical optical depths (cells,), bottom to top.
        mu: Signed nonzero direction cosines (directions,), positive upward.
        bottom_radiance: Incident lower-boundary radiance for upward rays.
        solar_mu: Optional direct-beam cosine activating solar source integration.

    Returns:
        Fresh cell-centre radiances (cells,directions) and exit radiances
        (directions,) in source radiance units.

    Raises:
        ValueError: Solar cosine is outside (0, 1] or source shape differs from
            the prepared cell and direction grid.
    """
    coefficients = propagation_coefficients(extinction, mu, solar_mu=solar_mu)
    return apply_propagation(
        source_depth,
        coefficients,
        bottom_radiance=bottom_radiance,
        solar_source=solar_mu is not None,
    )


def schwarzschild(
    source_function: NDArray[np.float64],
    absorption_depth: NDArray[np.float64],
    mu: float,
    incident_radiance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact transport through isothermal, non-scattering layers along one ray.

    Source function and vertical absorption depth are float64 [layer, point],
    bottom to top, finite and nonnegative. Source and incident [point] radiances
    have the same density units. Signed mu is in [-1, 1] excluding zero: positive
    travels upward from the bottom boundary, negative downward from the top.
    Return fresh writable exit [point] radiance; inputs are neither retained nor
    mutated. No Planck evaluation, angular integration or reflection occurs here.
    The recurrence is I_out = I_in*exp(-tau/abs(mu)) + B*(-expm1(-tau/abs(mu))).
    This avoids cancellation in thin layers and multiplication of B by tau.

    Args:
        source_function: Layer source radiances (layers,points), bottom to top.
        absorption_depth: Vertical optical depths (layers,points).
        mu: Signed ray cosine; positive rays propagate upward.
        incident_radiance: Boundary radiance (points,) entering the layer stack.

    Returns:
        Fresh exit radiance (points,) in the incident radiance units.

    Raises:
        ValueError: Ray cosine or optical depths leave supported physical bounds,
            or source and boundary arrays have incompatible shapes or values.
    """
    if (
        source_function.ndim != 2
        or source_function.shape != absorption_depth.shape
        or min(source_function.shape) < 1
        or incident_radiance.shape != (source_function.shape[1],)
    ):
        raise ValueError("Schwarzschild inputs require matching layer/point support")
    if not np.isfinite(mu) or not 0 < abs(mu) <= 1:
        raise ValueError("ray cosine must be finite, nonzero and in [-1, 1]")
    for values in (source_function, absorption_depth, incident_radiance):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Schwarzschild depths and radiances must be finite and nonnegative")
    running = incident_radiance.copy()
    layers = range(source_function.shape[0])
    if mu < 0:
        layers = range(source_function.shape[0] - 1, -1, -1)
    # Infinite slant depth from finite vertical depth is the opaque limit.
    with np.errstate(over="ignore", under="ignore"):
        for layer in layers:
            slant_depth = absorption_depth[layer] / abs(mu)
            running *= np.exp(-slant_depth)
            running += source_function[layer] * -np.expm1(-slant_depth)
    return running


def _prepare_numpy[Real: np.float32 | np.float64](
    extinction: NDArray[Real],
    mu: NDArray[Real],
    solar_mu: float | None,
    transmission: NDArray[Real],
    centre_transmission: NDArray[Real],
    source_weight: NDArray[Real],
    centre_source_weight: NDArray[Real],
    solar_source_weight: NDArray[Real] | None,
    solar_centre_source_weight: NDArray[Real] | None,
) -> None:
    """Fill full-grid coefficients with dtype-preserving NumPy/SciPy operations."""
    direction = np.abs(mu)
    above = None
    if solar_mu is not None:
        above = np.zeros_like(extinction)
        above[:-1] = np.cumsum(extinction[:0:-1])[::-1]
    depth = extinction[:, None]
    attenuation = depth / direction
    transmission[:] = np.exp(-attenuation)
    centre_transmission[:] = np.exp(-attenuation / 2)
    source_weight[:] = _constant_source_weight(attenuation) / direction
    centre_source_weight[:] = _constant_source_weight(attenuation / 2) / (2 * direction)
    if solar_mu is not None:
        assert above is not None
        assert solar_source_weight is not None and solar_centre_source_weight is not None
        # Endpoint exponents in travel order. Both are nonpositive, including
        # near-equal downward beam/ray cosines; never subtract exponentials.
        top = -above[:, None] / solar_mu
        bottom = -(above[:, None] + depth) / solar_mu
        entry = np.where(mu > 0, bottom, top)
        end = np.where(mu > 0, top, bottom)
        midpoint = -(above[:, None] + depth / 2) / solar_mu
        solar_source_weight[:] = integrated_exponential(entry - attenuation, end) / direction
        solar_centre_source_weight[:] = integrated_exponential(
            entry - attenuation / 2, np.broadcast_to(midpoint, attenuation.shape)
        ) / (2 * direction)


def _constant_source_weight[Real: np.float32 | np.float64](
    attenuation: NDArray[Real],
) -> NDArray[Real]:
    """Evaluate the constant-source weight at vacuum and weak optical depth.

    Returns:
        Stable phi(x)=-expm1(-x)/x at each attenuation value.
    """
    return exprel(-attenuation)


def _direction_selector(indices: NDArray[np.intp]) -> slice | NDArray[np.intp]:
    """Select contiguous directions by view and interleaved directions by gather.

    Returns:
        A slice spanning consecutive indices or the original index array.
    """
    if indices[-1] - indices[0] + 1 == indices.size:
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


__all__ = [
    "PropagationCoefficients",
    "integrated_exponential",
    "propagation_coefficients",
    "apply_propagation",
    "propagate",
    "schwarzschild",
]
