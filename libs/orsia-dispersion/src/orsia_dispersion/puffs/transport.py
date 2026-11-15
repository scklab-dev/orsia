"""Probe local forcing and integrate puff trajectories and affine covariance dynamics."""

from collections.abc import Callable

import numpy as np
from scipy.linalg import expm

from orsia_dispersion.puffs.population import GaussianPuffs, puff_principal_axes, validate_boundary
from orsia_dispersion.transport import Boundary, Diffusivity, FloatArray, Wind, validate_tensors


class SpatialResolutionError(RuntimeError):
    """Carry a sampled spatial failure from a trial state to its event loop.

    Attributes:
        indices: Failing population indices.
        directions: Physical ENU unit vectors of failed trial probes, shape (s,3).
        ratio: Maximum sampled residual divided by its tolerance.
        attempts: Attempted transport steps consumed before the failure.
        step_s: Duration of the failed transport trial, seconds; zero for state checks.
    """

    def __init__(
        self,
        indices: np.ndarray[tuple[int], np.dtype[np.int64]],
        directions: FloatArray,
        ratio: float,
    ) -> None:
        """Retain the failed spatial criterion for conservative subdivision.

        Args:
            indices: Population indices needing subdivision.
            directions: Corresponding physical unit directions, independent of eigenvalue order.
            ratio: Maximum normalized residual.
        """
        super().__init__(f"sampled spatial variation ratio={ratio}")
        self.indices = indices
        self.directions = directions
        self.ratio = ratio
        self.attempts = 0
        self.step_s = 0.0


def advance_puffs(
    state: GaussianPuffs,
    target_time_s: float,
    wind: Wind,
    diffusivity: Diffusivity,
    *,
    boundary: Boundary,
    rtol: float,
    atol_position_m: float,
    atol_covariance_m2: float,
    min_step_s: float,
    max_step_s: float,
    max_steps: int,
    envelope_sigma: float,
    discontinuities_s: tuple[float, ...] = (),
) -> GaussianPuffs:
    """Advance a snapshot without inserting releases or splitting puffs.

    Args:
        state: Stable active snapshot, copied before integration.
        target_time_s: Finite target at or after state.time_s, in seconds.
        wind: Pure velocity/gradient callable; see sample_forcing.
        diffusivity: Pure age-aware PSD diffusivity callable, SI units.
        boundary: 'free' or 'reflecting' under the documented separability contract.
        rtol: Positive relative tolerance on centre displacement and covariance norms.
        atol_position_m: Positive absolute centre tolerance in metres.
        atol_covariance_m2: Positive absolute Frobenius covariance tolerance in m2.
        min_step_s: Positive minimum refinement step. Short final event steps allowed.
        max_step_s: Positive maximum step, at least min_step_s.
        max_steps: Positive total attempted-step budget, including rejected attempts.
        envelope_sigma: Positive finite forcing-support radius in standard deviations.
        discontinuities_s: Strictly increasing finite forcing discontinuity times.
            Steps stop at these times; midpoint sampling never crosses an event.

    Returns:
        A fresh state at exactly target_time_s with unchanged mass and release ages.
        Empty populations advance directly between events without sampling forcing.

    Raises:
        ValueError: Inputs or forcing violate the physical/numerical contract.
        RuntimeError: Refinement or attempt budget exhausted; reports last accepted time.
    """
    validate_integration(
        rtol, atol_position_m, atol_covariance_m2, min_step_s, max_step_s, max_steps, envelope_sigma
    )
    validate_boundary(state, boundary)
    if not np.isfinite(target_time_s) or target_time_s < state.time_s:
        raise ValueError("target time must be finite and at or after current time")
    breaks = np.asarray(discontinuities_s, dtype=np.float64)
    if not np.all(np.isfinite(breaks)) or np.any(np.diff(breaks) <= 0):
        raise ValueError("discontinuities must be finite and strictly increasing")
    current = GaussianPuffs(
        state.time_s, state.release_times_s, state.masses_kg, state.centres_m, state.covariances_m2
    )
    events = [*breaks[(breaks > state.time_s) & (breaks < target_time_s)], target_time_s]
    attempts = 0
    for event in events:
        while current.time_s < event:
            current, _, used = transport_step(
                current,
                float(event),
                wind,
                diffusivity,
                boundary=boundary,
                rtol=rtol,
                atol_position_m=atol_position_m,
                atol_covariance_m2=atol_covariance_m2,
                min_step_s=min_step_s,
                max_step_s=max_step_s,
                max_steps=max_steps - attempts,
                envelope_sigma=envelope_sigma,
            )
            attempts += used
    return current


def validate_integration(
    rtol: float,
    atol_position_m: float,
    atol_covariance_m2: float,
    min_step_s: float,
    max_step_s: float,
    max_steps: int,
    envelope_sigma: float,
) -> None:
    """Check integration controls shared by advance and event simulation.

    Args:
        rtol: Relative tolerance.
        atol_position_m: Absolute trajectory tolerance, m.
        atol_covariance_m2: Absolute covariance tolerance, m2.
        min_step_s: Minimum refinement step, s.
        max_step_s: Maximum trial step, s.
        max_steps: Attempt budget.
        envelope_sigma: Forcing envelope radius.

    Raises:
        ValueError: Any positive finite bound or step ordering is invalid.
    """
    controls = (rtol, atol_position_m, atol_covariance_m2, min_step_s, max_step_s, envelope_sigma)
    if not np.all(np.isfinite(controls)) or min(controls) <= 0:
        raise ValueError("integration tolerances, steps and envelope must be finite and positive")
    if max_step_s < min_step_s or max_steps < 1:
        raise ValueError("step bounds or max_steps are invalid")


def transport_step(
    state: GaussianPuffs,
    target_time_s: float,
    wind: Wind,
    diffusivity: Diffusivity,
    *,
    boundary: Boundary,
    rtol: float,
    atol_position_m: float,
    atol_covariance_m2: float,
    min_step_s: float,
    max_step_s: float,
    max_steps: int,
    envelope_sigma: float,
    spatial_check: Callable[[GaussianPuffs], float] | None = None,
) -> tuple[GaussianPuffs, float, int]:
    """Take one accepted step with midpoint/step-doubling error control.

    Args:
        state: Validated snapshot.
        target_time_s: Next event time, strictly after state.time_s.
        wind: Pure velocity/gradient field.
        diffusivity: Pure diffusivity field.
        boundary: Supported boundary regime.
        rtol: Relative displacement and covariance tolerance.
        atol_position_m: Absolute position error scale, metres.
        atol_covariance_m2: Absolute covariance error scale, m2.
        min_step_s: Minimum refinement step, seconds.
        max_step_s: Maximum trial step, seconds.
        max_steps: Remaining attempted-step budget.
        envelope_sigma: Radius for forcing probes.
        spatial_check: Optional pure validation callback at every trial midpoint and
            endpoint. Simulation uses this to request subdivision before accepting transport.

    Returns:
        New state, normalized accepted local error estimate, and attempts consumed.
        The accepted state uses two half steps without Richardson extrapolation.
        Empty populations advance directly to the event without sampling or attempts.

    Raises:
        ValueError: A forcing sample violates its contract.
        RuntimeError: Attempt, floating-point time or minimum-step budget exhausted.
        SpatialResolutionError: Trial-state probes require subdivision; the exception
            records the consumed attempts and failed trial duration.
    """
    if not len(state.masses_kg):
        return (
            GaussianPuffs(
                target_time_s,
                state.release_times_s,
                state.masses_kg,
                state.centres_m,
                state.covariances_m2,
            ),
            0.0,
            0,
        )
    h = min(max_step_s, target_time_s - state.time_s)
    floor = min(min_step_s, h)
    for attempt in range(1, max_steps + 1):
        if state.time_s + h == state.time_s or state.time_s + h / 2 == state.time_s:
            raise RuntimeError(f"time resolution exhausted; last accepted time={state.time_s}")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                full = _midpoint(
                    state, h, wind, diffusivity, boundary, envelope_sigma, spatial_check
                )
                half = _midpoint(
                    state, h / 2, wind, diffusivity, boundary, envelope_sigma, spatial_check
                )
                fine = _midpoint(
                    half, h / 2, wind, diffusivity, boundary, envelope_sigma, spatial_check
                )
                position_scale = atol_position_m + rtol * np.linalg.norm(
                    fine.centres_m - state.centres_m, axis=1
                )
                covariance_scale = atol_covariance_m2 + rtol * np.linalg.norm(
                    fine.covariances_m2, axis=(1, 2)
                )
                position_error = np.linalg.norm(fine.centres_m - full.centres_m, axis=1)
                covariance_error = np.linalg.norm(
                    fine.covariances_m2 - full.covariances_m2, axis=(1, 2)
                )
                error = (
                    max(
                        float(np.max(position_error / position_scale, initial=0)),
                        float(np.max(covariance_error / covariance_scale, initial=0)),
                    )
                    / 3
                )
        except SpatialResolutionError as exc:
            exc.attempts = attempt
            exc.step_s = h
            raise
        except np.linalg.LinAlgError, FloatingPointError:
            error = np.inf
        if np.isfinite(error) and error <= 1:
            # Retain the exact event time rather than the sum of two half steps.
            time = target_time_s if h == target_time_s - state.time_s else state.time_s + h
            return (
                GaussianPuffs(
                    time, fine.release_times_s, fine.masses_kg, fine.centres_m, fine.covariances_m2
                ),
                error,
                attempt,
            )
        if h / 2 < floor:
            raise RuntimeError(
                f"minimum step exhausted; last accepted time={state.time_s}; error={error}"
            )
        h /= 2
    raise RuntimeError(f"step budget exhausted; last accepted time={state.time_s}")


def sample_forcing(
    state: GaussianPuffs,
    wind: Wind,
    diffusivity: Diffusivity,
    boundary: Boundary,
    envelope_sigma: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sample centres and six eigenvector probes with boundary and coverage checks.

    Args:
        state: Stable active distribution in the shared local frame.
        wind: Pure callable returning velocity (n,3) and gradient (n,3,3), SI units.
            Gradient indices are velocity component, then coordinate derivative.
        diffusivity: Pure callable returning PSD tensors (n,3,3) in m2/s, given
            points, time and original ages in seconds. Exceptions propagate.
        boundary: Free space or separable reflecting ground.
        envelope_sigma: Positive finite Mahalanobis radius of forcing support.

    Returns:
        Centre velocities, gradients, diffusivities, and per-axis maximum wind
        affine residuals (m/s) and diffusivity differences (m2/s). Last arrays
        have shape (p,3); axes follow ascending covariance eigenvalues, with
        horizontal eigenvectors first in reflecting mode.

    Raises:
        ValueError: Invalid samples, compressible flow or incompatible reflection.
            Successful probes leave unobserved variation between probes unresolved.
    """
    validate_boundary(state, boundary)
    if not np.isfinite(envelope_sigma) or envelope_sigma <= 0:
        raise ValueError("envelope_sigma must be finite and positive")
    p = len(state.masses_kg)
    values, vectors = puff_principal_axes(state, boundary)
    offsets = (vectors * (envelope_sigma * np.sqrt(values))[:, None, :]).swapaxes(1, 2)
    points = np.concatenate(
        (
            state.centres_m[:, None, :],
            state.centres_m[:, None, :] + offsets,
            state.centres_m[:, None, :] - offsets,
        ),
        axis=1,
    )
    if boundary == "reflecting":
        points[:, :, 2] = np.abs(points[:, :, 2])
    ages = np.broadcast_to((state.time_s - state.release_times_s)[:, None], (p, 7)).copy()
    # Callback buffers never alias the input snapshot.
    velocity, gradient = wind(points.reshape(-1, 3).copy(), state.time_s)
    diffusion = diffusivity(points.reshape(-1, 3).copy(), state.time_s, ages.ravel())
    velocity = np.asarray(velocity, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    diffusion = np.asarray(diffusion, dtype=np.float64)
    if velocity.shape != (7 * p, 3) or not np.all(np.isfinite(velocity)):
        raise ValueError("wind velocity must be finite with shape (n,3)")
    if gradient.shape != (7 * p, 3, 3) or not np.all(np.isfinite(gradient)):
        raise ValueError("wind gradient must be finite with shape (n,3,3)")
    if diffusion.shape != (7 * p, 3, 3):
        raise ValueError("diffusivity must have shape (n,3,3)")
    validate_tensors(diffusion, positive=False)
    if not np.allclose(np.trace(gradient, axis1=1, axis2=2), 0, atol=1e-12, rtol=0):
        raise ValueError("wind gradient must be locally incompressible")
    velocity = velocity.reshape(p, 7, 3)
    gradient = gradient.reshape(p, 7, 3, 3)
    diffusion = diffusion.reshape(p, 7, 3, 3)
    if boundary == "reflecting" and p:
        _check_reflecting(velocity, gradient, diffusion)
    displacement = points[:, 1:] - points[:, :1]
    predicted = velocity[:, :1] + np.einsum("pij,pqj->pqi", gradient[:, 0], displacement)
    residual = np.linalg.norm(velocity[:, 1:] - predicted, axis=-1)
    variation = np.linalg.norm(diffusion[:, 1:] - diffusion[:, :1], axis=(-2, -1))
    return (
        velocity[:, 0].copy(),
        gradient[:, 0].copy(),
        diffusion[:, 0].copy(),
        np.maximum(residual[:, :3], residual[:, 3:]),
        np.maximum(variation[:, :3], variation[:, 3:]),
    )


def _midpoint(
    state: GaussianPuffs,
    h: float,
    wind: Wind,
    diffusivity: Diffusivity,
    boundary: Boundary,
    envelope_sigma: float,
    spatial_check: Callable[[GaussianPuffs], float] | None,
) -> GaussianPuffs:
    velocity, _, _, _, _ = sample_forcing(state, wind, diffusivity, boundary, envelope_sigma)
    middle = GaussianPuffs(
        state.time_s + h / 2,
        state.release_times_s,
        state.masses_kg,
        state.centres_m + h / 2 * velocity,
        state.covariances_m2,
    )
    if spatial_check is not None:
        spatial_check(middle)
    velocity, gradient, diffusion, _, _ = sample_forcing(
        middle, wind, diffusivity, boundary, envelope_sigma
    )
    covariance = np.empty_like(state.covariances_m2)
    for i, (a, k, sigma) in enumerate(zip(gradient, diffusion, state.covariances_m2, strict=True)):
        block = np.zeros((6, 6))
        block[:3, :3], block[:3, 3:], block[3:, 3:] = a, 2 * k, -a.T
        exponential = expm(block * h)
        f = exponential[:3, :3]
        updated = f @ sigma @ f.T + exponential[:3, 3:] @ f.T
        covariance[i] = (updated + updated.T) / 2
    # Numerical loss of definiteness triggers refinement, never eigenvalue clipping.
    if not np.all(np.isfinite(covariance)):
        raise FloatingPointError("nonfinite affine covariance update")
    np.linalg.cholesky(covariance)
    result = GaussianPuffs(
        state.time_s + h,
        state.release_times_s,
        state.masses_kg,
        state.centres_m + h * velocity,
        covariance,
    )
    if spatial_check is not None:
        spatial_check(result)
    return result


def _check_reflecting(u: FloatArray, a: FloatArray, k: FloatArray) -> None:
    # The exact zeros define the supported separable model; variation comparisons
    # permit only floating-point evaluation differences, never physical shear.
    if np.any(u[:, :, 2] != 0) or np.any(a[:, :, 2, :] != 0) or np.any(a[:, :, :, 2] != 0):
        raise ValueError("reflection requires zero vertical wind and height-independent wind")
    if np.any(k[:, :, :2, 2] != 0):
        raise ValueError("reflecting diffusivity must be vertically separable")
    if not np.allclose(k[:, :, 2, 2], k[0, 0, 2, 2], rtol=1e-12, atol=0):
        raise ValueError("reflecting vertical diffusivity must be spatially uniform")
    for index in (3, 6):
        if not np.allclose(u[:, index], u[:, 0], rtol=1e-12, atol=0):
            raise ValueError("reflecting horizontal wind must be independent of height")
        if not np.allclose(k[:, index, :2, :2], k[:, 0, :2, :2], rtol=1e-12, atol=0):
            raise ValueError("reflecting horizontal diffusivity must be independent of height")
