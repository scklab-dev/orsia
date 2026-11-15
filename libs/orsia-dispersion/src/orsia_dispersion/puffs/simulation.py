"""Declare puff models and snapshots, and simulate release, forcing and output events."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import numpy as np

from orsia_dispersion.puffs.kernels import (
    compact_kernel_parameters,
    kernel_axis_cdf,
    kernel_envelope_tail,
)
from orsia_dispersion.puffs.population import (
    GaussianPuffs,
    PuffReleases,
    puff_principal_axes,
    split_puffs,
    validate_boundary,
)
from orsia_dispersion.puffs.transport import (
    SpatialResolutionError,
    sample_forcing,
    transport_step,
    validate_integration,
)
from orsia_dispersion.transport import Boundary, Diffusivity, FloatArray, Wind


@dataclass(frozen=True)
class PuffDiagnostics:
    """Mass accounting and sampled numerical evidence at an output time.

    Attributes:
        initial_mass_kg: Initial active mass; zero for release-driven simulations.
        released_mass_kg: Cumulative scheduled mass inserted through this output.
        active_mass_kg: Total represented mass over the selected kernel's full support.
        mass_residual_kg: Active minus initial minus released mass.
        below_ground_mass_kg: Free-space mass below z=0; zero for reflection.
        forcing_tail_probability: Probability outside the radius-sigma 3-D ellipsoid.
            This is excluded forcing support, not removed mass or a forcing-error bound.
        max_integration_error_ratio: Maximum accepted local step-doubling error/tolerance.
        sampled_variation_ratio: Maximum normalized spatial residual at this output.
        max_sampled_variation_ratio: Maximum residual seen, including pre-split trials.
        attempted_steps: Cumulative attempted transport steps, including refinements.
            Empty-population event advances consume no attempts.
        split_count: Cumulative count of parents replaced by pairs.
    """

    initial_mass_kg: float
    released_mass_kg: float
    active_mass_kg: float
    mass_residual_kg: float
    below_ground_mass_kg: float
    forcing_tail_probability: float
    max_integration_error_ratio: float
    sampled_variation_ratio: float
    max_sampled_variation_ratio: float
    attempted_steps: int
    split_count: int


@dataclass(frozen=True)
class GaussianPuffField:
    """Associate one puff population with its physical boundary and diagnostics.

    Attributes:
        state: Borrowed GaussianPuffs snapshot at state.time_s. Its owned buffers
            remain writable; sample_fields supplies independent snapshots.
        boundary: Free space or separable reflection at chart altitude zero.
        diagnostics: Optional transport diagnostics at the same output time.
            Keep state buffers stable throughout spatial evaluation.
        kernel_cutoff_radius: Optional positive ellipsoidal cutoff in inflated
            Mahalanobis units, preserving mass, centre and unreflected covariance.
    """

    state: GaussianPuffs
    boundary: Boundary
    diagnostics: PuffDiagnostics | None = None
    kernel_cutoff_radius: float | None = None

    def __post_init__(self) -> None:
        """Validate the supplied state and boundary.

        Raises:
            ValueError: The population or reflecting boundary is invalid.
        """
        validate_boundary(self.state, self.boundary)
        compact_kernel_parameters(self.kernel_cutoff_radius)


@dataclass(frozen=True, kw_only=True)
class GaussianPuffDispersion:
    """Own scheduled emissions and the numerical policy for their transport.

    Attributes:
        releases: Explicit emission history, including pre-observation releases.
            Construction snapshots arrays; keep the owned buffers unchanged during sampling.
        wind: Pure incompressible velocity/gradient callback in SI units.
        diffusivity: Pure diffusivity callback in m2/s, using original puff ages.
        boundary: Free space or separable reflection at z=0.
        rtol: Positive relative transport tolerance.
        atol_position_m: Positive absolute position tolerance, m.
        atol_covariance_m2: Positive absolute covariance tolerance, m2.
        min_step_s: Positive minimum refinement step, s.
        max_step_s: Maximum transport step, s, at least min_step_s.
        max_steps: Positive total attempted-step budget per sampling call.
        envelope_sigma: Positive forcing-envelope radius in standard deviations.
        wind_tolerance_m_s: Positive sampled affine-wind residual tolerance, m/s.
        diffusivity_tolerance_m2_s: Positive sampled diffusivity tolerance, m2/s.
        split_fraction: Eigenvalue fraction in (0,1) moved into child separation.
        max_puffs: Positive population budget.
        max_split_rounds: Positive consecutive subdivision budget.
        discontinuities_s: Increasing finite forcing discontinuity times, s.
            Each sampling call restarts from the declared history. Numerical policy
            has the same validation and failure contract as simulate_puffs.
        kernel_cutoff_radius: Optional positive moment-preserving compact cutoff.
            None keeps Gaussian tails. This differs from the forcing envelope and
            guarantees compact support, without a finite propagation-speed law.
    """

    releases: PuffReleases
    wind: Wind
    diffusivity: Diffusivity
    boundary: Boundary
    rtol: float
    atol_position_m: float
    atol_covariance_m2: float
    min_step_s: float
    max_step_s: float
    max_steps: int
    envelope_sigma: float
    wind_tolerance_m_s: float
    diffusivity_tolerance_m2_s: float
    split_fraction: float
    max_puffs: int
    max_split_rounds: int
    discontinuities_s: tuple[float, ...] = ()
    kernel_cutoff_radius: float | None = None

    def __post_init__(self) -> None:
        """Snapshot the release history independently of caller buffers.

        Raises:
            ValueError: The release history is invalid.
        """
        r = self.releases
        compact_kernel_parameters(self.kernel_cutoff_radius)
        object.__setattr__(
            self,
            "releases",
            PuffReleases(r.release_times_s, r.masses_kg, r.centres_m, r.covariances_m2),
        )


def simulate_puffs(
    releases: PuffReleases,
    output_times_s: FloatArray,
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
    wind_tolerance_m_s: float,
    diffusivity_tolerance_m2_s: float,
    split_fraction: float,
    max_puffs: int,
    max_split_rounds: int,
    discontinuities_s: tuple[float, ...] = (),
    kernel_cutoff_radius: float | None = None,
) -> Iterator[tuple[GaussianPuffs, PuffDiagnostics]]:
    """Yield independent snapshots with explicit mass and sampled-error diagnostics.

    Args:
        releases: Finite scheduled sources, copied when iteration starts.
        output_times_s: Finite strictly increasing times (o,), possibly empty.
            Output at a release includes that release; earlier outputs have zero mass.
        wind: Pure incompressible velocity/gradient callable in SI units.
        diffusivity: Pure PSD diffusivity callable with original puff ages in seconds.
        boundary: 'free' or 'reflecting' under the separable forcing contract.
        rtol: Positive relative displacement/covariance tolerance.
        atol_position_m: Positive absolute position tolerance in metres.
        atol_covariance_m2: Positive absolute covariance tolerance in m2.
        min_step_s: Positive minimum refinement step, seconds.
        max_step_s: Maximum step, at least min_step_s; each step probes spatial variation.
            Trial spatial failures subdivide the current state along the physical failed
            direction and halve trial duration, bounded by min_step_s. Event-time forcing
            is checked on the accepted event state, after release insertion.
        max_steps: Total attempted transport-step budget for the entire iterator.
        envelope_sigma: Positive forcing envelope Mahalanobis radius.
        wind_tolerance_m_s: Positive absolute norm tolerance for affine wind residuals.
        diffusivity_tolerance_m2_s: Positive Frobenius diffusivity variation tolerance.
        split_fraction: Fraction in (0,1) of an eigenvalue moved into child separation.
        max_puffs: Positive maximum population; required releases/splits fail on exhaustion.
        max_split_rounds: Positive consecutive subdivision budget between accepted advances.
        discontinuities_s: Strictly increasing finite times at which transport stops.
        kernel_cutoff_radius: Optional positive moment-preserving kernel cut for
            mass diagnostics. Moment transport is unchanged; use the same cutoff
            when evaluating the yielded state. None selects Gaussian tails.

    Yields:
        Fresh writable state and immutable scalar diagnostics at each requested time.
        Mutating a yielded state cannot change future evolution. No removals or merging.
        Empty populations advance between events without forcing samples or step attempts.

    Raises:
        ValueError: Inputs or sampled forcing violate documented contracts.
        RuntimeError: Integration, population or subdivision budget exhausted. Error
            includes failing event or last accepted time; previous outputs remain valid.
    """
    multiplier, _ = compact_kernel_parameters(kernel_cutoff_radius)
    tail = kernel_envelope_tail(envelope_sigma, kernel_cutoff_radius)
    model = GaussianPuffDispersion(
        releases=releases,
        wind=wind,
        diffusivity=diffusivity,
        boundary=boundary,
        rtol=rtol,
        atol_position_m=atol_position_m,
        atol_covariance_m2=atol_covariance_m2,
        min_step_s=min_step_s,
        max_step_s=max_step_s,
        max_steps=max_steps,
        envelope_sigma=envelope_sigma,
        wind_tolerance_m_s=wind_tolerance_m_s,
        diffusivity_tolerance_m2_s=diffusivity_tolerance_m2_s,
        split_fraction=split_fraction,
        max_puffs=max_puffs,
        max_split_rounds=max_split_rounds,
        discontinuities_s=discontinuities_s,
        kernel_cutoff_radius=kernel_cutoff_radius,
    )
    releases = model.releases
    times = np.array(output_times_s, dtype=np.float64, copy=True)
    breaks = np.asarray(discontinuities_s, dtype=np.float64)
    for name, values in (("outputs", times), ("discontinuities", breaks)):
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} must be finite strictly increasing vectors")
    validate_integration(
        rtol, atol_position_m, atol_covariance_m2, min_step_s, max_step_s, max_steps, envelope_sigma
    )
    tolerances = (wind_tolerance_m_s, diffusivity_tolerance_m2_s, split_fraction)
    if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0 or split_fraction >= 1:
        raise ValueError("spatial tolerances must be positive; split_fraction must be in (0,1)")
    if max_puffs < 1 or max_split_rounds < 1:
        raise ValueError("population and subdivision budgets must be positive")
    start = min(
        times[0] if len(times) else 0, releases.release_times_s[0] if len(releases.masses_kg) else 0
    )
    current = GaussianPuffs(
        float(start), np.empty(0), np.empty(0), np.empty((0, 3)), np.empty((0, 3, 3))
    )
    validate_boundary(current, boundary)
    if not len(times):
        return
    events = np.unique(np.concatenate((times, releases.release_times_s, breaks)))
    events = events[(events >= start) & (events <= times[-1])]
    evolution = _PuffEvolution(model, current)
    for event in events:
        evolution.advance_to(float(event))
        evolution.insert_releases(float(event))
        ratio = evolution.resolve_spatial()
        if np.any(times == event):
            current = evolution.current
            diagnostic = _puff_diagnostics(
                current,
                boundary=boundary,
                kernel_cutoff_radius=kernel_cutoff_radius,
                multiplier=multiplier,
                tail=tail,
                released_mass=evolution.released_mass,
                max_error=evolution.max_error,
                ratio=ratio,
                max_variation=evolution.max_variation,
                attempts=evolution.attempts,
                splits=evolution.splits,
            )
            yield (
                GaussianPuffs(
                    current.time_s,
                    current.release_times_s,
                    current.masses_kg,
                    current.centres_m,
                    current.covariances_m2,
                ),
                diagnostic,
            )


class _PuffEvolution:
    """Own one iterator's active population, release cursor and cumulative diagnostics.

    The model belongs to the iterator and owns its release-history copy. Current
    states are replaced after transport, release insertion or subdivision. Yielded
    snapshots copy those buffers; evolution objects are never shared between calls.
    """

    def __init__(self, model: GaussianPuffDispersion, current: GaussianPuffs) -> None:
        """Retain validated transport policy and an owned empty initial population."""
        self.model = model
        self.current = current
        self.attempts = self.splits = self.rounds = self.inserted = 0
        self.max_error = self.max_variation = self.released_mass = 0.0

    def advance_to(self, event: float) -> None:
        """Advance to an event, coordinating spatial splitting and trial duration.

        Args:
            event: Next release, output or forcing-break time in seconds.

        Raises:
            RuntimeError: Attempt, integration or subdivision budgets are exhausted.
            ValueError: Sampled forcing violates its physical contract.
        """
        trial_step_s = self.model.max_step_s
        check_trial = partial(self.check_trial, event_time=event)
        while self.current.time_s < event:
            try:
                self.check_spatial(self.current)
                if len(self.current.masses_kg) and self.attempts >= self.model.max_steps:
                    raise RuntimeError(
                        f"step budget exhausted; last accepted time={self.current.time_s}"
                    )
                next_state, error, used = transport_step(
                    self.current,
                    float(event),
                    self.model.wind,
                    self.model.diffusivity,
                    boundary=self.model.boundary,
                    rtol=self.model.rtol,
                    atol_position_m=self.model.atol_position_m,
                    atol_covariance_m2=self.model.atol_covariance_m2,
                    min_step_s=self.model.min_step_s,
                    max_step_s=trial_step_s,
                    max_steps=self.model.max_steps - self.attempts,
                    envelope_sigma=self.model.envelope_sigma,
                    spatial_check=check_trial,
                )
            except SpatialResolutionError as failure:
                self.attempts += failure.attempts
                self.subdivide(failure)
                if failure.step_s:
                    # Splitting creates current-state spatial headroom; shortening
                    # the trial bounds growth from diffusion before the next check.
                    # Neither operation alone guarantees progress at a size threshold.
                    trial_step_s = max(self.model.min_step_s, failure.step_s / 2)
                continue
            self.current = next_state
            self.attempts += used
            self.max_error = max(self.max_error, error)
            self.rounds = 0
            trial_step_s = self.model.max_step_s

    def insert_releases(self, event: float) -> None:
        """Insert all scheduled releases through the accepted event time.

        Args:
            event: Accepted event time in seconds, inclusive for release insertion.

        Raises:
            RuntimeError: Required releases exceed the population budget.
        """
        releases = self.model.releases
        end = int(np.searchsorted(releases.release_times_s, event, side="right"))
        if end > self.inserted:
            if len(self.current.masses_kg) + end - self.inserted > self.model.max_puffs:
                raise RuntimeError(f"release exceeds max_puffs at event time={event}")
            self.current = GaussianPuffs(
                float(event),
                np.concatenate(
                    (self.current.release_times_s, releases.release_times_s[self.inserted : end])
                ),
                np.concatenate((self.current.masses_kg, releases.masses_kg[self.inserted : end])),
                np.concatenate((self.current.centres_m, releases.centres_m[self.inserted : end])),
                np.concatenate(
                    (self.current.covariances_m2, releases.covariances_m2[self.inserted : end])
                ),
            )
            self.released_mass = float(np.sum(releases.masses_kg[:end]))
            self.inserted = end

    def resolve_spatial(self) -> float:
        """Subdivide the event population until its sampled forcing is resolved.

        Returns:
            Largest normalized spatial residual for the resolved event state.
        """
        while True:
            try:
                return self.check_spatial(self.current)
            except SpatialResolutionError as failure:
                self.subdivide(failure)

    def check_trial(self, state: GaussianPuffs, *, event_time: float) -> float:
        """Check upstream trials; event forcing is checked after release insertion.

        Returns:
            Sampled residual ratio, or zero at and beyond the next event.
        """
        return self.check_spatial(state) if state.time_s < event_time else 0.0

    def check_spatial(self, state: GaussianPuffs) -> float:
        """Measure spatial resolution and retain the maximum trial/event residual.

        Returns:
            Largest normalized sampled residual, zero for an empty population.

        Raises:
            ValueError: Reflecting forcing has unresolved vertical variation.
            SpatialResolutionError: A probe exceeds its spatial tolerance.
        """
        if not len(state.masses_kg):
            return 0.0
        _, _, _, residual, variation = sample_forcing(
            state,
            self.model.wind,
            self.model.diffusivity,
            self.model.boundary,
            self.model.envelope_sigma,
        )
        ratios = np.maximum(
            residual / self.model.wind_tolerance_m_s,
            variation / self.model.diffusivity_tolerance_m2_s,
        )
        ratio = float(np.max(ratios, initial=0))
        self.max_variation = max(self.max_variation, ratio)
        bad = np.flatnonzero(np.max(ratios, axis=1) > 1)
        if len(bad):
            axes = np.argmax(ratios[bad], axis=1).astype(np.int64)
            if self.model.boundary == "reflecting" and np.any(axes == 2):
                raise ValueError("reflecting forcing has unresolved vertical variation")
            _, vectors = puff_principal_axes(state, self.model.boundary)
            raise SpatialResolutionError(bad, vectors[bad, :, axes].copy(), ratio)
        return ratio

    def subdivide(self, failure: SpatialResolutionError) -> None:
        """Replace failing parents along the current axes matching failed probes.

        Raises:
            RuntimeError: Consecutive subdivision or population budget is exhausted.
        """
        self.rounds += 1
        if (
            self.rounds > self.model.max_split_rounds
            or len(self.current.masses_kg) + len(failure.indices) > self.model.max_puffs
        ):
            raise RuntimeError(
                f"split budget exhausted at event/last accepted time={self.current.time_s}; "
                f"unresolved sampled ratio={failure.ratio}"
            ) from failure
        self.splits += len(failure.indices)
        # Trial and accepted eigenvalue order can differ. Select the current
        # eigenvector with greatest physical overlap with the failed probe.
        _, vectors = puff_principal_axes(self.current, self.model.boundary)
        overlap = np.abs(np.einsum("pij,pi->pj", vectors[failure.indices], failure.directions))
        axes = np.argmax(overlap[:, :2] if self.model.boundary == "reflecting" else overlap, axis=1)
        self.current = split_puffs(
            self.current,
            failure.indices,
            axes,
            np.full(len(failure.indices), self.model.split_fraction),
            boundary=self.model.boundary,
        )


def _puff_diagnostics(
    state: GaussianPuffs,
    *,
    boundary: Boundary,
    kernel_cutoff_radius: float | None,
    multiplier: float,
    tail: float,
    released_mass: float,
    max_error: float,
    ratio: float,
    max_variation: float,
    attempts: int,
    splits: int,
) -> PuffDiagnostics:
    """Combine mass accounting with supplied scalar evolution diagnostics.

    State buffers are borrowed without mutation. The kernel multiplier and tail
    probability come from the same cutoff and forcing envelope as the simulation.

    Returns:
        Immutable scalar diagnostics at the supplied state's time.
    """
    mass = float(np.sum(state.masses_kg))
    below = (
        0.0
        if boundary == "reflecting"
        else float(
            np.sum(
                state.masses_kg
                * kernel_axis_cdf(
                    -state.centres_m[:, 2] / (np.sqrt(state.covariances_m2[:, 2, 2]) * multiplier),
                    kernel_cutoff_radius,
                )
            )
        )
    )
    return PuffDiagnostics(
        0.0,
        released_mass,
        mass,
        mass - released_mass,
        below,
        tail,
        max_error,
        ratio,
        max_variation,
        attempts,
        splits,
    )
