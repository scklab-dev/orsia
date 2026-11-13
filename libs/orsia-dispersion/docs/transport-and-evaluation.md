# Puff transport and evaluation reference

Use this reference to configure transport, interpret numerical controls and connect mass fields to atmospheric columns. For an introduction and a single-puff example, see the [library guide](../README.md).

## Transport and molecular-column example

This synthetic example releases a finite mass, transports it with a uniform wind, integrates height layers, and converts mass columns to molecular columns. The example gas has a caller-selected molar mass of 0.044 kg/mol.

```python
import numpy as np
from orsia_dispersion import discretize_puff_emissions, simulate_puffs, puff_layer_columns

releases = discretize_puff_emissions(
    np.array([0.0, 2.0]), np.array([0.5]),
    np.array([[0.0, 0.0, 10.0]]), np.array([np.diag([4.0, 4.0, 1.0])]),
    partition_times_s=np.array([1.0, 2.0]),
)

def wind(points_m, time_s):
    return np.tile([1.0, 0.0, 0.0], (len(points_m), 1)), np.zeros((len(points_m), 3, 3))

def diffusivity(points_m, time_s, ages_s):
    return np.tile(np.diag([0.2, 0.2, 0.1]), (len(points_m), 1, 1))

outputs = list(simulate_puffs(
    releases, np.array([1.0, 2.0, 4.0]), wind, diffusivity,
    boundary="reflecting", rtol=1e-6, atol_position_m=1e-6,
    atol_covariance_m2=1e-6, min_step_s=1e-5, max_step_s=1.0,
    max_steps=1000, envelope_sigma=4.0, wind_tolerance_m_s=0.01,
    diffusivity_tolerance_m2_s=0.01, split_fraction=0.5,
    max_puffs=100, max_split_rounds=10,
))
state, diagnostics = outputs[-1]
assert diagnostics.released_mass_kg == 1.0
assert diagnostics.mass_residual_kg == 0.0
mass_columns = puff_layer_columns(
    state, np.array([[3.0, 0.0]]), np.array([0.0, 5.0, 10.0, 20.0]),
    boundary="reflecting",
)
background_molecules_m2 = np.full((1, 3), 1e23)
excess_molecules_m2 = mass_columns * (6.02214076e23 / 0.044)
total_molecules_m2 = background_molecules_m2 + excess_molecules_m2
assert total_molecules_m2.shape == (1, 3)
assert np.all(total_molecules_m2 >= background_molecules_m2)
```

## State and time contracts

`PuffReleases` stores nondecreasing `release_times_s (r,)`, positive `masses_kg (r,)`, `centres_m (r,3)` and symmetric positive-definite `covariances_m2 (r,3,3)`. `GaussianPuffs` stores the same population arrays and one common `time_s`, at or after every original release time. Active release times can be unordered after splitting. Empty populations are valid. Coordinates and covariance axes follow east, north, up in metres and square metres. Time is elapsed SI seconds from a caller-owned timezone-aware UTC epoch. The caller supplies one common chart, height datum and epoch for emissions, forcing and observation support.

Construction copies physical arrays to float64. Output records and evaluation arrays own writable buffers and share no mutable storage with inputs. Frozen records protect field bindings; callers must keep array buffers stable during a call. Operations revalidate population buffers, including mutations after construction. The iterator copies releases and output times when iteration starts and keeps yielded snapshots separate from its internal state. Independent calls on stable snapshots have no hidden execution state, random draws, file access or concurrency policy.

`discretize_puff_emissions` accepts constant rates per increasing interval and one centre/covariance per interval. It integrates interval mass exactly and places each nonzero release at the interval midpoint. `partition_times_s` inserts output or acquisition times into that partition; `stop_time_s` clips the included interval at an earlier endpoint. Mass matches the continuous source integral at partition edges. Between edges, discrete impulses retain emission-time approximation error. Refine the partition for additional time queries and source-near observations.

## Transport and forcing

`simulate_puffs` yields a `GaussianPuffs` state and `PuffDiagnostics` at each requested time. Pass the state to concentration, layer-column or path-column evaluation. Applications supply physical conversion, background and radiative composition.

`advance_puffs` advances a stable state without releases or splitting. `simulate_puffs` starts empty, advances to each event, inserts simultaneous releases once, checks spatial resolution, and yields at requested times. Outputs at release times include the release. Empty populations advance directly between events without forcing samples or transport attempts. Both operations stop substeps at declared `discontinuities_s`. Discontinuity callbacks must define their event-time value consistently, usually by the right-hand branch; midpoint coefficients remain within each event interval. Simulation checks event-time forcing on the accepted event state after inserting releases. Right-hand forcing at a discontinuity therefore subdivides that event state.

A `Wind` callable receives `(points_m, time_s)` and returns velocity `(p,3)` in m/s and gradient `(p,3,3)` in 1/s. The gradient convention is `A[a,b] = du[a]/dx[b]`. A `Diffusivity` callable receives `(points_m, time_s, ages_s)` and returns symmetric positive-semidefinite tensors `(p,3,3)` in m2/s. Ages retain the original release time after splitting. Flow must be locally incompressible; the gradient trace check uses an absolute roundoff tolerance of 1e-12 1/s. Callbacks receive independent point and age buffers and must remain pure, deterministic and order-independent. They must reject unsupported positions or times; their exceptions propagate.

Centre integration uses explicit midpoint trajectories. Covariance integration uses the exact affine matrix-exponential update for midpoint-frozen gradients and diffusivity, including the diffusion integral through a 6-by-6 block exponential. Symmetrization removes roundoff asymmetry; Cholesky failures trigger refinement without eigenvalue clipping. Step doubling compares a full step with two half steps and divides the discrepancy by three. The accepted state uses the two half steps. The error ratio is the maximum of the position Euclidean and covariance Frobenius errors over puffs, scaled by the supplied absolute tolerances and relative tolerance. Relative position scaling uses displacement during the step, so moving the chart origin does not change the scale. The covariance relative scale uses the fine-step covariance norm.

`min_step_s`, `max_step_s`, `max_steps`, relative and absolute tolerances are explicit controls. Short final event steps are allowed below `min_step_s`; such a step cannot be refined further below its initial size. Every attempted step counts against the budget, including rejected refinements and subdivision-triggering trials. Minimum-step, floating-point time-resolution and attempt exhaustion raise `RuntimeError` with the last accepted time. Invalid physical inputs raise `ValueError`.

## Boundaries and spatial resolution

Free space retains full three-dimensional covariance, including vertical tilt. For atmospheric use, establish negligible ground interaction on the relevant support. `PuffDiagnostics.below_ground_mass_kg` reports the whole free-space distribution's mass below z=0; that mass remains represented.

Reflecting mode uses the separable image construction at z=0. Centres and evaluation support must have nonnegative height. Covariances and diffusivity must have exactly zero horizontal/vertical cross components. Vertical wind and its gradients must vanish, horizontal wind must be height-independent, the horizontal diffusivity block must be height-independent, and vertical diffusivity must be spatially uniform. Sampled independence comparisons use relative tolerance 1e-12 and zero absolute tolerance. These conditions also apply across puffs with different ages. Stored vertical moments describe the underlying unreflected Gaussian; physical reflected moments differ.

Spatial checks sample the centre and positive/negative offsets along each covariance eigenvector at `envelope_sigma` standard deviations. Reflecting checks use horizontal-block eigenvectors and a separate vertical direction, mapping negative-height probes into the upper half-space. Simulation checks release events, accepted-step endpoints and trial midpoints, including half-step midpoints. Wind residuals compare samples with the local affine prediction; diffusivity residuals use Frobenius differences from the centre tensor. A failure selects the largest normalized residual direction. Current-state failures trigger subdivision at the current time. A trial-state failure maps its physical direction to the current eigenvector with greatest absolute overlap, subdivides the current state, and halves the failed trial duration subject to `min_step_s`. Subdivision creates spatial headroom while step refinement limits subsequent diffusive growth. The next accepted advance restores the maximum trial step. After subdivision, the criterion is checked again; the attempt, subdivision and population budgets bound retries.

`split_puffs` replaces each selected parent by two equal-mass children with displacement `sqrt(alpha*lambda)*e` and covariance reduced by `alpha*lambda*e*e.T`. The split conserves mass and the mixture's first two moments. Reflecting splits use only eigenvectors of the horizontal 2-by-2 block, including degenerate horizontal/vertical eigenvalue cases. Splits change density and higher moments; refine source, temporal, spatial and observation discretization separately. Required-split failures at `max_puffs` or `max_split_rounds` raise `RuntimeError` with the last accepted time and unresolved sampled ratio. A release exceeding `max_puffs` raises `RuntimeError` with its event time. Required releases are never suppressed and puffs outside an observation window remain active.

The forcing envelope uses the stored covariance's Mahalanobis-radius ellipsoid. `forcing_tail_probability` gives the selected kernel's probability outside it, with Gaussian tails as the default. Reflecting mode reports the same latent-tail mass fraction, an upper bound on physical mass outside the folded envelope. This diagnostic describes forcing support excluded from the closure checks. Default Gaussian evaluation includes all tails; an explicit `kernel_cutoff_radius` selects the [normalized compact kernel](fields-and-layers.md#compact-gaussian-kernels). The six probes do not certify coverage or bound arbitrary variability between probes, including mixed-coordinate terms that vanish on principal axes. The caller must establish continuous forcing coverage and the global reflecting restrictions on the declared envelope, then assess unresolved spatial structure. `sampled_variation_ratio` is the output-time probe result; `max_sampled_variation_ratio` includes pre-split failures and trial states. Neither quantity is a certified model-error bound.

## Concentration and integrals

`puff_concentration(state, points_m, boundary=...)` returns kg/m3 at points shaped `(...,3)`. `puff_layer_columns(state, horizontal_points_m, height_edges_m, boundary=...)` returns kg/m2 shaped `(...,layers)` on finite increasing height support. The full-covariance layer calculation combines the horizontal marginal density with the conditional vertical Gaussian probability. `puff_path_columns(state, starts_m, ends_m, boundary=...)` returns kg/m2 along paired finite straight segments shaped `(...,3)`. Zero-length segments return zero; reflecting endpoints must lie in the upper half-space. Path length is included exactly once.

Evaluation uses Cholesky solves, log determinants and stable Gaussian interval probabilities, with positive quadrature for narrow intervals. `batch_size` bounds point or segment temporaries; puffs accumulate serially one at a time. Every puff contributes with its selected kernel, without mass removal. The optional `kernel_cutoff_radius` is shared by point, layer and path evaluation. Pass the same cutoff to direct `simulate_puffs` calls for consistent below-ground and forcing-tail diagnostics. Floating-point underflow can return zero for concentrations below float64 range. Point values, spatial cell averages and path integrals have distinct meanings. Applications perform converged area, volume, exposure and radiance quadrature on their observation support. Finite height support excludes mass above the model top; extend the top or compare summed layer columns across tops to quantify that excluded column at each horizontal point.

The example's mass-to-molecule conversion is `mass_column * Avogadro_constant / molar_mass`. Applications compute puff columns directly on their atmospheric height boundaries, preserve layer order and add the selected gas background. Radiative and instrument composition remains application-owned. Independent vertical-column transfer omits horizontal changes along slanted solar and viewing paths; compare matched path integrals before adopting that approximation. Water perturbations, altered thermodynamics and abundance-dependent spectroscopy require separate composition contracts.

## Evidence and limits

The package test suite uses synthetic forcing and analytical constant-wind, diffusion, shear, reflection, split-moment and Gaussian-integral references. It also checks release accounting, convergence, failure budgets, buffer ownership and bounded concurrent calls. These checks establish behavior on the tested cases. Field realism, meteorological closure, emissions, terrain, radiance accuracy and retrieval performance require case-specific independent evidence. The effective locally frozen diffusivity closure does not include general diffusivity-gradient moment terms. Chemistry, deposition, buoyancy, terrain-dependent transport, merging, concentration variance and near-ground vertical shear remain outside the puff core. The separate [bounded particle closure](fields-and-layers.md#bounded-particle-flights) supplies stochastic snapshots through the common scene path.

Run focused checks from the Orsia root with `devbox run -- just --justfile tests/justfile pytest libs/orsia-dispersion/tests`. The workspace `just check` includes this package's lint, types, tests and per-library coverage gate. Public signatures and parameter details are documented in the [package source](../src/orsia_dispersion/__init__.py) and its modules.
