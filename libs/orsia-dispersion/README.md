# Orsia Dispersion

`orsia-dispersion` models excess material with stationary Gaussian plumes, Gaussian or compact puffs, and bounded-velocity mass particles. All models share one time-sampling and pixel/layer-integration interface:

1. Construct `StationaryGaussianPlume`, `GaussianPuffDispersion` or `BoundedParticleDispersion` with explicit physical parameters.
2. Call `sample_fields(model, times_s, rng=...)` to obtain a mass field at each observation time. Particles require an explicit generator; Gaussian models consume no random draws.
3. Call `pixel_layer_moments` to integrate each field on the supplied pixels, altitude layers and terrain.
4. Pass the resulting `LayerMassMoments` to Scene's `gas_layer_columns`, then bind each `GasLayerColumns` snapshot to a prepared physical scene.

The package requires NumPy, SciPy and the NumPy-only Geometry base package. Dispersion owns mass distributions, transport and integration. [Scene](../orsia-scene/README.md#exact-level-grid-extinction-scenes) owns gas conversion and radiance composition. Applications select species, molar masses, atmosphere, instrument response and observation support.

## Common field and scene path

![StationaryGaussianPlume, GaussianPuffDispersion with Gaussian or compact kernels, and BoundedParticleDispersion feed sample_fields. Pixel integration or particle deposition produces LayerMassMoments. Scene converts the moments to GasLayerColumns for physical radiance evaluation.](docs/diagrams/architecture.svg)

Arrows carry model declarations, support or calculated arrays. Green blocks belong to Dispersion, purple blocks to Scene, and blue blocks are application inputs. The central operations name their outputs below the function names. The three model declarations feed the same operations. Puff declarations select Gaussian or compact kernels. Particle sampling consumes the supplied random generator. Scene preparation is independent of the gas snapshot selected by each binding. [Full-size SVG](docs/diagrams/architecture.svg) · [PNG](docs/diagrams/architecture.png) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

## Sample all models on the same support

This executable synthetic example constructs a constant 1 kg/s plume, a 10 kg puff with either kernel, and 256 particles carrying 10 kg in total. Their physics differ. All four configurations follow the same loop at two observation times. Numerical values are illustrative, without atmospheric calibration.

```python
from dataclasses import replace

import numpy as np
from orsia_dispersion import BoundedParticleDispersion, GaussianPuffDispersion, ParticleReleases, PuffReleases, StationaryGaussianPlume, pixel_layer_moments, sample_fields
from orsia_geometry import LayeredCells, LocalCartesianFrame, RectangularCells
from orsia_scene import gas_layer_columns

stationary = StationaryGaussianPlume(
    emission_rate_kg_s=1.0,
    wind_speed_m_s=5.0,
    source_regularization_m=25.0,
    stack_height_m=100.0,
    stack_inside_diameter_m=2.0,
    stack_exit_velocity_m_s=20.0,
    stack_exit_temperature_k=290.0,
    ambient_temperature_k=290.0,
)


def wind(points_m, time_s):
    return np.tile([5.0, 0.0, 0.0], (len(points_m), 1)), np.zeros((len(points_m), 3, 3))


def diffusivity(points_m, time_s, ages_s):
    return np.tile(np.eye(3), (len(points_m), 1, 1))


puffs = GaussianPuffDispersion(
    releases=PuffReleases(
        np.array([0.0]), np.array([10.0]), np.array([[0.0, 0.0, 100.0]]),
        np.array([np.diag([100.0, 100.0, 100.0])]),
    ),
    wind=wind, diffusivity=diffusivity, boundary="reflecting",
    rtol=1e-6, atol_position_m=1e-8, atol_covariance_m2=1e-8,
    min_step_s=1e-6, max_step_s=1.0, max_steps=1000,
    envelope_sigma=3.0, wind_tolerance_m_s=0.1,
    diffusivity_tolerance_m2_s=0.1, split_fraction=0.5,
    max_puffs=100, max_split_rounds=10,
)

compact = replace(puffs, kernel_cutoff_radius=3.0)
particles = BoundedParticleDispersion(
    releases=ParticleReleases(
        np.zeros(256), np.full(256, 10.0 / 256),
        np.tile([0.0, 0.0, 100.0], (256, 1)),
    ),
    wind=wind, wind_speed_bound_m_s=5.0,
    turbulent_covariance_m2_s2=np.diag([1.0, 1.0, 0.25]),
    correlation_time_s=2.0, boundary="reflecting",
    max_step_s=1.0, max_steps=10000, max_particles=1000,
)

lower = np.array([[[-25.0, -25.0], [25.0, -25.0], [75.0, -25.0]]])
upper = lower + 50.0
edges = np.array([0.0, 90.0, 110.0, 1000.0])
terrain = np.zeros((1, 3))
frame = LocalCartesianFrame("synthetic plume", "synthetic altitude")
support = LayeredCells(RectangularCells(lower, upper, frame), edges, terrain)

for model in (stationary, puffs, compact, particles):
    rng = np.random.default_rng(42)
    for field in sample_fields(model, np.array([0.0, 10.0]), rng=rng):
        mass = pixel_layer_moments(
            field, support,
            rtol=1e-7, atol_kg_m2=1e-12,
        )
        gas = gas_layer_columns(mass, species="CO2", molar_mass_kg_mol=0.0440095)
        assert gas.columns_molecules_m2.shape == (1, 3, 3)
        assert np.all(gas.columns_molecules_m2 >= 0)
        assert gas.columns_molecules_m2.sum() > 0
```

Pixel corners have `(row,column,2)` axes in east/north order, in metres. All altitude interfaces, terrain and fields share one vertical datum. The example produces `(row,column,layer)` arrays. `LayerMassMoments.columns_kg_m2` contains layer mass per full pixel area; `height_moments_kg_m` contains its first altitude moment per area. Their ratio gives the nonempty layer's mean altitude. Scene converts mass to molecular columns using the supplied molar mass and Avogadro constant. It preserves the centroid and the exact `LayeredCells` support owner.

For an already prepared `DirectSolarScene`, use `bound = scene.with_gas((gas,))` inside the loop, then evaluate `bound.radiance(xy_m, view_local, times_s, spectral)`. Bind an empty tuple for background only. Each binding shares unchanged physical preparation and asserts one frozen snapshot over its declared time interval. Instrument response and exposure integration remain separate.

## Spatial evaluation and numerical controls

- `concentration(field, points_m)` evaluates continuous plume or puff densities in kg/m3 at east/north/altitude points.
- `path_columns(field, starts_m, ends_m)` integrates straight paths through puff fields in kg/m2, preserving their reflection and cutoff. Stationary and raw particle paths raise `TypeError`.
- `layer_columns(field, horizontal_points_m, height_edges_m, ...)` integrates continuous fields at horizontal points, in kg/m2.
- `pixel_layer_moments(field, support, ...)` averages mass and first altitude moment over rectangular pixels and clipped layers.

`LayerMassMoments` owns immutable float64 amount arrays and retains the complete `LayeredCells` support, including horizontal bounds and frame. Terrain clips each layer before both integrals. Mass outside the horizontal domain, above the finite top or below the retained surface remains excluded, without renormalization. Continuous fields use analytic Gaussian integrals and adaptive quadrature; compact kernels couple the coordinates even for diagonal covariance. Particles deposit mass and altitude moments directly into half-open pixel/layer cells `[lower, upper)`. Particle pixels must have disjoint interiors. Raw particles have no point concentration or point column; those calls raise `TypeError`. The [field guide](docs/fields-and-layers.md) documents tolerances, integration budgets, ownership and failure behavior.

## Physical scope and limits

`StationaryGaussianPlume` describes a constant source in a constant eastward wind, with class-D widths, Briggs rise, optional first-order removal and a flat reflecting plane. Pixels align with east/north coordinates. Arbitrary wind directions and rotated footprints are unsupported. `sample_fields` yields the same immutable stationary declaration at each requested time.

`GaussianPuffDispersion` transports a declared emission history with supplied wind and diffusivity. Each sampled field owns an independent puff state and carries its boundary, kernel cutoff and transport diagnostics. `kernel_cutoff_radius=None` preserves Gaussian tails. A positive radius selects a normalized ellipsoidal cut, rescaled to preserve mass, centre and unreflected covariance. The cutoff differs from `envelope_sigma`, which controls forcing probes. Compact puffs retain the existing moment evolution and splitting; they have no bounded propagation-speed law. Free-space and flat reflecting-ground boundaries retain the [documented covariance and forcing restrictions](docs/transport-and-evaluation.md#boundaries-and-spatial-resolution).

`BoundedParticleDispersion` uses homogeneous turbulent velocity covariance `R` and Poisson renewal time `tau`, both supplied explicitly. Turbulent speed is bounded by `sqrt(3*lambda_max(R))`. The caller declares a global mean-wind speed bound; sampled checks reject violations but cannot certify a callback globally. Each numerical segment preserves the combined displacement bound. Uniform wind gives ballistic transport between renewals; variable wind uses first-order Euler segments and requires step convergence checks. Releases, renewals, wind discontinuities and step boundaries determine trajectories independently of observation times. A fresh identically seeded generator reproduces a sampling call. Particle snapshots own read-only arrays. Reflection is specular at altitude zero, with zero vertical mean wind and vertically separable turbulence.

In free space and uniform wind, the homogeneous random-flight closure has stationary turbulent velocity covariance `R` and long-time diffusivity `tau*R`. Synthetic tests check these declared statistics, mass, support, output-clock independence and projection. Stratified, convective or spatially varying turbulence needs a separately qualified closure. Chemistry, deposition and buoyant rise are outside the transient models.

Terrain clipping preserves each model's transport and reflection physics. Scene's centroid projection approximates the full within-cell distribution. Evolution during exposures requires a separate temporal policy. Software interchangeability and synthetic numerical tests do not establish physical equivalence between the models or accuracy for a real atmosphere.

## Source layout

The package root exports the common interface and the specialized puff and particle operations. Model-specific calculations live with their model; the shared sampling and spatial operations use those implementations.

| Module | Responsibility |
|---|---|
| [`emissions.py`](src/orsia_dispersion/emissions.py) | Integrate emission rates into release times and masses |
| [`forcing.py`](src/orsia_dispersion/forcing.py) | Shared flow callbacks, boundary choices and physical tensor validation |
| [`stationary.py`](src/orsia_dispersion/stationary.py) | Stationary Gaussian plume declaration and physical profile |
| [`particles.py`](src/orsia_dispersion/particles.py) | Particle emissions, states, model and transport |
| [`sampling.py`](src/orsia_dispersion/sampling.py) | Common model/field unions and time sampling |
| [`evaluation.py`](src/orsia_dispersion/evaluation.py) | Common concentrations, columns and pixel/layer mass moments |
| [`puffs/population.py`](src/orsia_dispersion/puffs/population.py) | Puff emissions, populations, invariants and conservative subdivision |
| [`puffs/kernels.py`](src/orsia_dispersion/puffs/kernels.py) | Gaussian and compact kernels, normalization and support |
| [`puffs/transport.py`](src/orsia_dispersion/puffs/transport.py) | Forcing probes and evolution of centres and covariances |
| [`puffs/simulation.py`](src/orsia_dispersion/puffs/simulation.py) | Puff models, field snapshots, event simulation and diagnostics |
| [`puffs/evaluation.py`](src/orsia_dispersion/puffs/evaluation.py) | Specialized puff concentrations and layer/path integrals |
| [`__init__.py`](src/orsia_dispersion/__init__.py) | Public root exports |

The current finite-cell API requires a complete `LayeredCells` support. Root imports expose the common operations; direct imports follow the responsibility layout below. Direct submodule imports use the paths above. `fields.py` is split between `stationary.py`, `puffs/simulation.py` and `sampling.py`. The former common `integration.py` becomes `evaluation.py`; puff evaluation moves to `puffs/evaluation.py`. The former `puffs.py` and `splitting.py` are combined in `puffs/population.py`. Puff kernels, transport and simulation move under `puffs/`. Forcing probes move from `forcing.py` to `puffs/transport.py`; shared tensor validation moves from `puffs.py` to `forcing.py`. The old paths have no compatibility aliases. Moving class definitions also changes their qualified module names; persisted Python pickles that reference old paths require explicit migration.

## Breaking migration and further use

Construct `LayeredCells(horizontal, height_edges_m, surface_altitude_m)` explicitly. Mass and gas results retain that same immutable support. `gas_layer_columns` replaces `prepare_gas_layers` and returns `GasLayerColumns.columns_molecules_m2` in molecules/m2; divide by `6.02214076e23` for mol/m2. Equal shapes, names or terrain arrays alone do not establish common coordinates. No implicit remapping or compatibility alias is supplied.

The historical `stationary_mass_column` and `stationary_vertical_weights` functions are removed. The common mass-moment path above replaces these functions; [migration details](docs/fields-and-layers.md#specialized-puff-operations-and-breaking-migration) cover units, finite support and archive ownership. The current `stationary.py` contains the stationary model and its physical profile. Historical reference conversion belongs to the consuming application's explicit archive importer.

- [Common fields and scene layers](docs/fields-and-layers.md): compact-kernel equations, bounded-flight contracts, support, units, ownership, numerical controls and migration.
- [Puff transport and evaluation reference](docs/transport-and-evaluation.md): direct puff operations, forcing callbacks and complete transport example.
- [Executable puff notebook](../../notebooks/gaussian_puff_dispersion.ipynb): synthetic transport and concentration maps.
- [Orsia workspace guide](../../README.md): development setup and library inventory.
