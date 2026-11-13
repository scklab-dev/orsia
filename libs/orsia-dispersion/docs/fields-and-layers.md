# Common mass fields and scene layers

`StationaryGaussianPlume`, `GaussianPuffDispersion` and `BoundedParticleDispersion` share time sampling and pixel/layer integration. Applications choose the model at construction, then call `sample_fields` and `pixel_layer_moments`. Continuous plume and puff fields also support `concentration` and `layer_columns`; raw particle fields raise `TypeError` for those point evaluations.

## Model, field and support

A model determines the distribution over time. A field represents one spatial distribution. `StationaryGaussianPlume` serves both roles because its scalar parameters remain constant. `GaussianPuffDispersion` owns the emission history and transport policy; each sampled `GaussianPuffField` binds one independent `GaussianPuffs` state to its physical boundary, kernel cutoff and transport diagnostics. `BoundedParticleDispersion` owns scheduled point releases, homogeneous turbulence, a bounded wind callback and numerical budgets. Each sampled `ParticleMassField` binds read-only `MassParticles` to its boundary and mass diagnostics.

`sample_fields(model, times_s, rng=None)` accepts finite, strictly increasing times, including an empty vector. Emission times and observation times share an explicit clock. Sampling includes every declared release at or before each observation. Each sampling call restarts from the declared history. Yielded states remain independent as iteration advances. Particles require an explicit `numpy.random.Generator`; Gaussian models consume no draws. Use separate generators for concurrent particle calls and fresh identically seeded generators for reproducibility.

Pixel corners have axes `(*pixels, 2)` in east/north order and units of metres. Atmospheric interfaces have axes `(layers + 1,)`. Interfaces, terrain and the field's altitude coordinate use the same vertical datum. Terrain is constant within each supplied pixel. The integration clips partially intersected layers before calculating both mass and its first altitude moment.

`pixel_layer_moments(field, support, ...)` receives Geometry's `LayeredCells`. `LayerMassMoments` retains this immutable support and owns immutable float64 amount arrays:

| Field | Axes | Meaning and units |
|---|---|---|
| `support.horizontal` | `(*pixels,2)` corners | Complete `RectangularCells`, including local frame and finite areas |
| `columns_kg_m2` | `(*pixels, layers)` | Layer mass divided by full horizontal pixel area, kg/m2 |
| `height_moments_kg_m` | `(*pixels, layers)` | Integral of altitude times mass divided by pixel area, kg/m |
| `support.height_edges_m` | `(layers + 1,)` | Increasing finite altitude interfaces, m |
| `support.surface_altitude_m` | `(*pixels,)` | Terrain altitude in the same datum, m |

The quotient of first moment and column gives the mean altitude in a nonempty layer. Empty layers have zero mass and moment. Finite-top, horizontal-domain and below-surface mass remain excluded. Integration performs no renormalization. Terrain clipping preserves the model's original reflection plane; it does not simulate transport over uneven terrain.

## Straight paths

`path_columns(field, starts_m, ends_m, batch_size=4096)` accepts `GaussianPuffField` and delegates to the [specialized path integral](transport-and-evaluation.md) with the field's boundary and compact cutoff. Endpoints have `(...,3)` metre coordinates; output has `(...)` kg/m2 and already includes length. Zero-length and empty segment sets are valid. The field snapshot remains borrowed and stable. Stationary fields and raw particle measures raise `TypeError`; point reconstruction for particles remains unspecified.

Scene's `puff_molecular_columns` now requires these bound puff fields. Replace a former bare population with `GaussianPuffField(state, boundary, kernel_cutoff_radius=...)`, using the actual physical declaration. The synthetic spatial-scene example explicitly selects free space and Gaussian tails. Molecular conversion and tangent scaling preserve the supplied field's path law.

## Executable common-path example

The [README example](../README.md#sample-all-models-on-the-same-support) constructs a stationary plume, Gaussian and compact puffs, and bounded particles. All configurations use the same pixel/layer integration and gas conversion. Their emission histories and physics can produce different fields.

A prepared `DirectSolarScene` accepts each `gas` through `scene.with_gas((gas,))`. Binding an empty tuple selects background only. The returned physical scene shares atmosphere, surface and illumination preparation. It evaluates explicit coordinates, outward views, times and a `SpectralGrid`; Instruments owns measurement integration.

## Numerical controls and limits

Vertical Gaussian integrals and unbounded diagonal-covariance puff rectangles use analytic probabilities and first moments. Other horizontal integrals use adaptive quadrature. Compact kernels clip the vertical interval by the ellipsoidal support and use sine-transformed horizontal quadrature near curved support edges. Gaussian-scale breakpoints resolve narrow puffs within wider pixels. `rtol` and `atol_kg_m2` control a block maximum norm; first moments are divided by a common altitude scale during error estimation. `max_intervals` limits each adaptive integral, and failure raises `RuntimeError`. Continuous-field columns below the smallest normal float64 value, about 2.23e-308 kg/m2, and their moments are set to zero together; `atol_kg_m2` must be at least that value. `batch_size` bounds the number of stationary or separable-kernel pixels evaluated together. Quadrature estimates require application-specific convergence checks, especially for small contributions beside much larger columns.

The stationary model retains eastward wind, class-D widths, Briggs rise and a flat reflecting plane at `ground_altitude_m`. Source east/north coordinates are explicit. Puff reflection uses the existing separable covariance and forcing contract at altitude zero. A caller with another physical reference altitude must express all puff and scene coordinates relative to that plane.

A `DirectSolarScene` binding freezes its supplied gas snapshot over a declared closed time interval. Repeated bindings can use successive snapshots while reusing stable preparation. Evolution within that interval requires a separately selected temporal model. The centroid projection retains only column mass and first altitude moment; it approximates the full within-cell spatial distribution.

The tests compare Gaussian rectangles, compact-kernel mass and covariance against independent spherical quadrature, partitioned moments, terrain clipping, translations and snapshot ownership. Particle tests compare homogeneous displacement statistics with the declared random-flight closure. These checks establish bounded numerical and software behavior. Physical equivalence between model families and accuracy in a real atmospheric scene remain unqualified.

## Compact Gaussian kernels

A positive `kernel_cutoff_radius=r` selects a radial cut in an inflated Gaussian covariance `B`. For stored covariance `Sigma`, define `Z=F_chi2(3)(r*r)` and `alpha=F_chi2(5)(r*r)/Z`, then set `B=Sigma/alpha`. The density is `mass * Normal(centre,B) / Z` inside `(x-centre).T @ inv(B) @ (x-centre) <= r*r`, and zero outside. This definition preserves the stored mass, centre and covariance before reflection. The radius uses the inflated kernel's Mahalanobis units. The support's radius in stored-covariance units is `r/sqrt(alpha)`. Cutoffs whose normalization underflows float64 raise `ValueError`.

`GaussianPuffField`, `GaussianPuffDispersion`, `puff_concentration`, `puff_layer_columns`, `puff_path_columns` and `simulate_puffs` accept the same optional cutoff. Direct users of `simulate_puffs` must pass that cutoff again during evaluation. The common field path carries the cutoff automatically. The sampler adjusts below-ground and forcing-tail diagnostics to the chosen kernel; transport still evolves the stored moments. Splitting preserves mixture mass and first two moments, while changing shape and support. Compact support at each instant provides no finite propagation-speed guarantee.

`envelope_sigma` continues to use the stored covariance for forcing probes. `forcing_tail_probability` measures the selected kernel's fraction outside that envelope, before folding; reflection retains the documented upper-bound interpretation. Normalization always applies to the entire kernel. Scene clipping retains only the intersecting mass.

## Bounded particle flights

`ParticleReleases` stores emission times `(particle,)`, positive masses `(particle,)` in kg and point positions `(particle,3)` in metres. `MassParticles` adds one `time_s` and turbulent velocities `(particle,3)` in m/s. Arrays use east/north/altitude order and independent read-only float64 storage. The state is a discrete mass measure. Finite-area integration deposits points without a density reconstruction or smoothing length.

`discretize_particle_emissions` shares the puff emission-interval integration rule. Each retained source subinterval contributes its integrated mass at its midpoint, divided among `particles_per_release` equal-mass particles. Source-time quadrature and random sampling are separate approximations. Keep the release partition fixed when testing output-clock independence.

For homogeneous velocity covariance `R`, the turbulent velocity is `sqrt(3)*sqrt(R)*n`, where `n` is uniform on the unit sphere. Independent Poisson renewals have mean interval `tau=correlation_time_s`. For this stationary closure, velocity autocovariance is `R*exp(-abs(dt)/tau)` and displacement covariance is `2*tau*R*(t-tau*(1-exp(-t/tau)))` in uniform wind and free space. The long-time diffusivity is `tau*R`. A diffusivity alone does not identify both `R` and `tau`.

The velocity bound is `vmax=sqrt(3*lambda_max(R))`. The supplied constant `wind_speed_bound_m_s=Umax` bounds the mean wind globally over the simulation. For a particle emitted at `xe,te`, the implemented path satisfies `norm(x(t)-xe) <= (Umax+vmax)*(t-te)` up to floating-point roundoff. Sampled wind checks reject observed bound violations; the caller remains responsible for the global claim. Reflection at altitude zero preserves this bound for nonnegative release altitudes.

The sampler holds wind velocity over each Euler segment. Releases, renewal clocks, declared wind discontinuities and `max_step_s` define segment boundaries. Output requests observe those paths without inserting transport steps or consuming random draws. Uniform wind segments are exact between physical events. Variable wind needs step refinement; no adaptive error estimate is supplied. Wind callbacks receive independent buffers and must be pure and order-independent. The gradient returned by the existing `Wind` contract is unused in particle transport. Reflected transport requires zero vertical mean wind everywhere and vertically separable turbulent covariance. Turbulence remains homogeneous in space and time.

`max_particles` bounds the entire declared population; `max_steps` bounds prepared transport segments over one sampling call. Exhaustion, non-advancing clocks and nonfinite propagated positions raise `RuntimeError`. No particle is discarded outside an image. `ParticleDiagnostics` reports released mass, represented mass, their signed residual, segment count, renewal count and turbulent speed bound. The transport mass balance uses the whole population. For a disjoint spatial partition, multiply columns by pixel areas and sum to recover in-support mass; subtract that mass from represented mass to obtain excluded mass, within summation roundoff.

Particle projection uses half-open horizontal pixels and altitude layers `[lower,upper)`. Exact lower faces, including terrain, are retained; exact upper faces, including the finite top and outer image edge, are excluded. Adjacent cells therefore count a shared-face particle once. Overlapping pixel interiors raise `ValueError`, independently of particle locations. Terrain clips deposition without changing trajectories or the reflection plane. Quadrature tolerances remain validated common-interface inputs, but do not control particle sampling error. Refine particle count and inspect multiple realizations separately from source partitioning and transport-step convergence.

The bounded homogeneous closure is an exploratory physical model. Stratification, convection, spatially varying turbulent statistics, chemistry, deposition and buoyant rise require additional physical treatment and qualification.

## Specialized puff operations and breaking migration

`simulate_puffs`, `advance_puffs`, `split_puffs`, `puff_concentration`, `puff_layer_columns` and `puff_path_columns` remain available for direct puff transport and evaluation. Their contracts are documented in the [puff reference](transport-and-evaluation.md).

The former `stationary_mass_column` and `stationary_vertical_weights` functions are removed without compatibility aliases. The `orsia_dispersion.stationary` module now owns `StationaryGaussianPlume` and its physical profile. Construct `StationaryGaussianPlume` with an emission rate in kg/s, then use the common field operations. Supply explicit pixel bounds, local frame, altitude interfaces and terrain through `LayeredCells`; use `gas_layer_columns` from Scene for molecular columns and centroids. Gas amounts are in molecules/m2 and preserve the same support owner. The new finite-support integration can differ from the historical full-column and normalized-weight calculation.

Scene also removes `plume_layer_geometry`. Applications that retain column/weight archives own their explicit import conversion. The historical QAIrbon reconstruction resides in its reference adapter; ordinary production and rendering use explicit gas layers.
