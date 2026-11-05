# Solar and LTE thermal transfer

Choose solar reflection or non-scattering LTE thermal emission explicitly. These bounded numerical models require application-specific convergence and scientific validation. [State and spectral contracts](state-and-spectra.md) define their inputs.

Run the Python blocks in order in one session with `orsia-radiative` installed. All examples use synthetic inputs without files or network access.

## Transfer boundary

`Radiance(grid, values, geometry)` contains physical values, spectral support and outgoing view geometry. Keep calculation labels, input lineage and numerical decompositions separately. [State and spectral contracts](state-and-spectra.md) specify physical ownership and explicit freshness.

The scene-level transfer entry points are `transfer.solar.reflected_radiance` and `transfer.thermal.thermal_radiance`. Separate `transfer.orders.scattering_orders` and `transfer.thermal.schwarzschild_terms` functions return fresh numerical arrays for decomposition. Import `transfer` from `orsia_radiative`, or `solar` / `thermal` from `orsia_radiative.transfer`. The `transfer` package exposes these two family submodules. Import `orders` explicitly from `orsia_radiative.transfer` for numerical diagnostics. `Radiance` carries the physical grid, values and observation geometry.

`solar` evaluates reflected sunlight from one Sun. `source: SolarSource` represents beam-normal irradiance along one downward direction; no source collection or implicit superposition remains. A resolved solar disc and external diffuse incident field are not implemented. Thermal emission is selected explicitly through the separate [thermal operation](#lte-thermal-transfer), never inferred by the solar methods from stored temperatures or emissivity.

`transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=2, angular_order=8, layer_subdivisions=8, delta_m_moments=None)` sums atmospheric orders 0 through a caller-selected nonnegative integer N, with a Lambertian boundary at every order. Physical direct reflection uses `max_order=0, delta_m_moments=None`. Order one includes direct-beam scattering, Rayleigh and HG ground coupling, and two ground encounters separated by one atmospheric event. The optional [Delta-M calculation](#experimental-aerosol-delta-m) counts residual events.

The radiance calculation and the numerical order decomposition each require one `source` and one `AtmosphericOptics`. Compose with `compose_atmospheric_optics(scene.atmosphere, source, absorption=total_or_none, rayleigh=rayleigh_or_none, aerosol=aerosol_or_none)`; `None` denotes an absent role. The physical scope is one conservative Rayleigh role and one untruncated HG aerosol role. Solver-local Delta-M adaptation preserves these physical inputs.

The two phase laws operate under one total extinction. SOS applies Rayleigh and HG redistributions to the same preceding angular field and sums their source terms before transport, preserving cross interactions. Physical roles are ordered absorption, Rayleigh, aerosol, regardless of keyword order; contributor provenance is caller-owned.

`solar_source(irradiance, direction=(zenith_deg, relative_azimuth_deg))` creates a `SolarSource`. Zenith is in `[0, 90)` toward the Sun and relative azimuth in `[0, 360)` degrees from incoming photon propagation to the outgoing view, under the [angular contract](#angular-contract-and-outward-ray-conversion) below. These are not geodetic coordinates. Sources own their incident directions relative to the scene's view; changing the absolute view azimuth requires recomputing each source's relative azimuth. The calculations consume both directly without constructing a temporary solar scene. A surface with emissivity may supply its reflected contribution without activating thermal emission; an emissivity-only surface cannot supply the reflection required by current solar methods.

Solar transfer returns `Radiance`. Use `orders.scattering_orders` for its float64 order array. Its rows are fresh writable arrays on the calculation grid. Callers retain input lineage and numerical controls alongside the calculation.

The solar methods implement downward reflected solar contributions only. They do not infer illumination from scene metadata or evaluate thermal emission. Neither family supports a combined thermal/solar call, derivatives, retrieval-coordinate mapping or parallel execution policy. Solar SOS also has no unresolved-tail closure, automatic order stopping or acceleration. `scattering_orders` returns rows for the selected fixed-order calculation; the last row is not a tail bound. These arrays identify numerical terms; callers retain input provenance separately.

The reflected-solar boundary in `transfer/solar.py` sums numerical orders and constructs `Radiance`. `transfer/orders.py` owns analytic direct initialization, fresh extinction, angular redistribution and the fixed-N recurrence. Initialization checks physical summed-depth overflow before optional Delta-M removal and calculates direct reflection once with the selected extinction. `transfer/thermal.py` validates and extracts physical inputs, supplies Gauss-Legendre nodes for quadrature, and wraps the serial C++ LTE calculation in `transfer/native/thermal_native.h`. The thermal kernel evaluates Planck radiometry, sums absorption and absorbing-aerosol depths, propagates upward emission and downward reflected emission, and returns the three additive terms. Source, optical, radiance and scene contracts remain outside `transfer`.

`formal.propagate(source_depth, extinction, mu, ...)` retains its signature and fresh centre/exit outputs. For repeated sources on one fixed grid, `formal.propagation_coefficients(extinction, mu, solar_mu=...)` returns call-owned `PropagationCoefficients` with read-only arrays; `formal.apply_propagation(source_depth, coefficients, solar_source=..., return_centres=False)` returns `(None, exits)` without calculating or allocating the centre field. Extinction is vertical optical depth, ordered bottom to top; directions are nonzero signed cosines, positive upward. Source values are integrated over vertical cell fraction and must use exactly the prepared cell/direction grid. Changing extinction, directions or solar cosine requires preparation again; changing source amplitude or the bottom boundary does not. Formal propagation performs no reflection; SOS applies its boundary separately.

Constant-source weights use `scipy.special.exprel(-x)`, the stable `-expm1(-x)/x` limit, including exactly one at zero optical depth. Endpoint exponential integrals use `exp(max(a,b))*exprel(-abs(b-a))`. Gauss-Legendre construction uses `scipy.special.roots_legendre` in SOS and thermal quadrature, preserving interval mappings and physical normalizations; last-bit differences from NumPy construction are possible. NumPy preparation evaluates full cell/direction arrays; application uses views for contiguous hemispheres and gathers interleaved directions once, rather than per cell. SOS refreshes one coefficient set per spectral point and reuses it across constituents and orders. Its six float64 solar coefficient matrices occupy 48 bytes per cell/direction pair; call-local scratch holds the other fields and boundary responses. Storage for coefficients scales with cells times directions and is reused across spectral points within each call; the NumPy fallback additionally allocates full-grid temporary arrays during preparation.

### Repeated solar evaluations

Call `solar.reflected_radiance(scene, source=source, optics=current_optics, **controls)` for each evaluation. Every call validates current inputs, constructs angular geometry, allocates scratch, computes the fixed-order recurrence and returns an independently owned `Radiance`. Its grid is `source.irradiance.grid`, and its observation geometry equals `scene.geometry`. Keep source, surface and optical arrays stable throughout the call, including concurrent calls; the caller maintains physical freshness and bottom-to-top layer correspondence.

```python
from orsia_radiative.transfer import solar


def repeated_solar_radiances(scene, source, optics_a, optics_b):
    controls = dict(max_order=4, angular_order=8, layer_subdivisions=8)
    radiance_a = solar.reflected_radiance(scene, source=source, optics=optics_a, **controls)
    radiance_b = solar.reflected_radiance(scene, source=source, optics=optics_b, **controls)
    # Each radiance retains its own values after subsequent calls.
    return radiance_a, radiance_b
```

`orders.scattering_orders` remains the independent numerical API, returning fresh writable float64 `[order, spectral point]` arrays in W m-2 sr-1 per (cm-1), with `(max_order + 1, point_count)` shape. `orders.angular_grid` and `orders.phase_function` retain their contracts. Solar sums these order contributions into total radiance. Both paths skip angular construction for order zero and globally inactive scattering. Native geometry, coefficients and fields belong to the invocation; phase matrices are reused across layers, orders and spectral points within that call, with a 16 MiB FIFO payload budget or one oversized working matrix. Concurrent calls own separate scratch.

## LTE thermal transfer

`transfer.thermal.thermal_radiance(scene, *, optics, angular_order=32, hemispheric_method="quadrature")` evaluates upward top-of-atmosphere radiance in a plane-parallel, horizontally homogeneous, non-scattering LTE atmosphere with isothermal input layers and zero incident radiance at the top. It uses atmospheric layer temperatures and vertical absorption depths on `optics.wavenumbers_cm1`; no solar object, provider, implicit grid, resampling, cloud model, non-LTE emission or observer-height selection is introduced. Any strictly positive Rayleigh or aerosol scattering depth raises `ValueError`, even if small or in an opaque layer. Zero scattering roles are accepted; purely absorbing aerosol extinction contributes once to absorption and emits at the corresponding layer temperature.

For a layer with source function B, ray cosine mu and vertical absorption depth tau, `formal.schwarzschild` applies `I_out = I_in * exp(-tau/abs(mu)) + B * (-expm1(-tau/abs(mu)))`. This is exact within each supplied constant-source layer; resolving temperature gradients requires a different vertical representation. The low-level function accepts float64 `(layers, points)` source-function/depth arrays and `(points,)` incident radiance, returning a fresh writable exit vector; positive mu travels bottom to top and negative mu top to bottom. It does not multiply B by tau, avoiding that intermediate overflow in opaque layers, and `expm1` preserves thin-layer emission.

The opaque Lambertian surface requires spectral emissivity and a positive temperature. It emits `epsilon * B(T_surface)` and reflects the descending atmospheric irradiance as `(1-epsilon) * E_down/pi`, then both terms are attenuated along the upward view ray. An emissivity-only surface is sufficient: reflectivity is its complement; when reflectance is also supplied, `SurfaceState` already checks Kirchhoff consistency. The implementation uses the emissivity complement in either case. Surface reflection does not create an additional atmospheric source in a non-scattering calculation. No external downwelling background is included.

`angular_order` is a Python integer >= 2 (booleans are rejected). Gauss-Legendre nodes on the open downward hemisphere integrate `E_down/pi = 2 * integral_0^1 mu * I_down(mu) dmu`; azimuth integrates analytically under horizontal symmetry. This numerical quadrature is needed only for nonblack surfaces with nonzero absorption, but its control is always validated. Refine it independently for the intended observable, particularly with weak absorption and near-grazing downwelling rays. The default is not a general accuracy certificate; there is no automatic convergence or error estimate. Layer subdivision is unnecessary for the supplied isothermal model.

`hemispheric_method="expn"` explicitly replaces angular quadrature with a layer sum `E_down/pi = 2 * sum(B_layer * [E3(below)-E3(below+depth)])`, where `below` is the summed vertical absorption between the layer and surface. Thin intervals satisfying `depth <= 0.02*min(below, 1)` use the integrated midpoint expansion of E2 through fourth order; intervals near zero use the integrated logarithmic series, and the origin uses the E3 recurrence with `expm1`. These expressions retain depth separately even when `below+depth` rounds to `below`. Well-separated intervals use direct E3 values, with an asymptotic tail beyond optical depth 700 to retain representable contributions past SciPy's underflow cutoff. Zero depths and final binary64 underflow return zero. The native calculation reuses call-local layer vectors across spectral points; scratch storage scales with layer count, while the three returned arrays scale with spectral size. The default method and its angular refinement remain unchanged. `angular_order` is validated for both methods; invalid method names raise `ValueError`.

Both thermal methods execute in the existing `transfer_native` extension, with one detached serial numerical region per call. The `expn` scalar comes from SciPy's Cython API through a signature-checked capsule; the bound callable retains its provider in the importing interpreter. At binding time, Python derives the scalar Planck coefficients `100*h*c/k` (cm K) and `2*k*c*1e6` (the Rayleigh-Jeans prefactor for radiance per cm-1) from `scipy.constants`; the native callable captures these values. Planck follows the logarithmic formula used by `radiometry.planck_wavenumber`, and ray transport follows `formal.schwarzschild`. Native scalar transcendental functions and reductions permit binary64 rounding differences from NumPy. Inputs remain stable for the duration of a call; native-endian float64 C-contiguous aligned inputs are borrowed, and other storage layouts receive temporary compatible copies. Each term owns fresh writable float64 storage. Concurrent calls share immutable inputs and own independent scratch.

Run `devbox run -- just rebuild-native`, then `devbox run -- just typecheck` and `devbox run -- just --justfile tests/justfile pytest libs/orsia-radiative/tests/test_thermal*.py` for thermal reference comparisons, independent interval integrals, physical limits, buffer validation and concurrent-call checks. The frozen pre-port NumPy/SciPy calculation lives in `tests/thermal_reference.py` as software regression evidence. These checks establish bounded software and numerical agreement; instrument-space accuracy and retrieval performance require application-specific validation.

`radiometry.planck_wavenumber(grid, temperature_k)` returns Planck radiance in W m-2 sr-1 per (cm-1), with exact SI h, c and k and the cm-1 density Jacobian. A positive finite scalar temperature returns `(points,)`; a float64 `(layers,)` temperature array returns `(layers, points)`. Coordinates must be positive, strictly increasing float64. Returned arrays are fresh and writable; inputs are not retained or mutated. Logarithmic amplitude evaluation avoids intermediate overflow; Rayleigh-Jeans and Wien limits are handled, underflow returns zero, and unrepresentable finite radiance raises `ValueError`. This is a point-density evaluation, not spectral integration or conversion to brightness temperature.

Thermal inputs are prepared with `compose_absorption_on_grid(atmosphere, grid, components)` and `atmospheric_optics(atmosphere, grid, ...)`. They retain the same exact-support, independent-storage, overlap and physical-freshness rules as their solar counterparts. The caller selects the grid once before spectroscopy; optical records retain its evaluation support, and transfer checks the surface grid and layer shape. Use `validate_scene_optics(scene, optics, reference_atmosphere=...)` for an explicit conservative identity check before thermal reuse. Mutable coordinate inputs are snapshotted. Existing source-based functions delegate to these common calculations and preserve their signatures and solar grid authority.

`thermal_radiance` returns the physical `Radiance`. `schwarzschild_terms` separately returns fresh writable arrays named `surface_emission`, `atmospheric_emission` and `reflected_atmospheric_emission`; their sum is the returned radiance. `view_transmission`, angular controls, method identity and ordered optical provenance are calculation-local and are not fields of `Radiance`. Numerical buffers are call-local without caches, RNG, I/O or parallel policy; storage scales with layers times spectral points and does not retain an angular cube.

```python
from orsia_radiative import wavenumber_grid
import numpy as np
from orsia_radiative import (
    Absorption,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    SurfaceState,
    compose_absorption_on_grid,
    atmospheric_optics,
)
from orsia_radiative.radiometry import planck_wavenumber
from orsia_radiative.transfer import thermal

grid = np.array([800.0, 1000.0], dtype=np.float64)
atmosphere = Atmosphere(
    np.array([0.0, 1000.0]),
    np.array([100000.0, 90000.0]),
    np.array([95000.0]),
    np.array([285.0]),
    np.array([2.0e29]),
    np.array([1.9e29]),
    np.array([1.0e28]),
)
scene = RadiativeScene(
    atmosphere,
    SurfaceState(wavenumber_grid(grid), emissivity=np.array([0.9, 0.95]), temperature_k=300.0),
    ObservationGeometry(20.0),
)
# Synthetic materialised absorption; a real caller evaluates spectroscopy on grid.
absorption = Absorption(wavenumber_grid(grid), np.array([[0.1, 0.5]]))
optics = atmospheric_optics(atmosphere, wavenumber_grid(grid), absorption=absorption)
result = thermal.thermal_radiance(scene, optics=optics, angular_order=64)
analytic_result = thermal.thermal_radiance(scene, optics=optics, hemispheric_method="expn")
np.testing.assert_allclose(analytic_result.values, result.values, rtol=1e-10)
terms = thermal.schwarzschild_terms(scene, optics=optics)
assert result.values.shape == (2,)
np.testing.assert_allclose(sum(terms.values()), result.values)
# Empty upstream contributions give a transparent atmosphere, without any Sun.
vacuum = compose_absorption_on_grid(atmosphere, wavenumber_grid(grid), ())
vacuum_optics = atmospheric_optics(atmosphere, wavenumber_grid(grid), absorption=vacuum)
vacuum_result = thermal.thermal_radiance(scene, optics=vacuum_optics)
np.testing.assert_allclose(
    vacuum_result.values, scene.surface.emissivity * planck_wavenumber(grid, 300.0)
)
```

`devbox run -- just --justfile tests/justfile pytest libs/orsia-radiative/tests/test_thermal.py libs/orsia-radiative/tests/test_radiometry.py` checks Planck against an independent frequency-density expression and a Stefan-Boltzmann integral, signed layer-ray propagation, vacuum/thin/opaque limits, black isothermal equilibrium, stratified source integrals, nonblack surface reflection against adaptive angular integrals and E3, relayering, quadrature refinement, purely absorbing aerosols, rejected scattering, explicit-grid composition/freshness, independent results and deterministic call-local concurrent evaluations. These are bounded synthetic numerical/software checks, not broad scientific qualification or representative performance. Historical standalone comparisons distinguished full reflecting-boundary radiance from SASKTRAN2 Standard's direct thermal terms and recorded DO vacuum failures. Current software checks do not regenerate those external observations or establish a general thermal accuracy envelope.

The retired scattering-omission diagnostic found that agreement in weak-scattering homogeneous cases did not extend across its tested scattering depths. Neither total scattering depth, transport scattering depth nor HG asymmetry provides a validated omission-error bound. Oblique-view and reversed-temperature sweeps were blocked by reference adapter gates, leaving those sensitivities unevaluated. Application acceptance requires an intended observable, representative optical states and an error budget. The production solver continues to reject nonzero scattering. When evaluating an absorption-only approximation, retain gas and aerosol absorption once and preserve the thermal boundary. Assess the omitted-scattering error separately from numerical error.
## Explicit spectral conversions and unscattered transfer

See [local spectroscopy data](../../orsia-data/README.md) for readers and downloads; this numerical package has no file/provider dependency.

`wavelength_to_wavenumber` and `wavenumber_to_wavelength` convert positive increasing point grids and return increasing grids in the target coordinate. `density_per_nm_to_per_cm1` and `density_per_cm1_to_per_nm` apply the pointwise Jacobian and reverse sample order; `reverse_spectral_points` only reverses finite dimensionless values. These functions do not interpolate, integrate bins or invent quadrature weights. Callers must use the resulting canonical cm-1 grid when constructing radiative records.

```python
from orsia_radiative import wavenumber_grid
import numpy as np
from orsia_radiative import (
    Absorber,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    transfer,
    wavelength_to_wavenumber,
)

grid = wavelength_to_wavenumber(np.array([760.0, 761.0], dtype=np.float64))
atmosphere = Atmosphere(
    np.array([0.0, 1000.0]),
    np.array([100000.0, 90000.0]),
    np.array([95000.0]),
    np.array([285.0]),
    np.array([2.0e29]),
    np.array([1.9e29]),
    np.array([1.0e28]),
    (Absorber("X", np.array([1.0e25]), "dry_air"),),
)
irradiance = SpectralIrradiance(wavenumber_grid(grid), np.array([10.0, 12.0]))
source = solar_source(irradiance, direction=(30.0, 20.0))
grid = source.irradiance.wavenumbers_cm1
scene = RadiativeScene(
    atmosphere,
    SurfaceState(wavenumber_grid(grid), reflectance=np.array([0.2, 0.3])),
    ObservationGeometry(10.0),
)
optics = compose_atmospheric_optics(atmosphere, source)
result = transfer.solar.reflected_radiance(
    scene, source=source, optics=optics, max_order=0, delta_m_moments=None
)
assert result.wavenumbers_cm1.flags.writeable is False
assert result.values.shape == (2,) and result.values.flags.writeable
```

The records validate numerical and physical correctness at composition boundaries: positive finite canonical grids, exact grid and layer support, finite values, nonnegative radiance or irradiance and optical-depth bounds. Provider identities and calculation lineage remain caller-owned; physical input records do not store artifact identifiers. Typed callers are checked statically by Pyrefly; runtime checks are not added merely to coerce or re-implement Python type annotations. Numerical values are copied to independent writable storage; safely immutable coordinate arrays may be shared. No provider calls, global mutable cache, implicit I/O, randomness or parallel execution policy is part of this package.

## Solar entry points

The reflected-solar public interface in `transfer.solar` is `reflected_radiance`. Numerical order decomposition belongs to `transfer.orders.scattering_orders`.

- Replace `direct_beam(scene, source=source, optics=optics)` with `reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)` to preserve physical unscattered reflection.
- Use `max_order=1, delta_m_moments=None` for first-order reflected sunlight. SOS includes common Rayleigh/HG ground coupling and the ground-atmosphere-ground path. The Lambertian boundary applies at every atmospheric order. Refine angular and vertical controls for the intended observable.
- Import `orders` from `orsia_radiative.transfer` and use `orders.scattering_orders(..., max_order=N)` for numerical diagnostics. Its rows aggregate all constituent contributions at each atmospheric order. With Delta-M enabled, rows count residual events and row zero uses transformed extinction.

## Typed scattering and experimental SOS

`RayleighComponent(grid, optical_depth)` represents conservative scalar Rayleigh scattering: scattering depth equals extinction and the phase is fixed. `AerosolComponent(grid, optical_depth, scattering_depth, asymmetry)` fixes the phase to untruncated HG, with `0 <= scattering_depth <= optical_depth` and `|g| <= 0.8`. All depth/asymmetry arrays have shape `(n_layers, n_points)`, bottom to top. Both types retain immutable grid references and own numerical snapshots, and are composed into atmospheric optics independently of transfer. `rayleigh_component(ScatteringProperties)` returns a `RayleighComponent` after checking untruncated scalar moments `(1, 0, 0.1)`; `aerosol_component(AerosolOptics)` returns an `AerosolComponent` with scattering depth equal to extinction times single-scattering albedo. No public generic constituent or conversion method is needed.

With `delta_m_moments=None`, `max_order=N` counts physical atmospheric scattering events, not surface reflections, and accepts integers N >= 0. Order zero is direct attenuated reflection. At order one the direct beam is scattered in addition to the order-zero surface field. Each subsequent order scatters the complete preceding field, summing all optical constituents before transport. The downward diffuse exits generate a Lambertian bottom boundary at the same atmospheric order; this reflected field contributes to both the observed radiance and the next order's illumination. Consecutive surface encounters without an intervening atmospheric event are not possible in this geometry. For example, sunlight -> ground -> atmosphere -> ground -> observer is included at atmospheric order one. No empirical tail is added. Zero scattering depth reduces every higher-order calculation to order-zero reflection. On a black surface, physical SOS-1 equals the analytic direct-beam volume integral.

The numerical angular field uses `angular_order` Gauss-Legendre points in each open hemisphere, `2 * angular_order` periodic midpoint azimuths and positive weights representing `dOmega / (4*pi)` with sum one. Each input layer is split uniformly into `layer_subdivisions` cells, with homogeneous input optical properties. Direct-beam forcing is integrated analytically inside cells. Diffuse forcing is evaluated at cell centres and transported as a piecewise-constant source in both directions across all cells. This retains inter-layer paths; order two integrates intermediate directions rather than squaring a final-angle phase. The view direction is evaluated separately and has no quadrature weight. `angular_order >= 2` and `layer_subdivisions >= 1` are required; increase both independently to measure convergence. Defaults, near-grazing geometries, optically thick profiles and strongly anisotropic aerosols have no qualified accuracy envelope.

Spectral points are processed serially. If `Q = 4 * angular_order**2` and `C = layer_count * layer_subdivisions`, call-local angular scratch scales as `C * Q` and each dense phase matrix as `Q**2`. Angular scratch retains the preceding field, sources and propagation storage for one spectral point. Every evaluation allocates an independent `O((N+1) * point_count)` order array; the solar boundary sums it and returns the total physical values. Numerical controls and convergence diagnostics remain caller-owned. Refine N, angular sampling and vertical cells separately. Work buffers live for one complete invocation.

Call-local native geometry contains quadrature directions and weights, traversal-cosine mappings, beam angles and seven-factor Rayleigh angular operators; one- and two-layer calculations additionally retain their dense Rayleigh matrix during the call. Aerosol redistribution uses current scattering depths with phase matrices keyed by exact asymmetry for that call's geometry and Delta-M moments. The FIFO cache retains at most 16 MiB of matrix payload. If a single matrix exceeds that budget, one working matrix of that size is used and rebuilt for each requested phase. Cache storage is therefore bounded by `max(16 MiB, 8*Q*(Q+1))`, separately from geometry and per-point scratch. Zero-depth layers skip redistribution. Extinction-dependent propagation coefficients and unit surface responses refresh from current optics at every point. Both `orders.scattering_orders` and `solar.reflected_radiance` skip angular construction for order zero and globally inactive scattering. See [repeated solar evaluations](#repeated-solar-evaluations) for call ownership and execution details.

```python
from orsia_radiative import wavenumber_grid
from orsia_radiative import AerosolComponent, RayleighComponent, compose_atmospheric_optics
from orsia_radiative.transfer import orders

rayleigh = RayleighComponent(wavenumber_grid(grid), np.array([[0.01, 0.02]]))
aerosol = AerosolComponent(
    wavenumber_grid(grid),
    np.array([[0.04, 0.05]]),
    np.array([[0.02, 0.03]]),
    np.array([[0.2, 0.4]]),
)
optics = compose_atmospheric_optics(atmosphere, source, rayleigh=rayleigh, aerosol=aerosol)
first_order = transfer.solar.reflected_radiance(
    scene, source=source, optics=optics, max_order=1, delta_m_moments=None
)
sos = transfer.solar.reflected_radiance(
    scene, source=source, optics=optics, max_order=5, angular_order=8, layer_subdivisions=8
)
order_values = orders.scattering_orders(
    scene, source=source, optics=optics, max_order=5, angular_order=8, layer_subdivisions=8
)
assert order_values.shape == (6, grid.size)
np.testing.assert_allclose(order_values[:2].sum(axis=0), first_order.values)
np.testing.assert_array_equal(order_values.sum(axis=0), sos.values)
repeated_a, repeated_b = repeated_solar_radiances(scene, source, optics, optics)
np.testing.assert_array_equal(repeated_a.values, repeated_b.values)
```

### Experimental aerosol Delta-M

`delta_m_moments=K` explicitly selects an uncorrected Delta-M SOS calculation; `None` preserves the physical calculation. K is the number of retained Legendre moments, including degree zero, not the number of scattering orders or an inferred stream count. Integer values must satisfy `2 <= K <= 2 * angular_order`, even for N=0 or absent aerosols. The phase has degree K-1, resolved by the per-hemisphere Gauss rule and periodic azimuth rule at that bound. K and angular sampling remain independent caller choices within this restriction; the restriction does not establish radiance accuracy or resolve the exact non-forward HG law.

For each layer/spectral aerosol sample with g>0, the solver uses HG moments `beta_l = g**l`, removes `f = g**K`, and retains `beta_l_star = (beta_l - f) / (1 - f)` for degrees 0 through K-1. The residual phase is `sum((2*l+1) * beta_l_star * P_l(cosine))`, with a 4π mean of one; it is not HG with a modified asymmetry. Scattering depth becomes `(1-f) * tau_s_a`, aerosol extinction becomes `tau_e_a - f * tau_s_a`, and total extinction changes by the same removed depth. Aerosol absorption, molecular absorption and Rayleigh optics remain unchanged. These are local numerical adaptations, not mutations or replacement physical records. For g<=0 no forward peak is removed and exact HG is retained, without polynomial truncation. Zero aerosol scattering leaves its absorption unchanged.

Both direct-beam and diffuse transport use the transformed common extinction, including the incident beam at the ground, the order-zero surface term, atmospheric redistribution, beam forcing and outgoing attenuation. Rayleigh and the aerosol residual still redistribute the same preceding field. The [ARTS core-solver documentation](https://atmtools.github.io/arts-docs-master/concept.disort.html#delta-m-scaling) gives the classical optical rescaling and the use of scaled coordinates for the internal beam; Orsia's aerosol-only split, non-forward policy and moment/quadrature restriction are explicit implementation choices, not claims of DISORT parity.

The fixed-order interpretation is unchanged: Delta-M rows from `scattering_orders` are `residual_order_0` through `residual_order_N` by caller convention, even if no forward peak is present. They count residual atmospheric events, not physical scattering events; removed exactly-forward events are incorporated into propagation. With this option N=0 returns direct reflection through the transformed extinction. No transmission, source lineage or control diagnostic is attached to the physical `Radiance`; callers that need such values must calculate and retain them explicitly.

The implemented Delta-M radiance uses the uncorrected residual expansion, including at the Lambertian boundary. Finite residual expansions can be signed internally; no clipping or discrete renormalization is applied. A negative observed contribution at any residual order raises `ValueError`, preserving the nonnegative physical-radiance contract rather than silently changing the phase or accepting cancellation between orders. Increase moments and angular resolution, or disable Delta-M, when that occurs. Positive returned orders alone do not certify a positive angular field or a physically accurate directional radiance. No speed gain, accuracy envelope or superiority at a fixed physical order is claimed.

```python
scaled = transfer.solar.reflected_radiance(
    scene,
    source=source,
    optics=optics,
    max_order=5,
    angular_order=8,
    layer_subdivisions=8,
    delta_m_moments=8,
)
residual_orders = orders.scattering_orders(
    scene,
    source=source,
    optics=optics,
    max_order=5,
    angular_order=8,
    layer_subdivisions=8,
    delta_m_moments=8,
)
np.testing.assert_array_equal(residual_orders.sum(axis=0), scaled.values)
```
### Separate numerical terms

`orders.scattering_orders` returns float64 `(N+1, points)` radiances on the source grid, including zero-valued rows. Row zero is the direct surface term; row n>0 aggregates all paths with n atmospheric events and their ground reflections. Delta-M rows count residual events instead. Summing axis zero reproduces `reflected_radiance`. Explicitly matched fixed-extinction experiments isolate constituent effects; each row aggregates the paths at its order. Omitted-tail assessment requires a separate convergence study.

`thermal.schwarzschild_terms` returns the three thermal terms described above. Thermal terms and solar order rows use W m-2 sr-1 per (cm-1), with the same grid and controls as their radiance calculation. Arrays are fresh, writable and independent of input storage. Each calculation borrows its inputs read-only; callers retain provenance separately.

### Angular contract and outward-ray conversion

Use a right-handed local frame with `+z` along the upward surface normal. Let `s` be the unit direction from the scattering point toward the Sun, `k_in = -s` the incoming photon propagation, and `k_out` the outgoing photon propagation toward the observer. Solar zenith is the angle between `s` and `+z`; view zenith is the angle between `k_out` and `+z`. Neither is the instrument's downward look direction. The scattering cosine supplied to Rayleigh and HG is `k_in dot k_out`; positive HG asymmetry favours forward photon propagation.

`SolarSource.relative_azimuth_deg` is the directed horizontal angle from `k_in` to `k_out`, positive about `+z` (counterclockwise when viewed from above), reduced to `[0, 360)`. Zero aligns their horizontal propagation projections; 180 degrees opposes them. The same geometry can be represented with `+x` along the incoming horizontal projection and `+y = +z cross +x`: `k_in = (sin(theta_s), 0, -cos(theta_s))` and `k_out = (sin(theta_v)*cos(a), sin(theta_v)*sin(a), cos(theta_v))`. This gives the retained scalar cosine `-cos(theta_s)*cos(theta_v) + sin(theta_s)*sin(theta_v)*cos(a)`, used by SOS beam-source construction. SOS quadrature directions are photon propagation vectors in this same frame, with positive vertical cosine upward.

For an adapter that supplies outward directions toward the Sun and observer, with absolute azimuths `phi_s` and `phi_v` measured in the same oriented frame, convert explicitly: `relative_azimuth_deg = (phi_v - phi_s - 180.0) % 360.0`. Equivalently, for the directed outward relative angle `delta_out = (phi_v - phi_s) % 360.0`, supply `(delta_out - 180.0) % 360.0`, not `delta_out`. If those azimuths increase clockwise, first reverse their orientation; a scalar unsigned separation does not encode that orientation. An adapter using an observer-to-scene look vector must reverse it before deriving view zenith and azimuth. At exact solar or view zenith the corresponding horizontal projection vanishes, so its azimuth is physically arbitrary; the finite input and `[0, 360)` checks remain unchanged. The analytic direct-beam volume integral is azimuth-independent there. At solar zenith, arbitrary rotations of the separately evaluated view relative to the finite SOS azimuth grid can leave a small numerical residual in diffuse terms; physical invariance is not a promise of roundoff equality for every quadrature. A bounded synthetic test checks that this residual decreases under angular refinement, without defining a general accuracy envelope.

| Solar zenith | View zenith | Directed outward Sun-to-view separation | Orsia relative azimuth | Scattering angle |
| --- | --- | --- | --- | --- |
| 45 degrees | 45 degrees | 0 degrees (same outward direction) | 180 degrees | 180 degrees (backscatter) |
| 45 degrees | 45 degrees | 180 degrees | 0 degrees | 90 degrees |
| 45 degrees | 45 degrees | 90 degrees | 270 degrees | 120 degrees |
| 0 degrees | 45 degrees | Arbitrary | Arbitrary | 135 degrees |

No automatic angular conversion or alternate convention flag is provided. Inputs already expressed in Orsia's convention retain their meaning. A caller that passed outward Sun-to-view separations must correct its inputs explicitly and expect changed radiances; previous shape or regression checks did not establish that caller's geometry. External adapters must still demonstrate their frame, direction and azimuth mapping, and no external solver agreement is claimed.

`test_angular_geometry.py` derives API angles from independently supplied outward Cartesian vectors and tests their dot products after reversing the solar vector. It checks cardinal, oblique, near-grazing and zenith geometries, signed azimuth wraparound and common rotations about the normal. Public analytic and black-surface SOS-1 volume terms are checked against independent attenuated-ray integrals with Rayleigh and signed HG constituents. Separate SOS checks inspect the actual beam/intermediate phase cosines on reflective scenes and compare the aggregate order-two volume term against the sum of every ordered pair in a one-cell discrete reference built from physical vectors, independent phase expressions and adaptive ray integrals. The discrete reference deliberately shares the angular nodes and cell-centre approximation: it tests geometry and assembly, not quadrature convergence or continuous-solver accuracy. These bounded local checks leave external scientific qualification outstanding.

## Evidence and limits

SOS uses deterministic angular quadrature and vertical cell subdivision. These are numerical controls, not physical response weights or processing weights; convergence is domain-specific and must be assessed independently. Work buffers are call-local, spectral points are processed serially and no global cache or hidden optical-depth basis is used.

The synthetic solar and aerosol fixtures under `tests/fixtures/` are project-authored numerical regression cases, not measured observations, independent solver parity or rights-cleared external data. The tests cover analytic Beer-Lambert limits, source-resolved single events, source composition, bounded SOS order reductions, per-order SOS provenance, independent continuous-ray SOS checks, vertical and angular refinement trends, line evaluation, bounded cross-section interpolation, explicit spectral Jacobians and rejected contract violations. These checks exercise small synthetic scenes and do not qualify a broad atmospheric domain, an accuracy envelope for the SOS controls, instrument-space response, derivative convergence, retrieval performance or live provider integration.

The [workspace coverage gate](../../../DEVELOPMENT.md#test-coverage-policy) requires at least 90% combined statement-and-branch execution coverage for this library independently of `orsia-uncertainty`. All radiative source modules, including unimported ones, contribute to the denominator. This engineering metric is separate from the scientific limits below.

The continuous-ray checks in `test_sos_numerics.py` use adaptive SciPy integrals for a homogeneous isotropic aerosol slab and disjoint Rayleigh/isotropic-aerosol layers. They compare the first two isotropic orders, including repeated-ground terms, and isolate the sum of ordered cross-constituent volume paths at fixed extinction against independently formulated integrals at a case-specific `2e-3` relative / `2e-8` absolute tolerance. `test_sos_orders.py` compares all orders through N=5 on a small stratified Rayleigh/HG grid, with reflectances 0, 0.4 and 1, against independent scalar ray integrals and phase expressions. That oracle shares angular nodes and cell-centre collocation: it verifies assembly, not continuum accuracy. Further checks cover scattering-depth scaling through N=6 at fixed extinction, prefix consistency, zero illumination/scattering, N=0, source-amplitude scaling, relayering, buffer/operator ownership and deterministic concurrent calls. Vertical and angular-refinement trends remain separately tested, including g = -0.8 and g = 0.8. These bounded checks do not qualify optically thick/high-order convergence, broad native thread safety, an external solver, instrument convolution or retrieval impact.

`devbox run -- just --justfile tests/justfile pytest libs/orsia-radiative/tests/test_sos*.py libs/orsia-radiative/tests/test_angular_geometry.py libs/orsia-radiative/tests/test_composable_transfer.py libs/orsia-radiative/tests/test_redistribution.py` runs focused fixed-N, composition, independent-ray and refinement regressions. Assess angular resolution, vertical resolution and atmospheric-order truncation separately. Check components as well as totals: cancellation can conceal an unresolved contribution. An internal finest grid supplies a comparison point with unknown error. The retired SOS-2 characterization used the former one-reflection closure; its reflective radiances and timings do not qualify the current Lambertian boundary at every order.

The retired standalone SASKTRAN2 comparison used shared homogeneous scalar optical inputs and explicit reference refinement checks. Its scattering cases retained radiance discrepancies and provisional or unavailable references; increasing SOS order on a fixed angular/vertical grid did not isolate truncation error. Those observations leave general accuracy, equal-accuracy speed, instrument response and retrieval impact unqualified. The maintained notebook's reference checks have their own limited scene/support scope. SASKTRAN2 remains a development dependency, excluded from library runtime requirements.
