# Orsia Radiative: composable atmospheric optics and transfer

Start with the [code guide, architecture diagram and progressive examples](src/orsia_radiative/README.md). This page retains the physical contracts, transfer details and scientific limitations.

`orsia-radiative` publishes materialised atmospheric and surface state, pure spectroscopy kernels, total absorption, explicit Rayleigh/aerosol roles, reflected-solar transfer and non-scattering LTE thermal transfer. Spectral records carry one immutable `SpectralGrid`; `SpectralDomain` separately declares physical coverage when known. The radiative core uses increasing float64 wavenumber points in cm-1. Wavelength coordinates and spectral densities are converted explicitly, never by a record constructor.

## Module organisation

Import the public API from `orsia_radiative` or from its defining scientific module. The root exports, signatures and numerical algorithms are unchanged by this reorganisation.

- `atmosphere.models` owns `Atmosphere`, `Absorber`, `AirBroadener` and physical-property validation; `atmosphere.layers` owns height/pressure coordinates, interfaces, layer means and conservative column redistribution; `atmosphere.hydrostatic` owns humidity/fraction conversion and hydrostatic construction. `surface`, `geometry`, `scene` and `profiles` own the other materialised physical inputs. Constructors validate their physical contracts and use `np.copy` directly for independent writable fields; absorption freshness uses `spectroscopy.components.values_equal` for targeted dependency comparisons.
- `spectral.grid`, `spectral.conversions` and `spectral.interpolation` separate support, point-density conversion and resampling. Radiometry remains in `radiometry`.
- `solar.source` owns solar incidence and irradiance materialisation; `solar.spectrum` owns native wavelength-point irradiance.
- `spectroscopy.lines`, `spectroscopy.line_tables`, `spectroscopy.cross_sections` and `spectroscopy.continuum` own their respective spectroscopy operations. `spectroscopy.components` owns species-resolved absorption contributions and explicit freshness checks.
- `optics.absorption` owns total absorption and its composition; `optics.phases`, `optics.scattering` and `optics.atmosphere` separate phase laws, scattering inputs and atmospheric optical composition.
- `transfer.solar` assembles solar radiance from `transfer.orders.scattering_orders`; `transfer.orders` owns direct initialization and order accumulation. `transfer.thermal` assembles LTE radiance, and `transfer.formal` owns source-independent transport.

Former root-level modules `arrays`, `sources`, `state`, `optical`, `absorption`, `continuum`, `line_tables`, `spectral_grid` and `solar_spectrum` are removed. `solar`, `spectral` and `spectroscopy` are packages without compatibility reexports. Old submodule imports and pickle globals are intentionally unsupported; use the root API or the defining modules above. The separate study checkout and parent design documents are not changed here.

The atmospheric split replaces `atmospheric_columns` with `atmosphere.layers` and `atmosphere.hydrostatic`; migrate direct imports to these owners or the existing root API. `orsia_radiative.atmosphere` continues to expose `Atmosphere`, `Absorber` and `AirBroadener`. Within the package, `hydrostatic` imports `models` and `layers`; `models` uses `profiles`, and `layers` depends only on NumPy and the standard library. `remap_layer_columns` is the public operation for conservative column redistribution. Its pressure-overlap calculation lives in the shared internal `layers._pressure_layer_overlap` helper, which hydrostatic construction evaluates once and reuses for its layer quantities. Vertical-support validation functions also live in `layers`.

## Compiled propagation

The `transfer/native/` package groups the compiled extension, its type stub and its C++ sources: `propagation_native.cpp`, `propagation_native.pyi`, `orders_native.h` and `thermal_native.h`. Python entry points remain in `transfer/formal.py`, `transfer/orders.py` and `transfer/thermal.py`, with the existing root and transfer APIs.

Package builds require a C++17 compiler and CPython headers; setuptools and the pinned pybind11 build dependency build `transfer.native.propagation_native` for the target interpreter, including CPython 3.14 free-threaded. Compilation errors are surfaced by the build. Wheels are platform/interpreter-specific. The extension uses the Python buffer protocol and pybind11's NumPy array allocation boundary. The build uses CPython and pybind11 headers, and packaged execution uses the declared NumPy/SciPy runtime dependencies. Numerical kernels take typed pointers and sizes independently of the binding; formal preparation and propagation preserve their floating-point formulas and operation order, while Rayleigh products and reflection use the arithmetic described below.

The internal native `prepare` and `apply` entry points remain positional-only and return `None`, writing directly into caller-provided buffers. They accept general C-contiguous buffer exporters with matching native `f` or `d` formats, including `array.array` vectors and multidimensional `memoryview` matrices, without conversion or copying. Inputs may be read-only; outputs must be writable and disjoint. Acquisition preserves exporter-specific contiguity/writeability exceptions; dtype and grid validation raise `ValueError`. Buffer exports remain live throughout the detached numerical call and are released with the Python thread state attached on both success and failure. Exit-only application does not acquire the unused centre coefficients. The extension declares free-threading and per-interpreter support; the isolated native module is tested in a subinterpreter with standard-library buffers, not the NumPy/SciPy package import surface.

`transfer.formal.apply_propagation` selects the serial native recurrence for matching native float32 or float64 arrays, copying noncontiguous inputs when necessary; its signature and physical model are unchanged. The NumPy recurrence handles mixed precision and arrays outside the native float/double ABI. No fast math or fused multiply-add is enabled. Sources and coefficients must not be mutated during a call; outputs are fresh, and concurrent calls may share immutable inputs. The internal buffer entry points are not supported consumer APIs and require fresh output buffers disjoint from each other and from inputs.

`transfer.formal.propagation_coefficients` also uses fused serial native preparation for matching float32/float64 vector inputs on small grids (at most 32 directions) or strongly repeated grids (at most 1024 directions, at least half exact repeated signed cosines). Reuse is call-local and exact, not an angular approximation or persistent cache. Stable nonpositive endpoint integration uses `exp(max(a,b))*exprel(-abs(b-a))`, with the near-zero limit and source-weight division order preserved; coefficients remain owned read-only snapshots. Large unique-angle grids use full-grid NumPy/SciPy vector operations because scalar native transcendental functions can be slower. Inputs must remain stable during preparation. The six solar coefficient matrices are unchanged; native preparation uses call-local scratch without matrix-sized temporaries, while the NumPy fallback builds full-grid temporary arrays.

Float32 depth and direction arrays select float32 coefficient buffers and native `expf`/`expm1f` arithmetic (or dtype-preserving NumPy/SciPy fallback); a float64 geometry input promotes preparation to float64. Application returns float32 centres and exits only when both source and coefficients are float32; mixed precision promotes both outputs to float64, without downcasting the higher-precision input. Solar cosine and bottom radiance scalars are rounded to the selected array precision; solar cosine must remain positive and at most one after rounding. Float64 callers retain their existing precision and defaults. Coefficient buffers remain owned read-only snapshots, and exit-only application and repeated reuse support both precisions. Float32 halves coefficient storage but has less exponent range and more accumulated roundoff; suitable precision depends on optical depth, direction cosines, cell count and cancellation in signed sources. This path covers the formal operator, not an end-to-end float32 solar or thermal solver, and introduces no changes to angular grids, subdivisions or scattering settings.

Run `devbox run -- just rebuild-propagation` after editing native source, then `devbox run -- just test-propagation typecheck-propagation runtime` for equivalence, dtype, edge, concurrent-call and imported-runtime checks, and `devbox run -- just clean-install` for independently installed float32/float64 wheel execution. The one-off propagation-dtype benchmark is retired; existing local measurements are historical formal-operator observations, not complete solar-call timings. These are software checks, not instrument-space or retrieval qualification; prototype timing evidence does not establish packaged speed on every platform.

Binding and dtype regressions remain in the propagation library tests (`devbox run -- just test-propagation`); the standalone before/after parity recorder is retired. These are software checks, not independent scientific references.

## Solar boundary-response reuse

Solar SOS prepares the discrete unit-amplitude Lambertian boundary response once per active spectral point and scales it for the initial surface field and subsequent reflected orders. The response uses the existing cell recurrence and centre convention; each atmospheric order still propagates its full diffuse source. Buffers belong to the current evaluation and spectral point. Any subnormal or zero upward unit exit selects the original actual-amplitude boundary sweeps, preserving exponent range in extreme attenuation. Scaling changes rounding at floating-point precision while preserving the physical model, public API, spectral support, angular grid and order controls. The additional retained response occupies one float64 cell/direction matrix and one exit vector, independently of the number of orders and spectral points.

Run `devbox run -- just test-solar-boundary` for repeated-sweep regression tests. The historical Git-baseline timing runner has been retired; numerical boundary-response regressions remain in the library test suite.

## Solar order accumulation

Packaged SOS evaluates the full spectral grid in one detached C++ invocation: physical extinction summation and overflow checking, optional Delta-M transformation, subcell preparation, direct reflection, transport coefficients, direct-beam scattering, diffuse redistribution and Lambertian surface coupling. Rayleigh components with at least three physical layers use the exact seven-factor expansion, including sparse and mixed scenes; smaller Rayleigh scenes and HG aerosol use dense phase operators. A call-owned FIFO phase cache retains up to 16 MiB of operators, with larger operators applied from temporary storage. Spectral points run serially and reuse the same transport and field buffers; successive orders retain the preceding angular field. Angular quadrature, public support validation, domain extraction and final `Radiance` assembly remain in Python.

The matrix products use SciPy's native `dgemm` pointer, resolved once when `orders` imports from the LP64 Cython capsule supplied by the supported environment. The interpreter-local callable retains the exporting module for the pointer's lifetime. Native balanced flux reduction and the shared formal sweeps complete each order. The detached region operates exclusively on exported buffers, scalar values and native-owned storage. Scalar source scaling and boundary operations use strict separate multiplication/addition; BLAS retains its backend arithmetic. Numerical regression uses `rtol=3e-14, atol=0` for retained boundary cases and `rtol=3e-13, atol=0` for expanded mixed/Delta-M cases; independent ray tests assess the same discrete physical model.

Keep borrowed inputs stable throughout the call, including concurrent evaluations. The native boundary supports native float64 C-layout, F-layout, signed strides and unaligned storage through retained buffer exports; other input precision is promoted at the Python boundary. Scratch and fresh writable float64 `(order, point)` output belong to one invocation. Public signatures and numerical controls retain their existing meanings. The compiled extension and fixed SciPy LP64 BLAS signature are supported-environment invariants; `orders` and `formal` import the extension directly, and `orders` binds BLAS at module load. The formal operator retains its separate float32/float64 behavior. BLAS thread policy remains caller-controlled.

Run `devbox run -- just test-solar-calls` for arithmetic, ownership, stride, error, portable-native and concurrent-call regressions, and `devbox run -- just clean-install` for independent wheel-install checks. The frozen Python recurrence lives in the test suite as an explicit reference. The one-off native migration runners have been retired without removing these numerical regressions or references.

## Public contract

- `Atmosphere` takes N+1 ascending height boundaries in m, N+1 descending pressure boundaries in Pa and N finite layer pressures, temperatures and molecular columns; layers are bottom to top. It does not reconstruct hydrostatic state or contain aerosol optics.
- `Absorber(species, columns_molecules_m2, air_broadener)` declares a gas column and whether its foreign broadener pool is `"dry_air"` or `"total_air"`. H2O columns must match the atmosphere's water-vapour columns.
- `LayerProfileBasis` and `evaluate_profile` provide named additive perturbation directions. Evaluation returns a fresh writable float64 array; callers rebuild and validate a new state so coupled columns and Kirchhoff consistency are never silently changed.
- `SurfaceState(grid, reflectance, emissivity, temperature_k, ...)` accepts a canonical wavenumber `SpectralGrid` and diffuse Lambertian values in [0, 1]. Thermal emissivity and temperature are supplied together. Reflectance plus emissivity must equal one when both are present.
- `ObservationGeometry(view_zenith_deg)` is a local flat viewing geometry in degrees, with zenith in `[0, 90)` measured from the upward normal toward the observer along outgoing photon propagation, not the instrument's downward look vector. Incident directions and their azimuth relative to this view belong to explicit radiative sources; no geodetic transformation, refraction or curvature is implied.
- `RadiativeScene` combines the materialised atmosphere, surface and view geometry. Reflectance and emissivity are physical boundary properties; an active source is selected by a transfer operation, not inferred from the scene.
- `Line` and `LineSet` are materialised isolated-line inputs in spectroscopy-native cm-1 units: air/self Lorentz broadening, no implicit continuum, and line strengths already weighted for isotopologue abundance relative to the whole-species column. Lorentz remains the default; supplying `LineSet.molecular_mass_kg` explicitly enables thermal Voigt profiles as described below. `CrossSectionField` is a materialised per-layer cross-section or line-excluding continuum table on a cm-1 grid, already weighted to that same species-column convention. `line_absorption` and `table_absorption` perform no provider access or temperature/pressure interpolation; `table_absorption` uses linear spectral interpolation without extrapolation. These fixed conventions are not configurable policy fields.
- `AbsorptionComponent` retains species-resolved spectroscopy and its physical dependencies upstream of transfer. `compose_absorption(atmosphere, source, components, reference_atmospheres=None)` derives the grid from `source.irradiance.wavenumbers_cm1` and checks exact grids, layers, species and duplicate or overlapping contributions, then validates finite summed depth in the physical value `Absorption(grid, optical_depth)`. No provenance or atmosphere reference is stored in either physical record. Pass separately retained evaluation atmospheres to `reference_atmospheres` when explicit freshness checking is required; otherwise the supplier is responsible for physical consistency. No resampling is performed. Total absorption is dimensionless vertical depth in independent writable float64 storage of shape `(n_layers, n_points)`, bottom to top. Solar irradiance values and direction are not absorption dependencies; same-grid illumination changes permit reuse. An empty collection composes zero absorption on the atmosphere/source support.
- `ScatteringProperties`, `AerosolOptics`, `RayleighComponent` and `AerosolComponent` retain explicit optical depths and fixed phase conventions. Aerosol extinction already includes its absorption; do not add it again to `Absorption`. Scattering depth contributes to the source term, not a second extinction term.
- `AtmosphericOptics` represents total absorption and optional explicit Rayleigh/aerosol roles on evaluated layer/spectral support. It retains no solar illumination, atmosphere identity, provenance, surface/view geometry or transfer solver. `compose_atmospheric_optics(atmosphere, source, ...)` validates the supplied atmosphere before assembling these physical contributions, using only the source's spectral coordinates.
- `validate_optical_inputs(atmosphere, source, absorption=None, *, rayleigh=None, aerosol=None)` checks singular total absorption on `source.irradiance.wavenumbers_cm1`, exact support, layer shape and finite summed extinction. `validate_atmospheric_optics(..., reference_atmosphere=...)` can check an explicitly retained atmosphere identity; without that argument, physical freshness remains the caller's responsibility. Species and spectroscopy overlap checks belong to `compose_absorption`, not transfer. All three roles absent means a transparent atmosphere.
- `SpectralIrradiance(grid, values)` is beam-normal irradiance in W m-2 per (cm-1), defined in `radiance.py`. `SolarSource` adds the Sun's local zenith and the incoming-propagation-to-view relative azimuth without choosing a transfer method. `Radiance(grid, values, geometry)` is only the physical directional spectral value and its support; it retains no transfer-method identity, input provenance, decomposition or diagnostics. Calculations that need additive terms expose them through separate term functions.

### Spectral domain and point grid

`SpectralDomain(coordinate, intervals)` represents declared physical coverage as positive, finite, closed intervals in either `"wavenumber_cm1"` or `"wavelength_nm"`. Intervals are ordered, nonoverlapping and separated; singleton intervals are allowed. `SpectralGrid(coordinate, points, domain=None)` represents the actual one-dimensional, positive, strictly increasing float64 point samples. When a domain is supplied, its coordinate must agree and every point must belong to it. `domain=None` means coverage is unknown: neither a continuous min/max envelope nor coverage across gaps is inferred from points.

`wavenumber_grid(points, domain=...)` explicitly adapts a raw array to the canonical radiative grid; passing an existing wavenumber grid retains that object. Grids snapshot unsafe coordinate buffers and may share safely immutable bytes-backed storage. Retained point arrays and their shape/dtype metadata must not be mutated. `grid.same_points(other)` tests exact coordinate/unit and point equality, without tolerance or interpolation. Full grid equality also includes the declared domain; evaluated spectra with different physical coverage can still be compatible at identical points.

Spectral irradiance, radiance, surface response, absorption components/totals, prepared cross sections, native cross-section tables and scattering/atmospheric optics store `grid` as their only spectral authority. Solar composition and transfer retain the irradiance's grid; explicit-grid composition and thermal transfer retain the supplied optics grid. The numerical `wavenumbers_cm1` property is only an alias of `grid.points`, not a second field or sampling choice. `compose_absorption_on_grid(atmosphere, grid, components)` and `atmospheric_optics(atmosphere, grid, ...)` require a `SpectralGrid`. Spectroscopy evaluation still accepts an explicit raw cm-1 array or a grid; passing a grid preserves its declared support in the returned record.

This is a breaking constructor and dataclass-field migration: use `grid=` instead of `wavenumbers_cm1=` and `dataclasses.replace(record, grid=new_grid)` to change sampling. Generic dataclass dictionaries now contain `grid`, not a coordinate-array field. Orsia callers and runnable examples are migrated; external callers, including the separate study checkout, must adopt the new constructors. The normalized cross-section NPZ keys remain unchanged, with array-to-grid adaptation at the reader boundary; the old spectroscopy artifact/manifest API is removed by the data redesign.

`convert_spectral_grid(grid)` explicitly converts nm to cm-1 or vice versa, reversing points and the order/endpoints of declared intervals. It does not transform values, interpolate or integrate. Convert spectral densities separately with `density_per_nm_to_per_cm1` or `density_per_cm1_to_per_nm`, including the Jacobian and reversed value order; dimensionless point values only reverse order. Fixed-table evaluation additionally rejects targets outside declared native support and interpolation stencils crossing a declared gap; exact native samples remain available.

`solar_irradiance(spectrum, target_grid, wavelength_domain=None)` consumes the physical `SolarIrradianceSpectrum` owned by radiative: increasing wavelength points [nm], nominal density [W m-2 nm-1] and optional supplied uncertainty/bandwidth. Arrays are independent writable float64 copies. The pure bridge applies the explicit density Jacobian and piecewise-linear point interpolation onto the increasing cm-1 target grid, rejecting extrapolation and gaps supplied through the optional wavelength domain. Without a domain only the native point envelope is known; undeclared gaps cannot be inferred. `interpolate_point_density` is available for already converted values. These operations do not perform conservative bin rebinning, instrument convolution, uncertainty propagation or provider access. `scale_spectral_irradiance(..., distance_au=..., reference_distance_au=...)` applies only the explicit inverse-square factor; date, ephemeris and geometry remain consumer policy.

```python
import numpy as np
from orsia_radiative import SpectralDomain, SpectralGrid, SpectralIrradiance

spectral_domain = SpectralDomain("wavenumber_cm1", ((999.0, 1000.0), (1001.0, 1003.0)))
spectral_grid = SpectralGrid("wavenumber_cm1", np.array([1000.0, 1001.0]), spectral_domain)
spectral_irradiance = SpectralIrradiance(spectral_grid, np.array([8.0, 10.0]))
assert spectral_irradiance.grid is spectral_grid
assert spectral_irradiance.wavenumbers_cm1 is spectral_grid.points
```

### Numerical arrays and ownership

Homogeneous numerical fields take ordinary `NDArray[np.float64]` arrays, not tuples: spectral values, surface properties, atmospheric profiles and gas columns, profile bases, partition ratios, optical depths, albedos, asymmetries and cross sections. Constructors validate dtype, finite values, support and physical bounds. Numerical values retain independent writable copies made with `np.copy`, without a C-order guarantee. Spectral coordinates retain their separate policy: they share an existing contiguous bytes-backed immutable array; mutable, merely read-only, memoryview-backed or strided coordinates receive an immutable snapshot. Inputs are never mutated, and later changes to caller buffers do not affect copied values. Callers own consistency after mutating retained numerical values and must keep shared records stable during calculations. Spectral coordinates, including shape/dtype metadata, must not be mutated; use `.copy()` for a writable coordinate buffer. Frozen dataclasses do not protect their writable array buffers.

Spectral vectors have shape `(n_points,)`; layer conditions, columns and partition ratios have shape `(n_layers,)`; atmospheric boundaries have shape `(n_layers + 1,)`. Layer/spectral depths, albedos, asymmetries and cross sections have shape `(n_layers, n_points)`, bottom to top on the first axis. Phase moments have shape `(n_layers, n_points, n_moments)`, beginning with the zeroth moment. A profile baseline has shape `(n_samples,)`, its directions `(n_directions, n_samples)` and evaluation coefficients `(n_directions,)`. The sample axis follows the declared layer or spectral domain. Units and physical conventions are unchanged.

This is a breaking caller-contract change: construct numerical inputs explicitly with `np.array(..., dtype=np.float64)` or `np.asarray(..., dtype=np.float64)`; no tuple compatibility coercion is installed. Numerical field `==` now gives elementwise NumPy comparison; use `np.array_equal` or `np.testing.assert_allclose` for whole arrays. Array-containing physical records are unhashable and no longer provide whole-record value equality. Identifiers, collections of records, validity-range pairs and the `(zenith_deg, relative_azimuth_deg)` pair remain tuples. Only `AerosolComponent` carries a same-support HG asymmetry array; `RayleighComponent` has no asymmetry or selectable phase field. Report/JSON writers convert arrays to lists only at serialization boundaries.

## Record comparison migration

The custom value `__eq__` methods on `Absorber`, `Atmosphere`, `LayerProfileBasis`, `SurfaceState`, `SpectralIrradiance`, `Radiance`, `LineSet`, `CrossSectionField`, `PreparedCrossSections`, `WaterContinuumReference`, `AbsorptionComponent`, `Absorption`, `ScatteringProperties`, `AerosolOptics`, `RayleighComponent`, `AerosolComponent` and `AtmosphericOptics` have been removed. Their dataclasses retain `eq=False`; `RadiativeScene` and `SolarSource` also disable generated equality. These objects use identity comparison, so distinct instances no longer compare equal merely because their fields have equal values. All remain unhashable. Constructor fields, array ownership and numerical calculations are unchanged by this equality migration.

Compare the fields relevant to the calculation explicitly: use `np.array_equal` or `np.testing.assert_array_equal` for exact numerical values, and `np.testing.assert_allclose` with scientifically justified tolerances for approximate agreement. Compare spectral support and geometry separately where required. `SpectralGrid` retains its specific value equality, including the declared domain, and its `same_points` method. Scalar-only record equality is unchanged. Absorption freshness still compares selected physical dependencies with `orsia_radiative.spectroscopy.components.values_equal`; it does not depend on whole-record equality. This function compares arrays, recursively nested tuples and scalar values exactly; NaNs are not equal. Its former `orsia_numerics.arrays.values_equal` location is removed without an alias. The generic `orsia_numerics.arrays.array_record_equal` helper is removed without an alias.

## Transfer boundary

`Radiance(grid, values, geometry)` now has exactly these three fields. This is a breaking constructor, dataclass-field and serialization change: `model_id`, `source_id`, `input_source_ids`, `radiative_source`, `radiative_source_id`, `components`, `diagnostics` and `surface_coupling_source_ids` are removed without aliases; `RadianceComponent` and `RadianceDiagnostic` are no longer exported. Compare radiance values, spectral support and geometry explicitly, not calculation labels. External consumers must keep their own calculation labels and input lineage, and request numerical decompositions separately. The input-side value/provenance migration and explicit freshness checks are described below. Orsia callers and examples are migrated; the separate study checkout and parent design documents are not modified by this child-scoped change.

The scene-level transfer entry points are `transfer.solar.reflected_radiance` and `transfer.thermal.schwarzschild_radiance`. Separate `transfer.orders.scattering_orders` and `transfer.thermal.schwarzschild_terms` functions return fresh numerical arrays for decomposition. Import `transfer` from `orsia_radiative`, or `solar` / `thermal` from `orsia_radiative.transfer`. The `transfer` package exposes these two family submodules. Import `orders` explicitly from `orsia_radiative.transfer` for numerical diagnostics. `Radiance` carries the physical grid, values and observation geometry.

`solar` evaluates reflected sunlight from one Sun. `source: SolarSource` represents beam-normal irradiance along one downward direction; no source collection or implicit superposition remains. A resolved solar disc and external diffuse incident field are not implemented. Thermal emission is selected explicitly through the separate [thermal operation](#lte-thermal-transfer), never inferred by the solar methods from stored temperatures or emissivity.

`transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=2, angular_order=8, layer_subdivisions=8, delta_m_moments=None)` sums atmospheric orders 0 through a caller-selected nonnegative integer N, with a Lambertian boundary at every order. Physical direct reflection uses `max_order=0, delta_m_moments=None`. Order one includes direct-beam scattering, Rayleigh and HG ground coupling, and two ground encounters separated by one atmospheric event. The optional [Delta-M calculation](#experimental-aerosol-delta-m) counts residual events.

The radiance calculation and the numerical order decomposition each require one `source` and one `AtmosphericOptics`. Compose with `compose_atmospheric_optics(scene.atmosphere, source, absorption=total_or_none, rayleigh=rayleigh_or_none, aerosol=aerosol_or_none)`; `None` denotes an absent role. The physical scope is one conservative Rayleigh role and one untruncated HG aerosol role. Solver-local Delta-M adaptation preserves these physical inputs.

The two phase laws operate under one total extinction. SOS applies Rayleigh and HG redistributions to the same preceding angular field and sums their source terms before transport, preserving cross interactions. Physical roles are ordered absorption, Rayleigh, aerosol, regardless of keyword order; contributor provenance is caller-owned.

`solar_source(irradiance, direction=(zenith_deg, relative_azimuth_deg))` creates a `SolarSource`. Zenith is in `[0, 90)` toward the Sun and relative azimuth in `[0, 360)` degrees from incoming photon propagation to the outgoing view, under the [angular contract](#angular-contract-and-outward-ray-conversion) below. These are not geodetic coordinates. Sources own their incident directions relative to the scene's view; changing the absolute view azimuth requires recomputing each source's relative azimuth. The calculations consume both directly without constructing a temporary solar scene. A surface with emissivity may supply its reflected contribution without activating thermal emission; an emissivity-only surface cannot supply the reflection required by current solar methods. `CollimatedSource` and `collimated_source` are removed, without compatibility aliases.

Solar transfer returns `Radiance`. Use `orders.scattering_orders` for its float64 order array. Its rows are fresh writable arrays on the calculation grid. Callers retain input lineage and numerical controls alongside the calculation.

The solar methods implement downward reflected solar contributions only. They do not infer illumination from scene metadata or evaluate thermal emission. Neither family supports a combined thermal/solar call, derivatives, retrieval-coordinate mapping or parallel execution policy. Solar SOS also has no unresolved-tail closure, automatic order stopping or acceleration. `scattering_orders` returns rows for the selected fixed-order calculation; the last row is not a tail bound. These arrays identify numerical terms, not public import paths or input provenance.

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

This simplification removes `solar.prepare_series_of_scatterers`, `solar.evaluate_series_of_scatterers`, `solar.SolarScatteringWorkspace`, `orders.prepare_scattering_orders`, `orders.allocate_scattering_buffers`, `orders.evaluate_scattering_orders` and `orders.PreparedScattering`. Migrate repeated evaluations to the complete call above. The private native persistent geometry/buffer types and their allocation, evaluation and BLAS-binding entry points are also removed. The physical validation, fixed-order recurrence, Delta-M conventions, spectral units and numerical controls retain their existing definitions. Use `devbox run -- just test-solar-calls` for the maintained complete-call regressions.

## LTE thermal transfer

`transfer.thermal.schwarzschild_radiance(scene, *, optics, angular_order=32, hemispheric_method="quadrature")` evaluates upward top-of-atmosphere radiance in a plane-parallel, horizontally homogeneous, non-scattering LTE atmosphere with isothermal input layers and zero incident radiance at the top. It uses atmospheric layer temperatures and vertical absorption depths on `optics.wavenumbers_cm1`; no solar object, provider, implicit grid, resampling, cloud model, non-LTE emission or observer-height selection is introduced. Any strictly positive Rayleigh or aerosol scattering depth raises `ValueError`, even if small or in an opaque layer. Zero scattering roles are accepted; purely absorbing aerosol extinction contributes once to absorption and emits at the corresponding layer temperature.

For a layer with source function B, ray cosine mu and vertical absorption depth tau, `formal.schwarzschild` applies `I_out = I_in * exp(-tau/abs(mu)) + B * (-expm1(-tau/abs(mu)))`. This is exact within each supplied constant-source layer, not an approximation to a resolved temperature gradient. The low-level function accepts float64 `(layers, points)` source-function/depth arrays and `(points,)` incident radiance, returning a fresh writable exit vector; positive mu travels bottom to top and negative mu top to bottom. It does not multiply B by tau, avoiding that intermediate overflow in opaque layers, and `expm1` preserves thin-layer emission.

The opaque Lambertian surface requires spectral emissivity and a positive temperature. It emits `epsilon * B(T_surface)` and reflects the descending atmospheric irradiance as `(1-epsilon) * E_down/pi`, then both terms are attenuated along the upward view ray. An emissivity-only surface is sufficient: reflectivity is its complement; when reflectance is also supplied, `SurfaceState` already checks Kirchhoff consistency. The implementation uses the emissivity complement in either case. Surface reflection does not create an additional atmospheric source in a non-scattering calculation. No external downwelling background is included.

`angular_order` is a Python integer >= 2 (booleans are rejected). Gauss-Legendre nodes on the open downward hemisphere integrate `E_down/pi = 2 * integral_0^1 mu * I_down(mu) dmu`; azimuth integrates analytically under horizontal symmetry. This numerical quadrature is needed only for nonblack surfaces with nonzero absorption, but its control is always validated. Refine it independently for the intended observable, particularly with weak absorption and near-grazing downwelling rays. The default is not a general accuracy certificate; there is no automatic convergence or error estimate. Layer subdivision is unnecessary for the supplied isothermal model.

`hemispheric_method="expn"` explicitly replaces angular quadrature with a layer sum `E_down/pi = 2 * sum(B_layer * [E3(below)-E3(below+depth)])`, where `below` is the summed vertical absorption between the layer and surface. Thin intervals satisfying `depth <= 0.02*min(below, 1)` use the integrated midpoint expansion of E2 through fourth order; intervals near zero use the integrated logarithmic series, and the origin uses the E3 recurrence with `expm1`. These expressions retain depth separately even when `below+depth` rounds to `below`. Well-separated intervals use direct E3 values, with an asymptotic tail beyond optical depth 700 to retain representable contributions past SciPy's underflow cutoff. Zero depths and final binary64 underflow return zero. The native calculation reuses call-local layer vectors across spectral points; scratch storage scales with layer count, while the three returned arrays scale with spectral size. The default method and its angular refinement remain unchanged. `angular_order` is validated for both methods; invalid method names raise `ValueError`.

Both thermal methods execute in the existing `propagation_native` extension, with one detached serial numerical region per call. The `expn` scalar comes from SciPy's Cython API through a signature-checked capsule; the bound callable retains its provider in the importing interpreter. At binding time, Python derives the scalar Planck coefficients `100*h*c/k` (cm K) and `2*k*c*1e6` (the Rayleigh-Jeans prefactor for radiance per cm-1) from `scipy.constants`; the native callable captures these values. Planck follows the logarithmic formula used by `radiometry.planck_wavenumber`, and ray transport follows `formal.schwarzschild`. Native scalar transcendental functions and reductions permit binary64 rounding differences from NumPy. Inputs remain stable for the duration of a call; native-endian float64 C-contiguous aligned inputs are borrowed, and other storage layouts receive temporary compatible copies. Each term owns fresh writable float64 storage. Concurrent calls share immutable inputs and own independent scratch.

Run `devbox run -- just rebuild-propagation`, then `devbox run -- just typecheck-thermal-native test-thermal-native` for thermal reference comparisons, independent interval integrals, physical limits, buffer validation and concurrent-call checks. The frozen pre-port NumPy/SciPy calculation lives in `tests/thermal_reference.py` as software regression evidence. `devbox run -- just clean-install` exercises built wheels independently of workspace resolution, including both thermal methods. These checks establish bounded software and numerical agreement; instrument-space accuracy and retrieval performance require application-specific validation.

`radiometry.planck_wavenumber(grid, temperature_k)` returns Planck radiance in W m-2 sr-1 per (cm-1), with exact SI h, c and k and the cm-1 density Jacobian. A positive finite scalar temperature returns `(points,)`; a float64 `(layers,)` temperature array returns `(layers, points)`. Coordinates must be positive, strictly increasing float64. Returned arrays are fresh and writable; inputs are not retained or mutated. Logarithmic amplitude evaluation avoids intermediate overflow; Rayleigh-Jeans and Wien limits are handled, underflow returns zero, and unrepresentable finite radiance raises `ValueError`. This is a point-density evaluation, not spectral integration or conversion to brightness temperature.

Thermal inputs are prepared with `compose_absorption_on_grid(atmosphere, grid, components)` and `atmospheric_optics(atmosphere, grid, ...)`. They retain the same exact-support, independent-storage, overlap and physical-freshness rules as their solar counterparts. The caller selects the grid once before spectroscopy; optical records retain its evaluation support, and transfer checks the surface grid and layer shape. Use `validate_scene_optics(scene, optics, reference_atmosphere=...)` for an explicit conservative identity check before thermal reuse. Mutable coordinate inputs are snapshotted. Existing source-based functions delegate to these common calculations and preserve their signatures and solar grid authority. No union of a solar source and array, new grid wrapper or fictitious solar irradiance is needed.

`schwarzschild_radiance` returns only the physical `Radiance`. `schwarzschild_terms` separately returns fresh writable arrays named `surface_emission`, `atmospheric_emission` and `reflected_atmospheric_emission`; their sum is the returned radiance. `view_transmission`, angular controls, method identity and ordered optical provenance are calculation-local and are not fields of `Radiance`. Numerical buffers are call-local without caches, RNG, I/O or parallel policy; storage scales with layers times spectral points and does not retain an angular cube.

```python

from orsia_radiative import wavenumber_grid
import numpy as np
from orsia_radiative import (
    Absorption, Atmosphere, ObservationGeometry, RadiativeScene, SurfaceState,
    compose_absorption_on_grid, atmospheric_optics,
)
from orsia_radiative.radiometry import planck_wavenumber
from orsia_radiative.transfer import thermal

grid = np.array([800.0, 1000.0], dtype=np.float64)
atmosphere = Atmosphere(
    np.array([0.0, 1000.0]), np.array([100000.0, 90000.0]),
    np.array([95000.0]), np.array([285.0]), np.array([2.0e29]),
    np.array([1.9e29]), np.array([1.0e28]),
)
scene = RadiativeScene(
    atmosphere, SurfaceState(wavenumber_grid(grid), emissivity=np.array([0.9, 0.95]), temperature_k=300.0),
    ObservationGeometry(20.0),
)
# Synthetic materialised absorption; a real caller evaluates spectroscopy on grid.
absorption = Absorption(wavenumber_grid(grid), np.array([[0.1, 0.5]]))
optics = atmospheric_optics(atmosphere, wavenumber_grid(grid), absorption=absorption)
result = thermal.schwarzschild_radiance(scene, optics=optics, angular_order=64)
analytic_result = thermal.schwarzschild_radiance(scene, optics=optics, hemispheric_method="expn")
np.testing.assert_allclose(analytic_result.values, result.values, rtol=1e-10)
terms = thermal.schwarzschild_terms(scene, optics=optics)
assert result.values.shape == (2,)
np.testing.assert_allclose(sum(terms.values()), result.values)
# Empty upstream contributions give a transparent atmosphere, without any Sun.
vacuum = compose_absorption_on_grid(atmosphere, wavenumber_grid(grid), ())
vacuum_optics = atmospheric_optics(atmosphere, wavenumber_grid(grid), absorption=vacuum)
vacuum_result = thermal.schwarzschild_radiance(scene, optics=vacuum_optics)
np.testing.assert_allclose(vacuum_result.values, scene.surface.emissivity * planck_wavenumber(grid, 300.0))
```

`devbox run -- just test-thermal` checks Planck against an independent frequency-density expression and a Stefan-Boltzmann integral, signed layer-ray propagation, vacuum/thin/opaque limits, black isothermal equilibrium, stratified source integrals, nonblack surface reflection against adaptive angular integrals and E3, relayering, quadrature refinement, purely absorbing aerosols, rejected scattering, explicit-grid composition/freshness, independent results and deterministic call-local concurrent evaluations. These are bounded synthetic numerical and software checks. Historical homogeneous SASKTRAN2 thermal comparisons remain outside this candidate; qualify an application-specific thermal accuracy envelope with representative external references.

## Physical values, provenance and explicit freshness

This is a breaking constructor, dataclass-field and serialization migration without compatibility aliases. `source_id` is removed from `SpectralIrradiance`, `RadiativeScene`, `AbsorptionComponent`, `ScatteringProperties`, `AerosolOptics`, `RayleighComponent` and `AerosolComponent`; `input_source_ids` is removed from `Absorption`; `reference_atmosphere` is removed from `AbsorptionComponent`, `Absorption` and `AtmosphericOptics`. Records contain physical values and their semantic support; they do not provide whole-record value equality. Keep artifact identity and evaluation references alongside a calculation rather than in its values. `species`, `kind`, `isotopologue`, `line_wing_convention`, `truncation` and phase parameters remain because they describe the values, not their provenance. `AerosolOptics` remains an extinction/albedo parameterisation with a public converter, not an additional transfer representation. `source_id` is also removed from `LineSet`, `CrossSectionField` and `WaterContinuumReference`; their constructor positions and serialized fields change without aliases. Provider identifiers are not accepted by these physical input records. Prepared numerical records and profile bases are outside this migration.

Automatic identity-collision checks and attached-reference freshness checks are removed. Composition and transfer still reject incompatible grids/layer shapes, invalid numerical values, overflow and semantic spectroscopy overlap. A supplied grid and layer shape cannot establish physical freshness. Before reusing a component, call `validate_absorption_freshness(atmosphere, component, reference_atmosphere=evaluation_atmosphere)` or pass `reference_atmospheres=(...)` in component order to either absorption composer; length mismatch raises `ValueError`. These compare exact physical dependencies, including line broadening and continuum density. For composed optics, `validate_atmospheric_optics(scene, source, optics, reference_atmosphere=evaluation_atmosphere)` and `validate_scene_optics(scene, optics, reference_atmosphere=evaluation_atmosphere)` conservatively reject a different atmosphere object. Without these explicit references, the caller owns physical consistency and must recompute affected optics; the library cannot infer their generating state. No hidden registry or mandatory provenance wrapper replaces the removed fields. Parent design documents and external consumers, including the separate study checkout, are not migrated here.

## Pure atmospheric column materialisation

`atmosphere.layers` defines `pressure_level_layer_means`, `remap_layer_columns` and `geometric_height`. `atmosphere.hydrostatic` defines `specific_humidity`, `gas_columns_from_fraction` and `hydrostatic_atmosphere`. All six functions are available from `orsia_radiative`. They consume explicit arrays and physical assumptions; hydrostatic construction returns the existing `Atmosphere` record. The [atmospheric boundary contract](../../docs/atmospheric-data.md#numerical-materialisation) defines units, dry/moist-air conventions, hydrostatic/height approximations, support, conservation and ownership. Source columns are materialised first; remapping conserves supplied extensive amounts over equal pressure support. Provider selection and additional physical inputs remain caller-owned.

## Atmospheric optics and freshness

`henyey_greenstein_phase(asymmetry, cosine)` and `rayleigh_phase(cosine)` in `optics/phases.py` own the phase laws used by SOS beam-source construction and diffuse redistribution; both are exported at the package root. They evaluate dimensionless phases with a 4π mean of one, using float64 NumPy broadcasting without Python element loops. HG accepts finite `|g| <= 0.8`; both reject nonfinite cosines or values outside `[-1, 1]` rather than silently clipping physical inputs. Scalar inputs return Python floats; if any input is an array, the result is a fresh writable array, including for zero-dimensional input, without mutating or retaining inputs. Incompatible HG shapes raise NumPy's broadcasting `ValueError`. `orders.phase_function` remains a thin Rayleigh/HG dispatcher for existing callers, retaining `g=None` and geometric cosine clipping but now applying the optical evaluator's HG bounds; it no longer defines either law. Transfer geometry bounds dot-product roundoff before optical evaluation. This factorisation does not change quadrature, path definitions or scattering orders.

`compose_atmospheric_optics(atmosphere, source, *, absorption=None, rayleigh=None, aerosol=None)` derives spectral support from `source.irradiance.wavenumbers_cm1` and validates the supplied layer shape before constructing the physical `AtmosphericOptics`. It takes neither a scene nor a separately selected grid. Numerical spectroscopy kernels still receive only the source's numerical grid and the atmosphere, not illumination properties. Atmospheric optics retain that original immutable coordinate array by reference and share optical records without copying; the retained `wavenumbers_cm1` fields identify evaluated support rather than independent sampling choices. Freshness is checked only when the caller supplies a separately retained `reference_atmosphere` to `validate_atmospheric_optics`. Common extinction and angular work buffers belong to transfer's one-shot call or explicit reusable buffer resource; atmospheric optics retain their physical values independently.

`AtmosphericOptics`, `compose_atmospheric_optics` and `validate_atmospheric_optics` replace `PreparedOpticalInputs`, `prepare_optical_inputs` and `validate_prepared_optical_inputs` without compatibility aliases. The names describe atmospheric optical state and its composition, independently of the transfer operation; fields and numerical behaviour are unchanged by this rename.

Migrate the former `prepare_optical_inputs(scene, grid, ...)` to `compose_atmospheric_optics(scene.atmosphere, source, ...)`, and `validate_optical_inputs(scene, grid, ...)` to `(scene.atmosphere, source, ...)`. Replace `compose_absorption(atmosphere, grid, components)` with `compose_absorption(atmosphere, source, components)`; the source is required and there is no independent grid argument or compatibility overload. Numerical spectroscopy kernels still receive `source.irradiance.wavenumbers_cm1`, not the source object. Optical composition requires neither a surface nor a view; surface support is checked when the selected transfer method receives the scene. `validate_atmospheric_optics(scene, source, optics)` also checks the current source against the original support.

`update_optical_inputs` is removed: constituent selection and reuse belong upstream, not to the optical state. Recompute affected spectroscopy contributions, reuse unaffected generated components, call `compose_absorption` for the current atmosphere, then compose atmospheric optics from the current total and explicit scattering roles. Recomposition can add or remove a role; `None` means absence, not retention. Selecting retrieved gases alone is insufficient to decide which contributions are unchanged. No inverse solver, retrieval-coordinate mapping or generic dependency framework is introduced.

`line_absorption`, `table_absorption` and `water_continuum_absorption` return physical components without an attached atmosphere reference. Composition can compare dependencies against separately retained evaluation atmospheres through `reference_atmospheres`; generated kernels validate their own evaluation inputs. Materialised tables and line sets likewise retain only physical fields; any source record stays outside radiative inputs and is not a freshness certificate. Rayleigh/aerosol records likewise contain no generating model; reuse or replacement remains explicit.

`cross_sections(grid, CrossSectionField)` interpolates a fixed table once into caller-owned `PreparedCrossSections`; `evaluate_table_absorption(atmosphere, prepared)` applies current molecular columns. The table's exact layer T/p and spectral support must remain fixed. `table_absorption` is the full prepare-and-evaluate path. This is a fixed-cross-section scientific contract, not general tau scaling: changing abundance in `line_absorption` also changes self/foreign widths and requires reevaluating that gas's lines. No line-profile basis or persistent numerical cache is introduced, and no retrieval speedup is claimed without representative benchmarks.

### Pure line-table materialisation

`LineDatabase(lines, molecule_id, isotopologue_id, coverage_cm1)` represents one molecule/local isotope's transitions and declared centre-coverage completeness [cm-1]. Empty lines are valid; bounds are not inferred from centres. `PartitionFunctionTable(molecule_id, isotopologue_id, temperatures_k, partition_sums)` represents dimensionless Q at positive increasing knots [K], with independent writable float64 arrays of shape `(n_knots,)`, at least two knots. `evaluate_partition_function(table, temperatures_k)` returns fresh writable Q(T) by explicit linear interpolation without extrapolation, not HAPI/TIPS interpolation.

`line_set(database, partition_table, atmosphere, grid, species=..., molecular_mass_kg=..., pressure_range_pa=..., wing_cutoff_cm1=..., subtract_lorentz_pedestal=False)` matches isotope identities, requires a declared atmospheric species and checks centre coverage beyond cutoff plus maximum absolute pressure shift across the declared pressure support. It selects lines, computes Q(296 K)/Q(T_layer) and returns a `LineSet` without file access, identity generation or caches. Mass is kg/molecule, pressure Pa, grid/cutoff cm-1. Covered empty selections raise `EmptyLineSelection`; insufficient support raises `ValueError`. Supplying species-to-molecule mappings and database completeness remains caller responsibility.

### Explicit thermal Voigt lines

`LineSet` represents physical transitions, layer partition ratios and reference/validity conditions for one species and isotopologue. Its appended optional `molecular_mass_kg` field is a positive finite mass in kg per molecule, not molar mass or atomic mass units. `None` preserves the pressure-broadened Lorentz model and `AbsorptionComponent.kind="lorentz_lines"`. A supplied mass enables `scipy.special.voigt_profile` with Gaussian standard deviation `line.centre_cm1 * sqrt(k*T/mass)/c` in cm-1 and the existing air/self Lorentz HWHM in cm-1, producing `kind="voigt_lines"`. The Doppler width uses the unshifted transition frequency; pressure shift still translates the profile centre. Mass is never inferred from a species name, and no provider access or partition approximation is added by this kernel. Optional wings and the separate materialised water-continuum API are described below; the default retains unbounded profiles and adds no continuum or line mixing. Unrepresentable or zero Doppler widths raise `ValueError`. Temperature correction uses `100*h*c/k` from SciPy's SI constants, with the cm-to-m conversion explicit.

The current positional constructors omit the former source identifier; consumers must migrate these calls. Lorentz and Voigt components for the same species/isotopologue are alternative line descriptions, so composition rejects their combination as a duplicate; either also overlaps a complete cross-section contribution for that species. Line-excluding continua and different isotopologues remain supported. Both profiles support explicit atmospheric freshness checks against a separately supplied evaluation atmosphere, including temperature, pressure, columns and broadener pools, and independent output ownership. Consumers that enumerate absorption kinds must accept the new `"voigt_lines"` value before receiving Voigt components; no persisted-schema or external-consumer migration is claimed.

## Materialised water continuum and explicit wings

`LineSet.wing_cutoff_cm1=None` retains unbounded Lorentz/Voigt profiles. A finite positive value truncates about the pressure-shifted centre without renormalising line strength. `subtract_lorentz_pedestal=True` is supported only for H2O with a 25 cm-1 cutoff: remove the Lorentz value at +/-25 cm-1 from the local profile, including when the local profile is Voigt. Default line-only numerical calculations are unchanged; constructors omit the former source identifier. Generated components declare `line_wing_convention`; a generated MT_CKD water continuum rejects composition with unbounded or plain-truncated water lines. The compatibility check is not proof of source-data completeness or of non-water continuum conventions.

`WaterContinuumReference` contains physical water self/foreign ABSCO [cm2/molecule/(cm-1)] without the radiation term, self-temperature exponents and positive reference P/T on an independent writable uniform float64 cm-1 grid. `water_continuum_absorption(atmosphere, grid, reference)` applies temperature scaling and the radiation term on the reference grid before four-point cubic interpolation. A complete stencil is required; no extrapolation or negative-value clipping is used. The result is dimensionless depth shaped `(n_layers, n_points)`. Homogeneous-layer self/foreign density products are `N_water**2/dz` and `N_water*(N_total-N_water)/dz` [molecule2/m5], using the supplied columns and geometric thickness, not an implicit hydrostatic column. Generated continuum freshness includes water and foreign-air columns as well as geometry and T/p; fixed continuum tables use a conservative version of the same density check.

This is the water continuum only, not full MT_CKD, CIA or CO2 line mixing. The caller declares a temperature envelope and materialises correctly normalised coefficients; the library never selects/downloads an edition. Readers and explicit download transport are in [orsia-data](../orsia-data/README.md); provenance is caller-owned. Software tests use synthetic files and analytic density/strength checks; they do not establish authentic-data agreement with the external Fortran reference or a representative WP7 accuracy budget.

The radiation term uses the stable exact `tanh` expression and SI constants rather than AER's small/large-x shortcuts and older radiation constant. A historical bounded MT_CKD 4.3 Fortran comparison found agreement after diagnostic radiation alignment and a criterion failure near zero: negative padding nodes affect the first positive interpolation stencil. That comparison remains outside this candidate; its tested conditions and failure intervals require owner review before citing an external equivalence claim.

## Explicit spectral conversions and unscattered transfer

See [local spectroscopy data](../orsia-data/README.md) for readers and downloads; this numerical package has no file/provider dependency.

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
    np.array([0.0, 1000.0]), np.array([100000.0, 90000.0]),
    np.array([95000.0]), np.array([285.0]), np.array([2.0e29]),
    np.array([1.9e29]), np.array([1.0e28]),
    (Absorber("X", np.array([1.0e25]), "dry_air"),),
)
irradiance = SpectralIrradiance(wavenumber_grid(grid), np.array([10.0, 12.0]))
source = solar_source(irradiance, direction=(30.0, 20.0))
grid = source.irradiance.wavenumbers_cm1
scene = RadiativeScene(atmosphere, SurfaceState(wavenumber_grid(grid), reflectance=np.array([0.2, 0.3])), ObservationGeometry(10.0))
optics = compose_atmospheric_optics(atmosphere, source)
result = transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
assert result.wavenumbers_cm1.flags.writeable is False
assert result.values.shape == (2,) and result.values.flags.writeable
```

The records validate numerical and physical correctness at composition boundaries: positive finite canonical grids, exact grid and layer support, finite values, nonnegative radiance or irradiance and optical-depth bounds. Input artifact identities remain on `LineSet`, `CrossSectionField` and `WaterContinuumReference`, not on evaluated physical values. Typed callers are checked statically by Pyrefly; runtime checks are not added merely to coerce or re-implement Python type annotations. Numerical values are copied to independent writable storage; safely immutable coordinate arrays may be shared. No provider calls, global mutable cache, implicit I/O, randomness or parallel execution policy is part of this package.

## Solar API migration

The reflected-solar public interface in `transfer.solar` is `reflected_radiance`. Numerical order decomposition belongs to `transfer.orders.scattering_orders`. This breaking migration retires `solar.direct_beam`, `solar.single_scattering`, `solar.single_scattering_terms` and the duplicate `solar.scattering_orders` entry.

- Replace `direct_beam(scene, source=source, optics=optics)` with `reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)` to preserve physical unscattered reflection.
- Use `max_order=1, delta_m_moments=None` for first-order reflected sunlight. This changes the historical selected-path physical model: SOS includes common Rayleigh/HG ground coupling and the ground-atmosphere-ground path. The former `surface_coupling` selection becomes the common Lambertian boundary at every atmospheric order. Refine angular and vertical controls for the intended observable.
- Import `orders` from `orsia_radiative.transfer` and use `orders.scattering_orders(..., max_order=N)` for numerical diagnostics, including callers of the former `solar.scattering_orders` wrapper. Its rows aggregate all constituent contributions at each atmospheric order. With Delta-M enabled, rows count residual events and row zero uses transformed extinction.

Historical reports and frozen selected-path fixtures retain their original model and measurement provenance. Current first-order tests use analytic black-ground volume integrals and independent ground-coupled SOS references. Git-based historical SOS benchmarks pair `orders.py` with `paths.py` from the selected revision whenever the order source imports that initializer; a saved orders file with this import requires its matching `--baseline-paths-file`. The performance runner names the new first-order workloads `sos-one` and `sos-one-varying`; archived `surface` measurements describe the retired selected-path calculation. The SciPy comparison runner retains covariance and thermal comparisons; its archived HG azimuth measurements describe the retired path implementation.

## Typed scattering and experimental SOS

`RayleighComponent(grid, optical_depth)` represents conservative scalar Rayleigh scattering: scattering depth equals extinction and the phase is fixed. `AerosolComponent(grid, optical_depth, scattering_depth, asymmetry)` fixes the phase to untruncated HG, with `0 <= scattering_depth <= optical_depth` and `|g| <= 0.8`. All depth/asymmetry arrays have shape `(n_layers, n_points)`, bottom to top. Both types retain immutable grid references and own numerical snapshots, and are composed into atmospheric optics independently of transfer. `rayleigh_component(ScatteringProperties)` returns a `RayleighComponent` after checking untruncated scalar moments `(1, 0, 0.1)`; `aerosol_component(AerosolOptics)` returns an `AerosolComponent` with scattering depth equal to extinction times single-scattering albedo. No public generic constituent or conversion method is needed.

With `delta_m_moments=None`, `max_order=N` counts physical atmospheric scattering events, not surface reflections, and accepts integers N >= 0. Order zero is direct attenuated reflection. At order one the direct beam is scattered in addition to the order-zero surface field. Each subsequent order scatters the complete preceding field, summing all optical constituents before transport. The downward diffuse exits generate a Lambertian bottom boundary at the same atmospheric order; this reflected field contributes to both the observed radiance and the next order's illumination. Consecutive surface encounters without an intervening atmospheric event are not possible in this geometry. For example, sunlight -> ground -> atmosphere -> ground -> observer is included at atmospheric order one. No empirical tail is added. Zero scattering depth reduces every higher-order calculation to order-zero reflection. On a black surface, physical SOS-1 equals the analytic direct-beam volume integral.

The numerical angular field uses `angular_order` Gauss-Legendre points in each open hemisphere, `2 * angular_order` periodic midpoint azimuths and positive weights representing `dOmega / (4*pi)` with sum one. Each input layer is split uniformly into `layer_subdivisions` cells, with homogeneous input optical properties. Direct-beam forcing is integrated analytically inside cells. Diffuse forcing is evaluated at cell centres and transported as a piecewise-constant source in both directions across all cells. This retains inter-layer paths; order two integrates intermediate directions rather than squaring a final-angle phase. The view direction is evaluated separately and has no quadrature weight. `angular_order >= 2` and `layer_subdivisions >= 1` are required; increase both independently to measure convergence. Defaults, near-grazing geometries, optically thick profiles and strongly anisotropic aerosols have no qualified accuracy envelope.

Spectral points are processed serially. If `Q = 4 * angular_order**2` and `C = layer_count * layer_subdivisions`, call-local angular scratch scales as `C * Q` and each dense phase matrix as `Q**2`. Angular scratch retains the preceding field, sources and propagation storage for one spectral point. Every evaluation allocates an independent `O((N+1) * point_count)` order array; the solar boundary sums it and returns the total physical values. Numerical controls and convergence diagnostics remain caller-owned. Refine N, angular sampling and vertical cells separately. Work buffers live for one complete invocation.

Call-local native geometry contains quadrature directions and weights, traversal-cosine mappings, beam angles and seven-factor Rayleigh angular operators; one- and two-layer calculations additionally retain their dense Rayleigh matrix during the call. Aerosol redistribution uses current scattering depths with phase matrices keyed by exact asymmetry for that call's geometry and Delta-M moments. The FIFO cache retains at most 16 MiB of matrix payload. If a single matrix exceeds that budget, one working matrix of that size is used and rebuilt for each requested phase. Cache storage is therefore bounded by `max(16 MiB, 8*Q*(Q+1))`, separately from geometry and per-point scratch. Zero-depth layers skip redistribution. Extinction-dependent propagation coefficients and unit surface responses refresh from current optics at every point. Both `orders.scattering_orders` and `solar.reflected_radiance` skip angular construction for order zero and globally inactive scattering. See [repeated solar evaluations](#repeated-solar-evaluations) for ownership and migration details.

```python

from orsia_radiative import wavenumber_grid
from orsia_radiative import AerosolComponent, RayleighComponent, compose_atmospheric_optics
from orsia_radiative.transfer import orders

rayleigh = RayleighComponent(wavenumber_grid(grid), np.array([[0.01, 0.02]]))
aerosol = AerosolComponent(
    wavenumber_grid(grid), np.array([[0.04, 0.05]]),
    np.array([[0.02, 0.03]]), np.array([[0.2, 0.4]])
)
optics = compose_atmospheric_optics(atmosphere, source, rayleigh=rayleigh, aerosol=aerosol)
first_order = transfer.solar.reflected_radiance(
    scene, source=source, optics=optics, max_order=1, delta_m_moments=None
)
sos = transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=5, angular_order=8, layer_subdivisions=8)
order_values = orders.scattering_orders(scene, source=source, optics=optics, max_order=5, angular_order=8, layer_subdivisions=8)
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
    scene, source=source, optics=optics,
    max_order=5, angular_order=8, layer_subdivisions=8,
    delta_m_moments=8,
)
residual_orders = orders.scattering_orders(
    scene, source=source, optics=optics,
    max_order=5, angular_order=8, layer_subdivisions=8, delta_m_moments=8,
)
np.testing.assert_array_equal(residual_orders.sum(axis=0), scaled.values)
```

`test_sos_delta_m.py` checks complete-radiance equality between omitted and explicitly disabled Delta-M, residual Legendre moments, discrete phase normalization at the declared quadrature bound, unchanged non-forward phases, absorption preservation, transformed propagation, direct-order buffer ownership, absent/absorbing aerosols, zero illumination, invalid K, signed-phase rejection, truncation-aware cache keys and deterministic call-local concurrent results. The scalar adaptive-ray oracle in `test_sos_orders.py` also checks all retained order rows through N=5 on small stratified mixed scenes with black, partial and fully reflecting Lambertian surfaces, using independently summed Legendre phases and transformed depths. It shares angular nodes and cell-centre collocation with SOS: these checks validate algebra and discrete assembly, not continuum convergence, physical energy balance of the complete collocated solver, external-solver agreement, intensity corrections or instrument/retrieval performance. Refine moments, angular/vertical sampling and residual orders independently for an intended observable.

## Example: absorption composition upstream

```python

from orsia_radiative import wavenumber_grid
from orsia_radiative import CrossSectionField, compose_absorption, table_absorption

table = CrossSectionField(
    "X", "1", "cross_section", wavenumber_grid(grid),
    atmosphere.temperature_k, atmosphere.pressure_pa,
    np.array([[1e-27, 2e-27]]),
)
gas = table_absorption(atmosphere, grid, table)
evaluation_atmosphere = atmosphere
# Retain evaluation state explicitly when freshness checks are needed.
absorption = compose_absorption(atmosphere, source, (gas,), reference_atmospheres=(evaluation_atmosphere,))
optics = compose_atmospheric_optics(
    atmosphere, source, absorption=absorption, rayleigh=rayleigh, aerosol=aerosol,
)
result = transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
assert optics.absorption is absorption
assert gas.wavenumbers_cm1 is absorption.wavenumbers_cm1 is optics.wavenumbers_cm1 is grid
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

This angular convention itself is unchanged by fixed-N SOS. No automatic conversion or alternate convention flag is added. Existing inputs already expressed in Orsia's convention need no migration. A caller that passed outward Sun-to-view separations must correct its inputs explicitly and expect changed radiances; previous shape or regression checks did not establish that caller's geometry. External adapters must still demonstrate their frame, direction and azimuth mapping, and no external solver agreement is claimed.

`test_angular_geometry.py` derives API angles from independently supplied outward Cartesian vectors and tests their dot products after reversing the solar vector. It checks cardinal, oblique, near-grazing and zenith geometries, signed azimuth wraparound and common rotations about the normal. Public analytic and black-surface SOS-1 volume terms are checked against independent attenuated-ray integrals with Rayleigh and signed HG constituents. Separate SOS checks inspect the actual beam/intermediate phase cosines on reflective scenes and compare the aggregate order-two volume term against the sum of every ordered pair in a one-cell discrete reference built from physical vectors, independent phase expressions and adaptive ray integrals. The discrete reference deliberately shares the angular nodes and cell-centre approximation: it tests geometry and assembly, not quadrature convergence or continuous-solver accuracy. These bounded local checks replace the former undefined azimuth interpretation, not the outstanding external scientific qualification.

## Evidence and limits

SOS uses deterministic angular quadrature and vertical cell subdivision. These are numerical controls, not physical response weights or processing weights; convergence is domain-specific and must be assessed independently. Work buffers are call-local, spectral points are processed serially and no global cache or hidden optical-depth basis is used.

The synthetic solar and aerosol fixtures under `tests/fixtures/` are project-authored numerical regression cases, not measured observations, independent solver parity or rights-cleared external data. The tests cover analytic Beer-Lambert limits, source-resolved single events, source composition, bounded SOS order reductions, per-order SOS provenance, independent continuous-ray SOS checks, vertical and angular refinement trends, line evaluation, bounded cross-section interpolation, explicit spectral Jacobians and rejected contract violations. These checks exercise small synthetic scenes and do not qualify a broad atmospheric domain, an accuracy envelope for the SOS controls, instrument-space response, derivative convergence, retrieval performance or live provider integration.

The [workspace coverage gate](../../README.md#test-coverage-policy) requires at least 90% combined statement-and-branch execution coverage for this library independently of `orsia-uncertainty`. All radiative source modules, including unimported ones, contribute to the denominator. This engineering metric is separate from the scientific limits below.

The continuous-ray checks in `test_sos_numerics.py` use adaptive SciPy integrals for a homogeneous isotropic aerosol slab and disjoint Rayleigh/isotropic-aerosol layers. They compare the first two isotropic orders, including repeated-ground terms, and isolate the sum of ordered cross-constituent volume paths at fixed extinction against independently formulated integrals at a case-specific `2e-3` relative / `2e-8` absolute tolerance. `test_sos_orders.py` compares all orders through N=5 on a small stratified Rayleigh/HG grid, with reflectances 0, 0.4 and 1, against independent scalar ray integrals and phase expressions. That oracle shares angular nodes and cell-centre collocation: it verifies assembly, not continuum accuracy. Further checks cover scattering-depth scaling through N=6 at fixed extinction, prefix consistency, zero illumination/scattering, N=0, source-amplitude scaling, relayering, buffer/operator ownership and deterministic concurrent calls. Vertical and angular-refinement trends remain separately tested, including g = -0.8 and g = 0.8. These bounded checks do not qualify optically thick/high-order convergence, broad native thread safety, an external solver, instrument convolution or retrieval impact.

The historical SOS-2 discretization characterization concerns the former one-reflection closure. `devbox run -- just test-sos` runs focused fixed-N, composition, independent-ray and refinement regressions for the current model. Its internal refinement checks bound software behavior; application-specific accuracy and retrieval qualification require independent evidence.

Historical development-only SASKTRAN2 comparisons used scalar discrete ordinates on shared homogeneous plane-parallel optical inputs. The comparison guides and bounded run reports remain outside this candidate. Application-specific accuracy, instrument response and retrieval impact require separately designed external comparisons; the external backend remains an optional development dependency group.

The previously referenced parent SOS design sketch (`../../../orsia-sos-integration.md`), library design (`../../../orsia-design.md`) and implementation plan (`../../../orsia-implementation-plan.md`) are unavailable in this checkout. Their contents cannot be rechecked here. This README and the source/tests describe implemented experimental behaviour, not promotion of the parent proposal to a qualified solver.

Standalone radiative/SciPy, propagation-dtype and complete-solar-call timing runners are retired. Numerical SciPy-method, float32 propagation and complete-call regressions remain in the library tests. Historical software timings are not scientific qualification or an ESA, instrument or campaign performance commitment.

Historical propagation and redistribution optimization runners have been retired. Their former one-reflection SOS measurements do not qualify the current fixed-N recurrence. Formal transport, redistribution, independent-ray and refinement regressions remain in the library test suite.
