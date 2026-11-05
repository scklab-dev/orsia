# Physical state, spectral support and ownership

Use this reference to construct materialised physical inputs, interpret spectral densities and manage independent numerical arrays. [Atmospheric columns](columns.md), [spectroscopy](spectroscopy.md) and [transfer](transfer.md) build on these contracts.

## Source responsibilities

| Source | Responsibility | Boundary |
| --- | --- | --- |
| `atmosphere/models.py`, `atmosphere/layers.py`, `atmosphere/hydrostatic.py`, `profiles.py` | Atmospheric column records, layer coordinates/remapping, hydrostatic construction and profile perturbations | Explicit in-memory inputs; layers are bottom to top. |
| `surface.py`, `geometry.py`, `scene.py` | Surface properties, outgoing view and assembled physical state | Illumination and transfer approximation are not scene fields. |
| `spectral/` | Point grids, optional coverage domains, coordinate/density conversion and interpolation | Grid spacing is not instrumental resolution. |
| `solar/`, `irradiance.py`, `radiance.py`, `radiometry.py` | Solar spectrum/source, physical spectral values and Planck radiometry | Solar irradiance and thermal emission are not interchangeable sources. |
| `spectroscopy/` | Line/table/continuum evaluation and species-resolved contributions | Pure numerical evaluation, not native-file reading or downloading. |
| `optics/` | Absorption composition, phase laws and Rayleigh/aerosol optical roles | Physical optical values remain independent of the selected transfer method. |
| `transfer/solar.py`, `transfer/orders.py` | Solar radiance assembly and experimental fixed-order SOS with analytic direct initialization | Refine the retained order count for the observable and scene. |
| `transfer/thermal.py`, `transfer/formal.py` | Non-scattering LTE transfer and source-independent transport kernels | Thermal transfer rejects nonzero atmospheric scattering. |

`transfer/native/` contains the packaged serial C++ kernels and Python binding stub. Python owns public validation and physical assembly; the [native execution reference](native-execution.md) specifies precision dispatch and buffer behavior. Data readers depend on radiative public contracts; radiative never imports `orsia-data`.

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
- `SpectralIrradiance(grid, values)` is beam-normal irradiance in W m-2 per (cm-1), defined in `irradiance.py`. `SolarSource` adds the Sun's local zenith and the incoming-propagation-to-view relative azimuth without choosing a transfer method. `Radiance(grid, values, geometry)` in `radiance.py` is the physical directional spectral value and its support; it retains no transfer-method identity, input provenance, decomposition or diagnostics. Calculations that need additive terms expose them through separate term functions.

### Spectral domain and point grid

`SpectralDomain(coordinate, intervals)` represents declared physical coverage as positive, finite, closed intervals in either `"wavenumber_cm1"` or `"wavelength_nm"`. Intervals are ordered, nonoverlapping and separated; singleton intervals are allowed. `SpectralGrid(coordinate, points, domain=None)` represents the actual one-dimensional, positive, strictly increasing float64 point samples. When a domain is supplied, its coordinate must agree and every point must belong to it. `domain=None` means coverage is unknown: neither a continuous min/max envelope nor coverage across gaps is inferred from points.

`wavenumber_grid(points, domain=...)` explicitly adapts a raw array to the canonical radiative grid; passing an existing wavenumber grid retains that object. Grids snapshot unsafe coordinate buffers and may share safely immutable bytes-backed storage. Retained point arrays and their shape/dtype metadata must not be mutated. `grid.same_points(other)` tests exact coordinate/unit and point equality, without tolerance or interpolation. Full grid equality also includes the declared domain; evaluated spectra with different physical coverage can still be compatible at identical points.

Spectral irradiance, radiance, surface response, absorption components/totals, prepared cross sections, native cross-section tables and scattering/atmospheric optics store `grid` as their only spectral authority. Solar composition and transfer retain the irradiance's grid; explicit-grid composition and thermal transfer retain the supplied optics grid. The numerical `wavenumbers_cm1` property is only an alias of `grid.points`, not a second field or sampling choice. `compose_absorption_on_grid(atmosphere, grid, components)` and `atmospheric_optics(atmosphere, grid, ...)` require a `SpectralGrid`. Spectroscopy evaluation still accepts an explicit raw cm-1 array or a grid; passing a grid preserves its declared support in the returned record.

Use `dataclasses.replace(record, grid=new_grid)` to change sampling. Serialized dataclass dictionaries contain `grid`; normalized cross-section NPZ files adapt arrays to this contract at the reader boundary.

`convert_spectral_grid(grid)` explicitly converts nm to cm-1 or vice versa, reversing points and the order/endpoints of declared intervals. It does not transform values, interpolate or integrate. Convert spectral densities separately with `density_per_nm_to_per_cm1` or `density_per_cm1_to_per_nm`, including the Jacobian and reversed value order; dimensionless point values only reverse order. Fixed-table evaluation additionally rejects targets outside declared native support and interpolation stencils crossing a declared gap; exact native samples remain available.

### Convert coordinates and densities together

Run this standalone example in the [development environment](../../../README.md#development-environment). Increasing wavelength points become increasing wavenumbers in reversed order; spectral density also needs the Jacobian.

```python
import numpy as np
from orsia_radiative import density_per_nm_to_per_cm1, wavelength_to_wavenumber

wavelengths_nm = np.array([500.0, 1000.0])
per_nm = np.array([1.0, 2.0])
wavenumbers_cm1 = wavelength_to_wavenumber(wavelengths_nm)
per_cm1 = density_per_nm_to_per_cm1(wavelengths_nm, per_nm)
np.testing.assert_allclose(wavenumbers_cm1, [10000.0, 20000.0])
np.testing.assert_allclose(per_cm1, [0.2, 0.025])
print(wavenumbers_cm1.tolist(), per_cm1.tolist())
```

Expected output: `[10000.0, 20000.0] [0.2, 0.025]`. Input density is per nm; output density is per cm-1. Coordinate conversion alone does not convert radiance or irradiance values. These are point densities, with bin integration and instrumental response handled separately.

### Solar point-density adaptation

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

Construct numerical inputs explicitly with `np.array(..., dtype=np.float64)` or `np.asarray(..., dtype=np.float64)`. Use NumPy comparisons for whole arrays. Identifiers, record collections, validity-range pairs and the `(zenith_deg, relative_azimuth_deg)` pair use tuples. `AerosolComponent` carries a same-support HG asymmetry array; `RayleighComponent` has a fixed Rayleigh phase law. JSON writers convert arrays to lists at serialization boundaries.

## Record comparison

Array-containing physical records use identity comparison and are unhashable. Compare the relevant fields explicitly with `np.array_equal`, `np.testing.assert_array_equal` or scientifically justified approximate tolerances. Compare spectral support and geometry separately. `SpectralGrid` defines value equality including its domain, and `same_points` compares numerical sampling. `orsia_radiative.spectroscopy.components.values_equal` compares selected physical dependencies exactly across arrays, nested tuples and scalars; NaNs compare unequal.

## Physical values, provenance and explicit freshness

Physical records contain values and their semantic support. Applications keep artifact identity and evaluation references alongside a calculation. Species, kind, isotopologue, line-wing convention, truncation and phase parameters describe the physical values. `AerosolOptics` supplies an extinction/albedo parameterisation with a public converter to aerosol optical depths.

Composition and transfer still reject incompatible grids/layer shapes, invalid numerical values, overflow and semantic spectroscopy overlap. A supplied grid and layer shape cannot establish physical freshness. Before reusing a component, call `validate_absorption_freshness(atmosphere, component, reference_atmosphere=evaluation_atmosphere)` or pass `reference_atmospheres=(...)` in component order to either absorption composer; length mismatch raises `ValueError`. These compare exact physical dependencies, including line broadening and continuum density. For composed optics, `validate_atmospheric_optics(scene, source, optics, reference_atmosphere=evaluation_atmosphere)` and `validate_scene_optics(scene, optics, reference_atmosphere=evaluation_atmosphere)` conservatively reject a different atmosphere object. Without these explicit references, the caller owns physical consistency and must recompute affected optics; the library cannot infer their generating state.
