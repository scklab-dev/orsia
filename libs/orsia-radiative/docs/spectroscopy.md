# Spectroscopy and atmospheric optics

This reference specifies line/table materialisation, absorption composition, water-continuum conventions and explicit freshness. Evaluate spectroscopy on the chosen grid before selecting a [transfer calculation](transfer.md).

## Reusable spectroscopy references

`LineSpectroscopy(species, databases, partition_table, molecular_mass_kg)` represents one molecule/local isotopologue's transitions, sampled Q(T) and mass. Each `SpectralLineTable` supplies a separate complete line-centre interval in cm-1. Intervals must be ordered and disjoint; gaps remain unsupported. Partition arrays are independent read-only float64 snapshots. Intensities retain natural-abundance weighting relative to whole-species columns at 296 K, with line widths and shifts referenced to 101325 Pa. Provider identities and hashes belong to [Data collections](../../orsia-data/docs/readers-and-acquisition.md#spectroscopy-collections).

`evaluate_line_absorption(atmosphere, grid, reference, profile=..., pressure_range_pa=..., wing_cutoff_cm1=..., subtract_lorentz_pedestal=...)` returns the existing `AbsorptionComponent`. All physical choices are explicit. `profile="voigt"` uses the reference mass; `"lorentz"` selects pressure broadening alone. One declared complete interval must cover the grid, cutoff and maximum pressure-shift margin. Linear Q(T) interpolation and the detailed kernels remain unchanged. A covered empty selection raises `EmptyLineSelection`; insufficient support raises `ValueError`. Empty line intervals remain valid stored references.

`CrossSectionMeasurements(species, isotopologue, spectra)` represents temperature-ordered laboratory spectra. Each `CrossSectionSpectrum` retains its own increasing vacuum-wavelength axis [nm], cross sections [m2/molecule], temperature [K], known pressure [Pa] or `None`, exact retained support intervals and optional resolution description. Arrays are independent read-only float64 snapshots. One spectrum per temperature is supported; pressure interpolation is outside this contract.

`measurement_absorption(atmosphere, grid, reference, spectral_interpolation="linear_native_wavelength", temperature_interpolation="linear", temperature_outside="error" | "hold_nearest", pressure_model="independent")` returns an `AbsorptionComponent` and `CrossSectionTemperatures`. The diagnostics retain requested/effective temperatures [K] and a boolean held-layer mask in bottom-to-top order, using independent read-only arrays. The pressure-independent model is an explicit approximation for known and unknown laboratory pressures. Interpolation first uses each contributing native wavelength axis, then laboratory temperature. Exact temperature knots require only that spectrum's support. Spectral extrapolation and interpolation through declared gaps raise `ValueError`; cross-section amplitudes receive no spectral-density Jacobian. `interpolate_cross_section_measurements` exposes the same interpolation without applying molecular columns.

The detailed `line_set`, `line_absorption`, `LayerCrossSections`, `resample_layer_cross_sections`, `table_absorption` and `evaluate_table_absorption` APIs expose the contracts specified below. Generated component depths remain independent writable float64 arrays. Composition retains duplicate, undeclared-species and line/complete-cross-section overlap checks.

## Cross-section optical-depth arrays

`orsia_radiative.optics.cross_sections` provides two pure array operations, also exported from `orsia_radiative`. Cross sections are fixed finite nonnegative native float64 arrays of shape `(layers, points)` in m2/molecule. Columns are finite nonnegative native float64 arrays of shape `(layers,)` in molecules/m2. Layers and points must be nonempty; the caller maintains their order and physical freshness.

- `cross_section_optical_depth(cross_sections_m2, columns_molecules_m2)` returns dimensionless `(layers, points)` depths, `tau[l,p] = sigma[l,p] * N[l]`. Its native direct path allocates the depth output without tangent storage or derivative work.
- `cross_section_optical_depth_tangent(cross_sections_m2, columns_molecules_m2, columns_tangent_molecules_m2)` returns `(tau, dtau)` in one native call. Signed finite native float64 column directions have shape `(layers, k)`. The tangent has shape `(layers, points, k)` and satisfies `dtau[l,p,d] = sigma[l,p] * dN[l,d]`.

Direction axes are last and `k` may be zero. Derivatives are with respect to molecular columns at fixed cross sections; temperature, pressure and cross-section derivatives are outside this contract.

Outputs are owned writable C-contiguous float64 arrays with no input aliasing. Inputs may be strided, unaligned or read-only and must remain stable during calls, including concurrent calls. Invalid dtype, shape, finite values, physical bounds or arithmetic overflow raises `ValueError`. Direct and tangent output-size overflow raises `OverflowError`. Native adaptation can copy inputs for compatible layout; inputs are preserved.

`optics.absorption` owns physical composition; numerical array operations live in `optics.cross_sections`. Import the two array operations from `optics.cross_sections` or the package root. The former `fixed_`-prefixed derivative names and their compatibility exports have been removed. Keyword parameters and numerical contracts are unchanged.

Table and laboratory-measurement evaluation apply the direct primitive after their existing species, grid, interpolation and layer-condition checks, then construct `AbsorptionComponent`. These adapters convert stored numerical arrays to float64 at the calculation boundary. Fixed tables own immutable snapshots. Constructors reject negative or nonfinite cross sections, including when the eventual gas column is zero. The fused line-spectroscopy calculation remains unchanged.

## Prescribed scattering optical-depth laws

`rayleigh_from_air_columns(grid, dry_air_columns_molecules_m2, reference_cross_section_m2, reference_wavelength_um)` returns a `RayleighComponent` from bottom-to-top dry-air columns of shape `(layers,)` in molecules/m2. It evaluates `tau = column * sigma_ref * (lambda_ref / lambda)**4`, with `sigma_ref` in m2/molecule and `lambda_ref` in um. The helper requires a wavenumber-cm-1 `SpectralGrid` and preserves its identity, domain and point order. This approximate dry-air law uses the caller's reference cross section and omits refractive-index and depolarization corrections. The existing conservative scalar Rayleigh phase convention applies.

`angstrom_optical_depth(grid, reference_optical_depth, reference_wavelength_nm, angstrom_exponent)` evaluates dimensionless spectral column extinction as `tau_ref * (lambda_nm / lambda_ref_nm)**(-alpha)`. It supports wavelength-nm and wavenumber-cm-1 grids and returns shape `(points,)` in the supplied point order, without a density Jacobian. Any finite exponent is supported, including zero and negative values. Vertical allocation, single-scattering albedo and phase remain separate caller inputs; `AerosolComponent` can consume the resulting allocated extinction and scattering depths.

Both functions are defined in `orsia_radiative.optics.scattering` and exported from `orsia_radiative`. All arguments accept keyword calls. Columns, reference cross section and reference optical depth must be finite and nonnegative; reference wavelengths must be finite and positive. Invalid values, empty or non-vector columns, and nonfinite evaluation results raise `ValueError`. Result arrays are independent writable float64 storage; Rayleigh depth has shape `(layers, points)`. Inputs remain unchanged and must remain stable during a call. Existing `RayleighComponent`, `AerosolComponent`, `rayleigh_component` and `aerosol_component` constructors and adapters remain available.

## Atmospheric optics and freshness

`henyey_greenstein_phase(asymmetry, cosine)` and `rayleigh_phase(cosine)` in `optics/phases.py` own the phase laws used by SOS beam-source construction and diffuse redistribution; both are exported at the package root. They evaluate dimensionless phases with a 4π mean of one, using float64 NumPy broadcasting without Python element loops. HG accepts finite `|g| <= 0.8`; both reject nonfinite cosines or values outside `[-1, 1]` rather than silently clipping physical inputs. Scalar inputs return Python floats; if any input is an array, the result is a fresh writable array, including for zero-dimensional input, without mutating or retaining inputs. Incompatible HG shapes raise NumPy's broadcasting `ValueError`. `orders.phase_function` dispatches Rayleigh/HG evaluation with `g=None` and geometric cosine clipping, applying the optical evaluator's HG bounds. Transfer geometry bounds dot-product roundoff before optical evaluation. This factorisation does not change quadrature, path definitions or scattering orders.

`compose_atmospheric_optics(atmosphere, source, *, absorption=None, rayleigh=None, aerosol=None)` derives spectral support from `source.irradiance.wavenumbers_cm1` and validates the supplied layer shape before constructing the physical `AtmosphericOptics`. It takes neither a scene nor a separately selected grid. Numerical spectroscopy kernels still receive only the source's numerical grid and the atmosphere, not illumination properties. Atmospheric optics retain that original immutable coordinate array by reference and share optical records without copying; the retained `wavenumbers_cm1` fields identify evaluated support rather than independent sampling choices. Freshness is checked only when the caller supplies a separately retained `reference_atmosphere` to `validate_atmospheric_optics`. Common extinction and angular work buffers belong to one transfer invocation; atmospheric optics retain their physical values independently.

Optical composition uses the atmosphere and the source spectral support. Surface and view support are checked when the selected transfer method receives the scene.

Constituent selection and reuse belong upstream of optical composition. Recompute affected spectroscopy contributions, reuse unaffected generated components, call `compose_absorption` for the current atmosphere, then compose atmospheric optics from the current total and explicit scattering roles. Recomposition can add or remove a role; `None` means absence, not retention. Selecting retrieved gases alone is insufficient to decide which contributions are unchanged. No inverse solver, retrieval-coordinate mapping or generic dependency framework is introduced.

`line_absorption`, `table_absorption` and `water_continuum_absorption` return physical components without an attached atmosphere reference. Composition can compare dependencies against separately retained evaluation atmospheres through `reference_atmospheres`; generated kernels validate their own evaluation inputs. Materialised tables and line sets likewise retain only physical fields; any source record stays outside radiative inputs and is not a freshness certificate. Rayleigh/aerosol records likewise contain no generating model; reuse or replacement remains explicit.

### Fixed layer cross sections

`LayerCrossSections` owns one species/component's coefficients, `SpectralGrid`, exact evaluation temperature/pressure and increasing `height_edges_m`. Coefficients have `(layer,spectral)` axes in m2/molecule and preserve their supplied numerical dtype. All retained numerical arrays are immutable snapshots. A one-sample table is valid. Source and resampled values share this concept; retain the original source table explicitly when another target grid is needed.

1. **Objective:** select supported source-coordinate samples for a requested grid. **Inputs:** immutable source table, target `SpectralGrid` in nm or cm-1, and the source's optional coverage intervals. **Outputs:** source-coordinate queries `q[k]` and an output-order reversal flag. **Processing:** same-coordinate requests use the target points; reciprocal requests use `q = reverse(10^7 / target.points)`. A target exactly equal to a stored reciprocal source knot maps to that original knot, avoiding a double-reciprocal rounding shift. Coverage is checked in the source coordinate.
2. **Objective:** construct the requested fixed coefficients. **Inputs:** source samples `s`, coefficients `sigma[layer,k]` and supported queries `q`. **Outputs:** `sampled_sigma` and a new `LayerCrossSections` on the requested grid. **Processing:** each exact knot copies its coefficient; otherwise bracket `q` by `s[i] < q < s[j]` and use `sampled_sigma = ((s[j]-q)*sigma[:,i] + (q-s[i])*sigma[:,j])/(s[j]-s[i])`. The implementation evaluates the nonnegative weighted terms separately to preserve dynamic range. Restore requested order for reciprocal coordinates. Extrapolation and stencils crossing declared gaps raise `ValueError`; a singleton source supports its exact knot only. Cross-section point values receive no density Jacobian. Known source coverage is retained when the requested grid omits a domain; an explicit target domain may narrow that coverage but cannot widen it. Unknown source coverage still cannot identify undeclared gaps. `resample_layer_cross_sections(grid, table)` returns immutable float64 coefficients with independent snapshots of the original conditions/interfaces.
3. **Objective:** calculate absorption for current molecular amounts. **Inputs:** the fixed table and a layer atmosphere with matching species. **Outputs:** a fresh `AbsorptionComponent` with `tau[layer,k] = sigma[layer,k] * N[layer]`. **Processing:** `evaluate_table_absorption` requires exact equality of layer interfaces, temperature and pressure, and a cm-1 grid for the current optical-component contract. It then invokes the shared direct optical-depth primitive and checks finite results. `table_absorption(atmosphere, grid, table)` combines resampling and application.

The fixed-table calculation retains its supplied spectroscopy. Abundance changes in a line model can change self/foreign broadening and require reevaluating that model. The tabulated path supplies no temperature/pressure interpolation, line-profile approximation, persistent cache or measured retrieval speedup.


### Pure line-table materialisation

`SpectralLineTable(lines, molecule_id, isotopologue_id, coverage_cm1)` represents one molecule/local isotope's transitions and declared centre-coverage completeness [cm-1]. Empty lines are valid; bounds are not inferred from centres. `PartitionFunctionTable(molecule_id, isotopologue_id, temperatures_k, partition_sums)` represents dimensionless Q at positive increasing knots [K], with independent writable float64 arrays of shape `(n_knots,)`, at least two knots. `evaluate_partition_function(table, temperatures_k)` returns fresh writable Q(T) by explicit linear interpolation without extrapolation, not HAPI/TIPS interpolation.

`line_set(database, partition_table, atmosphere, grid, species=..., molecular_mass_kg=..., pressure_range_pa=..., wing_cutoff_cm1=..., subtract_lorentz_pedestal=False)` matches isotope identities, requires a declared atmospheric species and checks centre coverage beyond cutoff plus maximum absolute pressure shift across the declared pressure support. It selects lines, computes Q(296 K)/Q(T_layer) and returns a `LineSet` without file access, identity generation or caches. Mass is kg/molecule, pressure Pa, grid/cutoff cm-1. Covered empty selections raise `EmptyLineSelection`; insufficient support raises `ValueError`. Supplying species-to-molecule mappings and database completeness remains caller responsibility.

### Explicit thermal Voigt lines

`LineSet` represents physical transitions, layer partition ratios and reference/validity conditions for one species and isotopologue. Its appended optional `molecular_mass_kg` field is a positive finite mass in kg per molecule, not molar mass or atomic mass units. `None` preserves the pressure-broadened Lorentz model and `AbsorptionComponent.kind="lorentz_lines"`. A supplied mass enables `scipy.special.voigt_profile` with Gaussian standard deviation `line.centre_cm1 * sqrt(k*T/mass)/c` in cm-1 and the existing air/self Lorentz HWHM in cm-1, producing `kind="voigt_lines"`. The Doppler width uses the unshifted transition frequency; pressure shift still translates the profile centre. Mass is never inferred from a species name, and no provider access or partition approximation is added by this kernel. Optional wings and the separate materialised water-continuum API are described below; the default retains unbounded profiles and adds no continuum or line mixing. Unrepresentable or zero Doppler widths raise `ValueError`. Temperature correction uses `100*h*c/k` from SciPy's SI constants, with the cm-to-m conversion explicit.

Lorentz and Voigt components for the same species/isotopologue are alternative line descriptions, so composition rejects their combination as a duplicate; either also overlaps a complete cross-section contribution for that species. Line-excluding continua and different isotopologues remain supported. Both profiles support explicit atmospheric freshness checks against a separately supplied evaluation atmosphere, including temperature, pressure, columns and broadener pools, and independent output ownership. The absorption kind is `"voigt_lines"` for a Voigt component.

## Materialised water continuum and explicit wings

`WaterContinuumCoefficients` stores immutable separate self/foreign arrays `(layer,spectral)` in m5/molecule2, exact interfaces, explicit coefficient evaluation T/p, the shared spectral grid and a matching line-wing convention. `water_continuum_coefficients(grid, reference, height_edges_m=..., temperature_k=..., pressure_pa=...)` prepares these values for either nm or cm-1 sampling. The pressure array records the selected layer context; the density-pair kernel's temperature dependence is evaluated directly. No representative layer conditions are selected internally.

**Objective:** prepare separate density-pair coefficients. **Inputs:** native MT_CKD reference knots, ABSCO values, temperature exponents, reference pressure/temperature, requested samples and explicit layer conditions. **Outputs:** self and foreign coefficients in m5/molecule2. **Processing:** on the native cm-1 grid, compute the exact-tanh radiation term `r(nu,T)=nu*tanh(100*h*c*nu/(2*k_B*T))`. Multiply self ABSCO by `(T_ref/T)^exponent * r` and foreign ABSCO by `r`. Apply the existing four-point cubic interpolation, then multiply by `1e-10/n_ref_cm3`, where `n_ref_cm3=p_ref/(k_B*T_ref*1e6)`. Wavelength output reverses the point coefficients without a density Jacobian. Missing complete stencils, unsupported temperatures and negative/nonfinite interpolated coefficients fail. The homogeneous absorption evaluator reuses the same coefficient calculation and retains its original arithmetic ordering.

A supplied H2O `LayerCrossSections` can declare `includes_water_continuum` and `line_wing_convention`. `None` means inclusion is unknown. Combining a fixed water table with separate continuum requires `includes_water_continuum=False`, matching local-line convention and, for column contraction, equal coefficient T/p. A true or unknown inclusion is rejected; a continuum-excluding-lines table plus separate water coefficients is also rejected as a duplicate. These declarations are checked numerical contracts, not source-provenance certificates. An evaluated component carries the declaration, and known continuum inclusion adds water/foreign-density dependencies to explicit freshness checks.

`LineSet.wing_cutoff_cm1=None` retains unbounded Lorentz/Voigt profiles. A finite positive value truncates about the pressure-shifted centre without renormalising line strength. `subtract_lorentz_pedestal=True` is supported only for H2O with a 25 cm-1 cutoff: remove the Lorentz value at +/-25 cm-1 from the local profile, including when the local profile is Voigt. Generated components declare `line_wing_convention`; a generated MT_CKD water continuum rejects composition with unbounded or plain-truncated water lines. The compatibility check is not proof of source-data completeness or of non-water continuum conventions.

`WaterContinuumReference` contains physical water self/foreign ABSCO [cm2/molecule/(cm-1)] without the radiation term, self-temperature exponents and positive reference P/T on an independent writable uniform float64 cm-1 grid. `water_continuum_absorption(atmosphere, grid, reference)` applies temperature scaling and the radiation term on the reference grid before four-point cubic interpolation. A complete stencil is required; no extrapolation or negative-value clipping is used. The result is dimensionless depth shaped `(n_layers, n_points)`. Homogeneous-layer self/foreign density products are `N_water**2/dz` and `N_water*(N_total-N_water)/dz` [molecule2/m5], using the supplied columns and geometric thickness, not an implicit hydrostatic column. Generated continuum freshness includes water and foreign-air columns as well as geometry and T/p; fixed continuum tables use a conservative version of the same density check.

This is the water continuum only, not full MT_CKD, CIA or CO2 line mixing. The caller declares a temperature envelope and materialises correctly normalised coefficients; the library never selects/downloads an edition. Readers and explicit download transport are in [orsia-data](../../orsia-data/README.md); provenance is caller-owned. Software tests use synthetic files and analytic density/strength checks; they do not establish authentic-data agreement with the external Fortran reference or a representative WP7 accuracy budget.

The radiation term uses the stable exact `tanh` expression and SI constants. AER's Fortran 4.3 calculation uses small/large-x shortcuts and an older radiation constant. The retired bounded comparison failed its unmodified all-points criterion near zero wavenumber: negative padding nodes affect the first positive interpolation stencil. Diagnostic radiation alignment produced agreement on the tested synthetic layers, while general numerical equivalence remains unaccepted. The library applies no automatic coefficient correction. [Current kernel arithmetic](../src/orsia_radiative/spectroscopy/continuum.py) and [padding/scaling regressions](../tests/test_water_continuum.py) preserve the exact-`tanh`, scaling-before-interpolation contract; those tests do not regenerate the external comparison.
## Example: absorption composition upstream

This synthetic fixed table uses exact layer T/p and a declared gas column. Keep the evaluated atmosphere separately when requesting freshness checks.

```python
import numpy as np
from orsia_radiative import Absorber, LayerAtmosphere, LayerCrossSections, compose_absorption_on_grid, table_absorption, wavenumber_grid

grid = wavenumber_grid(np.array([1000.0, 1001.0]))
atmosphere = LayerAtmosphere(
    np.array([0.0, 1000.0]),
    np.array([100000.0, 90000.0]),
    np.array([95000.0]),
    np.array([285.0]),
    np.array([2e29]),
    np.array([1.9e29]),
    np.array([1e28]),
    (Absorber("X", np.array([1e25]), "dry_air"),),
)
table = LayerCrossSections(
    "X",
    "1",
    "cross_section",
    grid,
    atmosphere.temperature_k,
    atmosphere.pressure_pa,
    np.array([[1e-27, 2e-27]]),  # m2/molecule
    height_edges_m=atmosphere.height_boundaries_m,
)
gas = table_absorption(atmosphere, grid, table)
total = compose_absorption_on_grid(
    atmosphere,
    grid,
    (gas,),
    reference_atmospheres=(atmosphere,),
)
np.testing.assert_allclose(total.optical_depth, [[0.01, 0.02]])
assert total.grid is grid
```
