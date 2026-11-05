# Spectroscopy and atmospheric optics

This reference specifies line/table materialisation, absorption composition, water-continuum conventions and explicit freshness. Evaluate spectroscopy on the chosen grid before selecting a [transfer calculation](transfer.md).

## Reusable spectroscopy references

`LineSpectroscopy(species, databases, partition_table, molecular_mass_kg)` represents one molecule/local isotopologue's transitions, sampled Q(T) and mass. Each `LineDatabase` supplies a separate complete line-centre interval in cm-1. Intervals must be ordered and disjoint; gaps remain unsupported. Partition arrays are independent read-only float64 snapshots. Intensities retain natural-abundance weighting relative to whole-species columns at 296 K, with line widths and shifts referenced to 101325 Pa. Provider identities and hashes belong to [Data collections](../../orsia-data/docs/readers-and-acquisition.md#spectroscopy-collections).

`evaluate_line_absorption(atmosphere, grid, reference, profile=..., pressure_range_pa=..., wing_cutoff_cm1=..., subtract_lorentz_pedestal=...)` returns the existing `AbsorptionComponent`. All physical choices are explicit. `profile="voigt"` uses the reference mass; `"lorentz"` selects pressure broadening alone. One declared complete interval must cover the grid, cutoff and maximum pressure-shift margin. Linear Q(T) interpolation and the detailed kernels remain unchanged. A covered empty selection raises `EmptyLineSelection`; insufficient support raises `ValueError`. Empty line intervals remain valid stored references.

`CrossSectionMeasurements(species, isotopologue, spectra)` represents temperature-ordered laboratory spectra. Each `CrossSectionSpectrum` retains its own increasing vacuum-wavelength axis [nm], cross sections [m2/molecule], temperature [K], known pressure [Pa] or `None`, exact retained support intervals and optional resolution description. Arrays are independent read-only float64 snapshots. One spectrum per temperature is supported; pressure interpolation is outside this contract.

`measurement_absorption(atmosphere, grid, reference, spectral_interpolation="linear_native_wavelength", temperature_interpolation="linear", temperature_outside="error" | "hold_nearest", pressure_model="independent")` returns an `AbsorptionComponent` and `CrossSectionTemperatures`. The diagnostics retain requested/effective temperatures [K] and a boolean held-layer mask in bottom-to-top order, using independent read-only arrays. The pressure-independent model is an explicit approximation for known and unknown laboratory pressures. Interpolation first uses each contributing native wavelength axis, then laboratory temperature. Exact temperature knots require only that spectrum's support. Spectral extrapolation and interpolation through declared gaps raise `ValueError`; cross-section amplitudes receive no spectral-density Jacobian. `interpolate_cross_section_measurements` exposes the same interpolation without applying molecular columns.

The detailed `line_set`, `line_absorption`, `CrossSectionField`, `cross_sections`, `table_absorption` and `evaluate_table_absorption` APIs remain available with their existing contracts. Generated component depths remain independent writable float64 arrays. Composition retains duplicate, undeclared-species and line/complete-cross-section overlap checks.

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

`cross_sections(grid, CrossSectionField)` interpolates a fixed table once into caller-owned `PreparedCrossSections`; `evaluate_table_absorption(atmosphere, prepared)` applies current molecular columns. The table's exact layer T/p and spectral support must remain fixed. `table_absorption` is the full prepare-and-evaluate path. This is a fixed-cross-section scientific contract, not general tau scaling: changing abundance in `line_absorption` also changes self/foreign widths and requires reevaluating that gas's lines. No line-profile basis or persistent numerical cache is introduced, and no retrieval speedup is claimed without representative benchmarks.

### Pure line-table materialisation

`LineDatabase(lines, molecule_id, isotopologue_id, coverage_cm1)` represents one molecule/local isotope's transitions and declared centre-coverage completeness [cm-1]. Empty lines are valid; bounds are not inferred from centres. `PartitionFunctionTable(molecule_id, isotopologue_id, temperatures_k, partition_sums)` represents dimensionless Q at positive increasing knots [K], with independent writable float64 arrays of shape `(n_knots,)`, at least two knots. `evaluate_partition_function(table, temperatures_k)` returns fresh writable Q(T) by explicit linear interpolation without extrapolation, not HAPI/TIPS interpolation.

`line_set(database, partition_table, atmosphere, grid, species=..., molecular_mass_kg=..., pressure_range_pa=..., wing_cutoff_cm1=..., subtract_lorentz_pedestal=False)` matches isotope identities, requires a declared atmospheric species and checks centre coverage beyond cutoff plus maximum absolute pressure shift across the declared pressure support. It selects lines, computes Q(296 K)/Q(T_layer) and returns a `LineSet` without file access, identity generation or caches. Mass is kg/molecule, pressure Pa, grid/cutoff cm-1. Covered empty selections raise `EmptyLineSelection`; insufficient support raises `ValueError`. Supplying species-to-molecule mappings and database completeness remains caller responsibility.

### Explicit thermal Voigt lines

`LineSet` represents physical transitions, layer partition ratios and reference/validity conditions for one species and isotopologue. Its appended optional `molecular_mass_kg` field is a positive finite mass in kg per molecule, not molar mass or atomic mass units. `None` preserves the pressure-broadened Lorentz model and `AbsorptionComponent.kind="lorentz_lines"`. A supplied mass enables `scipy.special.voigt_profile` with Gaussian standard deviation `line.centre_cm1 * sqrt(k*T/mass)/c` in cm-1 and the existing air/self Lorentz HWHM in cm-1, producing `kind="voigt_lines"`. The Doppler width uses the unshifted transition frequency; pressure shift still translates the profile centre. Mass is never inferred from a species name, and no provider access or partition approximation is added by this kernel. Optional wings and the separate materialised water-continuum API are described below; the default retains unbounded profiles and adds no continuum or line mixing. Unrepresentable or zero Doppler widths raise `ValueError`. Temperature correction uses `100*h*c/k` from SciPy's SI constants, with the cm-to-m conversion explicit.

Lorentz and Voigt components for the same species/isotopologue are alternative line descriptions, so composition rejects their combination as a duplicate; either also overlaps a complete cross-section contribution for that species. Line-excluding continua and different isotopologues remain supported. Both profiles support explicit atmospheric freshness checks against a separately supplied evaluation atmosphere, including temperature, pressure, columns and broadener pools, and independent output ownership. The absorption kind is `"voigt_lines"` for a Voigt component.

## Materialised water continuum and explicit wings

`LineSet.wing_cutoff_cm1=None` retains unbounded Lorentz/Voigt profiles. A finite positive value truncates about the pressure-shifted centre without renormalising line strength. `subtract_lorentz_pedestal=True` is supported only for H2O with a 25 cm-1 cutoff: remove the Lorentz value at +/-25 cm-1 from the local profile, including when the local profile is Voigt. Generated components declare `line_wing_convention`; a generated MT_CKD water continuum rejects composition with unbounded or plain-truncated water lines. The compatibility check is not proof of source-data completeness or of non-water continuum conventions.

`WaterContinuumReference` contains physical water self/foreign ABSCO [cm2/molecule/(cm-1)] without the radiation term, self-temperature exponents and positive reference P/T on an independent writable uniform float64 cm-1 grid. `water_continuum_absorption(atmosphere, grid, reference)` applies temperature scaling and the radiation term on the reference grid before four-point cubic interpolation. A complete stencil is required; no extrapolation or negative-value clipping is used. The result is dimensionless depth shaped `(n_layers, n_points)`. Homogeneous-layer self/foreign density products are `N_water**2/dz` and `N_water*(N_total-N_water)/dz` [molecule2/m5], using the supplied columns and geometric thickness, not an implicit hydrostatic column. Generated continuum freshness includes water and foreign-air columns as well as geometry and T/p; fixed continuum tables use a conservative version of the same density check.

This is the water continuum only, not full MT_CKD, CIA or CO2 line mixing. The caller declares a temperature envelope and materialises correctly normalised coefficients; the library never selects/downloads an edition. Readers and explicit download transport are in [orsia-data](../../orsia-data/README.md); provenance is caller-owned. Software tests use synthetic files and analytic density/strength checks; they do not establish authentic-data agreement with the external Fortran reference or a representative WP7 accuracy budget.

The radiation term uses the stable exact `tanh` expression and SI constants. AER's Fortran 4.3 calculation uses small/large-x shortcuts and an older radiation constant. The retired bounded comparison failed its unmodified all-points criterion near zero wavenumber: negative padding nodes affect the first positive interpolation stencil. Diagnostic radiation alignment produced agreement on the tested synthetic layers, while general numerical equivalence remains unaccepted. The library applies no automatic coefficient correction. [Current kernel arithmetic](../src/orsia_radiative/spectroscopy/continuum.py) and [padding/scaling regressions](../tests/test_water_continuum.py) preserve the exact-`tanh`, scaling-before-interpolation contract; those tests do not regenerate the external comparison.
## Example: absorption composition upstream

This synthetic fixed table uses exact layer T/p and a declared gas column. Keep the evaluated atmosphere separately when requesting freshness checks.

```python
import numpy as np
from orsia_radiative import (
    Absorber,
    Atmosphere,
    CrossSectionField,
    compose_absorption_on_grid,
    table_absorption,
    wavenumber_grid,
)

grid = wavenumber_grid(np.array([1000.0, 1001.0]))
atmosphere = Atmosphere(
    np.array([0.0, 1000.0]),
    np.array([100000.0, 90000.0]),
    np.array([95000.0]),
    np.array([285.0]),
    np.array([2e29]),
    np.array([1.9e29]),
    np.array([1e28]),
    (Absorber("X", np.array([1e25]), "dry_air"),),
)
table = CrossSectionField(
    "X",
    "1",
    "cross_section",
    grid,
    atmosphere.temperature_k,
    atmosphere.pressure_pa,
    np.array([[1e-27, 2e-27]]),  # m2/molecule
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
