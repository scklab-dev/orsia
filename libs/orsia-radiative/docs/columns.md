# Atmospheric columns and file-to-radiance composition

Pure layer calculations consume explicit pressure, temperature, humidity and gas arrays. File extraction belongs to [Orsia Data](../../orsia-data/docs/atmosphere.md); the application selects products and physical assumptions.

## Pure atmospheric column materialisation

`atmosphere.layers` defines `pressure_level_layer_means`, `remap_layer_columns` and `geometric_height`. `atmosphere.hydrostatic` defines `specific_humidity`, `gas_columns_from_fraction` and `hydrostatic_atmosphere`. All six functions are exported from `orsia_radiative`. They consume explicit arrays and physical assumptions; hydrostatic construction returns the existing bottom-to-top `LayerAtmosphere` with independent writable numerical copies. [Local atmospheric readers](../../orsia-data/docs/atmosphere.md) own native-file extraction and explicit acquisition. Provider selection and source combinations remain caller-owned.

| Function | Explicit convention |
| --- | --- |
| `pressure_level_layer_means(pressure_pa, values, pressure_boundaries_pa)` | Caller-selected reconstruction linear in pressure, integrated into pressure-mass layer means; fully bracketed support without interface inference or extrapolation. |
| `specific_humidity(ratio)` | Convert kg water/kg dry air to kg water/kg total moist air. |
| `gas_columns_from_fraction(..., convention=..., gravity_m_s2=..., molar_mass_kg_mol=...)` | Convert dry/total-air mole or mass fractions to molecules/m2; mass fractions require species molar mass [kg/mol]. |
| `remap_layer_columns(source_interfaces, columns, target_interfaces)` | Conserve extensive layer quantities by pressure overlap, assuming constant amount per Pa within each source layer. |
| `hydrostatic_atmosphere(...)` | Construct pressure [Pa], temperature [K], height [m] and column [molecules/m2] state from supplied layers and target pressure interfaces. |
| `geometric_height(..., reference_gravity_m_s2=..., earth_radius_m=...)` | Spherical conversion `z=R*H/(R-H)`, `H=geopotential/g0`; geoid/datum corrections remain separate. |

Hydrostatic layer mass per area is `delta_p/g`. Water and dry molecular columns are `q*delta_p/g/M_water*N_A` and `(1-q)*delta_p/g/M_dry*N_A`, with fixed molar masses 0.01801528 and 0.0289647 kg/mol and `N_A=6.02214076e23 mol-1`. Gravity is explicit and constant. Trace gases occupy the dry-air pool without changing its fixed mean molar mass. `hydrostatic_atmosphere` inserts H2O from q, remaps gas/dry/water columns by pressure overlap, preserves `total=dry+water` and uses pressure-mass weighted temperatures. Target midlayer pressures are arithmetic interface means.

Approximate geometric heights use a hypsometric calculation with pressure-mass weighted `R_specific*T` in each target layer, anchored by `bottom_height_m` at the target bottom. Native altitude reconstruction and height-grid invariance under remapping are outside this approximation. Zero model-top pressure may be reconstructed, but finite hydrostatic height requires a positive target top.

Convert composition on another product grid to columns using that product's air/humidity convention, then remap onto the meteorological source grid. Equal pressure support preserves integrated gas amount; a narrower target discards the outside column. Resolve surface pressure, terrain, coverage and time differences before combination. No gas climatology fills gaps. Filter knots below terrain before reconstruction. When remaining knots fail to bracket the surface, obtain valid surface/model-level data or construct an explicitly partial column. Hydrostatic construction cannot identify supplied native values that a provider extrapolated below terrain.

### Synthetic hydrostatic column

This offline example constructs a local layer column without files or network access:

```python
import numpy as np
from orsia_radiative import gas_columns_from_fraction, hydrostatic_atmosphere

p = np.array([100000.0, 60000.0, 20000.0])  # source interfaces, bottom to top [Pa]
t = np.array([280.0, 240.0])  # layer means [K]
q = np.array([0.01, 0.02])  # kg water/kg moist air
ch4 = gas_columns_from_fraction(
    p,
    q,
    np.array([1.8e-6, 1.8e-6]),
    convention="dry_air_mole",
    gravity_m_s2=9.80665,
)
atmosphere = hydrostatic_atmosphere(
    p,
    t,
    q,
    target_pressure_boundaries_pa=np.array([100000.0, 80000.0, 20000.0]),
    gravity_m_s2=9.80665,
    bottom_height_m=0.0,
    gas_columns_molecules_m2={"CH4": ch4},
    air_broadener="dry_air",
)
```

The column is truncated at 20000 Pa. Its columns and temperatures supply spectroscopy and transfer; plume winds, screening and prior covariance remain separate inputs. [Analytic column tests](../tests/test_atmospheric_columns.py) execute this snippet and check units, fraction conventions, ownership and conservative remapping. Synthetic tests leave authentic provider compatibility and atmospheric accuracy unqualified.

## Level atmospheres and column integrals

`LevelAtmosphere` in `atmosphere.levels` owns bottom-to-top level altitude [m], pressure [Pa], temperature [K], specific humidity [kg/kg moist air], explicit species identities and `(level,species)` dry-air molar mixing ratios. Its arrays are immutable snapshots. `LayerAtmosphere` in `atmosphere.layers` retains the separate layer-state representation and its independent writable numerical snapshots. Neither representation contains spectroscopy. The two reconstruction laws remain distinct.

`integrate_atmospheric_columns(atmosphere, surface_altitude_m=...)` returns `AtmosphericColumnIntegrals`. Gas columns use `(layer,species)` axes in molecules/m2. Dry-air columns and water-density products use `(layer,)`. The record retains original height interfaces and the reconstructed surface state. Its arrays are immutable. `partial_layer_index` and `first_complete_layer_index` are zero-based properties; `None` denotes absence. A partial top interval has no complete interval above it. `dry_air_column_mol_m2` is an explicit derived diagnostic using `N_A=6.02214076e23 molecules/mol`.

### Log-linear reconstruction

1. **Objective:** define the retained lower boundary. **Inputs:** increasing levels `z[i]`, their thermodynamics and species ratios, and surface altitude `h`. **Outputs:** retained altitude `h_ret`, surface pressure `p_ret`, first complete interval and optional partial interval. **Processing:** reject nonfinite `h`, `h < z[0]-1e-6` or `h >= z[-1]-1e-6`. If a level is within `1e-6 m`, set `h_ret=z[i]` and `p_ret=p[i]`. Otherwise locate `z[i] < h < z[i+1]`, set `f=(h-z[i])/(z[i+1]-z[i])`, and use `p_ret=exp((1-f)*log(p[i])+f*log(p[i+1]))`. Temperature, specific humidity and species ratios at this endpoint are linear interpolations with the same `f`.
2. **Objective:** reconstruct interval endpoint densities. **Inputs:** pressure `p`, temperature `T`, humidity `q` and dry-air ratios `x[s]` at each retained endpoint. **Outputs:** total-air, dry-air, water and species number densities [molecules/m3]. **Processing:** `n_total=p/(k_B*T)`, `r_water=q/(1-q)*M_dry/M_water`, `n_dry=n_total/(1+r_water)`, `n_water=n_dry*r_water`, and `n_species[s]=n_dry*x[s]`. Fixed constants are `k_B=1.380649e-23 J/K`, `M_dry=0.0289647 kg/mol` and `M_water=0.01801528 kg/mol`. `moist_air_number_densities` exposes this same conversion for arbitrary broadcastable sample axes.
3. **Objective:** integrate dry-air and species material. **Inputs:** lower/upper densities `a,b` and retained interval thickness `dz`. **Outputs:** molecular columns `N[layer,species]` and `N_dry[layer]`. **Processing:** use `N=dz*(a-b)/log(a/b)` for distinct positive endpoints. If either endpoint is zero or `abs(1-a/b) <= 1e-5`, use `N=dz*(a+b)/2`. `log_linear_density_integral` implements this fixed branch rule. Entirely excluded intervals receive zero.
4. **Objective:** retain water density-pair integrals for continuum. **Inputs:** endpoint water densities `w0,w1`, foreign densities `f0=max(n_total0-w0,0)`, `f1=max(n_total1-w1,0)` and retained thickness `dz`. **Outputs:** self product `Q_self=dz*(w0*w0+w0*w1+w1*w1)/3` and foreign product `Q_foreign=dz*(2*w0*f0+w0*f1+w1*f0+2*w1*f1)/6`, in molecules2/m5. **Processing:** evaluate these polynomial integrals of linearly interpolated endpoint densities and zero excluded intervals. This continuum law differs from the logarithmic-mean gas integral.

The existing constant-density `clip_atmosphere_height` law remains available for layer atmospheres. Converting level state to one layer state cannot generally preserve every partial-interval or water-product integral. Invalid physical domains and unsupported terrain raise `ValueError`; no profile extrapolation or representative coefficient temperature/pressure is inferred.

### Contract material with spectroscopy

**Objective:** evaluate vertical gas absorption on explicit spectral samples. **Inputs:** `AtmosphericColumnIntegrals`, a tuple of `LayerCrossSections`, the shared `SpectralGrid`, and optional `WaterContinuumCoefficients`. **Outputs:** a fresh writable dimensionless spectrum from `column_absorption_optical_depth(columns, tables, spectral=..., continuum=...)`. **Processing:** match interfaces, species identities and sampled coordinates; reject duplicate contributions. Each layer has `tau[l,k] = sum_s(sigma[l,s,k]*N[l,s]) + c_self[l,k]*Q_self[l] + c_foreign[l,k]*Q_foreign[l]`. Sum layers from top to bottom. Omitted contributions are zero. Supplied water tables paired with continuum must explicitly exclude that continuum and declare its matching local-line convention and evaluation T/p.

Cross sections have units m2/molecule; water coefficients have units m5/molecule2. Coefficient evaluation conditions are explicit inputs. This contraction selects no interval temperature/pressure, line model or continuum edition. Molecular amounts retain their supplied reconstruction law. Numerical tests cover exponential columns, terrain snapping, optional indices, species permutations, endpoint products and nonoverlapping water composition; physical spectroscopy accuracy requires separate evidence.

## From native profiles to solar or thermal radiance

The caller composes local readers from `orsia-data` with pure radiative functions. `read_netcdf_profiles` returns native pressure, temperature and humidity arrays; `pressure_level_layer_means` integrates the selected pressure reconstruction into layers. `gas_columns_from_fraction` converts explicit gas fractions to molecules/m2 before `hydrostatic_atmosphere` constructs the column. A dry-air CO2 mole fraction of `420e-6`, for example, gives `N_CO2=420e-6*N_dry`. Intermediate reader arrays remain caller-owned and writable; retained physical numerical arrays are independent writable copies.

`read_hitran_lines` and `read_partition_table` return `SpectralLineTable` and dimensionless `PartitionFunctionTable` inputs for the same molecule/local isotope. `line_set` combines them with atmosphere and target grid, checks pressure/temperature and line-centre coverage, and evaluates `Q(296 K)/Q(T_layer)` by linear interpolation. Supply mass in kg/molecule for Voigt profiles and an explicit wing cutoff when required. Intensities already include isotopologue abundance relative to whole-species columns. Numerical grid spacing specifies sampling; instrument resolution requires an explicit response model.

`line_absorption` evaluates temperature-adjusted strengths, self/foreign broadening and the chosen profile/wings. Each layer contributes `tau(layer, point)=column(layer)*cross_section(layer, point)`, with consistent spectroscopy-native cm-unit conversion. `compose_absorption_on_grid` validates and sums selected gas/continuum components without discovering missing constituents or interpolating mismatched grids. `atmospheric_optics` adds explicitly supplied Rayleigh/aerosol roles. For moist calculations, select H2O isotopologues and continuum explicitly; the supported MT_CKD combination requires [pedestal-subtracted local +/-25 cm-1 water lines](spectroscopy.md#materialised-water-continuum-and-explicit-wings). Other gases, continua and CO2 line mixing are caller selections.

For reflected sunlight, `read_solar_spectrum_csv` supplies wavelength-point beam-normal density [W m-2 nm-1]. `solar_irradiance` applies the density Jacobian and point interpolation to the selected cm-1 grid [W m-2 per (cm-1)]; bin integration and instrument convolution remain separate. `solar_source` adds solar zenith and incoming-to-outgoing relative azimuth. A Lambertian `ColumnScene` supplies atmosphere, surface and outgoing view. Geometry is local plane-parallel, with no implied geodetic conversion, curvature or refraction.

With `max_order=0, delta_m_moments=None`, `transfer.solar.reflected_radiance` returns direct surface reflection. For absorption only, `L=rho*E*mu_sun/pi*exp[-sum(tau)*(1/mu_sun+1/mu_view)]`: absorption acts on both solar and viewing paths. With scattering roles present, physical order zero uses their full extinction. Order one includes the first atmospheric event and common Lambertian ground coupling; higher orders add successive events. Refine order, angular sampling and vertical cells independently.

For thermal emission, select a thermal grid before evaluating spectroscopy, then call `transfer.thermal.thermal_radiance(scene, optics=optics)`. LTE gives each isothermal absorbing layer the Planck source `B(nu,T_layer)`. The opaque surface emits `epsilon*B(nu,T_surface)` and reflects downward atmospheric irradiance with reflectivity `1-epsilon`. With `mu=cos(view_zenith)` and `tau_above[k]` the depth above layer k, the direct terms are:

```text
surface_emission     = epsilon * B(nu, T_surface) * exp(-tau_total / mu)
atmospheric_emission = sum_k B(nu, T_k) * [1 - exp(-tau_k / mu)]
                             * exp(-tau_above[k] / mu)
```

The reflected term is `(1-epsilon)*(F_down/pi)*exp(-tau_total/mu)`, where `F_down` is the hemispheric downward atmospheric irradiance at the surface. The `expn` method evaluates this term with E3 exponential integrals; the alternative uses angular Gauss-Legendre quadrature requiring separate refinement. Atmospheric emission can produce absorption or emission features depending on temperatures and optical support. `schwarzschild_terms` returns the three fresh writable `(n_points,)` arrays in W m-2 sr-1 per (cm-1); their sum is the total `Radiance`. [LTE thermal transfer](transfer.md#lte-thermal-transfer) specifies the zero-top-incident, non-scattering closure and its rejected inputs.

Recompute temperature-dependent line inputs and spectroscopy after pressure/temperature changes. Gas-abundance changes require new columns and reevaluation of that gas's absorption, including self/foreign broadening. Surface, illumination amplitude or view changes can reuse unchanged same-grid atmospheric optics after rebuilding the relevant scene/source. A new spectral grid requires fresh spectral materialisation and optical evaluation; composition and transfer never resample silently.

The retired file-to-radiance scripts used invented dry CO2-only files and columns truncated at 20000 Pa. Their format demonstrations and analytic identities do not establish authentic ERA5/CAMS/HITRAN interoperability, spectroscopy accuracy, complete atmospheric physics, spectral convergence, retrieval performance or concurrent native-library safety. Current executable composition is in the [five-band notebook](../../../notebooks/README.md#five-band-transport-comparison); reader, materialisation and thermal regressions remain in the tests. Clouds and ice remain outside scope.
