# Atmospheric columns and file-to-radiance composition

Pure layer calculations consume explicit pressure, temperature, humidity and gas arrays. File extraction belongs to [Orsia Data](../../orsia-data/docs/atmosphere.md); the application selects products and physical assumptions.

## Pure atmospheric column materialisation

`atmosphere.layers` defines `pressure_level_layer_means`, `remap_layer_columns` and `geometric_height`. `atmosphere.hydrostatic` defines `specific_humidity`, `gas_columns_from_fraction` and `hydrostatic_atmosphere`. All six functions are exported from `orsia_radiative`. They consume explicit arrays and physical assumptions; hydrostatic construction returns the existing bottom-to-top `Atmosphere` with independent writable numerical copies. [Local atmospheric readers](../../orsia-data/docs/atmosphere.md) own native-file extraction and explicit acquisition. Provider selection and source combinations remain caller-owned.

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

## From native profiles to solar or thermal radiance

The caller composes local readers from `orsia-data` with pure radiative functions. `read_netcdf_profiles` returns native pressure, temperature and humidity arrays; `pressure_level_layer_means` integrates the selected pressure reconstruction into layers. `gas_columns_from_fraction` converts explicit gas fractions to molecules/m2 before `hydrostatic_atmosphere` constructs the column. A dry-air CO2 mole fraction of `420e-6`, for example, gives `N_CO2=420e-6*N_dry`. Intermediate reader arrays remain caller-owned and writable; retained physical numerical arrays are independent writable copies.

`read_hitran_lines` and `read_partition_table` return `LineDatabase` and dimensionless `PartitionFunctionTable` inputs for the same molecule/local isotope. `line_set` combines them with atmosphere and target grid, checks pressure/temperature and line-centre coverage, and evaluates `Q(296 K)/Q(T_layer)` by linear interpolation. Supply mass in kg/molecule for Voigt profiles and an explicit wing cutoff when required. Intensities already include isotopologue abundance relative to whole-species columns. Numerical grid spacing specifies sampling; instrument resolution requires an explicit response model.

`line_absorption` evaluates temperature-adjusted strengths, self/foreign broadening and the chosen profile/wings. Each layer contributes `tau(layer, point)=column(layer)*cross_section(layer, point)`, with consistent spectroscopy-native cm-unit conversion. `compose_absorption_on_grid` validates and sums selected gas/continuum components without discovering missing constituents or interpolating mismatched grids. `atmospheric_optics` adds explicitly supplied Rayleigh/aerosol roles. For moist calculations, select H2O isotopologues and continuum explicitly; the supported MT_CKD combination requires [pedestal-subtracted local +/-25 cm-1 water lines](spectroscopy.md#materialised-water-continuum-and-explicit-wings). Other gases, continua and CO2 line mixing are caller selections.

For reflected sunlight, `read_solar_spectrum_csv` supplies wavelength-point beam-normal density [W m-2 nm-1]. `solar_irradiance` applies the density Jacobian and point interpolation to the selected cm-1 grid [W m-2 per (cm-1)]; bin integration and instrument convolution remain separate. `solar_source` adds solar zenith and incoming-to-outgoing relative azimuth. A Lambertian `RadiativeScene` supplies atmosphere, surface and outgoing view. Geometry is local plane-parallel, with no implied geodetic conversion, curvature or refraction.

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
