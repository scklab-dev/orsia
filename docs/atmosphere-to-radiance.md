# From atmospheric and spectroscopic files to radiance

This guide explains the public file-to-radiance composition. The [radiative library guide](../libs/orsia-radiative/src/orsia_radiative/README.md) gives executable synthetic examples. Offline reader, materialisation and transport regressions remain in the library tests.

For emission rather than reflected sunlight, see the [thermal counterpart](atmosphere-to-thermal-radiance.md).

## The two data branches meet at optical depth

```text
meteorology.nc                 co2.par + partition.csv
       |                                  |
orsia-data: local readers       orsia-data: local readers
       |                                  |
native p, T, q                 LineDatabase + PartitionFunctionTable
       |                                  |
point-to-layer means                      |
CO2 fraction -> columns                   |
hydrostatic materialisation               |
       |                                  |
   Atmosphere --------> line_set <---- SpectralGrid
       |                                  |
       +-----------------> line_absorption
                                  |
                         AbsorptionComponent
                                  |
                      compose_absorption_on_grid
                                  |
                              Absorption
                                  |
                  atmospheric_optics
                                  |
                           AtmosphericOptics
                                  |
solar.csv -> SolarSource ----------+---> transfer.solar.reflected_radiance (order 0)
                                  |                  |
Atmosphere + surface + view -> RadiativeScene     Radiance
```

`orsia-data` owns files, native decoding and explicit acquisition. xarray is internal to its atmospheric readers; the numerical boundary is arrays and physical Orsia records, not an xarray dataset. `orsia-radiative` owns materialisation, spectroscopy, optical composition and transfer, and never opens a provider file or downloads data. This script is caller orchestration: it chooses the column, gas abundance, spectral grid, physical approximations and solver. The fixture writer uses xarray only to create a demonstrative native file.

## 1. Meteorology and gas composition become an Atmosphere

The native file has five pressure knots with temperature in kelvins and specific humidity in kg water/kg moist air. `read_netcdf_profiles` selects one exact time and location and returns detached NumPy profiles. `pressure_level_layer_means` integrates the explicitly selected linear-in-pressure reconstruction onto four layers; pressure interfaces descend from 100000 to 20000 Pa, while array layers run bottom to top.

The example selects a constant CO2 dry-air mole fraction of `420e-6`. `gas_columns_from_fraction` combines this with pressure thickness, humidity and explicit gravity to produce molecules/m2, not ppm values. `hydrostatic_atmosphere` builds pressure, temperature, approximate heights, total/dry/water columns and named absorbers. All retained numerical arrays are independent writable copies; intermediate reader arrays remain caller-owned and writable.

For dry air, layer mass per area is `delta_p / g`, dry molecular column is `delta_p / g / M_dry * N_A`, and CO2 molecular column is `420e-6` times the dry molecular column. In the example `q=0`, so H2O has zero column. The top is explicitly truncated at 20000 Pa: this is not a complete ground-to-space atmosphere.

For real files, obtain valid surface pressure and terrain support, ensure the native knots bracket every interface, and supply the actual composition. [The atmospheric contract](atmospheric-data.md) covers explicit ERA5/CAMS acquisition, GFS GRIB extraction, units, missing values, longitude policy and conservative remapping. Reading a CAMS fraction does not make it a molecular column: use its declared dry/moist and mass/mole convention and its own pressure support before remapping. No missing gas or surface knot is invented.

## 2. Spectroscopic tables become layer-dependent line inputs

`read_hitran_lines` returns a `LineDatabase` for molecule 2, local isotopologue 1, with declared line-centre coverage. `read_partition_table` returns a dimensionless Q(T) table for that same isotope. These are data, not evaluated optical depths.

`line_set` combines those tables with the atmosphere and the target grid. It selects transitions including the explicit 25 cm-1 wing margin, checks pressure/temperature support and evaluates `Q(296 K)/Q(T_layer)` by the implemented linear interpolation. The molecular mass is supplied in kg/molecule; it enables thermal Voigt profiles. Isotopologue abundance is already included in the line-intensity convention: do not multiply the whole-species column by a second isotope-abundance factor.

The target grid has 1001 increasing points from 6200 to 6202 cm-1. Its 0.002 cm-1 step is numerical sampling, not instrument resolution. The file declares centre coverage from 6174 to 6228 cm-1 to cover line wings; a real caller must establish that completeness rather than infer it from the lines present. The one transition and partition values here are invented, even though their file formats are valid.

## 3. Spectroscopy and molecular columns become optical state

`line_absorption(atmosphere, grid, lines)` evaluates temperature-adjusted strengths, self/foreign broadening, Doppler width and the chosen line-wing truncation. Conceptually each layer contributes `tau(layer, point) = column(layer) * cross_section(layer, point)`, with the spectroscopy-native cm units converted consistently. The returned `AbsorptionComponent` has dimensionless vertical optical depth of shape `(4, 1001)`.

`compose_absorption_on_grid` validates and sums explicitly supplied gas/continuum components into total `Absorption`. It does not discover missing constituents or interpolate mismatched grids. The example also passes the generating atmosphere separately for a freshness check. `atmospheric_optics` then combines this total with explicitly supplied Rayleigh/aerosol roles, if any, into solver-neutral `AtmosphericOptics`. There is no Rayleigh or aerosol in this deliberately absorption-only example.

For a moist calculation, evaluate the selected H2O isotopologues and continuum explicitly. The [spectroscopy contract](../libs/orsia-radiative/README.md#materialised-water-continuum-and-explicit-wings) requires the compatible local +/-25 cm-1 pedestal-subtracted H2O convention when adding the supported MT_CKD water continuum. Other gases, continua and CO2 line mixing are not automatically supplied.

## 4. Illumination and scene remain separate from spectroscopy

`read_solar_spectrum_csv` reads wavelength samples and beam-normal irradiance density in W m-2 nm-1. `solar_irradiance` applies the wavelength-to-wavenumber density Jacobian and point interpolation onto the same grid, returning W m-2 per (cm-1). It is not just an axis rename, instrument convolution or conservative bin integration. `solar_source` adds the explicit solar zenith and incoming-to-outgoing relative azimuth.

`RadiativeScene` combines the atmosphere, a Lambertian surface with reflectance 0.2 and an outgoing view zenith of 10 degrees. The solar zenith is 30 degrees. The geometry is local plane-parallel; no geodetic conversion, curvature or refraction is implied. The solar file is a synthetic flat density, not a reference solar product or distance-normalised observation.

## 5. The caller selects the transfer calculation

```python
radiance = solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
```

This call returns unscattered reflected-solar radiance at the top of the supplied column, with shape `(1001,)` and units W m-2 sr-1 per (cm-1). In this absorption-only case, `L = rho * E * mu_sun / pi * exp[-sum(tau) * (1/mu_sun + 1/mu_view)]`: absorption acts on both the downward solar path and upward viewing path. The script checks this identity, column conversion, finite/nonnegative output and a line-centre absorption dip.

With explicit Rayleigh/aerosol optics, physical order zero uses their full extinction for direct reflection. Select `max_order=1` for the first atmospheric event with common Lambertian ground coupling, or a larger order for successive scattering. Refine atmospheric order, angular sampling and vertical cells independently. Thermal emission uses the separate `transfer.thermal.schwarzschild_radiance` operation under its absorbing LTE contract.

## What changes require recomputation?

- A temperature or pressure change requires rematerialising temperature-dependent line inputs and reevaluating affected spectroscopy.
- A gas-abundance change requires new columns and reevaluation of that gas's line absorption, including self/foreign broadening; do not simply scale an old optical depth.
- A surface, illumination amplitude or view change can reuse unchanged same-grid atmospheric optics; rebuild the relevant scene/source and rerun transfer.
- A spectral-grid change requires rematerialising all spectra and optics on that grid; composition and transfer never resample silently.

This example tests the software chain and its bounded analytic transport identity, not authentic ERA5/CAMS/HITRAN compatibility, complete atmospheric physics, spectroscopy accuracy, retrieval performance or native-library concurrent-call safety. It stops before instrument response and inversion. The code and current package contracts are authoritative for the APIs used here; intended architecture alone does not establish implemented behavior or scientific qualification.
