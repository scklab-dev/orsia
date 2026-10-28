# Five-window trace-gas enrichment

The executable example is `notebooks/era5_hitran_sasktran2_five_bands.ipynb`; `notebooks/support/five_band_gases.py` prepares its reference profiles and measured spectra. Standalone species-removal and nested-sampling diagnostics are retired.

## Scope and atmospheric construction

| Window | Wavelength interval in vacuum | Included molecular absorption |
| --- | --- | --- |
| UV | 320-325 nm | H2O lines; O3, SO2 and NO2 measured cross sections |
| VNIR | 758-765 nm | H2O, O2 lines |
| SWIR | 1645-1655 nm | H2O, CO2, CH4 lines |
| MWIR | 4250-4300 nm | H2O, CO2 lines |
| LWIR | 9950-10050 nm | H2O, CO2, CH4, O3, exploratory NH3 lines |

The original windows, geometry, surface, H2O profile, O2/CO2/CH4 abundances and scalar Rayleigh law are retained. Pressure boundaries are the original 1000-1 hPa grid with eleven bottom-to-top layers. Temperature and specific humidity are pressure-layer means from the retained equatorial-ocean ERA5 column for January 15, 2024 at 12:00 UTC. The bottom height is the example's prescribed 0 m pressure surface. Orsia's public hydrostatic materializer supplies geometry and total/dry/water molecular columns.

The [MIPAS 2001 equatorial daytime reference](mipas-reference-atmospheres.md) supplies O3, SO2, NO2 and NH3. This choice matches the solar scene latitude and daytime NO2 convention. The thermal experiment applies dark illumination to the same frozen daytime composition. MIPAS supplies a historical reference mixture; ERA5 supplies the meteorology. The selected trace gases remain an explicit subset of the native profile, and their abundances retain independent identities alongside the unchanged baseline O2/CO2/CH4 assumptions.

At each MIPAS native pressure point, convert the supplied total-air molar fraction to dry-air basis using `x_dry = x_total / (1 - x_H2O_MIPAS)`. Integrate the resulting piecewise-linear-in-pressure profile over each ERA5 pressure interval with `pressure_level_layer_means`. Multiply the dry-fraction layer mean by the ERA5 dry-air molecular column using `gas_columns_from_fraction(..., convention='dry_air_mole')`. This replacement of native humidity by ERA5 humidity preserves the reference dry mixing ratio. The reported reconstructed total-air fraction is `x_dry_mean * N_dry_ERA5 / N_total_ERA5`. Interpolation requires full pressure coverage and every requested species; failed support or missing molecules stop execution. The retained ERA5 inputs have positive water fractions and the MIPAS native water fraction stays below unity.

## Spectroscopy and support

O3 UV uses the [Serdyuchenko-Gorshelev version 1.0 dataset, DOI 10.5281/zenodo.5793207](https://doi.org/10.5281/zenodo.5793207), linked from the [Bremen UVSAT data page](https://www.iup.uni-bremen.de/UVSAT/data/xsection/). The native file supplies vacuum wavelengths in nm, cross sections in cm2/molecule and eleven temperatures from 193 to 293 K. Its header gives 0.01 nm sampling and 1 cm-1 HWHM resolution at 290-350 nm. The acquisition verifies the provider's MD5 identity and records a local SHA-256. Scientific references are Gorshelev et al., AMT 7, 609-624 (2014), DOI 10.5194/amt-7-609-2014, and Serdyuchenko et al., AMT 7, 625-636 (2014), DOI 10.5194/amt-7-625-2014.

SO2 and NO2 use [Bremen SCIAMACHY PFM reference data](https://www.iup.uni-bremen.de/gruppen/molspec/databases/sciamachydata/index.html), version 1.0, at 203, 223, 243, 273 and 293 K. The downloaded `SO2_*K.dat` and `NO2_*K.dat` headers explicitly declare vacuum nm, cm2/molecule and 0.21 nm FWHM in the current UV window. SO2 was measured in an N2 mixture at 100 mbar; NO2 uses an N2 mixture at 150 mbar for 203 K and 100 mbar for the other four temperatures. The SO2 scale references Vandaele et al. (1994), DOI 10.1029/94JD02187; the NO2 scale references Vandaele et al. (1998), DOI 10.1016/S0022-4073(97)00168-4. The SCIAMACHY dataset is described by Bogumil et al. (2003), DOI 10.1016/S1010-6030(03)00062-5. This five-temperature selection provides directly declared vacuum coordinates and cold-temperature support for the demonstration.

UV preparation retains native samples spanning 319-326 nm plus bracketing points, checks increasing wavelength and finite values, and requires finite nonnegative cross sections on this support. It multiplies amplitudes by 1e-4 to obtain m2/molecule, linearly interpolates in native wavelength at `lambda_nm = 1e7 / nu_cm1`, then linearly interpolates temperature. Point cross sections retain their amplitudes through the coordinate change. Measured UV fields use the `all` isotopologue label for natural-abundance spectra.

The explicit `clip` temperature policy holds the nearest laboratory spectrum outside its temperature range. On this ERA5 column O3 holds 293 K at layer 0 (actual 294.290527 K). SO2 and NO2 hold 293 K at layer 0 and 203 K at layer 6 (actual 199.054611 K). All other layers interpolate between measured temperatures. Layer indices are zero-based, bottom to top. The retired species diagnostic reported every actual/effective temperature pair. This treatment defines a bounded approximation; a temperature-supported laboratory dataset or a separately validated extrapolation model would refine those layers. Pressure dependence is approximated by the fixed laboratory spectra. Refining the numerical grid retains the SO2/NO2 0.21 nm measurement resolution.

LWIR O3 (HITRAN molecule 3) and NH3 (molecule 11) use the public HAPI acquisition, local isotopologue 1, HAPI molecular masses and bundled partition sums at 70-400 K. The interval is the unchanged LWIR window plus a 26 cm-1 acquisition margin; evaluation retains the existing 25 cm-1 line-wing cutoff. The current HAPI endpoint establishes the downloaded bytes; the local receipt and retained native header specify replay identity. The acquisition mode `current` follows the [existing HITRAN adapter contract](../libs/orsia-data/src/orsia_data/services/hitran.py). NH3 remains exploratory within this specific window and climatological profile.

## Shared optics and counterfactuals

The notebook's `prepare_case(name, points=None)` returns the grid, scene, optics, source and shared reference inputs. It composes a common total absorption array for Orsia and the SASKTRAN2 adapter. Solar cases also share the same approximate scalar Rayleigh depth; thermal cases use zero scattering. In the enriched UV case, measured O3/SO2/NO2 and the original H2O line absorption coexist as distinct species. The IR and UV implementations of O3 occupy separate windows.

The retired diagnostic saved species components and used `remove_species` to recompose the remaining optical arrays while freezing the atmosphere, temperature, humidity, broadener columns, source, scattering and surface. That historical optical-removal definition isolated the species' direct marginal contribution at the reference mixture; a physical abundance perturbation can additionally alter self/foreign broadening and atmospheric thermodynamics. Each removed optical array was shared identically with both solvers. The signed radiance difference was `L_full - L_removed` in W m-2 sr-1 (cm-1)-1. Optical-depth arrays had shape `(layer, spectral_point)`, with the full vertical column given by their sum over layers. This is historical methodology, not a current callable diagnostic.

## Execution and evidence

Run from the repository root after following the [notebook preparation guide](../notebooks/README.md):

```sh
devbox run -- just prepare-five-band-enriched --download
devbox run -- just prepare-five-band-enriched
devbox run -- just execute-five-band
```

The acquisition creates `outputs/five-band-enriched-inputs/manifest.json`, references the baseline local inputs, and retains original spectroscopy files separately. `--download` permits the additional public transfers. With a complete receipt present, the recipe checks all input SHA-256 identities offline. The notebook also checks those identities before calculations. Inputs and receipts stay in ignored local storage; disposable generated plots, caches and timing histories are not required inputs. An earlier update saved a baseline notebook snapshot locally; neither that snapshot nor generated historical benchmark artifacts are maintained deliverables or prerequisites for the current workflow.

The standard comparison uses one SASKTRAN2 configuration (32/32 cells/streams) per band and the selected Orsia controls, followed by spectral/residual plots and readable agreement/timing tables. Its timing scope remains the complete Orsia call versus the prepared SASKTRAN2 `calculate_radiance` call, with historical cache times and current access times labelled separately. Earlier numerical-selection studies and enriched diagnostic runs are historical evidence, not current execution or artifact-retention guarantees.

The former species-removal diagnostic varied spectral sampling and removed individual optical contributions; it is no longer a maintained command. The notebook still exposes explicit spectral-point controls. Sampling and instrument-integrated sensitivity require dedicated evidence, not an inference from successful notebook execution.

Runtime validation is performed separately from the comparison display: check CPython 3.14, disabled GIL after scientific/native imports, CPU affinity within 0-7, and one-thread numerical pools. The isolated SASKTRAN2 worker retains its own runtime identity. These checks characterize the local execution configuration; concurrency safety and scientific accuracy require their corresponding validation.
