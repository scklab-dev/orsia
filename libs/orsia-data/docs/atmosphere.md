# Atmospheric acquisition and local profiles

`orsia-data` owns explicit acquisition, native-file decoding and the offline `read_pressure_level_atmosphere` composition. `orsia-radiative` owns pure hydrostatic/conservation calculations and `LayerAtmosphere`. Applications choose products, editions, source combinations, interpolation assumptions and missing-data policy. Acquisition, local extraction and numerical materialisation are separate operations. The supported file boundary remains experimental; provider compatibility and atmospheric accuracy require the evidence specified below.

## Acquisition

`download_era5(product=..., request=..., retrieve=..., destination=...)` accepts the explicit CDS identifiers `reanalysis-era5-pressure-levels`, `reanalysis-era5-single-levels` and `reanalysis-era5-complete`. `download_cams` distinguishes ADS `cams-global-greenhouse-gas-forecasts` from `cams-global-atmospheric-composition-forecasts`. Product selections do not pin editions or guarantee requested variable availability. Supply a configured client's synchronous `retrieve(dataset, request, target)` method. Orsia does not install a client, read credentials, accept licences, set retries or infer native request contents. Choose dates/cycle/lead, fields, area, levels and format explicitly. Archive/edition coverage and request schemas still require qualification against the selected service. See the official [ERA5 documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation), [CAMS GHG catalogue](https://ads.atmosphere.copernicus.eu/datasets/cams-global-greenhouse-gas-forecasts?tab=overview) and [CAMS composition catalogue](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview).

CDS/ADS retrieval writes in an exclusively owned private temporary directory, checks for a nonempty regular file and a post-transfer byte limit, optionally verifies SHA-256, then publishes by atomic no-clobber hard link. Existing destinations are preserved, including a racing writer. Every failure cleans up the private directory; parents may remain. The byte limit is checked after SDK transfer. SDK/network exceptions propagate; the caller owns credentials, concurrency, cancellation and deadlines. Native requests are deep-copied before submission and must remain stable during copying.

`download_gfs(cycle=..., lead_hours=..., variables=..., levels=..., area=..., destination=...)` builds a NOMADS 0.25-degree GRIB2 subset request. Use acquisition codes such as `TMP`, `SPFH`, `UGRD`, `VGRD`, and native level selectors such as `1000 mb` or `surface`; applications select their lists. `area=(west,east,south,north)` uses a non-wrapping rectangle with longitude in [0,360]. Cycle is an exact UTC 00/06/12/18 hour, distinct from the validity returned by `gfs_valid_time(cycle, lead_hours)`. Leads are hourly through 120 h and three-hourly through 384 h. Current NOMADS retention and field availability require separate verification; see the [NOAA product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/). Transport uses `download_file`'s bounded streaming/no-clobber semantics; an HTTP error page must still fail local GRIB decoding. No historical archive or ERA5 fallback is attempted.

Retain native bytes for offline replay. `file_sha256(path)` gives a streaming byte fingerprint; authenticity and coverage require separate evidence. Keep product/version, file identity, native request, cycle/lead/validity, actual spatial support, units, conversions and assumptions in caller-owned acquisition records, excluding credentials. Numerical atmospheric contracts contain no provider identity or generic manifest.

## NetCDF and GFS extraction

`read_netcdf_profiles(path, variables=..., coordinates=..., vertical_units=..., valid_time=..., latitude_deg=..., longitude_deg=..., backend=...)` returns `(native_vertical_coordinate, native_variable_profiles)`. `variables` maps native names to exact required native unit strings; `coordinates` names the one-dimensional `(time, vertical, latitude, longitude)` coordinate variables. Fields must have exactly those dimensions, in any order. Pressure reconstruction, species identity and unit conversions remain explicit. Extra ensemble/expver dimensions must be selected upstream. The default `forecast_dimensions=None` also rejects extra forecast axes. Surface/context fields use `read_netcdf_surface_fields` with `(time, latitude, longitude)` coordinates and return scalars, such as surface pressure, 10 m wind, boundary-layer height, surface temperature or cloud screening. Applications must establish their suitability as plume winds, surface parameters or screening inputs.

For one-cycle forecasts, pass `forecast_dimensions=(cycle_dimension, period_dimension)` to the profile, point-surface or grid-surface reader. A two-dimensional valid-time coordinate must use exactly those dimensions, in either order, with one cycle and a nonempty period axis. The reader selects the cycle by name in each field and preserves vertical and spatial singleton axes. One-dimensional time coordinates remain supported with an absent, scalar or singleton cycle coordinate. Multiple cycles, groups and unresolved extra field axes raise `ValueError`.

`read_netcdf_surface_grid` returns `(longitude_deg, latitude_deg, fields)` with fields in `(latitude, longitude)` order and native spatial-coordinate order retained. The reader opens the file once, prepares time interpolation once and reads only the selected plane or two bracketing planes for each requested field. Every selected grid value must be valid, including values outside a later application sampling stencil. Spatial coordinates must be finite and unique; singleton axes remain present. A nonempty tuple of unit strings in `variables` permits explicit native spellings without numerical conversion. Profile and point-surface callers retain their exact unit-string contract.

Spatial/time extraction is linear interpolation within declared coordinate coverage; ascending and descending coordinates work. Singleton axes require exact selection. Missing/nonfinite bracketing values fail; exact coordinates do not consume an adjacent missing cell. `vertical_interval=(min,max)` optionally selects inclusive native vertical coordinates before validating fields, so missing below-terrain levels can be excluded explicitly. Packed fields apply fill/missing/validity checks before scale and offset. `_Unsigned` reinterpretation is rejected. Time supports seconds/minutes/hours/days since an ISO epoch with `standard`, `gregorian` or `proleptic_gregorian` calendars within the stated modern date domain. Native floating offsets retain their submicrosecond spacing; ISO epoch fractions retain nanosecond digits. Epochs and native dates must be on or after 1582-10-15; other calendars fail. Periodic longitude requires explicit `periodic_longitude=True` and caller-established full global coverage. Regional subsets default to no wrapping or extrapolation. Returned buffers are fresh writable float64 arrays detached from the file; source files must remain stable during reading.

The default `backend="classic"` uses SciPy's public `netcdf_file` API for NetCDF CDF-1/2. `backend="hdf5"` uses optional `h5netcdf.File`/`h5py` for NetCDF4/HDF5 without selecting groups or extra dimensions implicitly. Install `orsia-data[atmosphere]` for HDF5 and GRIB. Orsia resolves named dimensions, selects bracketing columns and applies fill, packed validity, scale and offset locally. The classic reader memory-maps native arrays; successful reads release borrowed variables before closing the file and return detached arrays. A decoding exception can retain borrowed variables in its traceback. SciPy then closes the file descriptor and warns that the mapping remains until those references are released. These operations avoid importing xarray, cftime and netCDF4. Package imports defer optional backends and provider SDKs. Calls are synchronous; callers own native-backend serialization and execution policy. These readers implement the bounded CF subset described above.

`read_gfs_profiles` uses ecCodes directly and requires explicit native short names/units, decreasing `pressure_levels_pa`, cycle, lead and location. It supports deterministic instantaneous GRIB2 `regular_ll` fields on `isobaricInhPa` levels, converts requested hPa identity to Pa and requires every requested field/level exactly once for the requested cycle and validity. Missing/duplicate/ensemble/accumulated/malformed samples fail. Each selected message supplies one horizontal field, native coordinates and an optional bitmap. Orsia maps consecutive-j, westward, alternating-row and wrapped-longitude scans onto rectilinear axes, then samples the requested location. The bitmap identifies missing samples independently of the numeric sentinel. Declared in-band missing-value management, such as GRIB complex packing with `missingValueManagementUsed != 0`, is rejected explicitly. Bitmap-free fields without declared missing coding preserve finite values equal to the sentinel. Every handle is released on success and failure; disk indexes and persistent geometry caches are absent. Rerun generated scan-order tests when upgrading ecCodes. Hybrid/surface GRIB messages, interface inference and general GFS context fields remain unsupported. The Python extra requires a separately installed native ecCodes library. Devbox provisions ecCodes 2.48.0 and optional test recipes supply `ECCODES_DIR`.

## Pressure-level LayerAtmosphere composition

`read_pressure_level_atmosphere` combines offline NetCDF extraction with [pure radiative column materialisation](../../orsia-radiative/docs/columns.md#pure-atmospheric-column-materialisation). Declare native temperature/q variable names, q units, Pa/hPa coordinate units, location/time, surface pressure, target interfaces, gravity and bottom height. The function excludes knots below the supplied terrain surface pressure and requires remaining knots to bracket every target interface. It never manufactures a surface knot. Temperature must be K and q must be dimensionless SI specific humidity, kg water/kg moist air. RH, g/kg and dry-air mixing ratios need explicit conversion upstream. Gas columns must already be materialised on the selected target layers. For native hybrid levels, reconstruct interface pressures and supply them with layer fields to `hydrostatic_atmosphere`.

Convert composition from another product using its own pressure support and air/humidity convention before remapping columns. Resolve product surface pressure, terrain, coverage and time differences explicitly. A CAMS fraction requires its declared dry/moist and mass/mole convention; no missing gas climatology or surface knot is supplied. CAMS aerosol mass profiles can be remapped as mass columns, but require a caller-selected optical model to determine extinction, single-scattering albedo and phase. No CAMS-to-aerosol-optics converter, spectral AOD inversion, cloud or ice optical model is implemented. Wind, screening and prior covariance remain separate from radiative atmosphere construction.

## Verification and limits

From the repository root, run `devbox run -- just --justfile tests/justfile pytest libs/orsia-data/tests/test_atmosphere*.py` and `devbox run -- just --justfile tests/justfile pytest libs/orsia-radiative/tests/test_atmospheric_columns.py` in separate processes. Use `devbox run -- just --justfile tests/justfile test-atmosphere-backends` for optional native checks and `devbox run -- just check` for the full workspace gates. Offline tests use analytic columns, synthetic classic-NetCDF fields, mocked clients/transports/ecCodes boundaries, and generated native GRIB2/HDF5 files. They cover units/packing, reordered axes, spatial/time interpolation, longitude seams, validity/cycle identity, missing support, resource release, no-clobber/failure cleanup, fraction conventions and split/merged/partial-grid conservation. Import tests check deferred backends; generated-file tests check the GIL after reading. Disabled GIL and generated-file replay leave concurrent native-library safety and provider/scientific qualification unresolved.

The former xarray reader imported installed `cftime` 1.6.6.1 and enabled the GIL, even with classic files and Gregorian dates. Direct backend calls remove that import path. Fresh-process tests observe naturally disabled GIL state after classic reads; optional generated-file tests cover HDF5 and ecCodes. These observations leave concurrent native-library safety unqualified. Native ecCodes is provisioned by Devbox because its binary PyPI bundle lacked a wheel for this free-threaded platform. No live ERA5/GFS/CAMS request, authentic meteorological provider-file replay, archive-edition/field qualification, external atmospheric comparison, optical conversion or retrieval evaluation is established here. Public catalogue/endpoint references support request identity only. Provider fallback, campaign policy, automatic caching and persisted acquisition records remain application responsibilities. The separate authentic MIPAS evidence below concerns historical RFM profiles.

## MIPAS 2001 reference atmospheres

Explicit acquisition supports five historical reference atmospheres. The bounded RFM reader preserves native point levels and concentrations for caller-selected calculations. Layer construction, interpolation, integration, composition closure and concentration scaling remain separate scientific operations.

### Acquisition and local reading

```python
from pathlib import Path

from orsia_data.atmosphere.rfm import read_rfm_atmosphere
from orsia_data.services.mipas import download_mipas_profile

# Explicit network operation; choose a fresh destination.
path = download_mipas_profile(
    edition="2001",
    profile="equatorial_day",
    destination=Path("outputs/equatorial-2001.atm"),
)

# Local reading, also usable in a later offline session.
altitude_m, pressure_pa, temperature_k, gas_mole_fractions = read_rfm_atmosphere(path)
water_mol_mol = gas_mole_fractions["H2O"]
co2_mol_mol = gas_mole_fractions["CO2"]
```

Both functions are exported from `orsia_data`; the reader is also exported from `orsia_data.atmosphere`. The return tuple is `(altitude_m, pressure_pa, temperature_k, gas_mole_fractions)`: three vectors and a dictionary mapping uppercase RFM molecular labels to vectors. Each vector has shape `(n_levels,)` and owns an independent writable `numpy.float64` buffer, detached from the file, other vectors and other calls. Decimal text is decoded in double precision before SI conversion. Caller mutations affect only returned buffers and dictionary. Keep the source file stable while reading. The reader closes its file before returning and performs local decoding only.

`download_mipas_profile(*, edition="2001", profile="equatorial_day", destination, expected_sha256=None, timeout_s=30.0, max_bytes=134217728)` returns the destination `Path`. Defaults select the historical equatorial daytime profile; explicit arguments make the choice visible. Supported selections resolve to the Oxford collection:

| Profile | Native file |
| --- | --- |
| `midlatitude_day` | `day.atm` |
| `midlatitude_night` | `ngt.atm` |
| `polar_winter` | `win.atm` |
| `polar_summer` | `sum.atm` |
| `equatorial_day` | `equ.atm` |

The downloader uses `acquisition.download_file` for bounded streaming, optional lowercase SHA-256 verification and atomic no-clobber publication. Existing files and symlinks raise `FileExistsError` and remain intact, including a destination created during transfer. Failures clean temporary files; created parents can remain. Unsupported edition/profile choices raise `ValueError` before filesystem or network access. The timeout applies to network operations. Original bytes are transferred; the separate reader establishes format validity. Transport exceptions propagate. Retain native bytes and caller-recorded digests for offline replay; the provider controls mutable URLs.

### Supported RFM subset

The upstream [RFM atmospheric format](https://eodg.atm.ox.ac.uk/RFM/sum/atmfil.html) defines block-labelled profiles and standard units. This reader supports:

- UTF-8 text, optional blank records and full-line `!` comments between blocks, followed by a positive integer level count. Trailing count-record text is descriptive. One level represents a homogeneous sample.
- Each field starts on its own record with `*LABEL`, followed by exactly the declared count of values across one or more records. Field order is free; blank value records are accepted. Comment records belong outside value blocks.
- Case-insensitive labels separated from descriptive suffixes by whitespace, within the upstream 20-character bound. Unknown or overlong labels raise `ValueError`.
- Required `HGT`, `PRE` and `TEM` fields, each exactly once. Molecular labels must be unique after case normalization. Geometry/thermodynamics-only files return an empty gas dictionary.
- Mandatory `*END`, accepting a descriptive suffix and subsequent blank/comment records. Subsequent data raise `ValueError`.
- Decimal notation, signed `E`/`D` exponents, whitespace and single comma separators. Repetition expressions, empty comma fields, slash termination and inline data comments raise `ValueError`.
- Optional bracketed native units: `[km]` for height, `[mb]`/`[mbar]`/`[hPa]` for pressure, `[K]` for temperature and `[ppmv]` for gases, case-insensitively. Conflicting, malformed or repeated units raise `ValueError`. Parenthesized descriptions after whitespace, such as `*F14 (CF4) [ppmv]`, remain descriptive.

The molecular vocabulary is `N2`, `O2`, `CO2`, `O3`, `H2O`, `CH4`, `N2O`, `HNO3`, `CO`, `NO2`, `N2O5`, `CLO`, `HOCL`, `CLONO2`, `NO`, `HNO4`, `HCN`, `NH3`, `F11`, `F12`, `F14`, `F22`, `CCL4`, `COF2`, `H2O2`, `C2H2`, `C2H6`, `OCS`, `SO2` and `SF6`: the 30 species in the selected collection. `AEROSOL`, isotopologue qualifiers such as `H2O(4)`, vibrational qualifiers such as `CO2(1)(2)`, and additional species raise `ValueError`. Their interpretation, RFM's broader free-format grammar and ignored-field behavior remain unsupported.

Conversions are km times 1000 to m, hPa/mb/mbar times 100 to Pa, and ppmv times 1e-6 to mol/mol; temperature remains K. Gas fractions use total air, including water, as denominator. Native levels and order are preserved. Every decoded value and SI coordinate must be finite. Altitude increases strictly; pressure is positive and decreases strictly; temperature is positive; each gas fraction lies in `[0,1]`. Finite negative altitudes are accepted. Duplicate/missing fields, malformed data, truncation, invalid units and physical-support violations raise `ValueError`; filesystem and Unicode errors propagate.

The sum of tabulated species remains unchanged, including values above one. The format documentation describes a warning for such sums, and all five acquired snapshots contain them. The reader validates individual fractions while preserving composition. Calculations that require composition closure must select and document a treatment explicitly.

### Historical source and replay evidence

The [Oxford collection](https://eodg.atm.ox.ac.uk/RFM/atm/) identifies the MIPAS model atmospheres as the 2001 University of Leicester collection, with 121 levels from 0 to 120 km at 1 km spacing. CO2 values belong to that reference epoch. Collection and format pages were inspected on 2026-10-17; displayed revisions were 2026-02-12 and 2026-01-17 respectively.

The explicit five-file probe on 2026-10-17 recorded these byte identities and sums of all 30 fractions. Every native HGT/PRE/TEM and gas value agreed exactly with an independent block extraction after SI conversion.

| Profile | Bytes | Minimum sum | Maximum sum | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `midlatitude_day` | 42274 | 0.8032301822 | 1.0130309041 | `d3a8d9a664e45e587c336ef64ee7b5c57e8e4f1191b0ce98ce38fc061860464a` |
| `midlatitude_night` | 42276 | 0.8032308822 | 1.0130309038 | `539da9443f6d9d61610922c8b698ef8a7c147488bbcf827d58cf3ae8e1a76b4c` |
| `polar_winter` | 42270 | 0.8034446839 | 1.0043769063 | `5d230b97d33b7101b34beaaa8867ebd6ae01e893afbc3faefc243a7b54f45e1f` |
| `polar_summer` | 42270 | 0.8031835511 | 1.0091588181 | `27c01e6bd5d5469929707feb7cba62df0bc3efd2b54e09a3f16841eb7c2cf7fe` |
| `equatorial_day` | 42270 | 0.8033013288 | 1.0286207135 | `4f366cfc5de2bfde21a82ab27c5c1a3ad9cd2e25ee3296bb5d6109980196a9f6` |

Each file has 80 levels with sums above one. This establishes reader interoperability for those snapshots. Atmospheric accuracy, present-day representativeness, retrieval performance, future availability and concurrent-call qualification require separate evidence. Redistribution rights remain to be established; authentic inputs stay in ignored local storage and committed fixtures are synthetic.

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py` from the repository root for focused offline tests. The live probe/replay command is retired; acquisition and local reading remain public. The historical file identities describe original observations without fresh provider qualification.
