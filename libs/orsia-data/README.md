# Orsia Data

Explicit scientific acquisition, local product readers and geospatial preparation. `orsia-data` depends on the public `orsia-radiative` contracts; radiative never imports data, files or providers. Importing or reading never downloads. Existing provider-specific `fetch_*` functions select, download and decode spectroscopy/solar data; new scene-input families separate `search_*`, `download_*` and `read_*` operations. See [data products and geospatial preparation](../../docs/data-products.md) for product editions, authentication, raster contracts and qualification limits.

## Public API

Named service access now lives in `services/`, alongside the independent `stac/` protocol layer and the local scientific domains. See [service sessions, authentication and STAC](../../docs/data-services.md) for ownership, security, examples and compatibility limits. Existing top-level/domain acquisition imports remain available; new code can import the named service directly. No new SDK or dependency was added for sessions or STAC.

| Module | Public operations | Returned concept |
| --- | --- | --- |
| `acquisition` | `download_file`, `download_bearer_file`, `retrieve_file`, `file_sha256` | Common transfer/publication and byte identity; injected transfers have post-transfer rather than streaming bounds. |
| `services.cdse` | `CdseClient`, `CdseAuth`, `search_sentinel2_l2a`, `download_cdse` | OData catalogue, composed STAC access and explicit CDSE login/refresh. |
| `services.earthdata` | `EarthdataClient`, `EarthdataAuth`, `search_earthdata`, `earthdata_links`, `download_earthdata` | CMR pages and selected file transfers; explicit bearer token or caller-supplied token lifecycle. |
| `stac` | `StacClient`, `read_document`, `select_assets`, `asset_href` | Native documents, bounded API pagination and pure asset inspection; no download or scientific decoding. |
| `providers` | Existing catalogue/download imports | Compatibility facade for the named services and `acquisition.retrieve_file`. |
| `geospatial` | `RasterGrid`, `RasterBand`, `read_raster_band`, `read_raster_grid`, `window_grid`, `window_from_bounds`, `iter_raster_windows`, `crop_raster`, `sample_raster`, `reproject_raster`, `mosaic_rasters` | Explicit affine support, scalar fields, validity and bounded preparation. |
| `surface` | `read_sentinel2_metadata`, `read_sentinel2_l2a`, `read_ecostress_l2t`, `read_aster_ged` | Native-resolution reflectance, quality, thermal fields and emissivity climatology. |
| `services.copernicus_dem`, `terrain` | `download_copernicus_dem`, `read_copernicus_dem` | Service selects/transfers native DSM COGs; terrain reads local EGM2008 heights in metres. Old `terrain.download_copernicus_dem` import remains available. |
| `aerosols` | `read_cams_aod` | Native-wavelength total-column AOD samples, not reconstructed aerosol optics. |
| `services.cds`, `services.ads`, `services.nomads` | `download_era5`, `download_cams`, `download_gfs`, `gfs_valid_time` | Explicit native-file acquisition; old `atmosphere` imports remain available. |
| `atmosphere` | `read_netcdf_profiles`, `read_netcdf_surface_fields`, `read_gfs_profiles` | Native-unit profiles/context, with explicit field/time/support validation. |
| `atmosphere` | `read_pressure_level_atmosphere` | Existing radiative `Atmosphere` from pressure-level NetCDF, explicit terrain/interface support and caller-selected hydrostatic assumptions. |
| `services.mipas` | `download_mipas_profile(edition="2001", profile="equatorial_day", destination=...)` | Native `Path` for one of five historical MIPAS reference profiles. |
| `atmosphere.rfm` | `read_rfm_atmosphere(path)` | Native-level altitude [m], pressure [Pa], temperature [K] and uppercase species fractions [mol/mol total air]. |
| `services.lisird` (also `solar.lisird`) | `fetch_solar_spectrum(product=..., edition="current", directory=..., ...)` | Radiative `SolarIrradianceSpectrum`; experimental LISIRD adapter. |
| `solar.irradiance` | `read_solar_spectrum_csv(path)` | Radiative `SolarIrradianceSpectrum`. |
| `services.hitran` (also `spectroscopy.hitran`) | `fetch_hitran_lines(molecule_id=..., isotopologue_id=..., interval_cm1=..., edition="current", directory=...)` | Radiative `LineDatabase`, acquired through HAPI. |
| `spectroscopy.lines` | `read_hitran_lines(path, molecule_id=..., isotopologue_id=..., interval_cm1=...)`, `read_aer_lines(..., coupling="reject")` | Radiative `LineDatabase` for one molecule/local isotope and declared centre coverage. |
| `spectroscopy.partition_sums` | `read_partition_table(path, molecule_id=..., isotopologue_id=...)` | Radiative `PartitionFunctionTable`. |
| `spectroscopy.continuum` | `download_mt_ckd_water(edition="4.3", destination=..., ...)`, `fetch_mt_ckd_water(edition="4.3", directory=..., temperature_range_k=..., ...)`, `read_mt_ckd_water(path, temperature_range_k=...)` | Local `Path` for download-only; radiative `WaterContinuumReference` for fetch/read. |
| `spectroscopy.cross_sections` | `read_cross_section_table(path, species=..., isotopologue=..., kind=...)` | Radiative `CrossSectionField`. |

Existing function exports from `orsia_data` and their domain namespaces are preserved. The new client/authentication classes are imported from their named `services` namespaces; generic STAC operations are imported from `orsia_data.stac`. No generic scientific file record, manifest, automatic cache or provider registry is present. Atmospheric acquisition accepts explicitly selected native product requests; the caller retains their identity and supplies configured CDS/ADS retrieval methods. Atmospheric readers use internal xarray datasets with explicit SciPy/h5netcdf engines for NetCDF and optional cfgrib over ecCodes for GRIB, without exposing xarray in their public NumPy contracts. See the [atmospheric boundary](../../docs/atmospheric-data.md) for supported formats, optional native dependencies, pure numerical materialisation and qualification limits. The existing solar/spectroscopy fetch functions require scientific selections, not caller-defined column mappings. The separate catalogue/download interfaces retain native pages and explicitly selected product IDs or asset links; see [data products](../../docs/data-products.md).

## Acquisition

The [MIPAS reference-atmosphere guide](../../docs/mipas-reference-atmospheres.md) specifies the five supported 2001 profiles, the bounded RFM grammar, array ownership, SI conversions, explicit download/local-read example and authentic snapshot evidence. The local reader preserves all native levels and concentrations, including independently tabulated species sums above one. The downloader and reader are exported from `orsia_data` as well as their named modules.

`download_file` is an explicit unauthenticated HTTP(S) network operation. It writes a private temporary file beside the destination, checks a positive byte bound (128 MiB by default), rejects empty downloads, checks Content-Length when supplied and optionally verifies a lowercase SHA-256. Atomic no-clobber publication preserves an existing destination, including one created by another writer during transfer. Failure cleans up the temporary file; HTTP failures do not become empty scientific values. Retry starts from scratch.

The primitive's timeout is a network timeout, not a total elapsed-time deadline. Range-resume and archive extraction are not implemented. `download_file` remains unauthenticated; `download_bearer_file` and the new catalogue connectors are documented in [data products](../../docs/data-products.md#authentication-and-publication). Callers of the primitive supply an exact URL and retain licences, editions and any provenance separately; a checksum identifies bytes, not their authenticity or physical completeness. Download success does not imply parseability.

### Scientific acquisition

```python
from pathlib import Path
from orsia_data import fetch_hitran_lines, fetch_solar_spectrum

# Explicit network operations. Each directory must be new; no implicit reuse.
lines = fetch_hitran_lines(
    molecule_id=2, isotopologue_id=1, interval_cm1=(6100.0, 6400.0),
    edition="current", directory=Path("data/co2-lines"),
)
solar = fetch_solar_spectrum(
    product="tsis1_hsrs_p025nm", edition="current",
    directory=Path("data/solar-hsrs"),
)
```

Both functions create an exclusively owned new directory, reject an existing directory before network access, retain native files only on success and remove the new directory on acquisition or decode failure. Parent directories may remain after failure. Do not mutate a call's directory concurrently. Separate directories do not isolate HAPI's shared state; HITRAN execution policy belongs to the caller. Offline replay uses `read_hitran_lines(directory / "lines.data", ...)` or `read_solar_spectrum_csv(directory / "<product>.csv")` and requires neither network access nor HAPI. No resampling, line materialisation, convolution or distance scaling happens during acquisition.

Install `orsia-data[hitran]` for the optional pinned HAPI 1.3.0.0 backend; the PyPI distribution `hitran-api` exposes the import namespace `hapi`. The adapter uses the official [HAPI fetch API](https://github.com/hitranonline/hapi/blob/master/hapi/hapi.py), with one molecule/local isotopologue and an inclusive cm-1 interval per call. It retains HAPI `lines.data` and `lines.header`, rejects returned records outside the requested identity/interval and preserves a successful empty selection. HAPI is imported only when acquisition is requested and runs synchronously in the caller's process/thread, without a subprocess, lock or executor. `db_begin` and `fetch` mutate HAPI's shared module/database/cache state; the selected database remains changed even after filesystem cleanup on failure. Separate directories do not make concurrent HAPI calls safe, including calls outside Orsia. The caller owns serialization, process isolation and cancellation/deadline policy. Missing HAPI raises `ImportError`; upstream exceptions propagate unchanged; invalid decoded data raises the reader's error. HAPI has no total deadline, bounded-byte transfer or completeness verification here. The separate development runtime check observes the GIL before and after the optional import; a disabled GIL does not establish HAPI thread safety.

Both adapters require explicit `edition="current"`, meaning the mutable upstream service, not an archival release. HAPI `fetch` exposes no edition argument. The LISIRD adapter does not infer a selectable edition from SCKLab's `v2` label; other edition values fail before filesystem or network access. For reproducible offline use, retain the native bytes and any application-owned release/digest records separately.

The experimental LISIRD adapter supports the explicitly selected products `tsis1_hsrs` and `tsis1_hsrs_p025nm`, downloading `<product>.csv` from the LASP LaTiS endpoint and decoding only the existing strict solar CSV contract. It forwards the network timeout, byte limit and optional digest to `download_file`. LASP access timed out during this work, so product availability, live schema, edition identity, distance, rights and auxiliary-field meaning remain unverified; a synthetically tested endpoint resolver is not live-provider qualification. See the [HSRS evidence boundary](../../docs/hsrs.md#evidence-boundary). Unsupported provider bytes fail rather than being guessed or silently normalized.

### AER MT_CKD water reference

```python
from pathlib import Path
from orsia_data import download_mt_ckd_water, fetch_mt_ckd_water, read_mt_ckd_water

# Download only: returns a Path; does not require a temperature envelope.
path = download_mt_ckd_water(
    edition="4.3", destination=Path("data/mt-ckd-4.3.nc"),
)
reference = read_mt_ckd_water(path, temperature_range_k=(200.0, 350.0))

# Alternatively, download and decode in one call, retaining the native file.
reference = fetch_mt_ckd_water(
    edition="4.3", directory=Path("data/mt-ckd-water-4.3"),
    temperature_range_k=(200.0, 350.0),
)
```

Both acquisition functions require explicit `edition="4.3"`; no default edition, `current` lookup or other release is implemented. The file is obtained directly from the official [AER-RC/MT_CKD_H2O 4.3 reference](https://github.com/AER-RC/MT_CKD_H2O/blob/4.3/data/absco-ref_wv-mt-ckd.nc), not generated by a Fortran build or extracted from an archive. The [AER release documentation](https://github.com/AER-RC/MT_CKD_H2O/blob/4.3/README.md) describes this water self/foreign coefficient file. The adapter transfers original bytes without conversion or resampling and preserves their embedded copyright/licence. Supporting this edition does not select it for the consuming study or establish model accuracy.

`download_mt_ckd_water` returns the chosen destination and uses `download_file`'s atomic no-clobber publication; it does not parse the file. `fetch_mt_ckd_water` creates a new directory, retains `absco-ref_wv-mt-ckd.nc` on successful decoding and removes the owned directory on transport or reader failure. Existing directories fail before network access; parent directories may remain after failure. Do not mutate a call's directory concurrently. HTTP/transport and reader exceptions propagate. Both functions accept `timeout_s`, `max_bytes` and optional `expected_sha256` with the transport semantics above. The release-tag URL is not a cryptographic identity guarantee: retain the native bytes and supply an independently retained digest when reproducible byte identity is required. Offline replay uses `read_mt_ckd_water`; importing, reading and numerical evaluation never download.

`temperature_range_k` declares the caller's evaluation envelope in kelvins; it does not select a different upstream file or guarantee accuracy over that envelope. Only the existing classic-NetCDF water reader is used; other continua, CIA and HDF5/NetCDF4 conversion are outside this acquisition API.

## Scientific conventions

Solar CSV columns are exactly `wavelength (nm),irradiance (W/m^2/nm)`, optionally followed by `uncertainty (W/m^2/nm)`, then `bandwidth (nm)`. Wavelengths are increasing point centres; irradiance and uncertainty are W m-2 nm-1. Bandwidth [nm] is not sample spacing or a declared integration bin. Arrays are independent writable float64 copies. Missing ancillary fields remain `None`; their meaning is not inferred. The [solar example](../../docs/hsrs.md) demonstrates the physical spectrum → target irradiance → solar source path without manifests. The CSV contract is synthetic/local, not live LASP qualification.

HITRAN reading supports 160-character HITRAN2004+ records. AER/LNFL reading accepts UTF-8 `>` headers and exact `%%%%%%%%` separators and consumes mandatory negative-IFLAG coefficient rows. Selected flagged lines are rejected by default; `coupling="ignore"` explicitly selects independent lines, not full LBLRTM line mixing. Molecule and local-isotope IDs are caller-supplied; no species catalogue excludes N2O/O3. An empty selection returns a valid `LineDatabase` with `lines=()`, while malformed or missing files fail. `interval_cm1` declares centre-coverage completeness supplied by the caller; the reader cannot establish completeness from absent line centres. Read sufficient margin for the later wing/pressure-shift selection.

Partition CSV columns are `molecule_id,isotopologue_id,temperature_k,partition_sum`. `PartitionFunctionTable` contains positive increasing temperature knots [K] and dimensionless Q for one isotope, with independent writable float64 arrays of shape `(n_knots,)`, at least two knots. The reader performs no interpolation. `orsia_radiative.evaluate_partition_function` explicitly evaluates linear Q(T), without extrapolation; this is not HAPI/TIPS interpolation. `orsia_radiative.line_set` consumes `LineDatabase`, `PartitionFunctionTable`, atmosphere and target grid, enforcing coverage beyond cutoff plus maximum pressure shift. It computes Q(296 K)/Q(T_layer), requiring mass [kg/molecule], pressure support [Pa] and finite cutoff [cm-1]. Intensities already include isotope abundance relative to the whole-species column; no second abundance factor is applied. Covered empty materialisation raises radiative `EmptyLineSelection` rather than silently producing transparent absorption.

The MT_CKD water reader accepts only classic NetCDF CDF-1/CDF-2, converts reference pressure mbar → Pa and retains ABSCO units without the radiation term. HDF5/NetCDF4 is rejected; no heavy backend is imported implicitly. Reference nodes retain negative interpolation padding while evaluation points remain positive. `water_continuum_absorption` belongs to radiative, applies temperature/radiation scaling before four-point cubic interpolation and requires complete stencil support. Compose only with H2O local ±25 cm-1 lines with the Lorentz pedestal removed, without renormalisation. Other continua, CIA and CO2 line mixing are not supplied by this reader.

The cross-section reader accepts normalized pickle-free NPZ keys `wavenumbers_cm1`, `temperature_k`, `pressure_pa`, `cross_sections_m2_per_molecule`. Cross sections are m2/molecule with axes `(layer, point)`; layer T/p must match evaluation exactly. This is not arbitrary experimental-format conversion, temperature/pressure interpolation or a continuum model. All returned numerical buffers are independent writable copies, independent of the file and other calls. Raw files must not be mutated concurrently during reading.

## Breaking migration

`SpectroscopyArtifact`, `SolarArtifact`, both `files.py` modules, identify/verify/acquire-artifact functions and manifest readers/writers are removed without compatibility aliases. Existing manifests are application-owned historical records, not accepted reader arguments. Replace `root + artifact` with `path`; perform any external digest verification explicitly before reading. Replace `read_lines` with the named HITRAN or AER reader and `read_cross_section` with `read_cross_section_table`. Replace `read_partition_sums` with a table read followed by radiative evaluation. `read_partition_table` now returns `PartitionFunctionTable`, not bare array pairs.

`SolarIrradianceSpectrum`, `line_set` and `EmptyLineSelection` are owned/exported by `orsia-radiative`, not data. Radiative `LineSet`, `CrossSectionField` and `WaterContinuumReference` no longer accept/store `source_id`; constructor positions and serialized fields therefore change. Data readers retain species, isotope, units, support and physical conditions, not provider identities. See the [radiative contract](../orsia-radiative/README.md). Orsia callers, scripts and documentation are migrated; the study checkout, its environment and parent designs remain unchanged and require separate migration.

`fetch_hitran_lines` no longer accepts `timeout_s` or imposes a child process. Callers must choose their own execution and deadline policy; separate output directories are not a concurrency guarantee. Missing HAPI now raises `ImportError` and upstream exceptions propagate instead of being wrapped as child-process `RuntimeError`. HAPI output is no longer suppressed, and its shared state is not restored.

## Dataset baseline and evidence boundary

Tests use synthetic fixed-width, CSV, NPZ and NetCDF fixtures, analytic strengths/density products and mocked download failures/races. Acquisition tests exercise selection, failure cleanup, no-clobber behaviour, native field retention, offline replay and direct synchronous HAPI execution. `devbox run -- just test-hitran-api` installs the optional extra, checks its imported free-threaded runtime and exercises the real HAPI fetch/parser with synthetic intercepted HTTP bytes, including empty results; it never calls live HITRAN. Software contracts and bounded arithmetic are not authentic dataset coverage, external-solver equivalence, atmospheric or retrieval qualification, or native-backend thread-safety evidence.

The MT_CKD acquisition tests use synthetic NetCDF bytes and intercepted HTTP responses, including byte-limit, checksum, schema, HTTP and interruption failures. A separate live probe of the official 4.3 URL downloaded and decoded 83,576 bytes with 2,003 reference nodes, checked Git blob `ff75b0ed2b7409ab6f5cbaa90b0762e3a03b1903` (SHA-256 `69944eb8b045c268e2daeb2cddf536b99f9e9efe7e91c425067b46a5b287ddb3`), confirmed offline replay and observed the GIL disabled after native imports/reading. This establishes access and reader interoperability for that byte snapshot, not future endpoint availability, scientific accuracy or concurrent-call thread safety. The probe's temporary raw file was removed, not vendored.

The library tests exercise synthetic AER and MT_CKD contracts; the historical authentic-file probe and Fortran comparison are outside this candidate. The consuming study's AER 3.8.1 / MT_CKD 4.3 choices are separate from library defaults. Publish authentic data only under applicable provider terms.
