# Data products and geospatial preparation

`orsia-data` separates explicit discovery/acquisition, local product decoding and reusable geospatial preparation. Applications select products, editions, dates, fields, quality acceptance and scene composition; `orsia-radiative` owns physical reconstruction. Importing or reading never downloads. This document describes the bounded implemented contracts, not universal provider compatibility or a scientifically qualified scene.

## Families and access

Named service modules, composed CDSE/Earthdata clients and the reusable STAC layer are described in [data services and STAC](data-services.md). The existing function APIs below remain available. Their low-level token arguments do not gain implicit login; the optional client/authentication objects provide the separate session lifecycle.

| Family | Discovery and acquisition | Local decoding | Supported boundary |
| --- | --- | --- | --- |
| Meteorology and gas composition | Existing `download_era5`, `download_cams`, `download_gfs` | Existing NetCDF/GRIB readers and `read_pressure_level_atmosphere` | See [atmospheric data](atmospheric-data.md); CDS/ADS clients and native requests remain caller-owned. |
| Aerosols | `download_cams(product="cams-global-atmospheric-composition-forecasts", ...)` | `read_cams_aod` | Total-column AOD at selected native 469, 550, 670, 865 and 1240 nm; no mass-to-optics, vertical distribution, SSA or phase reconstruction. |
| Reflective surface | `search_sentinel2_l2a`, `download_cdse` | `read_sentinel2_metadata`, `read_sentinel2_l2a` | Extracted SAFE L2A metadata and native JP2 bands at an explicitly selected granule/resolution; no archive extraction or silent resampling. |
| Terrain | `download_copernicus_dem` | `read_copernicus_dem` | Public AWS GLO-30/GLO-90 COG tiles; DSM heights in metres above EGM2008, not bare earth or ellipsoidal height. |
| Thermal surface | `search_earthdata(product="ECO_L2T_LSTE.002", ...)`, `download_earthdata` | `read_ecostress_l2t` | Native `ECOv002_L2T_LSTE_..._<layer>.tif`: physical LST/LST_err, broadband EmisWB, height, view_zenith and categorical QC/cloud/water. Not packed swath or L2G HDF5. |
| Emissivity climatology | `search_earthdata(product="AG100.003", ...)`, `download_earthdata` | `read_aster_ged` | Native AG100.v003 HDF5: five ASTER band emissivities and standard deviations, plus temperature mean/standard deviation. |
| Solar and spectroscopy | Existing explicit `fetch_*` and local readers | Unchanged | See the [data package API](../libs/orsia-data/README.md). |

The connectors select explicit supported products; their availability, rights and quality remain provider responsibilities. Neither a product name nor a checksum authenticates scientific contents. The public DEM distribution is mutable and has no edition selector in this API. Retain native files and application-owned query/version/digest records when reproducibility requires them. A missing tile, unsupported file layout or failed request is an error, never a fallback to another provider, date, sea-level altitude or reference atmosphere. GASCON reference-file decoding is not added: no concrete local schema or licensed fixture was selected for this increment.

### Search pages

`search_sentinel2_l2a` queries CDSE OData with a non-wrapping WGS84 `bbox=(west,south,east,north)` and acquisition-start interval `[start,end)`. Instants must have explicit UTC offsets. It returns one native JSON page, including `value`, attributes, count and any pagination metadata. `top` and `skip` are explicit pagination controls; an empty `value` is a successful empty selection. There is no implicit cloud threshold or best-granule ranking.

`search_earthdata` queries CMR `granules.json` with a short name and version derived from the explicit product argument. It retains the provider's temporal-overlap semantics, not CDSE acquisition-start semantics. `temporal=None` explicitly omits temporal filtering, which is useful for a static climatology. `page_number` and `page_size` select one page; callers own traversal and should not assume pages form an immutable catalogue snapshot. `earthdata_links(entry)` returns deduplicated HTTPS data links excluding browse, metadata and inherited collection links. It does not choose a granule or distinguish interchangeable assets: ECOSTRESS layers remain separate files.

Search responses preserve native metadata rather than introducing a universal product record. Both operations bound each response to 16 MiB and use an explicit network timeout. HTTP, JSON and malformed-page failures propagate; an empty search is not an error. No search creates a directory or downloads scientific product bytes.

### Authentication and publication

`download_cdse` accepts a selected product UUID and either an explicit access token or a caller-owned `retrieve(url, path)` operation. The token path uses bounded native HTTPS transport and sends credentials only to the exact `download.dataspace.copernicus.eu` and `zipper.dataspace.copernicus.eu` hosts. `download_earthdata` accepts a selected data link and either an explicit Earthdata token plus `trusted_hosts`, or a configured transfer. It never discovers `.netrc`, reads environment credentials, logs in or refreshes a token. Supplying both authentication paths, or neither, fails before network access. An authenticated account is not needed for public catalogue searches or public DEM acquisition.

`download_bearer_file` is the shared lower-level transport. Its initial URL must be trusted HTTPS on port 443. HTTPS redirects may leave the trusted host set, for example to a signed object URL, but the Authorization header is removed unless the next host is explicitly trusted. Downgrades, embedded URL credentials and nonstandard ports are rejected. Trust is exact hostname matching, never wildcard or suffix matching. The caller selects the trust set and supplies the token in memory; do not persist tokens in URLs, examples or provenance.

Native HTTP downloads enforce `max_bytes` during transfer, optionally check SHA-256, reject incomplete/empty responses and publish without replacing an existing destination. Injected transfers instead write synchronously to a private temporary file, with size checked **after** the client returns; the client owns streaming limits, redirects, retries and deadlines. Both modes use atomic no-clobber publication and remove their private temporary files on failure or interruption. Parent directories may remain. Do not mutate a destination or client state concurrently. Transport timeouts are not whole-operation deadlines; SDK/network execution and concurrent-call safety remain caller responsibilities.

```python
from datetime import UTC, datetime
from pathlib import Path
from orsia_data import search_sentinel2_l2a, download_cdse

# Explicit network search, not a selected study scene.
page = search_sentinel2_l2a(
    bbox=(2.0, 48.0, 3.0, 49.0),
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 1, 8, tzinfo=UTC),
    top=100, skip=0,
)
# Application inspects candidates and obtains an authenticated token separately.
# archive = download_cdse(selected_uuid, Path("native/product.zip"),
#     access_token=token, max_bytes=2 * 1024**3)
# Archive extraction is an explicit caller step, outside the reader.
```

CDS/ADS continue to use `download_era5(..., retrieve=client.retrieve)` and `download_cams(..., retrieve=client.retrieve)`. There is no new client registry, generic manifest, implicit cache, provider fallback or campaign executor.

### DEM tiles for a geographic rectangle

Pass `bbox=(west,south,east,north)` in WGS84 degrees and a destination `directory` to `download_copernicus_dem`. It returns `list[Path]`, even when only one tile is selected, with native provider filenames. The rectangle is not a polygon or a projected-CRS geometry. `west > east` explicitly crosses the antimeridian; equal west/east and zero-height rectangles are rejected. Results are ordered south to north, then west to east through the rectangle, wrapping at 180 degrees without duplicate tiles.

```python
from pathlib import Path
from orsia_data import download_copernicus_dem, read_copernicus_dem

paths = download_copernicus_dem(
    bbox=(2.8, 48.8, 3.2, 49.2),  # west, south, east, north in degrees
    resolution_m=90,
    directory=Path("data/dem"),
)
# Four native files: N48/E002, N48/E003, N49/E002, N49/E003.
# Reading is separate and does not merge or crop them.
first_tile = read_copernicus_dem(paths[0])
```

Selection intersects the nominal one-degree tile cells with positive area: an exact integer north/east boundary does not add its neighbour. This is tile-grid coverage, not a guarantee of native pixel-edge coverage or valid heights: COG pixel edges are shifted relative to the nominal cells, and nodata remain possible. Use stored transforms and validity masks for subsequent support checks; this API does not certify that a downstream sampling/interpolation stencil is covered. No catalogue query, availability filtering, polygon clipping, mosaic or resampling is implicit.

`max_tiles=100` bounds the number selected before any filesystem or network effects; callers can explicitly raise it. `timeout_s` and `max_bytes` apply per file, not to the whole batch. Downloads run sequentially. Existing directories are allowed, but any existing selected target (including a symlink) causes `FileExistsError` before the first transfer; files are never silently reused or overwritten. A missing provider tile or other transfer failure propagates without returning a partial list. Previously completed files remain on disk, while the failing transfer's temporary file is cleaned up by the transport. The batch is not transactional; retry does not automatically resume or remove earlier files.

The original `south_deg=..., west_deg=..., destination=...` call still returns one `Path` and accepts `expected_sha256`. Do not mix it with `bbox`/`directory`. Bbox mode rejects a shared `expected_sha256`, since each native file has a distinct byte identity; retain or verify per-file digests separately if needed.

## Decoding and scientific support

### Sentinel-2 L2A

`read_sentinel2_l2a(safe, granule=..., bands=("B11", "SCL"), resolution_m=20, window=...)` reads only the requested native files identified by `MTD_MSIL2A.xml`. The resolution argument selects a native product representation, not an output resampling request; unavailable combinations fail. SAFE paths cannot escape the local product directory, including via symlinks. XML is bounded and DTD/entity declarations are rejected.

BOA reflectance is `(DN + band_offset) / quantification_value`, with scales taken from metadata. DN zero is absent before offset application; negative decoded reflectance is retained, not clipped. Baselines before 04.00 use zero offset only when the offset list is absent; later baselines require each requested band's offset. SCL remains categorical and class zero is absent. A cloud class is a valid decoded class, not an automatically accepted radiative surface. No BRDF assumption, cloud acceptance, spectral synthesis or hyperspectral interpolation is embedded in reading.

`read_sentinel2_metadata` exposes product URI, type, processing baseline and native start/stop strings. It does not infer per-pixel acquisition times or solar/view geometry not present in this bounded interface. Bands remain separate `RasterBand` values; differing native supports require an explicit preparation operation.

### ECOSTRESS and ASTER

The supported ECOSTRESS tiled v002 product stores physical float values; the reader does **not** apply packed-swath scale factors. LST and LST_err are kelvins, EmisWB is dimensionless broadband emissivity, height is metres and view_zenith is degrees. Invalid physical values are masked separately from native QC flags. Cloud, water and QC codes are retained without acceptance filtering. LST_err is the supplied error layer, not an inferred covariance. ECOSTRESS's selected CMR granule/sidecar retains its time and production metadata; the raster itself is not declared contemporaneous with another source.

The ASTER reader accepts native `AG100.v003.<north>.<west>.0001.h5` names and a bounded native layout: `Geolocation/Latitude`, `Geolocation/Longitude`, `Emissivity/Mean`, `Emissivity/SDev`, `Temperature/Mean` and `Temperature/SDev`. Emissivity fields have axes `(band,row,column)` with five bands numbered 10 through 14; other fields have `(row,column)`. A full tile is 1000 by 1000 cells at 0.001 degrees, with native geolocation checked against the filename-derived centre grid. Unsupported layouts fail instead of guessing axes. The schema and geolocation convention still require replay against authentic AG100 files; synthetic HDF5 fixtures alone do not establish supplier interoperability.

Requested ASTER fields use names `emissivity_10` through `emissivity_14`, `emissivity_sdev_10` through `emissivity_sdev_14`, `temperature` and `temperature_sdev`. Fill values are masked before scaling. Emissivity means use 0.001, emissivity standard deviations 0.0001, and temperature means/standard deviations 0.01 K per stored unit. Climatological variability is not per-observation measurement uncertainty; band emissivity is not a high-resolution spectral measurement. Temperature ECOSTRESS plus emissivity ASTER is an application composition, not a mandatory library concept.

ASTER observation counts and land-water datasets are not decoded by this bounded reader; their exact HDF5 paths/layouts need authentic-file verification rather than guessed group names. The emissivity/temperature paths and scales follow section 3.3 and table 2 of the official ASTER GED v3 guide.

### CAMS AOD and terrain

`read_cams_aod` delegates CF packing, exact coordinate validation and bounded linear time/space sampling to the existing NetCDF reader. Callers name the actual time/latitude/longitude coordinate variables and select native wavelengths. The file must contain corresponding `aod469`, `aod550`, `aod670`, `aod865` or `aod1240` fields with dimensionless units `1`. Unsupported cycle/lead/ensemble dimensions are not silently squeezed. No aerosol physical state is inferred from total-column AOD.

Copernicus DEM reading preserves native COG georeferencing and nodata. A valid zero altitude is not missing coverage. The fixed contract is a WGS84 horizontal DSM with EGM2008 orthometric heights, including vegetation/buildings; no vertical datum transformation or conversion to bare earth occurs. Tile coordinates identify provider keys, not exact pixel edges. Use the stored transform rather than constructing one from nominal 30/90 m labels.

## Geospatial contracts

`RasterGrid` represents only a two-dimensional horizontal affine grid: a direct `pyproj.CRS`, `(rows,columns)` and the six coefficients `(a,b,c,d,e,f)` mapping pixel edges by `x=a*column+b*row+c`, `y=d*column+e*row+f`. Centres use half-integers. `RasterBand` couples a scalar field with that grid, units, decoded-validity mask and categorical semantics. It is not a scene, response or generic artifact. Keeping support beside values prevents accidental grid/array mismatch across readers, displays, storage and resampling.

Values are float64 and validity is boolean, both shaped `(row,column)`. Valid values must be finite; categorical values must be integer codes. Constructors retain caller buffers without copying, and frozen records do not freeze array contents. Reader and preparation outputs are fresh writable buffers detached from files and input arrays. Inputs/files must not be mutated concurrently. Missing values use invalid masks and NaN outputs; quality acceptance remains a separate application operation.

| Operation | Contract |
| --- | --- |
| `read_raster_band(path, units=..., band=1, window=...)` | Existing local GeoTIFF/boxed JPEG2000 only; raw samples and file nodata, no scale/offset interpretation. Explicit one-based band; no remote GDAL paths or VRT execution. |
| `read_raster_grid(path)` | Affine support without materialising pixel arrays; the same local format restriction as the raster reader. |
| `window_from_bounds(grid, bounds, clip=...)` | Enclosing integer window for a rectangle in the source CRS; explicit clipping or rejection outside support, including rotated grids. |
| `iter_raster_windows(grid, block_shape)` | Disjoint row-major windows, clipping final blocks to the grid. |
| `window_grid(grid, window)` | Grid support for `(row_start,column_start,height,width)` wholly inside the source. |
| `crop_raster(band, window)` | Fresh data/mask copies with updated affine transform. |
| `sample_raster(band, x, y, method=...)` | Positions in the source CRS; nearest uses pixel footprints, bilinear uses centres and requires every nonzero-weight sample to be valid. No extrapolation or gap filling. |
| `reproject_raster(band, target, method=..., minimum_coverage=...)` | Explicit target grid and nearest/bilinear method; returns a raster plus kernel-valid weight in [0,1]. Categorical rasters require nearest. |
| `mosaic_rasters(bands, target, method=..., overlap=..., minimum_coverage=...)` | Explicit source order and `first`/`last` valid-pixel precedence, with matching units/categorical semantics. |

Reprojection coverage is the interpolation kernel's valid weight, **not** geometric area overlap, scientific quality or uncertainty. Missing values contribute zero numerator and zero weight. Retained values are normalized by valid weight; `minimum_coverage` in `(0,1]` explicitly chooses whether to accept partially supported kernels. Rasterio/GDAL define warp-kernel edge behaviour, which is distinct from the strict centre-stencil sampling function. No uncertainty layer is silently propagated as an independent standard deviation: callers must choose the scientific propagation model.

All readers accept bounded windows where applicable. Warp memory scales with supplied source/target windows, and mosaicking retains one warped source at a time. Large-scene orchestration should supply windows rather than ask for a full-scene materialisation. Native calls run synchronously, warp threads are set to one, and handles close within each operation. No Dask dependency or hidden executor is introduced. Disabled GIL observations do not establish concurrent GDAL/PROJ/HDF5 safety or performance.

## Dependencies and verification

Install `orsia-data[geospatial]` for Rasterio, pyproj and h5py. These backends are imported only by operations that need them, not by `import orsia_data`. The base package and existing atmospheric/spectroscopy interfaces remain usable without that extra; the library dependency direction remains application → data → public radiative contracts, never radiative → data.

```bash
devbox run -- just test-data
devbox run -- just typecheck-data
devbox run -- just clean-install-data
devbox run -- just check
```

Offline tests generate native-format JP2, GeoTIFF, HDF5 and NetCDF fixtures and intercept HTTP transfer bytes. They exercise offsets/scales, nodata versus quality, support checks, CRS changes, edge sampling, mosaics, failure cleanup, races, token redirect policy and lazy imports. The independent installation check builds only the data/radiative/numerics dependency chain, installs `orsia-data[geospatial]` into a fresh environment and replays the new contracts outside workspace resolution. The full check includes raster-backed tests in its library coverage gate.

The one-off public catalogue/DEM probe has been retired; the public query, download and reader APIs and their offline regressions remain. Its recorded observation below used a fixed test rectangle and did not select a study scene. Catalogue connectivity, authenticated downloads, authentic file decoding, scientific scene suitability and concurrency qualification are distinct evidence levels. Protected Sentinel-2/ECOSTRESS/ASTER file replay and real authenticated CDS/ADS/CDSE/Earthdata transfers remain unqualified unless separately observed; no credentials or proprietary reference-project code are bundled.

On October 8, 2026, the explicit public probe returned candidates from CDSE, CMR AG100.003 and CMR ECO_L2T_LSTE.002. It downloaded the public Copernicus GLO-90 N48/E002 COG (5,423,876 bytes; SHA-256 `8ddd564249bac5462a82a0305a0fe2bbbd987fae563271a55b3392d80f1385b6`), decoded an 8-by-8 window with 64 valid heights, observed the GIL disabled after reading and removed the temporary file. This establishes catalogue query interoperability and one authentic DEM byte snapshot, not authenticated-product access, future service availability or scientific suitability.

## Provider references

The implementation was checked against the CDSE OData documentation, Sentinel-2 product conventions, NASA product variable tables and the public DEM distribution documentation. These sources describe formats/services, not validation of the generated test fixtures.

- CDSE catalogue and download API: `https://documentation.dataspace.copernicus.eu/APIs/OData.html`.
- Sentinel-2 L2A product conventions: `https://sentiwiki.copernicus.eu/web/s2-products`.
- NASA CMR search API: `https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html`.
- ECOSTRESS L2T v002 product/variable table: `https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l2t-lste-002`.
- ASTER AG100 v003 product/variable table: `https://www.earthdata.nasa.gov/data/catalog/lpcloud-ag100-003`.
- ASTER GED v3 guide (native HDF5 paths, band order and scales): `https://lpdaac.usgs.gov/documents/120/ASTERGED_User_Guide_V3.pdf`.
- Copernicus public DEM COG distribution: `https://registry.opendata.aws/copernicus-dem/`.
- Copernicus COG provider grid, filenames and removed-edge conventions: `https://copernicus-dem-30m.s3.amazonaws.com/readme.html` (checked October 8, 2026).

The parent design still contains historical generic artifact/manifest requirements. This Orsia-scoped implementation follows the accepted explicit data boundary and does not reinstate those records; parent/study files are unchanged.
