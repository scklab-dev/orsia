# Data services and STAC

`orsia-data` exposes discovery and dissemination under named services, keeps STAC independent of those services, and leaves local scientific interpretation in the existing domain readers. Clients own only network/session policy; they do not choose a study scene, convert an asset into a physical model, impose a cache or launch an executor.

## Implemented source boundaries

| Source | Responsibility | Important boundary |
| --- | --- | --- |
| `services/cdse/` (`client.py`, `auth.py`, `odata.py`, `stac.py`) | `CdseClient`, `CdseAuth`, OData discovery/download and composed STAC access | OData download authentication is not forwarded to STAC assets. |
| `services/earthdata/` (`client.py`, `auth.py`, `cmr.py`) | `EarthdataClient`, `EarthdataAuth`, CMR discovery and selected downloads | The caller supplies tokens and explicitly trusted download hosts. |
| `services/cds.py`, `services/ads.py` | ERA5 and CAMS acquisition | Borrow a configured `retrieve` callable; do not construct an SDK client. |
| `services/nomads.py`, `services/copernicus_dem.py` | GFS subsets and public DEM tile acquisition | Caller-selected requests and destinations. |
| `services/lisird.py`, `services/hitran.py` | Existing solar and spectroscopy acquisition adapters | Explicit `fetch_*` operations may also decode; HAPI is optional. |
| `services/mipas.py`, `atmosphere/rfm.py` | Historical MIPAS 2001 reference-file acquisition and separate local RFM decoding | Native levels and concentrations in SI arrays; see the [contract and example](atmosphere.md#mipas-2001-reference-atmospheres). |
| `services/catalogue.py` | Shared bbox and UTC query conventions | Query preparation, not scientific interpretation. |
| `stac/catalog.py` | Read bounded local STAC JSON documents | Envelope checks, not complete STAC/extension schema validation. |
| `stac/client.py` | `StacClient`, GET/POST search and bounded pagination | Knows no named service; preserves native page dictionaries. |
| `stac/assets.py` | Filter assets and resolve a selected href | Pure inspection; no transfer or automatic band selection. |
| `http.py` | `JsonSession`, bounded JSON transport | Native requests use HTTPS and refuse redirects and environment proxies. |
| `acquisition.py` | Common file transfer, size/digest checks and publication | Does not decode products; existing destinations are not overwritten. |
| `atmosphere/`, `surface/`, `terrain.py`, `aerosols.py` | Local atmospheric, reflective, thermal, elevation and AOD inputs | Supported formats and physical support remain explicit. |
| `solar/`, `spectroscopy/` | Local irradiance, lines, partition tables, continua and cross sections | Scientific arrays/contracts, not catalogue metadata. |
| `geospatial.py` | Local raster IO, windows, sampling, reprojection and mosaics | Explicit grids and validity; no service selection. |

`stac/`, `http.py` and `acquisition.py` do not import a named service or scientific reader. CDSE composes the generic STAC client. Plain dictionaries preserve native metadata, extension fields, identifiers and links as fresh mutable outputs. Local interpretation remains separate from network/session policy.

LISIRD/HITRAN `fetch_*` functions explicitly acquire and decode a caller-selected product. MT_CKD download/fetch functions live in `spectroscopy.continuum`. Local reader functions do not accept or consult service sessions. Same-library root exports provide convenient access to readers and acquisition functions; named acquisition implementations live in their service modules. Reading never invokes acquisition.

## Resource and authentication ownership

`CdseClient`, `EarthdataClient`, `StacClient`, `JsonSession`, `CdseAuth` and `EarthdataAuth` support `with` and `close()`. Constructors do not contact a service. Clients own sessions they create, but borrow explicitly supplied sessions and authentication objects. Closing a client invalidates it and closes only its owned resources; `CdseClient.close()` also invalidates its `stac` view. Close borrowed authentication explicitly or give it its own `with` block. Calls are synchronous; these mutable clients are not safe for concurrent use, regardless of the GIL state.

The native `JsonSession` uses stdlib urllib without cookies, environment proxies, automatic credential lookup, retries or connection pooling. Responses close after each request. JSON pages are bounded to 16 MiB by default, with a positive finite per-operation timeout; neither this timeout nor a page count is a total elapsed-time deadline. An injected `open_url` is borrowed and must enforce the same redirect/security policy. Native JSON requests require HTTPS on port 443 and refuse redirects rather than forwarding tokens or POST bodies. Bearer callbacks run only for explicitly trusted hostnames. Link-supplied headers are restricted to `Accept` and `Content-Type`, without newline injection; other headers fail rather than silently weakening the policy.

### CDSE

`CdseAuth.login(username, password, totp=...)` performs one explicit password grant, optionally with a supplied 2FA code. It does not retain the password, read environment variables or inspect credential files. Access and refresh tokens remain in memory; their representations omit secrets. Authentication failures are reported without upstream response bodies. `CdseAuth(access_token, expires_in=..., refresh_token=...)` also accepts an explicitly obtained token, with remaining lifetime in seconds measured against a monotonic clock. Refresh is attempted once when the token expires or enters the refresh margin, using the current refresh token and accepting rotation. It does not silently re-login or retry a failed authenticated download. Expired refresh credentials require a new explicit login.

Use the object API when Orsia should manage CDSE token reuse and expiry. `client.stac` is public discovery; it does not assume that STAC Item IDs are OData UUIDs and does not send the OData download token to asset links. `download` requires an explicitly selected OData UUID and `max_bytes`; it delegates native-byte publication and trusted-host routing to the existing transfer layer.

```python
from getpass import getpass
from pathlib import Path
from orsia_data.services.cdse import CdseAuth, CdseClient

# Explicit interactive credentials; neither variable is persisted by Orsia.
# Running this example logs in and makes a real download.
username = input("CDSE username: ")
selected_uuid = input("Selected OData product UUID: ")
with CdseAuth.login(username, getpass("CDSE password: ")) as auth:
    with CdseClient(auth=auth) as client:
        archive = client.download(
            selected_uuid, Path("native/product.zip"), max_bytes=2 * 1024**3,
        )
# Archive extraction and local scientific reading remain separate operations.
```

The low-level `download_cdse(..., access_token=...)` contract remains available and does not gain implicit login or refresh. Token deletion on close means dropping local references, not cryptographic memory erasure or remote revocation. No authenticated live request was used to validate this implementation.

### Earthdata, CDS and ADS

`EarthdataAuth(token)` holds an explicit Earthdata bearer token. `EarthdataAuth(token_supplier)` borrows a callable invoked before each download, so an application can supply its configured token-renewal lifecycle. The supplier owns its resources and expiry decisions. Orsia has no generic NASA refresh endpoint, URS password login or `.netrc` lookup. `EarthdataClient(auth=auth, trusted_hosts=(...))` keeps CMR discovery separate from authenticated dissemination. Returned links do not extend the trust list; an untrusted initial download hostname fails before file transfer. Public search needs no authentication object.

`services.cds.download_era5` and `services.ads.download_cams` retain their caller-configured `retrieve=client.retrieve` boundary. SDK credentials, session lifetime, deadlines, retry policy and concurrency remain with that client. The native request is deep-copied before handing it to the SDK, and file size/checksum checks happen after the transfer. The supplied SDK client remains the resource boundary.

## STAC without a named service

`read_document(Path(...))` reads a bounded local Catalog, Collection, Item or ItemCollection without following links. It checks the supported document envelope, not the complete STAC or extension schema, and preserves unknown fields. `select_assets(item, role=..., media_type=...)` returns independent copies of every matching asset; it does not rank them or infer bands. `asset_href(item, key, document_url=...)` resolves one explicitly selected href against the actual source document URI, without opening it or granting credential trust. Local and non-HTTP asset schemes are not automatically converted into download requests.

### Inspect local metadata without downloading

Run this standalone example in the [development environment](../../../README.md#development-environment). It uses a temporary file without credentials or network requests.

This synthetic Item contains two assets. Filter by role, then resolve a relative href against the document URI. The referenced CSV is deliberately absent.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orsia_data.stac import asset_href, read_document, select_assets

with TemporaryDirectory() as directory:
    path = Path(directory).resolve() / "item.json"
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "teaching-item",
                "geometry": None,
                "properties": {"datetime": "2025-01-01T00:00:00Z"},
                "links": [],
                "assets": {
                    "spectrum": {
                        "href": "./solar.csv",
                        "type": "text/csv",
                        "roles": ["data"],
                    },
                    "preview": {
                        "href": "./preview.png",
                        "type": "image/png",
                        "roles": ["thumbnail"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    item = read_document(path)
    candidates = select_assets(item, role="data", media_type="text/csv")
    assert list(candidates) == ["spectrum"]
    href = asset_href(item, "spectrum", document_url=path.as_uri())
    assert href == (path.parent / "solar.csv").as_uri()
    assert not (path.parent / "solar.csv").exists()
    print(list(candidates))  # ['spectrum']
```

### Search and pagination

`StacClient(url)` uses the supplied API landing page. `landing_page()` returns native capabilities and links; `search(parameters, method="GET")` returns exactly one native ItemCollection. `pages(parameters, method="POST", max_pages=N)` explicitly traverses up to N pages, with no prefetch. Search follows the advertised method-specific `search` link instead of guessing `/search`. GET list parameters become comma-separated values, object-valued parameters become JSON, and POST preserves JSON bodies. Parameters and pagination state are copied so caller mutation does not rewrite a pending continuation.

Pagination follows relative or absolute `rel="next"` links, honors GET/POST, body replacement and `merge` with the original search body, rejects repeated requests and ambiguous next links, and never expands token trust. Empty pages are valid and do not terminate traversal if a next link exists. The page cap truncates intentionally and does not establish catalogue completeness or snapshot consistency. Unsupported link headers, methods, redirects and non-HTTPS API URLs raise rather than being ignored. The implementation does not claim support for every STAC extension or custom header-based pagination scheme.

```python
from orsia_data.services.cdse import CdseClient

# Explicit public network search. Collections, times and spatial filters belong
# in the caller's parameters; no mission/product choice is made by this client.
with CdseClient() as cdse:
    capabilities = cdse.stac.landing_page()
    for page in cdse.stac.pages({"limit": 10}, method="GET", max_pages=2):
        candidates = page["features"]
        # Application inspects candidates. No assets are downloaded here.
```

## Compatibility and verification

Keep discovery, selection, transfer, extraction and scientific reading explicitly callable. Local readers may construct public radiative values; radiative never imports data. Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-data/tests` from the workspace root after changing a service or reader.

Compatibility imports include top-level functions, `providers`, `atmosphere.providers`, `solar.lisird`, `spectroscopy.hitran` and `terrain.download_copernicus_dem`. They preserve scientific selections, outputs and file-publication semantics. Internal monkeypatch locations are outside the public contract; tests intercept named services or shared transport. CDSE/CMR search functions accept a borrowed `session`; its timeout/byte/security policy overrides the standalone timeout. Native catalogue requests fail on redirects, preventing implicit redirection of credential-bearing sessions and POST bodies.

Offline tests exercise native GET/POST pages, relative links, extension retention, empty results, page bounds, loop rejection, token rotation/expiry, owned versus borrowed closure, redacted representations, credential confinement and named-service composition. Optional HAPI tests still use intercepted synthetic HTTP bytes. These checks do not qualify live authenticated services, all provider formats, catalogue completeness, scientific retrieval performance or concurrent-native safety.

Use this guide, the [package API](../README.md) and [data products](products.md) for the implemented boundary.

## Protocol evidence

The implementation was checked against the primary [STAC API Item Search specification](https://github.com/radiantearth/stac-api-spec/blob/main/item-search/README.md), including method-specific search links, ItemCollection responses and GET/POST pagination semantics, and the official CDSE [STAC endpoint documentation](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) and [token documentation](https://documentation.dataspace.copernicus.eu/APIs/Token.html), accessed on 8 October 2026. CDSE documents the `stac.dataspace.copernicus.eu/v1/` endpoint and password/2FA token generation with refresh or explicit re-login after expiry. Reading documentation is not a live-provider qualification; no account credentials were used during this work.
