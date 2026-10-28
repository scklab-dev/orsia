# Data services and STAC

`orsia-data` exposes discovery and dissemination under named services, keeps STAC independent of those services, and leaves local scientific interpretation in the existing domain readers. Clients own only network/session policy; they do not choose a study scene, convert an asset into a physical model, impose a cache or launch an executor.

## Implemented source boundaries

```text
orsia_data/
├── services/
│   ├── cdse/                 # client.py, auth.py, odata.py, stac.py
│   ├── earthdata/            # client.py, auth.py, cmr.py
│   ├── cds.py                # ERA5 via an injected configured retrieval method
│   ├── ads.py                # CAMS via an injected configured retrieval method
│   ├── nomads.py             # Explicit GFS subset transport
│   ├── copernicus_dem.py     # Public tile selection and transfer
│   ├── lisird.py             # Existing solar acquisition adapter
│   ├── hitran.py             # Existing optional HAPI adapter
│   └── catalogue.py          # Shared bbox and UTC query conventions
├── stac/
│   ├── catalog.py            # Local native JSON documents
│   ├── client.py             # Search and bounded native-page traversal
│   └── assets.py             # Pure inspection/URI resolution
├── http.py                   # Bounded JSON HTTP session
├── acquisition.py            # Common native-byte transfer/publication
├── atmosphere/               # Local profiles and physical materialisation
├── surface/                  # Local reflective/thermal products
├── terrain.py                # Local DSM reading; legacy download import
├── solar/                    # Local solar reading; legacy acquisition imports
├── spectroscopy/             # Local spectroscopy; legacy acquisition imports
└── geospatial.py             # Local raster IO and spatial preparation
```

`stac/`, `http.py` and `acquisition.py` do not import a named service or scientific reader; the boundary checker enforces that direction. CDSE composes the generic STAC client rather than subclassing it. No `BaseProvider`, service registry, new data-record hierarchy or external client SDK is introduced. Plain dictionaries preserve native metadata, extension fields, identifiers and links; they are fresh mutable outputs, not immutable scientific records.

The existing LISIRD/HITRAN `fetch_*` functions still explicitly acquire and decode a caller-selected product; moving them does not silently change their return or cleanup contracts. The existing MT_CKD download/fetch functions in `spectroscopy.continuum` are unchanged by this migration. Local reader functions do not accept or consult service sessions. Compatibility imports may expose an acquisition function beside a reader, but reading itself never invokes acquisition.

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

`EarthdataAuth(token)` holds an explicit Earthdata bearer token. `EarthdataAuth(token_supplier)` borrows a callable invoked before each download, so an application can provide its configured token-renewal lifecycle without passing tokens at every call. The supplier owns its resources and expiry decisions; Orsia does not invent a generic NASA refresh endpoint, perform URS password login or read `.netrc`. `EarthdataClient(auth=auth, trusted_hosts=(...))` keeps CMR discovery separate from authenticated dissemination. Returned links do not extend the trust list; an untrusted initial download hostname fails before file transfer. Public search needs no authentication object.

`services.cds.download_era5` and `services.ads.download_cams` retain their caller-configured `retrieve=client.retrieve` boundary. SDK credentials, session lifetime, deadlines, retry policy and concurrency remain with that client. The native request is deep-copied before handing it to the SDK, and file size/checksum checks happen after the transfer. No redundant wrapper client is introduced for these already-injected resource boundaries.

## STAC without a named service

`read_document(Path(...))` reads a bounded local Catalog, Collection, Item or ItemCollection without following links. It checks the supported document envelope, not the complete STAC or extension schema, and preserves unknown fields. `select_assets(item, role=..., media_type=...)` returns independent copies of every matching asset; it does not rank them or infer bands. `asset_href(item, key, document_url=...)` resolves one explicitly selected href against the actual source document URI, without opening it or granting credential trust. Local and non-HTTP asset schemes are not automatically converted into download requests.

```python
from pathlib import Path
from orsia_data.stac import read_document, select_assets, asset_href

path = Path("native/item.json").resolve()
item = read_document(path)
candidates = select_assets(item, role="data")
hrefs = {
    key: asset_href(item, key, document_url=path.as_uri())
    for key in candidates
}
# Inspect/select hrefs explicitly; this code does not download any asset.
```

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

Existing top-level function imports, `providers`, `atmosphere.providers`, `solar.lisird`, `spectroscopy.hitran` and `terrain.download_copernicus_dem` remain available. Their scientific selections, outputs and file-publication semantics are preserved. Import locations for internal monkeypatching are not a public contract; tests now intercept the named service or shared transport instead. CDSE/CMR search functions additionally accept a borrowed `session`; its timeout/byte/security policy overrides the standalone timeout. Native catalogue requests now fail on redirects instead of relying on urllib's automatic redirects. This deliberate behavior change prevents credential-bearing sessions and POST bodies from being redirected implicitly.

Offline tests exercise native GET/POST pages, relative links, extension retention, empty results, page bounds, loop rejection, token rotation/expiry, owned versus borrowed closure, redacted representations, credential confinement and compatibility exports. The data clean-install recipe builds wheels, installs them outside workspace resolution and replays the STAC/client tests alongside existing raster/transfer contracts. Optional HAPI tests still use intercepted synthetic HTTP bytes. These checks do not qualify live authenticated services, all provider formats, catalogue completeness, scientific retrieval performance or concurrent-native safety.

The parent `orsia-design.md` section 4.8 predates the implemented reader/acquisition contracts and still describes removed artifact requirements and an earlier maturity state. This repository migration does not update that parent document; use this guide, the [package API](../libs/orsia-data/README.md) and [data products](data-products.md) for the implemented boundary.

## Protocol evidence

The implementation was checked against the primary [STAC API Item Search specification](https://github.com/radiantearth/stac-api-spec/blob/main/item-search/README.md), including method-specific search links, ItemCollection responses and GET/POST pagination semantics, and the official CDSE [STAC endpoint documentation](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) and [token documentation](https://documentation.dataspace.copernicus.eu/APIs/Token.html), accessed on 8 October 2026. CDSE documents the `stac.dataspace.copernicus.eu/v1/` endpoint and password/2FA token generation with refresh or explicit re-login after expiry. Reading documentation is not a live-provider qualification; no account credentials were used during this work.
