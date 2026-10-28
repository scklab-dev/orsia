"""NASA CMR discovery and explicit Earthdata file dissemination."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit

from ...acquisition import download_bearer_file, retrieve_file
from ...http import JsonSession
from ..catalogue import geographic_bbox, utc_interval

type EarthdataProduct = Literal["ECO_L2T_LSTE.002", "AG100.003"]


def search_earthdata(
    *,
    product: EarthdataProduct,
    bbox: tuple[float, float, float, float],
    temporal: tuple[datetime, datetime] | None,
    page_number: int = 1,
    page_size: int = 100,
    timeout_s: float = 30.0,
    session: JsonSession | None = None,
) -> dict[str, Any]:
    """One native CMR page; temporal=None omits time selection explicitly.

    CMR overlap semantics and edition identity are retained. No link selection,
    download or automatic page traversal. A supplied JSON session is borrowed;
    its policy overrides timeout_s. Page traversal is not a catalogue snapshot.

    Args:
        product: Supported CMR short name and edition identifier.
        bbox: Geographic (west,south,east,north) query bounds in degrees.
        temporal: Optional UTC start/end time selection; None omits time filtering.
        page_number: One-based native CMR page number.
        page_size: Page length between 1 and 2000.
        timeout_s: Timeout for the session created when ``session`` is absent.
        session: Optional borrowed JSON session with its own request policy.

    Returns:
        Native CMR response mapping with an ``entry`` list in its ``feed``.

    Raises:
        ValueError: Product, geographic or temporal selection, page bounds or
            returned granule page is invalid.
    """
    if product not in ("ECO_L2T_LSTE.002", "AG100.003"):
        raise ValueError("unsupported Earthdata product/edition")
    geographic_bbox(bbox)
    if page_number < 1 or not 1 <= page_size <= 2000:
        raise ValueError("invalid CMR page selection")
    short_name, version = product.split(".")
    query: dict[str, object] = {
        "short_name": short_name,
        "version": version,
        "bounding_box": ",".join(map(str, bbox)),
        "page_num": page_number,
        "page_size": page_size,
    }
    if temporal is not None:
        query["temporal"] = ",".join(utc_interval(*temporal))
    url = "https://cmr.earthdata.nasa.gov/search/granules.json?" + urlencode(query)
    if session is None:
        with JsonSession(timeout_s=timeout_s) as owned:
            page = owned.request(url)
    else:
        page = session.request(url)
    if not isinstance(page.get("feed", {}).get("entry"), list):
        raise ValueError("invalid CMR granule page")
    return page


def earthdata_links(granule: dict[str, Any]) -> tuple[str, ...]:
    """Candidate HTTPS data links, excluding inherited/browse links; no IO.

    Several links can be distinct native layers, not interchangeable files.

    Args:
        granule: Native CMR granule entry containing optional link metadata.

    Returns:
        Distinct HTTPS data hrefs in first-seen order, as a tuple of strings.
    """
    return tuple(
        dict.fromkeys(
            link["href"]
            for link in granule.get("links", ())
            if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"
            and not link.get("inherited", False)
            and urlsplit(link["href"]).scheme == "https"
        )
    )


def download_earthdata(
    url: str,
    destination: Path,
    *,
    retrieve: Callable[[str, Path], object] | None = None,
    access_token: str | None = None,
    trusted_hosts: tuple[str, ...] = (),
    timeout_s: float = 30.0,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> Path:
    """Download an explicit link with a token/trust list OR a borrowed transfer.

    No implicit login or credential files. Native token transport has streaming
    bounds and safe redirects; an injected transfer owns those policies and is
    checked only after transfer. Both publish atomically without clobbering.

    Args:
        url: Explicit selected HTTPS data link.
        destination: New local output path for the native file.
        retrieve: Optional borrowed transfer callable taking URL and path.
        access_token: Optional token for the native bearer transfer path.
        trusted_hosts: Redirect hosts trusted for native bearer token transfer.
        timeout_s: Native HTTP timeout in seconds.
        max_bytes: Positive maximum transfer size in bytes.
        expected_sha256: Optional expected lowercase hexadecimal SHA-256 digest.

    Returns:
        Destination path after bounded transfer and atomic publication.

    Raises:
        ValueError: Transfer policy mixes token and injected client, or byte
            bound, URL or content integrity checks fail.
    """
    if (retrieve is None) == (access_token is None):
        raise ValueError("supply exactly one of access_token and retrieve")
    if access_token is not None:
        return download_bearer_file(
            url,
            destination,
            access_token=access_token,
            trusted_hosts=trusted_hosts,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )
    if trusted_hosts:
        raise ValueError("trusted_hosts belongs to native token transfer, not an injected client")
    assert retrieve is not None
    return retrieve_file(
        url, destination, retrieve=retrieve, max_bytes=max_bytes, expected_sha256=expected_sha256
    )
