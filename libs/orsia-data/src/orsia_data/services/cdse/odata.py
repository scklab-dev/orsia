"""CDSE OData catalogue queries and explicit product dissemination."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

from ...acquisition import download_bearer_file, retrieve_file
from ...http import JsonSession
from ..catalogue import geographic_bbox, utc_interval


def search_sentinel2_l2a(
    *,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    top: int = 100,
    skip: int = 0,
    timeout_s: float = 30.0,
    session: JsonSession | None = None,
) -> dict[str, Any]:
    """One native CDSE L2A page, UTC [start,end), bbox=(west,south,east,north).

    No cloud threshold, nearest-time selection or pagination loop. Native
    @odata.nextLink/@odata.count remain available for explicit continuation.
    A supplied session is borrowed and its policy overrides timeout_s.

    Args:
        bbox: Geographic (west,south,east,north) rectangle in degrees.
        start: Inclusive UTC acquisition start time.
        end: Exclusive UTC acquisition end time.
        top: Positive number of records requested, at most 1000.
        skip: Nonnegative number of records skipped for this page.
        timeout_s: Session timeout in seconds when constructing a transport.
        session: Optional borrowed JSON session overriding timeout_s.

    Returns:
        Native OData product page including its value list and pagination metadata.

    Raises:
        ValueError: Geographic/time selection, paging or response envelope is invalid.
    """
    west, south, east, north = geographic_bbox(bbox)
    first, last = utc_interval(start, end)
    if not 1 <= top <= 1000 or skip < 0:
        raise ValueError("invalid CDSE page size/offset")
    polygon = f"{west} {south},{east} {south},{east} {north},{west} {north},{west} {south}"
    query = {
        "$filter": "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and "
        "a/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"ContentDate/Start ge {first} and ContentDate/Start lt {last} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({polygon}))')",
        "$top": top,
        "$skip": skip,
        "$count": "true",
        "$expand": "Attributes",
    }
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?" + urlencode(query)
    if session is None:
        with JsonSession(timeout_s=timeout_s) as owned:
            page = owned.request(url)
    else:
        page = session.request(url)
    if not isinstance(page.get("value"), list):
        raise ValueError("invalid CDSE product page")
    return page


def download_cdse(
    product_id: str,
    destination: Path,
    *,
    retrieve: Callable[[str, Path], object] | None = None,
    access_token: str | None = None,
    timeout_s: float = 30.0,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> Path:
    """Download an explicit UUID with a token OR borrowed authenticated transfer.

    Native bearer transfer trusts only download/zipper.dataspace.copernicus.eu.
    An injected transfer owns streaming bounds, redirects, retries and deadlines;
    its size is checked after transfer. Both publish without clobbering. No
    extraction, token refresh or scientific decoding is performed here.

    Args:
        product_id: Explicit CDSE UUID of the selected product.
        destination: New output path for the native product bytes.
        retrieve: Optional borrowed authenticated URL-to-path transfer callback.
        access_token: Alternative explicit bearer token for native transfer.
        timeout_s: Native transfer timeout in seconds.
        max_bytes: Maximum download size in bytes.
        expected_sha256: Optional expected hexadecimal content digest.

    Returns:
        Destination path following bounded transfer and optional digest verification.

    Raises:
        ValueError: UUID or exclusive authentication choice is invalid.
        FileExistsError: Destination path already exists.
    """
    identifier = str(UUID(product_id))
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({identifier})/$value"
    if (retrieve is None) == (access_token is None):
        raise ValueError("supply exactly one of access_token and retrieve")
    if access_token is not None:
        return download_bearer_file(
            url,
            destination,
            access_token=access_token,
            trusted_hosts=("download.dataspace.copernicus.eu", "zipper.dataspace.copernicus.eu"),
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )
    assert retrieve is not None
    return retrieve_file(
        url, destination, retrieve=retrieve, max_bytes=max_bytes, expected_sha256=expected_sha256
    )
