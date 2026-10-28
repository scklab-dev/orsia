"""Search advertised STAC API endpoints and traverse a bounded set of pages."""

import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

from ..http import JsonSession


class StacClient:
    """STAC API client; no implicit discovery, downloads, credentials or retries.

    A supplied session is borrowed, otherwise owned. Not thread-safe. Landing
    page capabilities are returned natively; search uses advertised GET/POST
    links, not a guessed endpoint. Close invalidates the client, not a borrowed
    session. Links may traverse HTTPS hosts but never expand credential trust.
    """

    def __init__(self, url: str, *, session: JsonSession | None = None) -> None:
        """Bind explicit STAC landing URL to a borrowed or owned JSON session.

        Args:
            url: Landing-page HTTPS URL whose advertised links govern search.
            session: Optional borrowed JSON session; otherwise create an owned one.
        """
        self.url = url
        self._owned = session is None
        self._session = session or JsonSession()
        self._closed = False

    def landing_page(self) -> dict[str, Any]:
        """Fetch the native STAC landing page to inspect advertised capabilities.

        Returns:
            JSON mapping from the configured landing URL.

        Raises:
            RuntimeError: The client has been closed.
        """
        self._check_open()
        return self._session.request(self.url)

    def search(self, parameters: dict[str, Any], *, method: str = "GET") -> dict[str, Any]:
        """Fetch one native STAC ItemCollection page for explicit search parameters.

        Args:
            parameters: STAC search parameters, including supported spatial filters.
            method: GET or POST, advertised by the landing page's search link.

        Returns:
            Native ItemCollection page, including extensions and pagination links.

        Raises:
            ValueError: Search method, spatial filter combination, advertised
                link or returned ItemCollection envelope is invalid.
            RuntimeError: The client has been closed.
        """
        return next(self.pages(parameters, method=method, max_pages=1))

    def pages(
        self, parameters: dict[str, Any], *, method: str = "GET", max_pages: int
    ) -> Iterator[dict[str, Any]]:
        """Yield at most max_pages pages (explicit truncation), never fetch ahead.

        GET encodes list parameters as comma-separated values; object parameters
        as JSON. POST pagination honors method/body/merge relative to the initial
        search request. Relative hrefs resolve against the current response URL.
        Unsupported headers/methods and repeated requests fail instead of looping.
        No snapshot consistency or completeness is implied by the page cap.

        Args:
            parameters: Initial STAC search filters; copied before pagination.
            method: Advertised GET or POST search method.
            max_pages: Positive maximum number of native pages to request.

        Yields:
            Each fetched ItemCollection mapping in pagination order.

        Raises:
            ValueError: Page limit or filters are invalid, search link is ambiguous,
                a response is not an ItemCollection or pagination repeats a request.
            RuntimeError: The client has been closed before a request.
        """
        if max_pages < 1 or method not in ("GET", "POST"):
            raise ValueError("positive page bound and GET/POST method required")
        original = deepcopy(parameters)
        if "bbox" in original and "intersects" in original:
            raise ValueError("bbox and intersects are mutually exclusive")
        landing = self.landing_page()
        links = [
            link
            for link in landing.get("links", [])
            if link.get("rel") == "search"
            and link.get("method", "GET") == method
            and link.get("type", "application/geo+json") == "application/geo+json"
        ]
        if len(links) != 1:
            raise ValueError("landing page must advertise one search link for this method")
        url = urljoin(self.url, links[0]["href"])
        if method == "GET":
            query = urlencode({key: _query_value(value) for key, value in original.items()})
            url += ("&" if urlsplit(url).query else "?") + query
        body = original if method == "POST" else None
        headers: dict[str, str] = {}
        seen: set[str] = set()
        for index in range(max_pages):
            self._check_open()
            identity = json.dumps([url, method, body, headers], sort_keys=True)
            if identity in seen:
                raise ValueError("repeated STAC pagination request")
            seen.add(identity)
            page = self._session.request(url, method=method, body=body, headers=headers)
            if page.get("type") != "FeatureCollection" or not isinstance(
                page.get("features"), list
            ):
                raise ValueError("search must return an ItemCollection")
            following = deepcopy(
                [link for link in page.get("links", []) if link.get("rel") == "next"]
            )
            yield page
            if not following or index + 1 == max_pages:
                return
            if len(following) != 1:
                raise ValueError("ambiguous STAC next link")
            link = following[0]
            url = urljoin(url, link["href"])
            method = link.get("method", "GET")
            body = link.get("body")
            headers = link.get("headers", {})
            if link.get("merge", False):
                body = original | (body or {})

    def close(self) -> None:
        """Release an owned session and invalidate future client requests."""
        if self._owned:
            self._session.close()
        self._closed = True

    def __enter__(self) -> StacClient:
        self._check_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("STAC client is closed")


def _query_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    return str(value)
