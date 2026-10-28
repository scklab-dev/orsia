"""Composed CMR catalogue session and explicitly trusted Earthdata downloads."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ...http import JsonSession
from .auth import EarthdataAuth
from .cmr import EarthdataProduct, download_earthdata, search_earthdata


class EarthdataClient:
    """Borrow explicit auth and optional JSON session; never trust catalogue hosts automatically.

    Catalogue access is public by default. trusted_hosts applies only to native
    bearer downloads. No SDK, credential discovery, retry or scientific decoding.
    Not thread-safe; close invalidates the client but not borrowed resources.
    """

    def __init__(
        self,
        *,
        auth: EarthdataAuth | None = None,
        trusted_hosts: tuple[str, ...] = (),
        session: JsonSession | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Set up public CMR queries and explicitly trusted bearer downloads.

        Args:
            auth: Borrowed credential handle for selected downloads.
            trusted_hosts: Hostnames allowed to receive the bearer token.
            session: Borrowed catalogue JSON transport or an owned default transport.
            timeout_s: Timeout in seconds when constructing the default transport.
        """
        self._auth = auth
        self._hosts = trusted_hosts
        self._owned = session is None
        self._session = session or JsonSession(timeout_s=timeout_s)
        self._closed = False

    def search(
        self,
        *,
        product: EarthdataProduct,
        bbox: tuple[float, float, float, float],
        temporal: tuple[datetime, datetime] | None,
        page_number: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """Query a CMR product page using caller-selected spatial and time limits.

        Args:
            product: Product collection and CMR endpoint selection.
            bbox: Geographic (west, south, east, north) bounds in degrees.
            temporal: Optional start and end datetimes for the acquisition period.
            page_number: One-based CMR page number.
            page_size: Maximum items requested per page.

        Returns:
            Native CMR JSON response for the requested page.

        Raises:
            RuntimeError: The client has been closed.
            ValueError: The search selection or page bounds are invalid.
        """
        self._check_open()
        return search_earthdata(
            product=product,
            bbox=bbox,
            temporal=temporal,
            page_number=page_number,
            page_size=page_size,
            session=self._session,
        )

    def download(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download a selected asset with explicit bearer trust and byte limits.

        Args:
            url: Absolute native asset URL on an explicitly trusted host.
            destination: New local path for the downloaded bytes.
            max_bytes: Upper bound on the transferred body in bytes.
            expected_sha256: Optional hexadecimal digest to verify against the bytes.

        Returns:
            Destination path after successful transfer and validation.

        Raises:
            RuntimeError: The client has been closed.
            ValueError: Authentication is absent or the download selection is invalid.
        """
        self._check_open()
        if self._auth is None:
            raise ValueError("Earthdata download requires explicit authentication")
        return download_earthdata(
            url,
            destination,
            access_token=self._auth.get_token(),
            trusted_hosts=self._hosts,
            timeout_s=self._session.timeout_s,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )

    def close(self) -> None:
        """Invalidate this client and close only a session it owns."""
        if self._owned:
            self._session.close()
        self._auth = None
        self._closed = True

    def __enter__(self) -> EarthdataClient:
        self._check_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Earthdata client is closed")
