"""Composed CDSE catalogue session, optional authentication and native downloads."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ...http import JsonSession
from .auth import CdseAuth
from .odata import download_cdse, search_sentinel2_l2a
from .stac import stac_client


class CdseClient:
    """Own public catalogue access; borrow explicitly supplied auth/session.

    auth is used only for explicit OData downloads, never STAC-linked assets.
    No implicit credentials/login, cache, scene construction or retries. Calls
    are synchronous and the client is not thread-safe. Close invalidates this
    client and its STAC view, but does not close borrowed auth/session objects.
    """

    def __init__(
        self,
        *,
        auth: CdseAuth | None = None,
        session: JsonSession | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Compose public catalogue access and optional authenticated downloads.

        Args:
            auth: Borrowed token handle used only for explicit OData downloads.
            session: Borrowed JSON transport, or create an owned one when omitted.
            timeout_s: Timeout in seconds for the newly created transport.
        """
        self._auth = auth
        self._owned = session is None
        self._session = session or JsonSession(timeout_s=timeout_s)
        self.stac = stac_client(session=self._session)
        self._closed = False

    def search_sentinel2_l2a(
        self,
        *,
        bbox: tuple[float, float, float, float],
        start: datetime,
        end: datetime,
        top: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]:
        """Find a native OData product page within a spatial and temporal window.

        Args:
            bbox: Geographic (west, south, east, north) bounds in degrees.
            start: UTC interval start.
            end: UTC interval end.
            top: Maximum entries to request in this page.
            skip: Entries to skip before this page.

        Returns:
            Native OData JSON response from the public catalogue.

        Raises:
            RuntimeError: The client has been closed.
            ValueError: The search bounds or paging parameters are invalid.
        """
        self._check_open()
        return search_sentinel2_l2a(
            bbox=bbox, start=start, end=end, top=top, skip=skip, session=self._session
        )

    def download(
        self,
        product_id: str,
        destination: Path,
        *,
        max_bytes: int,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download an explicitly selected CDSE product with current credentials.

        Args:
            product_id: Selected CDSE product UUID.
            destination: Path for the downloaded product bytes.
            max_bytes: Maximum permitted response size in bytes.
            expected_sha256: Optional expected hexadecimal content digest.

        Returns:
            Destination path following a verified download.

        Raises:
            RuntimeError: The client is closed or the auth token cannot be refreshed.
            ValueError: Authentication is absent or the product/destination is invalid.
        """
        self._check_open()
        if self._auth is None:
            raise ValueError("CDSE download requires explicit authentication")
        return download_cdse(
            product_id,
            destination,
            access_token=self._auth.get_token(),
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
            timeout_s=self._session.timeout_s,
        )

    def close(self) -> None:
        """Invalidate this client and STAC view; close only its owned session."""
        self.stac.close()
        if self._owned:
            self._session.close()
        self._auth = None
        self._closed = True

    def __enter__(self) -> CdseClient:
        self._check_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("CDSE client is closed")
