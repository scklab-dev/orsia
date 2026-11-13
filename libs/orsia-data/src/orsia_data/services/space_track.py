"""Acquire bounded Space-Track GP payloads with explicit credentials and local provenance.

The default transport rejects redirects and proxies and keeps cookies in memory.
Clients are serial, account-local sessions; callers coordinate all other account
activity and persist/reuse downloads. No network activity occurs at construction.
"""

import json
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import (
    HTTPCookieProcessor,
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from orsia_data.acquisition import retrieve_file
from orsia_data.orbits import parse_orbital_elements

_ORIGIN = "https://www.space-track.org"


class SpaceTrackClient:
    """Own one explicit authenticated session with bounded requests and no credential discovery.

    The session retains a borrowed credential supplier until close. Passwords and
    cookies remain in memory only and are omitted from representations/errors and
    receipts. Closing drops references; it cannot guarantee memory erasure or
    remote revocation. The session is unsuitable for concurrent use.
    """

    def __init__(
        self,
        credentials: Callable[[], tuple[str, str]],
        *,
        timeout_s: float = 30,
        max_bytes: int = 16 * 1024 * 1024,
        attempts: int = 1,
        max_wait_s: float = 30,
        open_url: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure a session without logging in or looking up environment credentials.

        Args:
            credentials: Borrowed zero-argument supplier returning identity and password.
            timeout_s: Positive finite timeout per transport operation, seconds.
            max_bytes: Positive maximum response body size, bytes.
            attempts: Request attempts, 1 to 3; only transport/5xx/429 failures retry.
            max_wait_s: Maximum cumulative pacing/backoff wait per request, seconds.
                GP's hourly spacing also applies to retries; exceeding this bound fails.
            open_url: Borrowed test/transport opener; its owner must reject redirects,
                disable credential discovery and enforce the default origin/cookie policy.
            clock: Monotonic clock for this client's request limits.
            sleep: Synchronous wait function used for bounded pacing.

        Raises:
            ValueError: Request, retry or waiting bounds are invalid.
        """
        if (
            not math.isfinite(timeout_s)
            or timeout_s <= 0
            or max_bytes <= 0
            or attempts not in (1, 2, 3)
            or not math.isfinite(max_wait_s)
            or max_wait_s < 0
        ):
            raise ValueError("invalid Space-Track request bounds")
        self._credentials: Callable[[], tuple[str, str]] | None = credentials
        self._cookies = CookieJar(DefaultCookiePolicy(allowed_domains=("www.space-track.org",)))
        self._open = (
            open_url
            or build_opener(
                ProxyHandler({}), _NoRedirect(), HTTPCookieProcessor(self._cookies)
            ).open
        )
        self._timeout = timeout_s
        self._max_bytes = max_bytes
        self._attempts = attempts
        self._max_wait = max_wait_s
        self._clock, self._sleep = clock, sleep
        self._requests: list[float] = []
        self._gp_next = float("-inf")
        self._history: set[str] = set()
        self._authenticated = False
        self._closed = False

    def download(
        self,
        object_ids: tuple[int, ...],
        destination: Path,
        *,
        format: Literal["omm", "tle"] = "omm",
        history_utc: tuple[datetime, datetime] | None = None,
    ) -> Path:
        """Acquire selected native elements and publish a credential-free receipt beside them.

        Args:
            object_ids: Distinct positive catalogue identifiers combined in one query.
            destination: New payload path; parent directories can be created.
                Receipt path is destination with '.provenance.json' appended.
            format: GP OMM JSON (recommended) or native TLE text. The required SatKit
                parser is loaded before authentication for payload validation.
            history_utc: Closed UTC epoch interval for gp_history; None selects newest gp.

        Returns:
            Payload path after validation and receipt publication. Existing payloads
            are never refreshed implicitly. The receipt identifies bytes, query,
            acquisition time, object IDs and element epochs. Failure publishing a
            receipt can leave the complete payload; callers may reuse local bytes.

        Raises:
            RuntimeError: Closed session, authentication/service/throttle failure,
                retry/wait exhaustion or duplicate history retrieval in this session.
            ValueError: Selection, format or provider payload is invalid/empty.
            FileExistsError: Payload or receipt already exists.
            OSError: Local publication fails.
            ImportError: The required SatKit installation cannot load; no request is sent.
        """
        self._check_open()
        if (
            not object_ids
            or len(set(object_ids)) != len(object_ids)
            or any(i <= 0 for i in object_ids)
        ):
            raise ValueError("distinct positive object identifiers required")
        if format not in ("omm", "tle"):
            raise ValueError("format must be omm or tle")
        selection = ",".join(str(i) for i in sorted(object_ids))
        element_class = "gp_history" if history_utc else "gp"
        query = f"/basicspacedata/query/class/{element_class}/NORAD_CAT_ID/{selection}"
        if history_utc is not None:
            start, end = history_utc
            if (
                any(t.tzinfo is None or t.utcoffset() != timedelta(0) for t in history_utc)
                or start > end
                or start.microsecond
                or end.microsecond
            ):
                raise ValueError("history requires ordered whole-second UTC boundaries")
            interval = f"{start:%Y-%m-%dT%H:%M:%S}--{end:%Y-%m-%dT%H:%M:%S}"
            query += "/EPOCH/" + quote(interval, safe="-:")
        query += f"/orderby/NORAD_CAT_ID,EPOCH/format/{'json' if format == 'omm' else 'tle'}"
        if history_utc is not None and query in self._history:
            raise RuntimeError("history selection already retrieved; reuse the local payload")
        receipt = destination.with_name(destination.name + ".provenance.json")
        if destination.exists() or receipt.exists():
            raise FileExistsError("orbital payload or receipt already exists")
        # Load the required parser before credentials or requests are used.
        from importlib import import_module

        import_module("satkit")
        self._login()
        payload = self._request(query, gp=history_utc is None)
        records = parse_orbital_elements(payload, format=format)
        if any(r.norad_id not in object_ids for r in records):
            raise ValueError("provider returned an unrequested object")
        if {r.norad_id for r in records} != set(object_ids):
            raise ValueError("provider selection is incomplete")
        if history_utc is not None:
            self._history.add(query)
        metadata = {
            "provider": "Space-Track",
            "query": query,
            "acquired_utc": datetime.now(UTC).isoformat(),
            "format": format,
            "payload_sha256": sha256(payload).hexdigest(),
            "object_ids": list(object_ids),
            "element_epochs": [r.epoch.isoformat() for r in records],
        }
        retrieve_file(
            _ORIGIN + query,
            destination,
            retrieve=lambda _url, path: path.write_bytes(payload),
            max_bytes=self._max_bytes,
            expected_sha256=metadata["payload_sha256"],
        )
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        return destination

    def close(self) -> None:
        """Drop credentials, cookies and transport references without remote revocation."""
        self._credentials = None
        self._cookies.clear()
        self._open = None
        self._authenticated = False
        self._closed = True

    def __enter__(self) -> SpaceTrackClient:
        self._check_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Space-Track session is closed")

    def _login(self) -> None:
        if self._authenticated:
            return
        assert self._credentials is not None
        try:
            identity, password = self._credentials()
        except Exception:
            raise RuntimeError("Space-Track credential supplier failed") from None
        if not identity or not password:
            raise ValueError("explicit Space-Track identity and password required")
        body = urlencode({"identity": identity, "password": password}).encode()
        result = self._request("/ajaxauth/login", body=body)
        # Space-Track success can return an empty JSON string, object or body;
        # explicit failures and HTML/login pages must remain rejected.
        try:
            parsed = json.loads(result) if result.strip() else {}
        except ValueError:
            raise RuntimeError("Space-Track authentication failed") from None
        if parsed != {} and parsed != "":
            raise RuntimeError("Space-Track authentication failed")
        self._authenticated = True

    def _request(self, path: str, *, body: bytes | None = None, gp: bool = False) -> bytes:
        waited = 0.0
        retry_after = 0.0
        for attempt in range(self._attempts):
            now = self._clock()
            self._requests = [t for t in self._requests if now - t < 3600]
            delay = max(0.0, self._gp_next - now) if gp else 0.0
            if len(self._requests) >= 299:
                delay = max(delay, 3600 - (now - self._requests[-299]) + 0.01)
            recent = [t for t in self._requests if now - t < 60]
            if len(recent) >= 29:
                delay = max(delay, 60 - (now - recent[-29]) + 0.01)
            if attempt:
                delay = max(delay, retry_after, 2.0 ** (attempt - 1))
            if waited + delay > self._max_wait:
                raise RuntimeError("Space-Track pacing/retry exceeds the waiting budget")
            self._sleep(delay)
            waited += delay
            now = self._clock()
            self._requests.append(now)
            if gp:
                self._gp_next = now + 3600
            request = Request(_ORIGIN + path, data=body)
            if body is not None:
                request.add_header("Content-Type", "application/x-www-form-urlencoded")
            try:
                with self._open(request, timeout=self._timeout) as response:
                    if response.geturl() != _ORIGIN + path:
                        raise RuntimeError("Space-Track redirect refused")
                    payload = response.read(self._max_bytes + 1)
                if len(payload) > self._max_bytes:
                    raise ValueError("Space-Track response exceeds byte bound")
                return payload
            except HTTPError as exc:
                status = exc.code
                header = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if header is not None:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        try:
                            date = parsedate_to_datetime(header)
                            retry_after = max(0.0, (date - datetime.now(UTC)).total_seconds())
                        except TypeError, ValueError:
                            raise RuntimeError("Space-Track retry policy is invalid") from None
                    if not math.isfinite(retry_after) or retry_after < 0:
                        raise RuntimeError("Space-Track retry policy is invalid") from None
                if status in (401, 403):
                    self._authenticated = False
                    raise RuntimeError("Space-Track authentication failed") from None
                if status != 429 and status < 500:
                    raise RuntimeError("Space-Track request/redirect refused") from None
            except URLError, TimeoutError, OSError:
                pass
        raise RuntimeError("Space-Track service/throttle failure; retry budget exhausted") from None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
