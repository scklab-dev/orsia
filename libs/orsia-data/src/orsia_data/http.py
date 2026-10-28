"""Bounded JSON HTTP session, independent of catalogues and named services.

No environment credentials, cookies, retries or implicit redirects. Connections
are closed after each response (stdlib transport, not a connection pool). An
injected opener is borrowed; its owner must enforce the same redirect policy.
Sessions are context managers, not safe for concurrent use or mutation.
"""

import json
from collections.abc import Callable
from math import isfinite
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class JsonSession:
    """Own request policy; retain no secrets after close. Network starts on request.

    token is called only for explicitly trusted HTTPS origins on port 443.
    Redirects fail rather than forwarding credentials or POST bodies. Timeout is
    per network operation, not an overall deadline; max_bytes bounds each page.
    """

    def __init__(
        self,
        *,
        token: Callable[[], str] | None = None,
        trusted_hosts: tuple[str, ...] = (),
        timeout_s: float = 30.0,
        max_bytes: int = 16 * 1024 * 1024,
        open_url: Callable[..., Any] | None = None,
    ) -> None:
        """Configure bounded HTTPS JSON transport with explicit bearer trust.

        Args:
            token: Optional callback supplying a bearer token for trusted hosts.
            trusted_hosts: Bare lowercase HTTPS hostnames permitted to receive tokens.
            timeout_s: Positive timeout in seconds for each network operation.
            max_bytes: Positive maximum JSON response size in bytes.
            open_url: Borrowed opener for an explicit transport implementation.

        Raises:
            ValueError: Timeout/byte limit or token-host trust configuration is invalid.
        """
        if not isfinite(timeout_s) or timeout_s <= 0 or max_bytes <= 0:
            raise ValueError("positive finite timeout and byte bound required")
        if token is not None and not trusted_hosts:
            raise ValueError("token requires explicit trusted hosts")
        if any(not h or h != h.lower() or any(c in h for c in "/:@*?#") for h in trusted_hosts):
            raise ValueError("trusted hosts must be bare lowercase hostnames")
        self._token = token
        self._hosts = trusted_hosts
        self._open = open_url or build_opener(ProxyHandler({}), _NoRedirect()).open
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self._closed = False

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        form: bytes | None = None,
    ) -> dict[str, Any]:
        """Return a fresh native JSON object; preserve unknown fields/extensions.

        headers cannot supply credentials or transport-routing headers. form is
        reserved for explicit form POSTs; never taken from catalogue metadata.
        HTTP errors propagate; no automatic login or HTTP-error retry occurs.

        Args:
            url: Credential-free HTTPS endpoint on the standard TLS port.
            method: GET or POST, with GET requiring an empty request body.
            body: Optional JSON object to serialize as a POST body.
            headers: Optional Accept and Content-Type headers for this request.
            form: Optional URL-encoded POST bytes instead of a JSON body.

        Returns:
            Newly decoded native JSON object, including service-specific fields.

        Raises:
            RuntimeError: The session is closed.
            ValueError: URL, method, headers, credentials or response exceeds policy.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            raise ValueError("JSON transport requires credential-free HTTPS on port 443")
        if method not in ("GET", "POST") or (
            method == "GET" and (body is not None or form is not None)
        ):
            raise ValueError("only GET without body or POST is supported")
        if form is not None and body is not None:
            raise ValueError("supply JSON or form, not both")
        outgoing = dict(headers or {})
        for name, value in outgoing.items():
            if name.lower() not in ("accept", "content-type") or "\r" in value or "\n" in value:
                raise ValueError("only Accept and Content-Type link headers are supported")
        data = form
        if body is not None:
            data = json.dumps(body, allow_nan=False).encode()
            outgoing["Content-Type"] = "application/json"
        elif form is not None:
            outgoing["Content-Type"] = "application/x-www-form-urlencoded"
        if self._token is not None and parsed.hostname in self._hosts:
            token = self._token()
            if not token or any(c.isspace() for c in token):
                raise ValueError("invalid bearer token")
            outgoing["Authorization"] = f"Bearer {token}"
        request = Request(url, data=data, headers=outgoing, method=method)
        with self._open(request, timeout=self.timeout_s) as response:
            data = response.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("JSON response exceeds byte bound")
        page = json.loads(data)
        if not isinstance(page, dict):
            raise ValueError("JSON response must be an object")
        return page

    def close(self) -> None:
        """Invalidate this session and release its token callback/opener references."""
        self._closed = True
        self._token = None
        self._open = None

    def __enter__(self) -> JsonSession:
        if self._closed:
            raise RuntimeError("session is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
