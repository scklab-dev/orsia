"""Explicit CDSE password login and expiring bearer/refresh-token lifecycle."""

from collections.abc import Callable
from math import isfinite
from time import monotonic
from typing import Any
from urllib.parse import urlencode

from ...http import JsonSession

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


class CdseAuth:
    """In-memory CDSE tokens; no environment access, persistence or thread safety.

    expires_in is remaining lifetime in seconds, measured from construction with
    a monotonic clock. Refresh happens once when expired (or within refresh_margin_s).
    No automatic password re-login or HTTP-error retry. A supplied session is
    borrowed; otherwise owned. repr never includes tokens. close drops references,
    not a guarantee of cryptographic memory erasure or remote revocation.
    """

    def __init__(
        self,
        access_token: str,
        *,
        expires_in: float,
        refresh_token: str | None = None,
        session: JsonSession | None = None,
        refresh_margin_s: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Store expiring credentials for explicit CDSE downloads.

        Args:
            access_token: Initial nonempty bearer token without whitespace.
            expires_in: Remaining positive token lifetime in seconds.
            refresh_token: Optional token for one refresh exchange near expiry.
            session: Borrowed JSON transport, or create an owned one when omitted.
            refresh_margin_s: Requested nonnegative seconds before expiry to refresh.
            clock: Monotonic seconds supplier for token lifetime accounting.

        Raises:
            ValueError: Token, lifetime or refresh margin is invalid.
        """
        _valid_token(access_token)
        if refresh_token is not None:
            _valid_token(refresh_token)
        if not isfinite(expires_in) or expires_in <= 0:
            raise ValueError("positive finite remaining token lifetime required")
        if not isfinite(refresh_margin_s) or refresh_margin_s < 0:
            raise ValueError("nonnegative finite refresh margin required")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._clock = clock
        self._expires = clock() + expires_in
        self._margin = min(refresh_margin_s, expires_in / 2)
        self._requested_margin = refresh_margin_s
        self._owned = session is None
        self._session = session or JsonSession(max_bytes=64 * 1024)
        self._closed = False

    @classmethod
    def login(
        cls,
        username: str,
        password: str,
        *,
        totp: str | None = None,
        session: JsonSession | None = None,
        refresh_margin_s: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> CdseAuth:
        """One explicit password grant (optional 2FA); never retain the password.

        Authentication errors are sanitized without response bodies/credentials.
        Fresh credentials/2FA must be supplied explicitly if refresh later fails.

        Args:
            username: Explicit CDSE account username for the password grant.
            password: Password used for this exchange, discarded after the request.
            totp: Optional current second-factor code submitted with the grant.
            session: Borrowed JSON transport, or create an owned one when omitted.
            refresh_margin_s: Nonnegative seconds before expiry to attempt refresh.
            clock: Monotonic seconds supplier for lifetime accounting.

        Returns:
            Auth handle holding the returned access and optional refresh tokens.

        Raises:
            ValueError: Credentials or refresh margin are invalid.
            RuntimeError: The token exchange fails or returns invalid credentials.
        """
        if not username or not password:
            raise ValueError("explicit username and password required")
        if not isfinite(refresh_margin_s) or refresh_margin_s < 0:
            raise ValueError("nonnegative finite refresh margin required")
        transport = session or JsonSession(max_bytes=64 * 1024)
        form = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        }
        if totp is not None:
            form["totp"] = totp
        started = clock()
        try:
            reply = _exchange(transport, form)
            remaining = float(reply["expires_in"]) - (clock() - started)
            result = cls(
                reply["access_token"],
                expires_in=remaining,
                refresh_token=reply.get("refresh_token"),
                session=transport,
                refresh_margin_s=refresh_margin_s,
                clock=clock,
            )
        except BaseException:
            if session is None:
                transport.close()
            raise
        result._owned = session is None
        return result

    def get_token(self) -> str:
        """Provide a current bearer token for one authenticated request.

        Returns:
            Current access token, refreshed once near expiry when configured.

        Raises:
            RuntimeError: Handle is closed, token has expired without refresh,
                or the bounded refresh exchange fails.
        """
        if self._closed:
            raise RuntimeError("CDSE authentication is closed")
        now = self._clock()
        if self._refresh_token is None:
            if now >= self._expires:
                raise RuntimeError("CDSE token expired; explicit login required")
            return self._access_token
        if now >= self._expires - self._margin:
            started = self._clock()
            reply = _exchange(
                self._session,
                {
                    "client_id": "cdse-public",
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                },
            )
            lifetime = float(reply["expires_in"])
            if self._clock() >= started + lifetime:
                raise RuntimeError("CDSE token expired during refresh")
            self._access_token = reply["access_token"]
            self._refresh_token = reply.get("refresh_token", self._refresh_token)
            self._expires = started + lifetime
            self._margin = min(self._requested_margin, lifetime / 2)
        return self._access_token

    def close(self) -> None:
        """Invalidate local tokens and close the owned HTTP session, if any."""
        self._access_token = ""
        self._refresh_token = None
        if self._owned:
            self._session.close()
        self._closed = True

    def __enter__(self) -> CdseAuth:
        if self._closed:
            raise RuntimeError("CDSE authentication is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _valid_token(value: str) -> None:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("invalid CDSE token")


def _exchange(session: JsonSession, form: dict[str, str]) -> dict[str, Any]:
    try:
        reply = session.request(TOKEN_URL, method="POST", form=urlencode(form).encode())
        _valid_token(reply["access_token"])
        if "refresh_token" in reply:
            _valid_token(reply["refresh_token"])
        lifetime = float(reply["expires_in"])
        if not isfinite(lifetime) or lifetime <= 0:
            raise ValueError("invalid expiry")
        return reply
    except Exception:
        raise RuntimeError(
            "CDSE token exchange failed; check credentials and service availability"
        ) from None
