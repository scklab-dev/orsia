"""Explicit Earthdata bearer credentials; no invented universal refresh protocol."""

from collections.abc import Callable


class EarthdataAuth:
    """Hold a token or borrow a supplier called before each download.

    A supplier owns expiry, renewal and its resources. No URS login, environment,
    .netrc lookup or retry is implicit. repr omits credentials. Not thread-safe.
    close drops local references, not remote revocation or secure memory erasure.
    """

    def __init__(self, token: str | Callable[[], str]) -> None:
        """Retain explicit credentials or a caller-managed credential supplier.

        Args:
            token: Bearer string or zero-argument token supplier invoked per request.
        """
        self._token: str | Callable[[], str] | None = token

    def get_token(self) -> str:
        """Obtain a bearer token from the stored value or borrowed supplier.

        Returns:
            A nonempty token without whitespace for an explicit download.

        Raises:
            RuntimeError: This auth handle has been closed.
            ValueError: The supplied token is empty or contains whitespace.
        """
        if self._token is None:
            raise RuntimeError("Earthdata authentication is closed")
        token = self._token() if callable(self._token) else self._token
        if not token or any(c.isspace() for c in token):
            raise ValueError("invalid Earthdata token")
        return token

    def close(self) -> None:
        """Drop the local token or supplier reference without revoking credentials."""
        self._token = None

    def __enter__(self) -> EarthdataAuth:
        if self._token is None:
            raise RuntimeError("Earthdata authentication is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
