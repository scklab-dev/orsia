"""Explicit bounded HTTP(S) transport with optional caller-supplied bearer authentication."""

import os
import shutil
import tempfile
from collections.abc import Callable
from copy import deepcopy
from hashlib import file_digest, sha256
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
) -> Path:
    """Download to a private temporary file, then atomically publish without clobber.

    Explicit network call, not offline lookup or a cache. Return destination only
    after bounded complete transfer and optional lowercase SHA-256 verification.
    Existing destinations raise FileExistsError. Errors leave no partial published
    file; retry starts from scratch. HTTP errors/unsupported redirects propagate.
    No format parsing, credentials, manifests, archive extraction or provider
    discovery. Digest identifies bytes, not scientific coverage or authenticity.

    Args:
        url: Explicit HTTPS resource URL to transfer.
        destination: New local path for the completed native file.
        expected_sha256: Optional lowercase hexadecimal checksum of file bytes.
        timeout_s: Positive per-operation network timeout in seconds.
        max_bytes: Positive maximum accepted response size in bytes.

    Returns:
        Destination path once transfer, size and optional checksum succeed.

    Raises:
        FileExistsError: Destination already exists.
        ValueError: Transport selection, bounds, response or checksum is invalid.
    """
    return _download(url, destination, expected_sha256, timeout_s, max_bytes, urlopen)


def download_bearer_file(
    url: str,
    destination: Path,
    *,
    access_token: str,
    trusted_hosts: tuple[str, ...],
    expected_sha256: str | None = None,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
) -> Path:
    """Bounded HTTPS download with an explicit token and exact trusted hostnames.

    No credential discovery/login, cookies, SDK, token refresh or persisted token.
    The initial URL must be trusted HTTPS on port 443. HTTPS redirects are allowed
    but Authorization is stripped unless the next hostname is explicitly trusted;
    HTTP downgrades and embedded URL credentials fail. Trust lists contain bare
    lowercase hostnames, never suffix/wildcard matching. A fresh opener is owned by
    this call; urllib's transport timeout is not a whole-download deadline.
    Publication/checksum semantics match download_file. Never put tokens in URLs.

    Args:
        url: Explicit HTTPS resource on a trusted hostname.
        destination: New local path for the completed native file.
        access_token: Nonempty bearer credential provided for this request.
        trusted_hosts: Exact lowercase hostnames allowed to receive the token.
        expected_sha256: Optional lowercase hexadecimal checksum of file bytes.
        timeout_s: Positive transport timeout for each network operation, in seconds.
        max_bytes: Positive upper bound on transferred response bytes.

    Returns:
        Destination after bounded transfer and integrity validation.

    Raises:
        FileExistsError: Destination already exists.
        ValueError: Credential, trust policy, transport or file validation fails.
    """
    if not access_token or any(c.isspace() for c in access_token):
        raise ValueError("explicit nonempty bearer token without whitespace required")
    if not trusted_hosts or any(
        not host or host != host.lower() or any(c in host for c in "/:@*?#")
        for host in trusted_hosts
    ):
        raise ValueError("explicit lowercase trusted hostnames required")
    _https(url)
    if urlsplit(url).hostname not in trusted_hosts:
        raise ValueError("initial download host is not trusted for this token")
    opener = build_opener(_BearerRedirect(access_token, trusted_hosts))
    request = Request(url, headers={"Authorization": f"Bearer {access_token}"})
    return _download(
        url,
        destination,
        expected_sha256,
        timeout_s,
        max_bytes,
        lambda _url, timeout: opener.open(request, timeout=timeout),
    )


def file_sha256(path: Path) -> str:
    """Fingerprint a local file for explicit byte-level integrity comparison.

    Args:
        path: Existing local file kept unchanged while it is read.

    Returns:
        Lowercase hexadecimal SHA-256 digest of the complete file bytes.
    """
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def retrieve_file(
    url: str,
    destination: Path,
    *,
    retrieve: Callable[[str, Path], object],
    max_bytes: int,
    expected_sha256: str | None = None,
) -> Path:
    """Publish a borrowed synchronous HTTPS transfer with post-transfer checks.

    The injected client owns streaming bounds, redirects, credentials, retries
    and deadlines. Only our private directory is removed on failure. Existing
    destinations are never replaced; parent directories may remain.

    Args:
        url: Explicit HTTPS resource passed to the borrowed transfer callable.
        destination: New local path for the published native file.
        retrieve: Caller-owned synchronous transfer into a temporary path.
        max_bytes: Positive post-transfer file-size limit in bytes.
        expected_sha256: Optional lowercase SHA-256 digest to verify after transfer.

    Returns:
        Destination path after a bounded regular file is published.

    Raises:
        FileExistsError: Destination already exists.
        ValueError: URL or produced file violates the declared validation policy.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("retrieval requires HTTPS without embedded credentials")
    return _publish_retrieval(
        lambda path: retrieve(url, path), destination, max_bytes, expected_sha256
    )


def retrieve_dataset(
    product: str,
    request: dict[str, object],
    retrieve: Callable[[str, dict[str, object], str], object],
    destination: Path,
    max_bytes: int,
    expected_sha256: str | None,
) -> Path:
    """Publish an injected SDK retrieval; isolate nested caller request values.

    No SDK construction, credentials, timeouts or retries are introduced.
    Size/digest are checked after transfer, not enforced during SDK execution.

    Args:
        product: Native dataset or product identifier passed to the SDK callable.
        request: Nonempty nested service request, deep-copied before SDK transfer.
        retrieve: Caller-owned SDK function receiving product, request and local path.
        destination: New local path for the retrieved native dataset.
        max_bytes: Positive post-transfer file-size limit in bytes.
        expected_sha256: Optional lowercase SHA-256 digest to verify after transfer.

    Returns:
        Destination after the transferred regular file passes validation.

    Raises:
        FileExistsError: Destination already exists.
        ValueError: Request or produced file violates the declared validation policy.
    """
    if not request:
        raise ValueError("explicit request required")
    return _publish_retrieval(
        lambda path: retrieve(product, deepcopy(request), str(path)),
        destination,
        max_bytes,
        expected_sha256,
    )


def _publish_retrieval(
    retrieve: Callable[[Path], object],
    destination: Path,
    max_bytes: int,
    expected_sha256: str | None,
) -> Path:
    if max_bytes <= 0:
        raise ValueError("positive post-transfer byte bound required")
    if expected_sha256 is not None and (
        len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("expected digest must be lowercase SHA-256")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=destination.parent))
    try:
        target = directory / "native"
        retrieve(target)
        if (
            target.is_symlink()
            or not target.is_file()
            or not 0 < target.stat().st_size <= max_bytes
        ):
            raise ValueError("retrieval did not produce a bounded nonempty regular file")
        if expected_sha256 is not None and file_sha256(target) != expected_sha256:
            raise ValueError("retrieval checksum mismatch")
        os.link(target, destination)
        return destination
    finally:
        shutil.rmtree(directory)


def _download(
    url: str,
    destination: Path,
    expected_sha256: str | None,
    timeout_s: float,
    max_bytes: int,
    open_url: Callable[..., Any],
) -> Path:
    _validate_url(url)
    if not isfinite(timeout_s) or timeout_s <= 0 or max_bytes <= 0:
        raise ValueError("timeout and byte bound must be positive")
    if expected_sha256 is not None and (
        len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("expected digest must be lowercase SHA-256")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with open_url(url, timeout=timeout_s) as response:
            _validate_url(response.geturl())
            length = response.headers.get("Content-Length")
            declared_size = None
            if length is not None:
                try:
                    declared_size = int(length)
                except ValueError as error:
                    raise ValueError("invalid response Content-Length") from error
                if declared_size <= 0 or declared_size > max_bytes:
                    raise ValueError("response Content-Length exceeds byte bound or is empty")
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                digest = sha256()
                size = 0
                while block := response.read(min(1024 * 1024, max_bytes - size + 1)):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("download exceeds byte bound")
                    stream.write(block)
                    digest.update(block)
                if size == 0:
                    raise ValueError("download is empty")
                if declared_size is not None and size != declared_size:
                    raise ValueError("incomplete response Content-Length")
                stream.flush()
                os.fsync(stream.fileno())
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError("download checksum mismatch")
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("download requires an unauthenticated HTTP(S) URL")


def _https(url: str) -> None:
    _validate_url(url)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.port not in (None, 443):
        raise ValueError("bearer transfer requires HTTPS on port 443")


class _BearerRedirect(HTTPRedirectHandler):
    """Per-transfer urllib resource policy; never share token-bearing openers."""

    def __init__(self, token: str, hosts: tuple[str, ...]) -> None:
        self.token = token
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
            if urlsplit(newurl).hostname in self.hosts:
                redirected.add_header("Authorization", f"Bearer {self.token}")
        return redirected
