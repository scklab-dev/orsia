"""Mocked transport safety and atomic no-clobber publication."""

from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from orsia_data import acquisition, download_file


class Response(BytesIO):
    headers = Message()

    def geturl(self) -> str:
        return "https://example.invalid/data"


def test_download_digest_and_no_clobber(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: Response(b"data"))
    path = tmp_path / "sub" / "data"
    assert (
        download_file(
            "https://example.invalid/data", path, expected_sha256=sha256(b"data").hexdigest()
        )
        == path
    )
    assert path.read_bytes() == b"data" and list(path.parent.iterdir()) == [path]
    with pytest.raises(FileExistsError):
        download_file("https://example.invalid/data", path)
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        download_file("https://example.invalid/data", path)


@pytest.mark.parametrize(
    "body,options",
    [
        (b"", {}),
        (b"data", {"max_bytes": 2}),
        (b"data", {"expected_sha256": "0" * 64}),
    ],
)
def test_failed_transfer_cleanup(tmp_path: Path, monkeypatch, body: bytes, options: dict) -> None:
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: Response(body))
    with pytest.raises(ValueError):
        download_file("https://example.invalid/data", tmp_path / "data", **options)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("length", ["999", "-1", "bad", "1", "0"])
def test_invalid_or_incomplete_content_length(tmp_path: Path, monkeypatch, length: str) -> None:
    response = Response(b"data")
    response.headers = Message()
    response.headers["Content-Length"] = length
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="Content-Length"):
        download_file("https://example.invalid/data", tmp_path / "data", max_bytes=10)
    assert list(tmp_path.iterdir()) == []


def test_exact_byte_bound_and_content_length(tmp_path: Path, monkeypatch) -> None:
    response = Response(b"data")
    response.headers = Message()
    response.headers["Content-Length"] = "4"
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: response)
    assert (
        download_file("https://example.invalid/data", tmp_path / "data", max_bytes=4).read_bytes()
        == b"data"
    )


@pytest.mark.parametrize(
    "url,options",
    [
        ("file:///x", {}),
        ("https://", {}),
        ("https://user:pass@host/data", {}),
        ("https://host/data", {"timeout_s": 0}),
        ("https://host/data", {"timeout_s": float("nan")}),
        ("https://host/data", {"max_bytes": 0}),
        ("https://host/data", {"expected_sha256": "bad"}),
        ("https://host/data", {"expected_sha256": "G" * 64}),
    ],
)
def test_invalid_transport(tmp_path: Path, url: str, options: dict) -> None:
    with pytest.raises(ValueError):
        download_file(url, tmp_path / "data", **options)
    assert list(tmp_path.iterdir()) == []


def test_failures_redirects_and_publication_race(tmp_path: Path, monkeypatch) -> None:
    def fail(*a, **kw):
        raise HTTPError("https://example.invalid", 404, "absent", Message(), None)

    monkeypatch.setattr(acquisition, "urlopen", fail)
    with pytest.raises(HTTPError):
        download_file("https://example.invalid/data", tmp_path / "data")

    class BadRedirect(Response):
        def geturl(self) -> str:
            return "file:///bad"

    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: BadRedirect(b"data"))
    with pytest.raises(ValueError):
        download_file("https://example.invalid/data", tmp_path / "data")

    class Partial(Response):
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError("partial")
            return super().read(size)

    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: Partial(b"data"))
    with pytest.raises(TimeoutError):
        download_file("https://example.invalid/data", tmp_path / "data")
    assert list(tmp_path.iterdir()) == []

    def race(*a, **kw):
        (tmp_path / "data").write_bytes(b"other writer")
        return Response(b"data")

    monkeypatch.setattr(acquisition, "urlopen", race)
    with pytest.raises(FileExistsError):
        download_file("https://example.invalid/data", tmp_path / "data")
    assert (tmp_path / "data").read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [tmp_path / "data"]
