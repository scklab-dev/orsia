"""Credential routing and bounded transfer, using intercepted native urllib IO."""

import io
from urllib.request import Request

import pytest
from orsia_data import acquisition, download_bearer_file, download_cdse, download_earthdata


class Response(io.BytesIO):
    headers = {"Content-Length": "6"}

    def geturl(self):
        return "https://data.example/file"


def test_bearer_download_and_redirect_policy(monkeypatch, tmp_path):
    calls = []

    class Opener:
        def open(self, request, *, timeout):
            assert request.get_header("Authorization") == "Bearer example-token"
            calls.append(request.full_url)
            return Response(b"native")

    def build(handler):
        initial = Request(
            "https://data.example/file", headers={"Authorization": "Bearer example-token"}
        )
        same = handler.redirect_request(initial, None, 302, "", {}, "https://data.example/other")
        other = handler.redirect_request(
            initial, None, 302, "", {}, "https://signed.example/object"
        )
        assert same.get_header("Authorization") == "Bearer example-token"
        assert other.get_header("Authorization") is None
        for bad in (
            "http://data.example/file",
            "https://data.example:444/file",
            "https://user:pass@data.example/file",
            "file:///etc/passwd",
        ):
            with pytest.raises(ValueError):
                handler.redirect_request(initial, None, 302, "", {}, bad)
        return Opener()

    monkeypatch.setattr(acquisition, "build_opener", build)
    path = download_bearer_file(
        "https://data.example/file",
        tmp_path / "out",
        access_token="example-token",
        trusted_hosts=("data.example",),
    )
    assert path.read_bytes() == b"native"
    with pytest.raises(FileExistsError):
        download_bearer_file(
            "https://data.example/file",
            path,
            access_token="example-token",
            trusted_hosts=("data.example",),
        )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="byte bound"):
        download_bearer_file(
            "https://data.example/file",
            tmp_path / "too-large",
            access_token="example-token",
            trusted_hosts=("data.example",),
            max_bytes=1,
        )
    assert set(p.name for p in tmp_path.iterdir()) == {"out"}


@pytest.mark.parametrize(
    "token,hosts,url",
    [
        ("", ("data.example",), "https://data.example/a"),
        ("contains\nspace", ("data.example",), "https://data.example/a"),
        ("token", (), "https://data.example/a"),
        ("token", ("*.example",), "https://data.example/a"),
        ("token", ("DATA.example",), "https://data.example/a"),
        ("token", ("data.example",), "https://untrusted.example/a"),
        ("token", ("data.example",), "http://data.example/a"),
    ],
)
def test_invalid_bearer_before_io(tmp_path, token, hosts, url):
    with pytest.raises(ValueError):
        download_bearer_file(url, tmp_path / "out", access_token=token, trusted_hosts=hosts)
    assert not list(tmp_path.iterdir())


def test_provider_token_paths(monkeypatch, tmp_path):
    from orsia_data.services.cdse import odata
    from orsia_data.services.earthdata import cmr

    calls = []

    def download(url, destination, **kwargs):
        calls.append((url, kwargs))
        return destination

    monkeypatch.setattr(odata, "download_bearer_file", download)
    monkeypatch.setattr(cmr, "download_bearer_file", download)
    download_cdse(
        "00112233-4455-6677-8899-aabbccddeeff",
        tmp_path / "cdse",
        access_token="token",
        max_bytes=1000,
    )
    assert calls[-1][1]["trusted_hosts"] == (
        "download.dataspace.copernicus.eu",
        "zipper.dataspace.copernicus.eu",
    )
    download_earthdata(
        "https://data.example/a",
        tmp_path / "earth",
        access_token="token",
        trusted_hosts=("data.example",),
        max_bytes=1000,
    )
    assert calls[-1][1]["trusted_hosts"] == ("data.example",)
    for function, argument in (
        (download_cdse, "00112233-4455-6677-8899-aabbccddeeff"),
        (download_earthdata, "https://data.example/a"),
    ):
        with pytest.raises(ValueError, match="exactly one"):
            function(argument, tmp_path / "out", max_bytes=1000)
        with pytest.raises(ValueError, match="exactly one"):
            function(
                argument,
                tmp_path / "out",
                access_token="token",
                retrieve=lambda *a: None,
                max_bytes=1000,
            )
    with pytest.raises(ValueError, match="trusted_hosts"):
        download_earthdata(
            "https://data.example/a",
            tmp_path / "out",
            retrieve=lambda *a: None,
            trusted_hosts=("data.example",),
            max_bytes=1000,
        )
