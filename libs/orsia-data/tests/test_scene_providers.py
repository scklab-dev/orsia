"""Offline endpoint, selection and atomic publication contracts."""

import io
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from orsia_data import (
    download_cdse,
    download_copernicus_dem,
    download_earthdata,
    earthdata_links,
    http,
    retrieve_file,
    search_earthdata,
    search_sentinel2_l2a,
)
from orsia_data.services import copernicus_dem as terrain

START, END = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
BBOX = (-1.0, 40.0, 2.0, 42.0)


def test_catalogue_selection(monkeypatch):
    calls = []

    def open_page(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(
            json.dumps({"value": [], "@odata.count": 0, "feed": {"entry": []}}).encode()
        )

    monkeypatch.setattr(http, "build_opener", lambda *a: SimpleNamespace(open=open_page))
    result = search_sentinel2_l2a(bbox=BBOX, start=START, end=END, top=2, skip=4)
    assert result["value"] == []
    query = parse_qs(urlsplit(calls[-1]).query)
    assert query["$top"] == ["2"] and query["$skip"] == ["4"]
    assert "S2MSI2A" in query["$filter"][0]
    assert "POLYGON((-1.0 40.0,2.0 40.0,2.0 42.0,-1.0 42.0,-1.0 40.0))" in query["$filter"][0]
    assert "2024-01-01T00:00:00Z" in query["$filter"][0]
    search_earthdata(product="ECO_L2T_LSTE.002", bbox=BBOX, temporal=(START, END))
    query = parse_qs(urlsplit(calls[-1]).query)
    assert query["short_name"] == ["ECO_L2T_LSTE"] and query["version"] == ["002"]
    search_earthdata(product="AG100.003", bbox=BBOX, temporal=None, page_number=2)
    query = parse_qs(urlsplit(calls[-1]).query)
    assert query["version"] == ["003"] and "temporal" not in query
    assert query["page_num"] == ["2"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bbox": (2.0, 0.0, -2.0, 1.0)},
        {"top": 0},
        {"skip": -1},
        {"start": datetime(2024, 1, 1)},
        {"end": START},
        {"timeout_s": float("nan")},
    ],
)
def test_cdse_bad_selection(kwargs):
    with pytest.raises(ValueError):
        search_sentinel2_l2a(**({"bbox": BBOX, "start": START, "end": END} | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [{"product": "AG100.002"}, {"page_number": 0}, {"page_size": 2001}, {"bbox": (-181, 0, 0, 1)}],
)
def test_cmr_bad_selection(kwargs):
    with pytest.raises(ValueError):
        search_earthdata(**({"product": "AG100.003", "bbox": BBOX, "temporal": None} | kwargs))


@pytest.mark.parametrize("content", [b"[]", b"{}", b"not json", b"x" * (16 * 1024 * 1024 + 1)])
def test_bad_pages(monkeypatch, content):
    monkeypatch.setattr(
        http, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **kw: io.BytesIO(content))
    )
    with pytest.raises(ValueError):
        search_sentinel2_l2a(bbox=BBOX, start=START, end=END)
    with pytest.raises(ValueError):
        search_earthdata(product="AG100.003", bbox=BBOX, temporal=None)


def test_links():
    data = {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": "https://example.test/a"}
    granule = {
        "links": [
            data,
            data,
            data | {"inherited": True},
            data | {"rel": "browse"},
            data | {"href": "s3://bucket/a"},
        ]
    }
    assert earthdata_links(granule) == ("https://example.test/a",)
    assert earthdata_links({}) == ()


def test_authenticated_publication(tmp_path: Path):
    calls = []

    def transfer(url, path):
        calls.append(url)
        path.write_bytes(b"native bytes")

    destination = tmp_path / "native"
    identifier = "00112233-4455-6677-8899-aabbccddeeff"
    download_cdse(
        identifier,
        destination,
        retrieve=transfer,
        max_bytes=100,
        expected_sha256=sha256(b"native bytes").hexdigest(),
    )
    assert f"Products({identifier})/$value" in calls[0]
    assert destination.read_bytes() == b"native bytes"
    with pytest.raises(FileExistsError):
        download_cdse(identifier, destination, retrieve=transfer, max_bytes=100)
    assert len(calls) == 1
    download_earthdata(
        "https://example.test/native", tmp_path / "earth", retrieve=transfer, max_bytes=100
    )
    assert calls[-1] == "https://example.test/native"
    assert set(p.name for p in tmp_path.iterdir()) == {"native", "earth"}


@pytest.mark.parametrize(
    "failure",
    ["empty", "oversize", "digest", "missing", "symlink", "race", "interrupt", "directory"],
)
def test_publication_failures(tmp_path: Path, failure):
    destination = tmp_path / "final"

    def transfer(url, path):
        if failure == "missing":
            return
        if failure == "directory":
            path.mkdir()
            return
        if failure == "symlink":
            path.symlink_to(tmp_path / "absent")
            return
        path.write_bytes(b"" if failure == "empty" else b"native")
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "race":
            destination.write_bytes(b"winner")

    with pytest.raises((ValueError, FileExistsError, KeyboardInterrupt)):
        retrieve_file(
            "https://example.test/a",
            destination,
            retrieve=transfer,
            max_bytes=1 if failure == "oversize" else 100,
            expected_sha256="0" * 64 if failure == "digest" else None,
        )
    if failure == "race":
        assert destination.read_bytes() == b"winner"
    else:
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url,max_bytes,digest",
    [
        ("http://example.test/a", 2, None),
        ("https://user:secret@example.test/a", 2, None),
        ("https://example.test/a", 0, None),
        ("https://example.test/a", 2, "BAD"),
    ],
)
def test_transfer_validation(tmp_path, url, max_bytes, digest):
    with pytest.raises(ValueError):
        retrieve_file(
            url,
            tmp_path / "out",
            retrieve=lambda *a: pytest.fail("not called"),
            max_bytes=max_bytes,
            expected_sha256=digest,
        )


def test_dem_endpoint(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        terrain, "download_file", lambda url, target, **kw: calls.append(url) or target
    )
    result = download_copernicus_dem(
        south_deg=-1, west_deg=-2, resolution_m=30, destination=tmp_path / "dem"
    )
    assert result == tmp_path / "dem"
    assert calls[-1].endswith("Copernicus_DSM_COG_10_S01_00_W002_00_DEM.tif")
    download_copernicus_dem(south_deg=1, west_deg=2, resolution_m=90, destination=result)
    assert "90m" in calls[-1] and "COG_30_N01_00_E002_00" in calls[-1]
    with pytest.raises(ValueError):
        download_copernicus_dem(south_deg=90, west_deg=1, resolution_m=30, destination=result)
    with pytest.raises(ValueError):
        download_copernicus_dem(south_deg=1, west_deg=180, resolution_m=30, destination=result)
    with pytest.raises(ValueError):
        download_copernicus_dem(south_deg=1, west_deg=1, resolution_m=10, destination=result)  # type: ignore
