"""Offline geographic tile-selection and native transfer contracts."""

from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.response import addinfourl

import pytest
from orsia_data import acquisition, download_copernicus_dem
from orsia_data.services import copernicus_dem as terrain


@pytest.mark.parametrize(
    ("bbox", "keys"),
    [
        ((2.2, 48.2, 2.8, 48.8), ["N48_00_E002_00"]),
        ((2.0, 48.0, 3.0, 49.0), ["N48_00_E002_00"]),
        (
            (2.8, 48.8, 3.2, 49.2),
            ["N48_00_E002_00", "N48_00_E003_00", "N49_00_E002_00", "N49_00_E003_00"],
        ),
        (
            (-1.2, -1.2, -0.2, -0.2),
            ["S02_00_W002_00", "S02_00_W001_00", "S01_00_W002_00", "S01_00_W001_00"],
        ),
        (
            (-0.2, -0.2, 0.2, 0.2),
            ["S01_00_W001_00", "S01_00_E000_00", "N00_00_W001_00", "N00_00_E000_00"],
        ),
        ((179.2, 0.2, -179.2, 0.8), ["N00_00_E179_00", "N00_00_W180_00"]),
        ((180.0, 0.2, -179.2, 0.8), ["N00_00_W180_00"]),
        ((179.2, 0.2, -180.0, 0.8), ["N00_00_E179_00"]),
        ((179.0, 89.0, 180.0, 90.0), ["N89_00_E179_00"]),
        ((-180.0, -90.0, -179.0, -89.0), ["S90_00_W180_00"]),
    ],
)
@pytest.mark.parametrize("resolution_m", [30, 90])
def test_bbox_selection(monkeypatch, tmp_path, bbox, keys, resolution_m):
    calls = []

    def download(url, path, **kwargs):
        calls.append((url, path, kwargs))
        return path

    monkeypatch.setattr(terrain, "download_file", download)
    directory = tmp_path / "dem"
    paths = download_copernicus_dem(
        bbox=bbox,
        resolution_m=resolution_m,
        directory=directory,
        timeout_s=12.0,
        max_bytes=1024,
        max_tiles=len(keys),
    )
    code = 10 if resolution_m == 30 else 30
    names = [f"Copernicus_DSM_COG_{code}_{key}_DEM" for key in keys]
    assert paths == [directory / f"{name}.tif" for name in names]
    assert calls == [
        (
            f"https://copernicus-dem-{resolution_m}m.s3.amazonaws.com/{name}/{name}.tif",
            path,
            {"timeout_s": 12.0, "max_bytes": 1024},
        )
        for name, path in zip(names, paths, strict=True)
    ]
    assert not directory.exists()  # Only the transport creates directories.


@pytest.mark.parametrize("bbox", [(-180.0, 0.0, 180.0, 1.0), (0.8, 0.0, 0.2, 1.0)])
def test_world_and_near_global_wrap_have_no_duplicates(monkeypatch, tmp_path, bbox):
    monkeypatch.setattr(terrain, "download_file", lambda url, path, **kw: path)
    paths = download_copernicus_dem(bbox=bbox, resolution_m=90, directory=tmp_path, max_tiles=360)
    assert len(paths) == len(set(paths)) == 360


@pytest.mark.parametrize(
    "bbox",
    [
        (float("nan"), 0.0, 1.0, 1.0),
        (0.0, float("inf"), 1.0, 1.0),
        (0.0, 0.0, float("-inf"), 1.0),
        (0.0, 0.0, 1.0, float("nan")),
        (-181.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 181.0, 1.0),
        (0.0, -91.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 91.0),
        (0.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
        (180.0, 0.0, -180.0, 1.0),
    ],
)
def test_invalid_bbox_has_no_effects(monkeypatch, tmp_path, bbox):
    monkeypatch.setattr(terrain, "download_file", lambda *a, **kw: pytest.fail("no download"))
    directory = tmp_path / "dem"
    with pytest.raises(ValueError, match="bbox"):
        download_copernicus_dem(bbox=bbox, resolution_m=90, directory=directory)
    assert not directory.exists()


@pytest.mark.parametrize("max_tiles", [0, -1, 3])
def test_tile_limit_has_no_effects(monkeypatch, tmp_path, max_tiles):
    monkeypatch.setattr(terrain, "download_file", lambda *a, **kw: pytest.fail("no download"))
    directory = tmp_path / "dem"
    with pytest.raises(ValueError, match="selects 4 tiles"):
        download_copernicus_dem(
            bbox=(2.8, 48.8, 3.2, 49.2),
            resolution_m=90,
            directory=directory,
            max_tiles=max_tiles,
        )
    assert not directory.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"directory": Path("dem")},
        {"south_deg": 48, "destination": Path("dem.tif")},
        {"south_deg": 48, "west_deg": 2},
        {"south_deg": 48, "west_deg": 2, "destination": Path("dem.tif"), "directory": Path("d")},
        {"bbox": (2.0, 48.0, 3.0, 49.0)},
        {"bbox": (2.0, 48.0, 3.0, 49.0), "directory": Path("dem"), "south_deg": 48},
        {"bbox": (2.0, 48.0, 3.0, 49.0), "directory": Path("dem"), "west_deg": 2},
        {"bbox": (2.0, 48.0, 3.0, 49.0), "directory": Path("dem"), "destination": Path("x")},
        {"bbox": (2.0, 48.0, 3.0, 49.0), "directory": Path("dem"), "expected_sha256": "0" * 64},
    ],
)
def test_incomplete_or_mixed_modes(monkeypatch, arguments):
    monkeypatch.setattr(terrain, "download_file", lambda *a, **kw: pytest.fail("no download"))
    with pytest.raises(ValueError):
        download_copernicus_dem(resolution_m=90, **arguments)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_all_destinations_checked_before_first_transfer(monkeypatch, tmp_path, kind):
    target = tmp_path / "Copernicus_DSM_COG_30_N49_00_E003_00_DEM.tif"
    if kind == "file":
        target.write_bytes(b"keep me")
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(terrain, "download_file", lambda *a, **kw: pytest.fail("no download"))
    with pytest.raises(FileExistsError):
        download_copernicus_dem(bbox=(2.8, 48.8, 3.2, 49.2), resolution_m=90, directory=tmp_path)
    assert list(tmp_path.iterdir()) == [target]
    if kind == "file":
        assert target.read_bytes() == b"keep me"


@pytest.mark.parametrize("failure", [None, "missing", "truncated", "interrupted", "limit"])
def test_bbox_real_transport_publication(monkeypatch, tmp_path, failure):
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        if len(calls) == 2:
            if failure == "missing":
                raise HTTPError(url, 404, "missing tile", {}, None)
            if failure == "interrupted":
                raise KeyboardInterrupt
        headers = Message()
        headers["Content-Length"] = "8" if failure == "truncated" and len(calls) == 2 else "4"
        return addinfourl(BytesIO(b"tile"), headers, url)

    monkeypatch.setattr(acquisition, "urlopen", open_url)
    directory = tmp_path / "nested" / "dem"

    def download():
        return download_copernicus_dem(
            bbox=(2.8, 48.8, 3.2, 49.2),
            resolution_m=90,
            directory=directory,
            max_bytes=3 if failure == "limit" else 10,
        )

    if failure is None:
        paths = download()
        assert len(paths) == 4
        assert all(path.read_bytes() == b"tile" for path in paths)
        assert set(directory.iterdir()) == set(paths)
    else:
        error = {"missing": HTTPError, "interrupted": KeyboardInterrupt}.get(failure, ValueError)
        with pytest.raises(error):
            download()
        retained = list(directory.iterdir())
        if failure == "limit":
            assert retained == []
            assert len(calls) == 1
        else:
            assert len(calls) == 2
            assert [path.name for path in retained] == [
                "Copernicus_DSM_COG_30_N48_00_E002_00_DEM.tif"
            ]
            assert retained[0].read_bytes() == b"tile"


def test_original_single_tile_forwards_digest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(terrain, "download_file", lambda url, path, **kw: calls.append(kw) or path)
    target = tmp_path / "chosen.tif"
    path = download_copernicus_dem(
        south_deg=48,
        west_deg=2,
        resolution_m=90,
        destination=target,
        expected_sha256="a" * 64,
    )
    assert path == target
    assert calls == [
        {"timeout_s": 30.0, "max_bytes": 128 * 1024 * 1024, "expected_sha256": "a" * 64}
    ]
