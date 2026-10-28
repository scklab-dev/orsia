"""MIPAS resolver and real transport publication using synthetic HTTP bytes."""

from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from orsia_data import acquisition, download_mipas_profile, read_rfm_atmosphere
from orsia_data.services.mipas import download_mipas_profile as service_download

BODY = b"1\n*HGT [km]\n0\n*PRE [mb]\n1000\n*TEM [K]\n290\n*CO2 [ppmv]\n363\n*END\n"


class Response(BytesIO):
    def __init__(self, body: bytes = BODY, length: str | None = None):
        super().__init__(body)
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = length

    def geturl(self) -> str:
        return "https://eodg.atm.ox.ac.uk/RFM/atm/equ.atm"


@pytest.mark.parametrize(
    "profile,filename",
    [
        ("midlatitude_day", "day.atm"),
        ("midlatitude_night", "ngt.atm"),
        ("polar_winter", "win.atm"),
        ("polar_summer", "sum.atm"),
        ("equatorial_day", "equ.atm"),
    ],
)
def test_resolve_transfer_and_replay(tmp_path: Path, monkeypatch, profile, filename) -> None:
    assert service_download is download_mipas_profile
    calls = []

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        return Response(length=str(len(BODY)))

    monkeypatch.setattr(acquisition, "urlopen", open_url)
    path = tmp_path / "native" / filename
    assert (
        download_mipas_profile(
            edition="2001",
            profile=profile,
            destination=path,
            timeout_s=7.0,
            max_bytes=len(BODY),
            expected_sha256=sha256(BODY).hexdigest(),
        )
        == path
    )
    assert calls == [(f"https://eodg.atm.ox.ac.uk/RFM/atm/{filename}", 7.0)]
    assert path.read_bytes() == BODY and list(path.parent.iterdir()) == [path]
    assert read_rfm_atmosphere(path)[3]["CO2"][0] == 363e-6
    assert len(calls) == 1


def test_defaults_download_only(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def open_url(url, **kwargs):
        calls.append(url)
        return Response(b"arbitrary transport bytes")

    monkeypatch.setattr(acquisition, "urlopen", open_url)
    path = tmp_path / "default.atm"
    assert download_mipas_profile(destination=path).read_bytes() == b"arbitrary transport bytes"
    assert calls == ["https://eodg.atm.ox.ac.uk/RFM/atm/equ.atm"]


@pytest.mark.parametrize(
    "options", [{"edition": "2011"}, {"profile": "../day"}, {"profile": "day"}]
)
def test_invalid_selection_before_effects(tmp_path: Path, monkeypatch, options: dict) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected transfer")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    with pytest.raises(ValueError):
        download_mipas_profile(destination=tmp_path / "new" / "native", **options)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "body,length,options",
    [
        (BODY, None, {"expected_sha256": "0" * 64}),
        (BODY, None, {"expected_sha256": "bad"}),
        (BODY, None, {"max_bytes": 1}),
        (BODY, str(len(BODY) + 1), {}),
        (b"", None, {}),
        (BODY, None, {"timeout_s": 0}),
    ],
)
def test_integrity_failure_cleanup(
    tmp_path: Path, monkeypatch, body, length, options: dict
) -> None:
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: Response(body, length))
    with pytest.raises(ValueError):
        download_mipas_profile(destination=tmp_path / "native", **options)
    assert list(tmp_path.iterdir()) == []


def test_existing_file_and_symlink_before_network(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    path = tmp_path / "native"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        download_mipas_profile(destination=path)
    assert path.read_bytes() == b"existing"
    path.unlink()
    path.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        download_mipas_profile(destination=path)
    assert path.is_symlink()


def test_publication_race_and_http_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "native"

    def fail(*args, **kwargs):
        raise HTTPError("https://eodg.atm.ox.ac.uk", 503, "unavailable", Message(), None)

    monkeypatch.setattr(acquisition, "urlopen", fail)
    with pytest.raises(HTTPError):
        download_mipas_profile(destination=path)
    assert list(tmp_path.iterdir()) == []

    def race(*args, **kwargs):
        path.write_bytes(b"other writer")
        return Response()

    monkeypatch.setattr(acquisition, "urlopen", race)
    with pytest.raises(FileExistsError):
        download_mipas_profile(destination=path)
    assert path.read_bytes() == b"other writer" and list(tmp_path.iterdir()) == [path]


def test_interrupted_transfer(tmp_path: Path, monkeypatch) -> None:
    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError("transfer interrupted")
            return super().read(size)

    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: Interrupted())
    with pytest.raises(TimeoutError):
        download_mipas_profile(destination=tmp_path / "native")
    assert list(tmp_path.iterdir()) == []
