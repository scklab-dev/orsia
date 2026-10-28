"""Offline acquisition tests with synthetic NetCDF bytes; never call live AER."""

from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import numpy as np
import pytest
from orsia_data import (
    acquisition,
    download_mt_ckd_water,
    fetch_mt_ckd_water,
    read_mt_ckd_water,
)
from orsia_data.spectroscopy import continuum
from scipy.io import netcdf_file

URL = "https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/4.3/data/absco-ref_wv-mt-ckd.nc"
FILENAME = "absco-ref_wv-mt-ckd.nc"


@pytest.fixture
def native_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "fixture.nc"
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("wavenumber", 7)
        dataset.createDimension("scalar", 1)
        for name, values in {
            "wavenumbers": np.arange(800.0, 1101.0, 50.0),
            "self_absco_ref": np.full(7, 1e-24),
            "for_absco_ref": np.full(7, 2e-24),
            "self_texp": np.full(7, 3.0),
        }.items():
            dataset.createVariable(name, "d", ("wavenumber",))[:] = values
        for name, value in (("ref_press", 1013.0), ("ref_temp", 296.0)):
            dataset.createVariable(name, "d", ("scalar",))[:] = value
    return path.read_bytes()


@pytest.mark.parametrize("failure", [None, "bound", "digest", "schema", "http", "interrupt"])
def test_transfer_decode_cleanup_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_bytes: bytes,
    failure: str | None,
) -> None:
    body = b"<html>error</html>" if failure == "schema" else native_bytes
    calls: list[str] = []

    class Response(BytesIO):
        headers = Message()

        def geturl(self) -> str:
            return URL

    def response(url: str, *, timeout: float) -> Response:
        assert url == URL and timeout == 3.0
        calls.append(url)
        if failure == "http":
            raise HTTPError(url, 404, "missing", Message(), None)
        if failure == "interrupt":
            raise KeyboardInterrupt
        return Response(body)

    monkeypatch.setattr(acquisition, "urlopen", response)
    directory = tmp_path / "new"
    keep = tmp_path / "keep"
    keep.write_bytes(b"keep")
    options: dict[str, Any] = dict(
        edition="4.3",
        directory=directory,
        temperature_range_k=(200.0, 350.0),
        timeout_s=3.0,
        max_bytes=2 if failure == "bound" else len(body),
        expected_sha256="0" * 64 if failure == "digest" else sha256(body).hexdigest(),
    )
    if failure is not None:
        with pytest.raises((ValueError, HTTPError, KeyboardInterrupt)):
            fetch_mt_ckd_water(**options)
        assert not directory.exists()
    else:
        reference = fetch_mt_ckd_water(**options)
        assert directory.joinpath(FILENAME).read_bytes() == body
        assert reference.reference_pressure_pa == 101300.0
        assert reference.reference_temperature_k == 296.0
        assert reference.temperature_range_k == (200.0, 350.0)
        assert reference.self_absco_ref.flags.writeable
        with pytest.raises(FileExistsError):
            fetch_mt_ckd_water(**options)
        assert calls == [URL]

        def offline(*args: Any, **kwargs: Any) -> None:
            pytest.fail("offline replay accessed the network")

        monkeypatch.setattr(acquisition, "urlopen", offline)
        replay = read_mt_ckd_water(directory / FILENAME, temperature_range_k=(200.0, 350.0))
        np.testing.assert_array_equal(replay.self_absco_ref, reference.self_absco_ref)
    assert keep.read_bytes() == b"keep"


def test_download_file_only_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_bytes: bytes
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def download(url: str, destination: Path, **options: Any) -> Path:
        calls.append((url, options))
        destination.write_bytes(native_bytes)
        return destination

    monkeypatch.setattr(continuum, "download_file", download)
    path = tmp_path / "chosen-name.nc"
    assert (
        download_mt_ckd_water(
            edition="4.3",
            destination=path,
            timeout_s=4.0,
            max_bytes=100000,
            expected_sha256="a" * 64,
        )
        == path
    )
    assert path.read_bytes() == native_bytes
    assert calls == [(URL, dict(timeout_s=4.0, max_bytes=100000, expected_sha256="a" * 64))]

    monkeypatch.setattr(continuum, "download_file", acquisition.download_file)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("existing file reached the network")

    monkeypatch.setattr(acquisition, "urlopen", unexpected)
    # The real transport rejects existing bytes before trying to reach AER.
    with pytest.raises(FileExistsError):
        download_mt_ckd_water(edition="4.3", destination=path)
    assert path.read_bytes() == native_bytes


@pytest.mark.parametrize("edition", ["current", "4.2", "../master", ""])
def test_unsupported_edition_has_no_effects(tmp_path: Path, edition: Any) -> None:
    directory = tmp_path / "new"
    with pytest.raises(ValueError, match="edition 4.3"):
        fetch_mt_ckd_water(edition=edition, directory=directory, temperature_range_k=(200.0, 350.0))
    with pytest.raises(ValueError, match="edition 4.3"):
        download_mt_ckd_water(edition=edition, destination=directory / FILENAME)
    assert not directory.exists()


def test_existing_empty_directory_fails_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("existing directory reached transport")

    monkeypatch.setattr(continuum, "download_file", unexpected)
    with pytest.raises(FileExistsError):
        fetch_mt_ckd_water(edition="4.3", directory=tmp_path, temperature_range_k=(200.0, 350.0))
    assert list(tmp_path.iterdir()) == []


def test_public_exports() -> None:
    from orsia_data import spectroscopy

    assert spectroscopy.download_mt_ckd_water is download_mt_ckd_water
    assert spectroscopy.fetch_mt_ckd_water is fetch_mt_ckd_water
