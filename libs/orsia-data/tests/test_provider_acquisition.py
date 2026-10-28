"""Provider acquisition contracts with synthetic bytes; no live provider access."""

import sys
from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import get_ident
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pytest
from orsia_data import acquisition, fetch_hitran_lines, fetch_solar_spectrum, read_hitran_lines
from orsia_data.services import lisird

SOLAR = (
    "wavelength (nm),irradiance (W/m^2/nm),uncertainty (W/m^2/nm),bandwidth (nm)\n"
    "500,1,0.1,0.025\n750,2,0.2,0.025\n1000,3,0.3,0.025\n"
)
LINE = (
    f"{1:2d}1{950.0:12.6f}{1e-25:10.3E}{1.0:10.3E}"
    f"{0.1:5.3f}{0.2:5.3f}{0.0:10.3f}{0.7:4.2f}{0.0:8.5f}"
).ljust(160) + "\n"


@pytest.mark.parametrize("failure", [None, "bound", "digest", "schema"])
def test_solar_full_transport_decode_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    url = "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs.csv"
    body = ("<html>error</html>" if failure == "schema" else SOLAR).encode()

    class Response(BytesIO):
        headers = Message()

        def geturl(self) -> str:
            return url

    def response(requested_url: str, *, timeout: float) -> Response:
        assert requested_url == url and timeout == 3.0
        return Response(body)

    monkeypatch.setattr(acquisition, "urlopen", response)
    directory = tmp_path / "new"
    options: dict[str, Any] = dict(
        product="tsis1_hsrs",
        edition="current",
        directory=directory,
        timeout_s=3.0,
        max_bytes=2 if failure == "bound" else len(body),
        expected_sha256="0" * 64 if failure == "digest" else sha256(body).hexdigest(),
    )
    if failure is None:
        spectrum = fetch_solar_spectrum(**options)
        assert directory.joinpath("tsis1_hsrs.csv").read_bytes() == body
        np.testing.assert_array_equal(spectrum.irradiance_per_nm, [1, 2, 3])
    else:
        with pytest.raises(ValueError):
            fetch_solar_spectrum(**options)
        assert not directory.exists()


@pytest.mark.parametrize("product", ["tsis1_hsrs", "tsis1_hsrs_p025nm"])
def test_solar_selection_native_fields_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    product: Literal["tsis1_hsrs", "tsis1_hsrs_p025nm"],
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def download(url: str, destination: Path, **options: Any) -> Path:
        calls.append((url, options))
        destination.write_text(SOLAR)
        return destination

    monkeypatch.setattr(lisird, "download_file", download)
    directory = tmp_path / "new"
    spectrum = fetch_solar_spectrum(
        product=product,
        edition="current",
        directory=directory,
        timeout_s=4.0,
        max_bytes=4000,
        expected_sha256="a" * 64,
    )
    assert calls == [
        (
            f"https://lasp.colorado.edu/lisird/latis/dap/{product}.csv",
            dict(timeout_s=4.0, max_bytes=4000, expected_sha256="a" * 64),
        )
    ]
    assert (directory / f"{product}.csv").read_text() == SOLAR
    np.testing.assert_array_equal(spectrum.wavelengths_nm, [500, 750, 1000])
    np.testing.assert_array_equal(spectrum.irradiance_per_nm, [1, 2, 3])
    np.testing.assert_array_equal(spectrum.uncertainty_per_nm, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(spectrum.bandwidth_nm, [0.025] * 3)
    with pytest.raises(FileExistsError):
        fetch_solar_spectrum(product=product, edition="current", directory=directory)
    assert len(calls) == 1
    assert (directory / f"{product}.csv").read_text() == SOLAR


@pytest.mark.parametrize("failure", ["transport", "schema", "interrupt"])
def test_solar_failure_removes_only_owned_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    unrelated = tmp_path / "keep"
    unrelated.write_text("keep")

    def download(url: str, destination: Path, **options: Any) -> Path:
        destination.write_text("partial")
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "transport":
            raise OSError("network")
        return destination

    monkeypatch.setattr(lisird, "download_file", download)
    directory = tmp_path / "new"
    with pytest.raises((OSError, ValueError, KeyboardInterrupt)):
        fetch_solar_spectrum(product="tsis1_hsrs", edition="current", directory=directory)
    assert not directory.exists()
    assert unrelated.read_text() == "keep"


def test_unsupported_selections_do_not_create_directories(tmp_path: Path) -> None:
    directory = tmp_path / "new"
    invalid: Any = "unsupported"
    with pytest.raises(ValueError, match="product"):
        fetch_solar_spectrum(product=invalid, edition="current", directory=directory)
    with pytest.raises(ValueError, match="current"):
        fetch_solar_spectrum(product="tsis1_hsrs", edition=invalid, directory=directory)
    with pytest.raises(ValueError, match="current"):
        fetch_hitran_lines(
            molecule_id=1,
            isotopologue_id=1,
            interval_cm1=(900.0, 1000.0),
            edition=invalid,
            directory=directory,
        )
    assert not directory.exists()


@pytest.fixture
def fake_hapi(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Synthetic shared HAPI state, imported directly by the acquisition call."""
    module = ModuleType("hapi")

    def db_begin(directory: str) -> None:
        module.root = Path(directory)  # type: ignore[missing-attribute]

    def fetch(name: str, molecule: int, isotope: int, lower: float, upper: float) -> None:
        assert (name, molecule, isotope, lower, upper) == ("lines", 1, 1, 900, 1000)
        root = module.root  # type: ignore[missing-attribute]
        (root / "lines.data").write_text(LINE)
        (root / "lines.header").write_text("{}")

    monkeypatch.setattr(module, "db_begin", db_begin, raising=False)
    monkeypatch.setattr(module, "fetch", fetch, raising=False)
    monkeypatch.setitem(sys.modules, "hapi", module)
    return module


def acquire(directory: Path):
    return fetch_hitran_lines(
        molecule_id=1,
        isotopologue_id=1,
        interval_cm1=(900.0, 1000.0),
        edition="current",
        directory=directory,
    )


def test_hapi_direct_execution_and_offline_replay(
    tmp_path: Path,
    fake_hapi: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = [tmp_path / "first", tmp_path / "second"]
    thread = get_ident()
    cwd = Path.cwd()
    calls: list[Path] = []
    original_fetch = fake_hapi.fetch  # type: ignore[missing-attribute]

    def fetch(*args: Any) -> None:
        assert get_ident() == thread
        assert Path.cwd() == cwd
        calls.append(fake_hapi.root)  # type: ignore[missing-attribute]
        original_fetch(*args)

    monkeypatch.setattr(fake_hapi, "fetch", fetch)
    for directory in directories:
        database = acquire(directory)
        assert fake_hapi.root == directory.resolve()  # type: ignore[missing-attribute]
        assert database.coverage_cm1 == (900.0, 1000.0)
        assert len(database.lines) == 1
        assert database.lines[0].centre_cm1 == 950.0
        assert database == read_hitran_lines(
            directory / "lines.data",
            molecule_id=1,
            isotopologue_id=1,
            interval_cm1=(900.0, 1000.0),
        )
    assert calls == directories
    with pytest.raises(FileExistsError):
        acquire(directories[0])
    assert calls == directories
    assert directories[0].joinpath("lines.data").read_text() == LINE


def test_hapi_missing_extra_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hapi", None)
    with pytest.raises(ImportError):
        acquire(tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("failure", ["begin", "provider", "interrupt", "header", "decode"])
def test_hapi_or_decode_failure_cleanup(
    tmp_path: Path,
    fake_hapi: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    directory = tmp_path / "new"
    unrelated = tmp_path / "keep"
    unrelated.write_text("keep")
    error = KeyboardInterrupt() if failure == "interrupt" else OSError("provider error")

    def fail_begin(root: str) -> None:
        raise error

    def fetch(*args: Any) -> None:
        (directory / "lines.data").write_text("partial")
        if failure in {"provider", "interrupt"}:
            raise error
        if failure == "decode":
            (directory / "lines.header").write_text("{}")

    if failure == "begin":
        monkeypatch.setattr(fake_hapi, "db_begin", fail_begin)
    monkeypatch.setattr(fake_hapi, "fetch", fetch)
    with pytest.raises((OSError, KeyboardInterrupt, ValueError)) as caught:
        acquire(directory)
    if failure in {"begin", "provider", "interrupt"}:
        assert caught.value is error
    if failure != "begin":
        # Filesystem cleanup does not restore HAPI's shared database selection.
        assert fake_hapi.root == directory.resolve()  # type: ignore[missing-attribute]
    assert not directory.exists()
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize("record", ["", LINE.replace("950.000000", "850.000000"), " 2" + LINE[2:]])
def test_hapi_empty_or_wrong_coverage_or_identity(
    tmp_path: Path,
    fake_hapi: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
) -> None:
    directory = tmp_path / "new"

    def fetch(*args: Any) -> None:
        (directory / "lines.data").write_text(record)
        (directory / "lines.header").write_text("{}")

    monkeypatch.setattr(fake_hapi, "fetch", fetch)
    if record:
        with pytest.raises(ValueError, match="outside"):
            acquire(directory)
        assert not directory.exists()
    else:
        assert acquire(directory).lines == ()
        assert (directory / "lines.header").is_file()


def test_hapi_invalid_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fetch_hitran_lines(
            molecule_id=1,
            isotopologue_id=1,
            interval_cm1=(1000.0, 900.0),
            edition="current",
            directory=tmp_path / "new",
        )
    assert not (tmp_path / "new").exists()
