"""Real HAPI, synthetic HTTP bytes; optional extra, not live HITRAN qualification."""

import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from orsia_data import fetch_hitran_lines

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("hapi") is None, reason="requires optional hitran extra"
)


@pytest.mark.parametrize("empty", [False, True])
def test_real_hapi_fetch_with_synthetic_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty: bool,
) -> None:
    record = (
        (
            f"{1:2d}1{950.0:12.6f}{1e-25:10.3E}{1.0:10.3E}"
            f"{0.1:5.3f}{0.2:5.3f}{0.0:10.3f}{0.7:4.2f}{0.0:8.5f}"
        )
        + " " * 60
        + "0" * 18
        + " "
        + f"{1.0:7.1f}{1.0:7.1f}\n"
    )
    assert len(record.rstrip("\n")) == 160
    payload = b"" if empty else record.encode("ascii")
    import hapi  # type: ignore[import-not-found]

    def response(url: str) -> BytesIO:
        assert "/lbl/api?" in url
        assert "iso_ids_list=1" in url
        assert "numin=900.0" in url and "numax=1000.0" in url
        return BytesIO(payload)

    monkeypatch.setattr(hapi.urllib2, "urlopen", response)
    directory = tmp_path / "acquired"
    database = fetch_hitran_lines(
        molecule_id=1,
        isotopologue_id=1,
        interval_cm1=(900.0, 1000.0),
        edition="current",
        directory=directory,
    )
    assert len(database.lines) == (0 if empty else 1)
    assert database.coverage_cm1 == (900.0, 1000.0)
    assert directory.joinpath("lines.data").read_bytes() == payload
    assert directory.joinpath("lines.header").is_file()
    if not empty:
        assert database.lines[0].centre_cm1 == 950.0
        assert database.lines[0].strength_ref_cm_per_molecule == 1e-25
