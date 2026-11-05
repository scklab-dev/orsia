"""Offline scenario loading, manifest compatibility and natural-GIL checks."""

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import pytest
from orsia_data.acquisition import file_sha256
from pydantic import ValidationError

from notebooks.support.five_band_scenario import FiveBandScenario, load_scenario

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_resolves_named_inputs_and_preserves_manifest(tmp_path, monkeypatch):
    path, manifest_path, payload = _write_scenario(tmp_path)
    original = manifest_path.read_bytes()
    monkeypatch.chdir(tmp_path.parent)
    scenario, paths, provenance = load_scenario(path)
    inputs = scenario.inputs
    assert isinstance(scenario, FiveBandScenario)
    assert inputs.era5 == payload
    assert inputs.solar_csv == payload
    assert inputs.mipas.file == payload
    assert inputs.spectroscopy[0].lines == payload
    assert inputs.spectroscopy[0].partition == payload
    assert inputs.cross_sections[0].files == [payload]
    assert inputs.valid_time == datetime.fromisoformat("2024-01-15T12:00:00+00:00")
    assert inputs.coordinates == ("time", "pressure", "latitude", "longitude")
    assert inputs.spectroscopy[0].dry_air_mole_fraction == [0.00042]
    assert scenario.reference.cache_directory == tmp_path / "cache"
    assert paths == [manifest_path, payload, payload, payload, payload, payload, payload]
    assert provenance == {str(p): file_sha256(p) for p in [*paths, path]}
    assert manifest_path.read_bytes() == original
    entry = inputs.cross_sections[0].model_dump(mode="json")
    assert entry["source"] == "synthetic test fixture"
    assert entry["files"] == [str(payload)]
    assert FiveBandScenario.model_validate_json(scenario.model_dump_json()) == scenario


def test_scenario_verifies_original_checksums(tmp_path):
    path, _, payload = _write_scenario(tmp_path)
    payload.write_text("changed input\n")
    with pytest.raises(AssertionError, match="Input checksum mismatch"):
        load_scenario(path)


@pytest.mark.parametrize(
    "change", ["missing_manifest", "bad_manifest", "bad_points", "bad_band", "nonzero_azimuth"]
)
def test_scenario_reports_invalid_fields(tmp_path, change):
    path, _, _ = _write_scenario(tmp_path)
    settings = json.loads(path.read_text())
    if change == "missing_manifest":
        del settings["manifest"]
    elif change == "bad_manifest":
        settings["manifest"] = 42
    elif change == "bad_points":
        settings["points"] = 1
    elif change == "nonzero_azimuth":
        settings["physics"]["relative_azimuth_deg"] = 45.0
    else:
        settings["bands"]["UV"]["high_nm"] = 300
    path.write_text(json.dumps(settings))
    with pytest.raises(ValidationError):
        load_scenario(path)


@pytest.mark.parametrize("text", ["[]", "{"])
def test_scenario_reports_invalid_json_object(tmp_path, text):
    path = tmp_path / "scenario.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_scenario(path)


def test_pydantic_is_development_only():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for package in lock["package"]:
        if package["name"].startswith("orsia-"):
            assert all(entry["name"] != "pydantic" for entry in package.get("dependencies", []))


def test_scenario_native_import_validation_and_serialization_keep_gil_off(tmp_path):
    path, _, _ = _write_scenario(tmp_path)
    code = """
import json, os, sys
from pathlib import Path
observed = {"initial": sys._is_gil_enabled()}
import pydantic_core._pydantic_core as native
observed["native_import"] = sys._is_gil_enabled()
from notebooks.support.five_band_scenario import FiveBandScenario, load_scenario
scenario, _, _ = load_scenario(Path(sys.argv[1]))
observed["validation"] = sys._is_gil_enabled()
assert FiveBandScenario.model_validate_json(scenario.model_dump_json()) == scenario
observed["serialization"] = sys._is_gil_enabled()
print(json.dumps(dict(gil=observed, override=os.environ.get("PYTHON_GIL"), native=native.__file__)))
"""
    env = os.environ.copy()
    env.pop("PYTHON_GIL", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["override"] is None
    assert not any(report["gil"].values()), report
    assert "cpython-314t" in report["native"]


def _write_scenario(directory):
    payload = directory / "input.txt"
    payload.write_text("synthetic local bytes\n")
    manifest = dict(
        era5=payload.name,
        solar_csv=payload.name,
        coordinates=["time", "pressure", "latitude", "longitude"],
        backend="classic",
        valid_time="2024-01-15T12:00:00+00:00",
        latitude_deg=0,
        longitude_deg=0,
        pressure_interfaces_pa=[100000, 100],
        spectroscopy=[
            dict(
                species="CO2",
                lines=payload.name,
                partition=payload.name,
                molecule_id=2,
                isotopologue_id=1,
                mass_u=43.98983,
                pressure_range_pa=[100, 105000],
                included_cases=["LWIR"],
                dry_air_mole_fraction=[0.00042],
            )
        ],
        mipas=dict(file=payload.name, species=["O3"]),
        cross_sections=[
            dict(
                species="O3",
                format="serdyuchenko",
                files=[payload.name],
                temperatures_k=[193, 203],
                included_cases=["UV"],
                temperature_policy="clip",
                pressure_policy="fixed laboratory spectrum",
                source="synthetic test fixture",
            )
        ],
        sha256={payload.name: file_sha256(payload)},
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    settings = json.loads((ROOT / "notebooks/data/five-band.json").read_text())
    settings["manifest"] = manifest_path.name
    settings["reference"]["cache_directory"] = "cache"
    path = directory / "scenario.json"
    path.write_text(json.dumps(settings))
    return path, manifest_path, payload
