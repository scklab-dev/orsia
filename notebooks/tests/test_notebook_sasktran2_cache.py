"""Cache boundary regressions and bounded isolated-backend miss/hit evidence."""

import copy
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "notebook_cache", ROOT / "notebooks/support/notebook_sasktran2_cache.py"
)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


@pytest.fixture
def shared():
    return {
        "height_m": np.array([0.0, 1000.0, 2000.0]),
        "temperature_k": np.array([280.0, 275.0]),
        "wavenumbers_cm1": np.array([1000.0, 1100.0, 1200.0]),
        "absorption_depth": np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]),
        "rayleigh_depth": np.zeros((2, 3)),
        "thermal": False,
        "mu0": 0.8,
        "muv": 0.7,
        "irradiance": np.ones(3),
        "albedo": np.full(3, 0.2),
        "emissivity": np.full(3, 0.8),
        "surface_temperature_k": 280.0,
    }


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("# worker version 1\n")
    backend = {"python": "test-python", "sasktran2": "test-sk", "gil_enabled": True}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--identity" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps(backend), stderr="")
        with np.load(command[2], allow_pickle=False) as data:
            size = data["wavenumbers_cm1"].size
            thermal = bool(data["thermal"])
            record = dict(
                backend,
                radiance=list(np.arange(size, dtype=float)),
                los_optical_depth=(
                    (data["absorption_depth"] + data["rayleigh_depth"]).sum(axis=0) / data["muv"]
                ).tolist(),
                cells=int(command[4]),
                streams=int(command[6]),
                timing={
                    "first_seconds": 0.2,
                    "seconds": [0.1] * int(command[8]),
                    "median_seconds": 0.1,
                    "scope": "test scope",
                },
                thermal_surface_coupling=(
                    "explicit_lambertian_boundary" if thermal else "native_solar"
                ),
                reflected_boundary_radiance=[0.01] * size if thermal else None,
            )
        return SimpleNamespace(returncode=0, stdout=json.dumps(record), stderr="")

    monkeypatch.setattr(cache.subprocess, "run", run)
    options = dict(
        cells=2,
        streams=4,
        runs=1,
        interpreter=Path("isolated-python"),
        worker=worker,
        cache_directory=tmp_path / "cache",
    )
    return options, backend, calls


def test_hit_complete_and_owned(shared, fake_backend):
    options, _, calls = fake_backend
    first = cache.reference(shared, **options)
    second = cache.reference(shared, **options)
    assert [item["cache"]["status"] for item in (first, second)] == ["miss", "hit"]
    assert len([cmd for cmd in calls if "--identity" not in cmd]) == 1
    assert len([cmd for cmd in calls if "--identity" in cmd]) == 2
    assert second["cache"]["timing_provenance"] == "historical"
    assert second["cache"]["load_wall_seconds"] > 0
    assert second["cache"]["identity_wall_seconds"] > 0
    for name in first.keys() - {"cache"}:
        assert first[name] == second[name]
    second["radiance"][0] = -99
    assert cache.reference(shared, **options)["radiance"] == first["radiance"]
    persisted = json.loads(next(options["cache_directory"].glob("*.json")).read_text())
    assert "cache" not in persisted["record"]
    assert persisted["signature"]["schema"] == cache.SCHEMA


def test_input_order_and_layout_are_stable(shared, fake_backend):
    options, _, _ = fake_backend
    first = cache.reference(shared, **options)
    shared = dict(reversed(list(shared.items())))
    shared["absorption_depth"] = np.asfortranarray(shared["absorption_depth"])
    second = cache.reference(shared, **options)
    assert second["cache"]["status"] == "hit"
    assert first["cache"]["key"] == second["cache"]["key"]


def test_backend_fingerprints_actual_installed_files(tmp_path, monkeypatch):
    worker = runpy.run_path(str(ROOT / "notebooks/support/notebook_sasktran2_layers.py"))
    monkeypatch.setitem(sys.modules, "sasktran2", SimpleNamespace())
    metadata = worker["importlib"].metadata
    distribution = SimpleNamespace(
        files=[Path("backend.so"), Path("generated.pyc")],
        metadata={"Name": "sasktran2"},
        version="1.0",
        locate_file=lambda item: tmp_path / item,
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])
    monkeypatch.setattr(metadata, "version", lambda name: "1.0")
    binary = tmp_path / "backend.so"
    binary.write_bytes(b"original binary")
    first = worker["backend_identity"]()
    (tmp_path / "generated.pyc").write_bytes(b"generated bytecode")
    assert worker["backend_identity"]() == first
    binary.write_bytes(b"modified binary")
    second = worker["backend_identity"]()
    assert second["packages"] != first["packages"]
    distribution.version = "2.0"
    assert worker["backend_identity"]()["packages"] != second["packages"]
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert worker["backend_identity"]()["thread_environment"] != first["thread_environment"]


def test_atomic_replacement_publishes_complete_entry(shared, fake_backend, monkeypatch):
    options, _, _ = fake_backend
    first = cache.reference(shared, **options)
    path = Path(first["cache"]["path"])
    original = path.read_bytes()
    replace = cache.os.replace

    def inspect_replace(temporary, destination):
        assert destination.read_bytes() == original
        entry = json.loads(temporary.read_text())
        assert entry["record_sha256"] == cache._digest(entry["record"])
        assert temporary.parent == destination.parent
        replace(temporary, destination)

    monkeypatch.setattr(cache.os, "replace", inspect_replace)
    cache.reference(shared, **options, mode="refresh")


def test_invalid_controls_fail_before_probe(shared, fake_backend):
    options, _, calls = fake_backend
    with pytest.raises(ValueError, match="cache mode"):
        cache.reference(shared, **options, mode="typo")
    options["runs"] = 0
    with pytest.raises(ValueError, match="runs"):
        cache.reference(shared, **options)
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "height_m",
        "temperature_k",
        "wavenumbers_cm1",
        "absorption_depth",
        "rayleigh_depth",
        "thermal",
        "mu0",
        "muv",
        "irradiance",
        "albedo",
        "emissivity",
        "surface_temperature_k",
    ],
)
def test_every_shared_input_invalidates(shared, fake_backend, name):
    options, _, _ = fake_backend
    first = cache.reference(shared, **options)
    changed = copy.deepcopy(shared)
    changed[name] = True if name == "thermal" else np.asarray(shared[name]) + 0.001
    second = cache.reference(changed, **options)
    assert second["cache"]["status"] == "miss"
    assert first["cache"]["key"] != second["cache"]["key"]


@pytest.mark.parametrize(
    "change", ["cells", "streams", "runs", "worker", "backend", "dtype", "shape", "extra"]
)
def test_configuration_identity_and_representation(shared, fake_backend, change):
    options, backend, _ = fake_backend
    first = cache.reference(shared, **options)
    if change in ("cells", "streams", "runs"):
        options[change] *= 2
    elif change == "worker":
        options["worker"].write_text("# updated worker\n")
    elif change == "backend":
        backend["package_digest"] = "updated binary"
    elif change == "dtype":
        shared["height_m"] = shared["height_m"].astype(np.float32)
    elif change == "shape":
        shared["height_m"] = shared["height_m"].reshape(1, 3)
    else:
        shared["future_worker_input"] = 1
    second = cache.reference(shared, **options)
    assert second["cache"]["status"] == "miss"
    assert first["cache"]["key"] != second["cache"]["key"]


@pytest.mark.parametrize(
    "damage",
    ["json", "signature", "checksum", "missing", "shape", "nan", "timing", "los", "date", "type"],
)
def test_corrupt_entry_recovers(shared, fake_backend, damage):
    options, _, _ = fake_backend
    first = cache.reference(shared, **options)
    path = Path(first["cache"]["path"])
    entry = json.loads(path.read_text())
    if damage == "json":
        path.write_text("{interrupted")
    else:
        if damage == "signature":
            entry["signature"]["runs"] = 8
        elif damage == "checksum":
            entry["record"]["radiance"][0] = 42
        elif damage == "missing":
            del entry["record"]["radiance"]
        elif damage == "shape":
            entry["record"]["radiance"] = []
        elif damage == "nan":
            entry["record"]["radiance"][0] = float("nan")
        elif damage == "timing":
            entry["record"]["timing"]["seconds"] = []
        elif damage == "los":
            entry["record"]["los_optical_depth"][0] = 900
        elif damage == "date":
            entry["record"]["computed_at_utc"] = "bad date"
        elif damage == "type":
            entry["record"]["timing"] = None
        if damage not in {"checksum", "nan"}:
            entry["record_sha256"] = cache._digest(entry["record"])
        path.write_text(json.dumps(entry))
    recovered = cache.reference(shared, **options)
    assert recovered["cache"]["status"] == "invalid"
    assert recovered["radiance"] == first["radiance"]
    assert cache.reference(shared, **options)["cache"]["status"] == "hit"


def test_refresh_and_off(shared, fake_backend):
    options, _, calls = fake_backend
    first = cache.reference(shared, **options)
    path = Path(first["cache"]["path"])
    original = path.read_bytes()
    off = cache.reference(shared, **options, mode="off")
    assert off["cache"]["status"] == "off"
    assert off["cache"]["path"] is None
    assert path.read_bytes() == original
    fresh = cache.reference(shared, **options, mode="refresh")
    assert fresh["cache"]["timing_provenance"] == "fresh"
    assert path.read_bytes() != original
    assert cache.reference(shared, **options)["computed_at_utc"] == fresh["computed_at_utc"]
    assert len([cmd for cmd in calls if "--identity" not in cmd]) == 3
    options["cache_directory"] /= "unused"
    cache.reference(shared, **options, mode="off")
    assert not options["cache_directory"].exists()


@pytest.mark.parametrize("failure", ["process", "incomplete", "write"])
def test_failures_preserve_entry(shared, fake_backend, monkeypatch, failure):
    options, _, _ = fake_backend
    first = cache.reference(shared, **options)
    path = Path(first["cache"]["path"])
    original = path.read_bytes()
    original_run = cache.subprocess.run

    def run(command, **kwargs):
        if "--identity" in command:
            return original_run(command, **kwargs)
        return SimpleNamespace(
            returncode=1 if failure == "process" else 0, stdout="{}", stderr="failed solve"
        )

    if failure == "write":

        def fail_replace(*args):
            raise OSError("simulated failed replacement")

        monkeypatch.setattr(cache.os, "replace", fail_replace)
    else:
        monkeypatch.setattr(cache.subprocess, "run", run)
    with pytest.raises((RuntimeError, KeyError, OSError)):
        cache.reference(shared, **options, mode="refresh")
    assert path.read_bytes() == original
    assert list(options["cache_directory"].glob("*.tmp")) == []
    path.unlink()
    with pytest.raises((RuntimeError, KeyError, OSError)):
        cache.reference(shared, **options)
    assert list(options["cache_directory"].glob("*.json")) == []


@pytest.mark.parametrize("thermal", [False, True])
def test_real_backend_cache_roundtrip(tmp_path, shared, thermal, monkeypatch, capsys):
    interpreter = ROOT / "outputs/sasktran2-env/bin/python"
    if not interpreter.is_file():
        pytest.skip("isolated SASKTRAN2 environment not provisioned")
    shared["thermal"] = thermal
    if thermal:
        shared["irradiance"][:] = 0
    # The notebook wrapper retains provenance in records for its readable table.
    notebook = json.loads((ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "materialize")
    namespace: dict = dict(
        ROOT=ROOT,
        REFERENCE_PYTHON=interpreter,
        TIMING_RUNS=1,
        REFERENCE_CACHE=tmp_path,
        REFERENCE_CACHE_MODE="reuse",
        cached_reference=cache.reference,
    )
    exec("".join(cell["source"]), namespace)
    first = namespace["reference"](shared, 2, 4)
    real_run = cache.subprocess.run
    calls = []

    def identity_only(command, **kwargs):
        assert "--identity" in command, "cache hit launched the expensive solve worker"
        calls.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(cache.subprocess, "run", identity_only)
    second = namespace["reference"](shared, 2, 4)
    assert len(calls) == 1
    assert first["cache"]["status"] == "miss"
    assert second["cache"]["status"] == "hit"
    for name in first.keys() - {"cache"}:
        assert first[name] == second[name]
    assert np.all(np.isfinite(second["radiance"]))
    assert first["cache"]["timing_provenance"] == "fresh"
    assert second["cache"]["timing_provenance"] == "historical"
    assert second["computed_at_utc"] == first["computed_at_utc"]
    assert capsys.readouterr().out == ""
