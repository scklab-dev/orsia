"""Notebook input integrity, preparation effects and runtime setting propagation."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from orsia_data.acquisition import file_sha256

from notebooks.support.notebook_five_band_support import sasktran2_inputs, verify_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("available", [False, True])
def test_notebook_runtime_records_optional_affinity(monkeypatch, available):
    from notebooks.support.notebook_instrument_support import runtime_settings

    # Isolate affinity recording from atmospheric imports during collection.
    # test_core_imports checks real Orsia native-import GIL effects separately.
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False)
    if available:
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {12, 15}, raising=False)
    else:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    report = runtime_settings()
    assert report["affinity"] == ([12, 15] if available else None)


def test_input_verification_checks_files_and_retains_provenance(tmp_path):
    payload = tmp_path / "input.txt"
    payload.write_text("local synthetic input\n")
    manifest = {
        "era5": payload.name,
        "solar_csv": payload.name,
        "spectroscopy": [],
        "mipas": {"file": payload.name},
        "cross_sections": [],
        "sha256": {payload.name: file_sha256(payload)},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded, base, paths, provenance = verify_inputs(path)
    assert loaded == manifest
    assert base == tmp_path
    assert paths == [path, payload, payload, payload]
    assert provenance == {str(p): file_sha256(p) for p in paths}
    payload.write_text("changed synthetic input\n")
    with pytest.raises(AssertionError, match="Input checksum mismatch"):
        verify_inputs(path)


@pytest.mark.parametrize("thermal", [False, True])
def test_reference_adapter_retains_evaluated_arrays(thermal):
    height = np.array([0.0, 1000.0, 2000.0])
    temperature = np.array([280.0, 250.0])
    axis = np.array([1000.0, 1100.0])
    absorption = np.array([[0.1, 0.2], [0.3, 0.4]])
    rayleigh = np.zeros_like(absorption)
    surface = SimpleNamespace(emissivity=np.full(2, 0.97), reflectance=np.full(2, 0.2))
    scene = SimpleNamespace(
        atmosphere=SimpleNamespace(height_boundaries_m=height, temperature_k=temperature),
        surface=surface,
    )
    optics = SimpleNamespace(
        grid=SimpleNamespace(points=axis),
        absorption=SimpleNamespace(optical_depth=absorption),
        rayleigh=SimpleNamespace(optical_depth=rayleigh),
        aerosol=None,
    )
    irradiance = np.array([1.0, 2.0])
    source = None if thermal else SimpleNamespace(irradiance=SimpleNamespace(values=irradiance))
    shared = sasktran2_inputs(
        scene, optics, source, mu0=0.8, muv=0.9, emissivity=0.97, surface_temperature_k=300.0
    )
    assert shared["height_m"] is height
    assert shared["temperature_k"] is temperature
    assert shared["wavenumbers_cm1"] is axis
    assert shared["absorption_depth"] is absorption
    assert shared["rayleigh_depth"] is rayleigh
    assert shared["thermal"] is thermal
    np.testing.assert_array_equal(shared["irradiance"], np.zeros(2) if thermal else irradiance)
    np.testing.assert_array_equal(shared["emissivity"], np.full(2, 0.97))
    np.testing.assert_allclose(shared["albedo"], 0.03 if thermal else 0.2)
    assert shared["surface_temperature_k"] == 300.0
    assert shared["mu0"] == 0.8 and shared["muv"] == 0.9
