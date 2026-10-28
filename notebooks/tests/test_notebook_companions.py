"""Scientific/API visibility and local side-effect boundaries of notebook companions."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from orsia_data.acquisition import file_sha256

from notebooks.support.notebook_five_band_support import verify_inputs
from notebooks.support.notebook_instrument_responses import preparation_sources

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "era5_hitran_instrument_responses.ipynb",
            {
                "RadiativeScene",
                "ObservationGeometry",
                "solar_source",
                "solar.reflected_radiance",
                "Radiance",
                "wavenumber_grid",
                "gaussian_along_track_response",
                "along_track_raster_reads",
                "filter_imager",
                "spatial_filter_imager",
                "lvf_imager",
                "dispersive_imager",
                "interferometric_imager",
                "cavity_imager",
                "response_electrons",
                "reconstruction",
                "correct_digital_readings",
                "electron_covariance",
                "propagate_covariance",
            },
        ),
        (
            "era5_hitran_sasktran2_five_bands.ipynb",
            {
                "read_netcdf_profiles",
                "pressure_level_layer_means",
                "gas_columns_from_fraction",
                "hydrostatic_atmosphere",
                "read_hitran_lines",
                "read_partition_table",
                "line_set",
                "line_absorption",
                "compose_absorption_on_grid",
                "atmospheric_optics",
                "solar_irradiance",
                "solar.reflected_radiance",
                "thermal.schwarzschild_radiance",
            },
        ),
    ],
)
def test_notebook_keeps_explicit_scientific_steps(name, expected):
    cells = json.loads((ROOT / "notebooks" / name).read_text())["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    trees = [ast.parse(source) for source in sources]
    calls = {
        ast.unparse(node.func)
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert expected <= calls
    if name == "era5_hitran_instrument_responses.ipynb":
        assert {"scan_templates", "evaluate_instrument", "along_track_strip_templates"} <= calls
        assert "notebook_scan_support" not in "\n".join(sources)
    assert all("np.savez" not in source for source in sources)
    if name == "era5_hitran_instrument_responses.ipynb":
        assert all(
            "plt.subplots" not in "".join(cell["source"])
            for cell in cells
            if cell["cell_type"] == "code"
            and cell["id"] not in {"scan-preview", "scan-timing", "native-image"}
        )
    else:
        assert all("plt.subplots" not in source for source in sources)


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


def test_preparation_selection_skips_display_and_reference_setup():
    sources = preparation_sources(ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb")
    for source in sources.values():
        tree = ast.parse(source)
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "show_atmosphere" not in calls
        assert "cached_reference" not in calls
    assert "verify_inputs(MANIFEST)" in sources["config"]
    assert "atmospheric_optics" in sources["materialize"]


def test_reference_and_timing_hooks_use_current_notebook_settings(tmp_path):
    path = ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"
    cell = next(c for c in json.loads(path.read_text())["cells"] if c["id"] == "materialize")
    hooks = [
        node
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name in ("reference", "time_orsia")
    ]
    cached = Mock(return_value={"radiance": [1.0]})
    measure = Mock()
    namespace = {"cached_reference": cached, "measure_orsia": measure}
    exec(compile(ast.Module(body=hooks, type_ignores=[]), str(path), "exec"), namespace)
    for runs, mode in ((1, "reuse"), (3, "refresh"), (2, "off")):
        namespace.update(
            ROOT=ROOT,
            TIMING_RUNS=runs,
            REFERENCE_PYTHON=tmp_path / "python",
            REFERENCE_CACHE=tmp_path / mode,
            REFERENCE_CACHE_MODE=mode,
        )
        shared, calculate = {}, Mock()
        namespace["reference"](shared, 32, 32)
        cached.assert_called_with(
            shared,
            32,
            32,
            runs=runs,
            interpreter=tmp_path / "python",
            worker=ROOT / "notebooks/support/notebook_sasktran2_layers.py",
            cache_directory=tmp_path / mode,
            mode=mode,
        )
        namespace["time_orsia"](calculate)
        measure.assert_called_with(calculate, runs=runs)
