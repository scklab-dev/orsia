"""Notebook structure and optional real isolated-backend transport smoke checks."""

import inspect
import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import numpy as np
import pytest
import xarray as xr
from orsia_radiative.transfer import solar
from scipy.constants import c, h, k
from scipy.special import expn

from notebooks.support import notebook_five_band_support as support

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"


def test_notebook_structure():
    notebook = json.loads(NOTEBOOK.read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
            assert cell["execution_count"] is not None
            assert all(output["output_type"] != "error" for output in cell["outputs"])
    config = next(cell for cell in notebook["cells"] if cell["id"] == "config")
    assert all(
        f"'{name}'" in "".join(config["source"]) for name in ("UV", "VNIR", "SWIR", "MWIR", "LWIR")
    )
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for name in ("config", "atmosphere", "compare", "summary", "timing"):
        assert cells[name]["outputs"]
    assert sum("image/png" in output.get("data", {}) for output in cells["compare"]["outputs"]) == 5
    assert set(cells) == {
        "intro",
        "imports",
        "inputs-doc",
        "config",
        "atmosphere",
        "materialize",
        "controls-doc",
        "compare",
        "summary",
        "timing-doc",
        "timing",
    }
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            assert "application/json" not in output.get("data", {})
            text = "".join(output.get("text", []))
            assert "MIPAS projection:" not in text
            assert "GIL" not in text
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for removed in ("json.dumps", "diagnose(", "UV_REFINEMENT_CONTROLS", "refined", "SAVE"):
        assert removed not in code
    assert "text/markdown" in cells["summary"]["outputs"][0]["data"]
    assert "text/markdown" in cells["timing"]["outputs"][0]["data"]


def test_notebook_execution_recipe():
    recipe = (ROOT / "justfile").read_text().split("execute-five-band:\n", 1)[1].split("\n\n", 1)[0]
    install, execute = [shlex.split(line) for line in recipe.splitlines()]
    assert (
        install[:7]
        == execute[2:9]
        == ["uv", "run", "--no-sync", "--locked", "--all-packages", "--extra", "atmosphere"]
    )
    assert install[7:] == [
        "python",
        "-m",
        "ipykernel",
        "install",
        "--prefix",
        "outputs/jupyter",
        "--name",
        "orsia-314t",
        "--display-name",
        "Orsia CPython 3.14 free-threaded",
    ]
    assert execute[:2] == [
        "JUPYTER_PATH=$PWD/outputs/jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}",
        "MPLBACKEND=Agg",
    ]
    assert execute[9:] == [
        "jupyter",
        "execute",
        "--inplace",
        "--timeout=3600",
        "--kernel_name=orsia-314t",
        str(NOTEBOOK.relative_to(ROOT)),
    ]


def test_reference_group_setup_recipe():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["dependency-groups"]["sasktran2-reference"] == ["sasktran2==2026.10.0"]
    assert project.get("tool", {}).get("uv", {}).get("default-groups", ["dev"]) == ["dev"]
    workspace = next(package for package in lock["package"] if package["name"] == "orsia-workspace")
    assert workspace["dev-dependencies"]["sasktran2-reference"] == [{"name": "sasktran2"}]
    assert all(entry["name"] != "sasktran2" for entry in workspace["dev-dependencies"]["dev"])
    assert any(package["name"] == "sasktran2" for package in lock["package"])
    recipe = (ROOT / "justfile").read_text().split("setup-sasktran2-reference:\n", 1)[1]
    command = shlex.split(recipe.splitlines()[0])
    assert command == [
        "UV_PROJECT_ENVIRONMENT=outputs/sasktran2-env",
        "uv",
        "sync",
        "--locked",
        "--only-group",
        "sasktran2-reference",
        "--python",
        "3.14.3+gil",
        "--no-build",
    ]


def test_notebook_solar_controls_keep_delta_m_disabled():
    notebook = json.loads(NOTEBOOK.read_text())
    config = next(cell for cell in notebook["cells"] if cell["id"] == "config")
    namespace = {"np": np, "ROOT": ROOT}
    exec("CASES =" + "".join(config["source"]).split("CASES =", 1)[1], namespace)
    signature = inspect.signature(solar.reflected_radiance)
    assert namespace["POINTS"] == 201
    assert namespace["EMISSIVITY"] == 0.97
    assert namespace["SOLAR_CONTROLS"] == {
        "UV": dict(max_order=16, angular_order=4, layer_subdivisions=2),
        "VNIR": dict(max_order=2, angular_order=2, layer_subdivisions=1),
        "SWIR": dict(max_order=1, angular_order=2, layer_subdivisions=1),
    }
    for controls in namespace["SOLAR_CONTROLS"].values():
        arguments = signature.bind_partial(**controls)
        arguments.apply_defaults()
        assert arguments.arguments["delta_m_moments"] is None


@pytest.mark.parametrize("name", ["UV", "VNIR", "SWIR", "MWIR", "LWIR"])
@pytest.mark.parametrize("runs", [1, 3])
def test_notebook_complete_timing_scope(name, runs, monkeypatch):
    """Execute the comparison cell with an event clock around each solver call."""
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    namespace: dict = {"np": np, "ROOT": ROOT}
    exec("CASES =" + cells["config"].split("CASES =", 1)[1], namespace)
    namespace["CASES"] = {name: namespace["CASES"][name]}
    namespace["TIMING_RUNS"] = runs
    is_thermal = name in ("MWIR", "LWIR")
    scene, source, optics = object(), object(), object()
    shared = {"thermal": is_thermal}
    grid = SimpleNamespace(points=np.array([1000.0, 1100.0]))
    values = np.array([1.0, 2.0])
    events = []

    def clock():
        events.append("clock")
        return float(events.count("clock"))

    def series(actual_scene, *, source, optics, **controls):
        assert actual_scene is scene
        assert source is namespace["source"]
        assert optics is namespace["optics"]
        assert controls == dict(
            max_order={"UV": 16, "VNIR": 2, "SWIR": 1}[name],
            angular_order=4 if name == "UV" else 2,
            layer_subdivisions=2 if name == "UV" else 1,
        )
        events.append("series")
        return SimpleNamespace(values=values.copy())

    def schwarzschild(actual_scene, *, optics, **controls):
        assert actual_scene is scene
        assert optics is namespace["optics"]
        assert controls == {"hemispheric_method": "expn"}
        events.append("thermal")
        return SimpleNamespace(values=values.copy())

    reference = Mock(
        side_effect=lambda shared, cells, streams: dict(
            radiance=values.tolist(), cells=cells, streams=streams
        )
    )
    plotting = Mock()
    plotting.subplots.return_value = (None, [Mock(), Mock(), Mock()])
    namespace.update(
        measure_orsia=support.time_orsia,
        cached_reference=Mock(),
        show_comparison=Mock(),
        solar=SimpleNamespace(reflected_radiance=series),
        thermal=SimpleNamespace(schwarzschild_radiance=schwarzschild),
        plt=plotting,
    )
    monkeypatch.setattr(support, "perf_counter", clock)
    exec(cells["materialize"], namespace)
    namespace.update(
        prepare_case=lambda name: (grid, scene, optics, source, shared), reference=reference
    )
    exec(cells["compare"], namespace)
    expected = ["clock", "thermal" if is_thermal else "series", "clock"] * (runs + 1)
    assert events == expected
    result = namespace["results"][name]
    timing = result["orsia_timing"]
    assert timing["first_seconds"] == 1.0
    assert timing["seconds"] == [1.0] * runs
    assert timing["median_seconds"] == 1.0
    expected_scope = "thermal.schwarzschild_radiance" if is_thermal else "solar.reflected_radiance"
    assert timing["scope"].startswith(expected_scope + ";")
    if not is_thermal:
        assert "validation, geometry, scratch allocation, transport" in timing["scope"]
    np.testing.assert_array_equal(result["orsia"], values)
    assert "orsia_refined" not in result
    assert reference.call_args_list == [((shared, 32, 32), {})]


def test_notebook_readable_agreement_summary(monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "summary")
    display = Mock()
    monkeypatch.setattr(support, "display", display)
    monkeypatch.setattr(support, "Markdown", lambda text: text)
    namespace = {
        "np": np,
        "Markdown": lambda text: text,
        "display": display,
        "show_agreement": support.show_agreement,
        "results": {"UV": {"orsia": [3.0, 1.0, 0.0], "reference": [{"radiance": [2.0, 2.0, 0.0]}]}},
    }
    exec("".join(cell["source"]), namespace)
    text = display.call_args.args[0]
    assert "| UV | 1 | 50 |" in text
    assert "W m⁻² sr⁻¹ (cm⁻¹)⁻¹" in text


def test_notebook_timing_linear_panels(monkeypatch):
    """Render the timing cell in isolation with synthetic timing records."""
    from copy import deepcopy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "timing")
    results: dict = {}
    # Include both large contrasts and a ratio below one; thermal seconds stay small.
    for name, seconds in zip(
        ("UV", "VNIR", "SWIR", "MWIR", "LWIR"),
        ((3.0, 75.0), (0.5, 74.0), (0.6, 76.0), (0.0003, 0.004), (0.004, 0.002)),
        strict=True,
    ):
        timings: list[dict] = [
            {"median_seconds": value, "seconds": [0.9 * value, value, 1.1 * value]}
            for value in seconds
        ]
        timings[0]["scope"] = (
            "solar.reflected_radiance; complete call including geometry and scratch allocation"
            if name in ("UV", "VNIR", "SWIR")
            else "thermal.schwarzschild_radiance; includes returned Radiance and values"
        )
        results[name] = {
            "orsia_timing": timings[0],
            "reference": [
                {
                    "timing": timings[1],
                    "worker_wall_seconds": 1.0,
                    "python": "synthetic",
                    "sasktran2": "synthetic",
                    "gil_enabled": True,
                    "computed_at_utc": "2026-01-01T00:00:00+00:00",
                    "cache": {
                        "timing_provenance": "historical" if name == "UV" else "fresh",
                        "access_wall_seconds": 0.03,
                        "load_wall_seconds": 0.001,
                    },
                }
            ],
        }
    original = deepcopy(results)
    monkeypatch.setattr(plt, "show", lambda: None)
    display = Mock()
    monkeypatch.setattr(support, "display", display)
    monkeypatch.setattr(support, "Markdown", lambda text: text)
    namespace: dict = {
        "results": results,
        "np": np,
        "plt": plt,
        "TIMING_RUNS": 3,
        "Markdown": lambda text: text,
        "display": display,
        "show_timing": support.show_timing,
    }
    exec("".join(cell["source"]), namespace)
    fig = plt.gcf()
    try:
        fig.canvas.draw()
        assert results == original
        assert len(fig.axes) == 6
        limits = []
        for ax, (name, record) in zip(fig.axes, results.items(), strict=False):
            timings = [record["orsia_timing"], record["reference"][-1]["timing"]]
            medians = [timing["median_seconds"] for timing in timings]
            assert ax.get_yscale() == "linear"
            assert ax.get_ylim()[0] == 0
            limits.append(ax.get_ylim()[1])
            np.testing.assert_allclose([bar.get_height() for bar in ax.patches], medians)
            assert all(bar.get_y() == 0 for bar in ax.patches)
            assert [text.get_text() for text in ax.texts] == [f"{v:.4g} s" for v in medians]
            assert ax.get_title() == f"{name}\nSASKTRAN2 / Orsia = {medians[1] / medians[0]:.3g}"
            label = "historical" if name == "UV" else "fresh"
            assert ax.get_xticklabels()[1].get_text() == f"SASKTRAN2\n{label}"
            orsia_label = (
                "complete solar call" if name in ("UV", "VNIR", "SWIR") else "Schwarzschild"
            )
            assert ax.get_xticklabels()[0].get_text() == f"Orsia\n{orsia_label}"
            segments = ax.containers[-1].errorbar.lines[2][0].get_segments()
            np.testing.assert_allclose(
                [segment[:, 1] for segment in segments],
                [[min(t["seconds"]), max(t["seconds"])] for t in timings],
            )
        assert max(limits[:3]) > 1000 * max(limits[3:])
        note = fig.axes[-1].texts[0].get_text()
        assert "Independent scale per band" in note
        assert "Different solver settings;\naccuracy equivalence remains unmeasured." in note
        assert "complete reflected_radiance\nwith call-local geometry and scratch" in note
        assert "SASKTRAN2: calculate_radiance" in note
        assert "Thermal boundary adapter runs separately" in note
        output = display.call_args.args[0]
        assert "Orsia current median [s]" in output
        assert "Current access [s]" in output
        assert "| UV | 3 | 75 | historical | 0.03 |" in output
        assert "UV: 2026-01-01T00:00:00+00:00" in output
        assert "| LWIR | 0.004 | 0.002 | fresh | 0.03 |" in output
        # Earlier one-shot records lack scope; require a fresh comparison-cell run.
        for record in results.values():
            del record["orsia_timing"]["scope"]
        with pytest.raises(KeyError, match="scope"):
            exec("".join(cell["source"]), namespace)
    finally:
        plt.close(fig)


def test_notebook_materialization_with_synthetic_files(tmp_path):
    """Exercise actual cell APIs, not authentic ERA5/HITRAN or plot rendering."""
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    namespace = {}
    imports = cells["imports"].replace("import matplotlib.pyplot as plt\n", "")
    exec(imports, namespace)
    exec("CASES =" + cells["config"].split("CASES =", 1)[1], namespace)
    namespace["POINTS"] = 5
    namespace["REFERENCE_CACHE"] = tmp_path / "reference-cache"
    namespace["base"] = tmp_path
    namespace["local"] = lambda name: tmp_path / name
    for name, (low, high, _) in namespace["CASES"].items():
        directory = tmp_path / name
        directory.mkdir()
        _write_synthetic_files(
            directory, centre_cm1=(1e7 / low + 1e7 / high) / 2, include_solar=True
        )
        (directory / "solar.csv").write_text(
            "wavelength (nm),irradiance (W/m^2/nm)\n300,1.0\n11000,1.0\n"
        )
        namespace["manifest"] = {
            "era5": f"{name}/meteorology.nc",
            "solar_csv": f"{name}/solar.csv",
            "coordinates": ["time", "pressure", "latitude", "longitude"],
            "backend": "classic",
            "valid_time": "2026-01-01T00:00:00+00:00",
            "latitude_deg": 45.0,
            "longitude_deg": 5.0,
            "pressure_interfaces_pa": [100000, 80000, 60000, 40000, 20000],
            "spectroscopy": [
                {
                    "species": "CO2",
                    "molecule_id": 2,
                    "isotopologue_id": 1,
                    "lines": f"{name}/co2.par",
                    "partition": f"{name}/partition.csv",
                    "mass_u": 43.98983,
                    "pressure_range_pa": [20000, 100000],
                    "dry_air_mole_fraction": [420e-6] * 4,
                }
            ],
        }
        exec(cells["atmosphere"].split("show_atmosphere(")[0], namespace)
        exec(cells["materialize"], namespace)
        grid, scene, optics, source, shared = namespace["prepare_case"](name)
        assert shared["absorption_depth"].shape == (4, 5)
        assert np.all(shared["absorption_depth"] >= 0)
        if shared["thermal"]:
            # Both MWIR and LWIR use the same nonblack thermal surface.
            np.testing.assert_allclose(shared["emissivity"], 0.97)
            np.testing.assert_allclose(shared["albedo"], 0.03)
            result = namespace["thermal"].schwarzschild_radiance(
                scene, optics=optics, hemispheric_method="expn"
            )
        else:
            solar = namespace["solar"]
            controls = dict(max_order=1, angular_order=4, layer_subdivisions=2)
            result = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
            saved = result.values.copy()
            repeated = solar.reflected_radiance(
                scene,
                source=source,
                optics=optics,
                **controls,
            )
            assert not np.shares_memory(result.values, repeated.values)
            np.testing.assert_array_equal(result.values, saved)
            np.testing.assert_array_equal(repeated.values, saved)
        assert result.values.shape == grid.points.shape
        assert np.all(np.isfinite(result.values))
        interpreter = ROOT / "outputs/sasktran2-env/bin/python"
        if interpreter.is_file():
            namespace["REFERENCE_PYTHON"] = interpreter
            record = namespace["reference"](shared, 16, 16)
            assert np.all(np.isfinite(record["radiance"]))
            if shared["thermal"]:
                assert record["thermal_surface_coupling"] == "explicit_lambertian_boundary"
                assert np.all(np.asarray(record["reflected_boundary_radiance"]) > 0)


@pytest.mark.parametrize(
    "is_thermal, reflectance, wavenumbers",
    [
        (False, 0.2, [1000.0, 1100.0, 1200.0]),
        (True, 0.0, [1000.0, 1100.0, 1200.0]),
        (True, 0.3, [1000.0, 1100.0, 1200.0]),
        (True, 0.03, [1e7 / 4300, 1e7 / 4275, 1e7 / 4250]),
    ],
    ids=["solar", "thermal-black", "lwir-reflecting", "mwir-reflecting"],
)
def test_real_reference_absorption_limit(tmp_path, is_thermal, reflectance, wavenumbers):
    interpreter = ROOT / "outputs/sasktran2-env/bin/python"
    if not interpreter.is_file():
        pytest.skip("isolated SASKTRAN2 environment not provisioned")
    axis = np.array(wavenumbers)
    tau = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    muv, mu0 = 0.7, 0.8
    path = tmp_path / "shared.npz"
    np.savez(
        path,
        height_m=[0.0, 1000.0, 2000.0],
        temperature_k=[280.0, 280.0],
        wavenumbers_cm1=axis,
        absorption_depth=tau,
        rayleigh_depth=np.zeros_like(tau),
        thermal=is_thermal,
        mu0=mu0,
        muv=muv,
        irradiance=np.zeros(3) if is_thermal else np.ones(3),
        albedo=np.full(3, reflectance),
        emissivity=np.full(3, 1 - reflectance) if is_thermal else np.ones(3),
        surface_temperature_k=280.0,
    )
    result = subprocess.run(
        [
            str(interpreter),
            str(ROOT / "notebooks/support/notebook_sasktran2_layers.py"),
            str(path),
            "--cells",
            "32",
            "--streams",
            "32",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    record = json.loads(result.stdout)
    timing = record["timing"]
    assert len(timing["seconds"]) == 3
    assert all(value > 0 for value in timing["seconds"])
    assert timing["first_seconds"] > 0
    assert timing["median_seconds"] == np.median(timing["seconds"])
    np.testing.assert_allclose(record["los_optical_depth"], tau.sum(axis=0) / muv, rtol=1e-5)
    if is_thermal:
        frequency = 100 * c * axis
        expected = 2 * h * frequency**3 / c**2 / np.expm1(h * frequency / (k * 280)) * 100 * c
        boundary = reflectance * expected * (1 - 2 * expn(3, tau.sum(axis=0)))
        expected = expected * (
            1 - reflectance * np.exp(-tau.sum(axis=0) / muv)
        ) + boundary * np.exp(-tau.sum(axis=0) / muv)
        np.testing.assert_allclose(
            record["reflected_boundary_radiance"], boundary, rtol=1e-10, atol=1e-12
        )
    else:
        expected = reflectance * mu0 / np.pi * np.exp(-tau.sum(axis=0) * (1 / muv + 1 / mu0))
    np.testing.assert_allclose(record["radiance"], expected, rtol=1e-5, atol=1e-12)
    assert sys.version_info >= (3, 14)


def _write_synthetic_files(directory: Path, *, centre_cm1: float, include_solar: bool) -> None:
    """Invent format fixtures; none of these values are authentic provider data."""
    pressure = np.array([100000.0, 80000.0, 60000.0, 40000.0, 20000.0])
    dimensions = ("time", "pressure", "latitude", "longitude")
    dataset = xr.Dataset(
        {
            "t": (dimensions, np.array([290.0, 280.0, 265.0, 250.0, 230.0])[None, :, None, None]),
            "q": (dimensions, np.zeros((1, 5, 1, 1))),
        },
        coords={
            "time": np.array([0.0]),
            "pressure": pressure,
            "latitude": np.array([45.0]),
            "longitude": np.array([5.0]),
        },
    )
    dataset["time"].attrs = {"units": "hours since 2026-01-01 00:00:00", "calendar": "standard"}
    for name, units in (
        ("pressure", "Pa"),
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
        ("t", "K"),
        ("q", "kg kg-1"),
    ):
        dataset[name].attrs["units"] = units
    try:
        dataset.to_netcdf(directory / "meteorology.nc", engine="scipy", format="NETCDF3_CLASSIC")
    finally:
        dataset.close()
    prefix = (
        f"{2:2d}1{centre_cm1:12.6f}{1e-23:10.3E}{1.0:10.3E}"
        f"{0.07:5.3f}{0.1:5.3f}{100.0:10.3f}{0.7:4.2f}{0.0:8.5f}"
    )
    record = prefix.ljust(160)
    assert len(record) == 160
    (directory / "co2.par").write_text(record + "\n", encoding="utf-8")
    (directory / "partition.csv").write_text(
        "molecule_id,isotopologue_id,temperature_k,partition_sum\n"
        "2,1,200,100\n2,1,296,148\n2,1,350,175\n",
        encoding="utf-8",
    )
    if include_solar:
        (directory / "solar.csv").write_text(
            "wavelength (nm),irradiance (W/m^2/nm)\n1600,1.0\n1610,1.0\n1620,1.0\n",
            encoding="utf-8",
        )
