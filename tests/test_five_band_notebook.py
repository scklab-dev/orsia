"""Offline notebook execution, timing data and workspace reference-backend transport."""

import ast
import inspect
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
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
from notebooks.support.five_band_scenario import FiveBandInputs, FiveBandScenario

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"


def test_notebook_code_is_valid():
    notebook = json.loads(NOTEBOOK.read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")


def test_sasktran2_is_not_a_library_runtime_dependency():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for package in lock["package"]:
        if package["name"].startswith("orsia-"):
            assert all(entry["name"] != "sasktran2" for entry in package.get("dependencies", []))


@pytest.mark.parametrize("module", ["prepare_five_band", "prepare_five_band_enriched"])
def test_preparation_paths_remain_workspace_relative(module):
    # Importing these entrypoints must leave acquisition to main().
    from importlib import import_module

    entrypoint = import_module(f"notebooks.support.{module}")
    assert entrypoint.ROOT == ROOT
    destination = entrypoint.BASE if module == "prepare_five_band" else entrypoint.DESTINATION
    assert destination.parent == ROOT / "outputs"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"CDSAPI_URL": "env-url"},
        {"CDSAPI_KEY": "env-key"},
        {"CDSAPI_URL": "env-url", "CDSAPI_KEY": "env-key"},
        {"CDSAPI_URL": "", "CDSAPI_KEY": ""},
    ],
)
def test_preparation_cds_credentials_are_local_and_scoped(tmp_path, monkeypatch, overrides):
    from notebooks.support import prepare_five_band as preparation

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".env").write_text(
        "CDSAPI_URL='file-url'\nCDSAPI_KEY='literal-${ORSIA_TEST_EXPANSION}-#-key'\n"
        "SPACETRACK_USERNAME='unused-user'\nSPACETRACK_PASSWORD='unused-password'\n"
        "ORSIA_TEST_UNRELATED='unused'\n"
    )
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    (other_directory / ".env").write_text("CDSAPI_URL=wrong-url\nCDSAPI_KEY=wrong-key\n")
    monkeypatch.setattr(preparation, "ROOT", repository)
    monkeypatch.chdir(other_directory)
    monkeypatch.setenv("ORSIA_TEST_EXPANSION", "must-not-expand")
    monkeypatch.delenv("ORSIA_TEST_UNRELATED", raising=False)
    for name in ("CDSAPI_URL", "CDSAPI_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)

    assert (
        preparation._cds_credentials()
        == {
            "CDSAPI_URL": "file-url",
            "CDSAPI_KEY": "literal-${ORSIA_TEST_EXPANSION}-#-key",
        }
        | overrides
    )
    # Compare as a boolean so a failure cannot display the inherited environment.
    assert bool(dict(os.environ) == before)


@pytest.mark.parametrize("contents", [None, "", "CDSAPI_URL\nCDSAPI_KEY\n", "CDSAPI_URL=\n"])
@pytest.mark.parametrize("environment_present", [False, True])
def test_preparation_cds_credentials_allow_missing_settings(
    tmp_path, monkeypatch, contents, environment_present
):
    from notebooks.support import prepare_five_band as preparation

    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    if contents is not None:
        (tmp_path / ".env").write_text(contents)
    for name in ("CDSAPI_URL", "CDSAPI_KEY"):
        monkeypatch.delenv(name, raising=False)
        if environment_present:
            monkeypatch.setenv(name, "environment-value")
    expected = {
        "CDSAPI_URL": "" if contents == "CDSAPI_URL=\n" else None,
        "CDSAPI_KEY": None,
    }
    if environment_present:
        expected = dict.fromkeys(expected, "environment-value")
    assert preparation._cds_credentials() == expected


def test_preparation_passes_cds_credentials_explicitly(tmp_path, monkeypatch):
    from notebooks.support import prepare_five_band as preparation

    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "BASE", tmp_path / "outputs")
    (tmp_path / ".env").write_text("CDSAPI_URL=file-url\nCDSAPI_KEY=file-key\n")
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.setenv("CDSAPI_KEY", "env-key")

    class ClientIntercepted(Exception):
        pass

    def client(**kwargs):
        assert kwargs["url"] == "file-url"
        assert kwargs["key"] == "env-key"
        raise ClientIntercepted

    monkeypatch.setattr(preparation.cdsapi, "Client", client)
    with pytest.raises(ClientIntercepted):
        preparation.main()


def test_notebook_reference_uses_kernel_interpreter():
    notebook = json.loads(NOTEBOOK.read_text())
    config = next(cell for cell in notebook["cells"] if cell["id"] == "config")
    namespace = _settings_namespace()
    exec("".join(config["source"]), namespace)
    assert namespace["REFERENCE_PYTHON"] == Path(sys.executable)


def test_notebook_solar_controls_keep_delta_m_disabled():
    notebook = json.loads(NOTEBOOK.read_text())
    config = next(cell for cell in notebook["cells"] if cell["id"] == "config")
    namespace = _settings_namespace()
    exec("".join(config["source"]), namespace)
    signature = inspect.signature(solar.reflected_radiance)
    for band in namespace["scenario"].bands.values():
        if band.thermal:
            continue
        arguments = signature.bind_partial(**band.solar.model_dump())
        arguments.apply_defaults()
        assert arguments.arguments["delta_m_moments"] is None


@pytest.mark.parametrize("name", ["UV", "VNIR", "SWIR", "MWIR", "LWIR"])
@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("mode", ["reuse", "refresh", "off"])
def test_notebook_complete_timing_scope(name, runs, mode, monkeypatch):
    """Execute each real case comparison cell with a clock around complete calls."""
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    namespace = _settings_namespace()
    exec(cells["config"], namespace)
    scenario = namespace["scenario"]
    scenario.bands = {name: scenario.bands[name]}
    scenario.timing_runs = runs
    scenario.reference.cache_mode = mode
    is_thermal = name in ("MWIR", "LWIR")
    prefix = name.lower()
    scene, optics = object(), object()
    source = None if is_thermal else object()
    shared = {"thermal": is_thermal}
    grid = SimpleNamespace(points=np.array([1000.0, 1100.0]))
    values = np.array([1.0, 2.0])
    events = []

    def clock():
        events.append("clock")
        return float(events.count("clock"))

    def series(actual_scene, *, source, optics, **controls):
        assert actual_scene is scene
        assert source is namespace[f"{prefix}_source"]
        assert optics is namespace[f"{prefix}_optics"]
        assert controls == scenario.bands[name].solar.model_dump()
        events.append("series")
        return SimpleNamespace(values=values.copy())

    def thermal_radiance(actual_scene, *, optics, **controls):
        assert actual_scene is scene
        assert optics is namespace[f"{prefix}_optics"]
        assert controls == {"hemispheric_method": "expn"}
        events.append("thermal")
        return SimpleNamespace(values=values.copy())

    reference = Mock(
        side_effect=lambda shared, cells, streams, **settings: dict(
            radiance=values.tolist(), cells=cells, streams=streams
        )
    )
    namespace.update(
        time_orsia=support.time_orsia,
        cached_reference=reference,
        sasktran2_inputs=Mock(return_value=shared),
        show_comparison=Mock(),
        solar=SimpleNamespace(reflected_radiance=series),
        thermal=SimpleNamespace(thermal_radiance=thermal_radiance),
    )
    namespace.update(
        {
            f"{prefix}_band": scenario.bands[name],
            f"{prefix}_grid": grid,
            f"{prefix}_scene": scene,
            f"{prefix}_optics": optics,
            f"{prefix}_source": source,
        }
    )
    monkeypatch.setattr(support, "perf_counter", clock)
    exec(cells[f"{prefix}-compare"], namespace)
    expected = ["clock", "thermal" if is_thermal else "series", "clock"] * (runs + 1)
    assert events == expected
    result = namespace[f"{prefix}_result"]
    timing = result["orsia_timing"]
    assert timing["first_seconds"] == 1.0
    assert timing["seconds"] == [1.0] * runs
    assert timing["median_seconds"] == 1.0
    np.testing.assert_array_equal(result["orsia"], values)
    expected_scope = "thermal.thermal_radiance;" if is_thermal else "solar.reflected_radiance;"
    assert timing["scope"].startswith(expected_scope)
    namespace["sasktran2_inputs"].assert_called_once_with(
        scene,
        optics,
        source,
        mu0=namespace["MU0"],
        muv=namespace["MUV"],
        emissivity=scenario.physics.emissivity,
        surface_temperature_k=scenario.physics.surface_temperature_k,
    )
    namespace["show_comparison"].assert_called_once()
    plotted_name, plotted_grid, plotted_orsia, plotted_reference = namespace[
        "show_comparison"
    ].call_args.args
    assert plotted_name == name and plotted_grid is grid
    np.testing.assert_array_equal(plotted_orsia, values)
    np.testing.assert_array_equal(plotted_reference, values)
    reference.assert_called_once_with(
        shared,
        32,
        32,
        runs=runs,
        interpreter=Path(sys.executable),
        worker=ROOT / "notebooks/support/notebook_sasktran2_layers.py",
        cache_directory=scenario.reference.cache_directory,
        mode=mode,
    )


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
        "results": {"UV": {"orsia": [3.0, 1.0, 0.0], "reference": {"radiance": [2.0, 2.0, 0.0]}}},
    }
    exec("".join(cell["source"]), namespace)
    text = display.call_args.args[0]
    assert "| UV | 1 | 50 |" in text
    assert "W m⁻² sr⁻¹ (cm⁻¹)⁻¹" in text


@pytest.mark.parametrize("aod", [0.0, 0.1])
def test_notebook_materialization_with_synthetic_files(tmp_path, aod):
    """Exercise each explicit recipe through public APIs on synthetic local files."""
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    namespace = {}
    exec(cells["imports"], namespace)
    namespace.update(_settings_namespace())
    exec(cells["config"], namespace)
    scenario = namespace["scenario"]
    cases = scenario.bands.copy()
    scenario.physics.solar_aerosol.optical_depth = aod
    scenario.points = 5
    scenario.reference.cache_directory = tmp_path / "reference-cache"
    from notebooks.support.prepare_spectroscopy import prepare_water_continuum

    native = tmp_path / "synthetic-continuum.nc"
    _write_synthetic_continuum(native)
    scenario.water_continuum_manifest = tmp_path / "continuum/manifest.json"
    prepare_water_continuum(scenario.water_continuum_manifest, native=native)
    namespace["show_atmosphere"] = Mock()
    for name, band in cases.items():
        scenario.bands = {name: band}
        prefix = name.lower()
        low, high = band.low_nm, band.high_nm
        directory = tmp_path / name
        directory.mkdir()
        _write_synthetic_files(
            directory, centre_cm1=(1e7 / low + 1e7 / high) / 2, include_solar=True
        )
        (directory / "solar.csv").write_text(
            "wavelength (nm),irradiance (W/m^2/nm)\n300,1.0\n11000,1.0\n"
        )
        (directory / "reference.atm").write_text(
            "3\n*HGT [km]\n0 5 12\n*PRE [hPa]\n1000 600 200\n"
            "*TEM [K]\n290 265 230\n*H2O [ppmv]\n0 0 0\n*END\n"
        )
        # Each named recipe selects its own species from these format fixtures.
        molecules = {"H2O": 1, "CO2": 2, "CH4": 6, "O2": 7, "O3": 3, "NH3": 11, "SO2": 9, "NO2": 10}
        spectroscopy = []
        for species, molecule in molecules.items():
            line = (directory / "co2.par").read_text()
            (directory / f"{species}.par").write_text(f"{molecule:2d}" + line[2:])
            (directory / f"{species}-partition.csv").write_text(
                "molecule_id,isotopologue_id,temperature_k,partition_sum\n"
                f"{molecule},1,200,100\n{molecule},1,296,148\n{molecule},1,350,175\n"
            )
            spectroscopy.append(
                dict(
                    species=species,
                    included_cases=[name],
                    molecule_id=molecule,
                    isotopologue_id=1,
                    lines=directory / f"{species}.par",
                    partition=directory / f"{species}-partition.csv",
                    mass_u=44.0,
                    pressure_range_pa=[20000, 100000],
                    dry_air_mole_fraction=None if species == "H2O" else [420e-6] * 4,
                )
            )
        cross_sections = []
        for species in ("O3", "SO2", "NO2"):
            files = []
            for temperature in (200, 350):
                filename = directory / f"{species}-{temperature}.dat"
                filename.write_text(
                    f"vacuum nm, cm2/molec, Temperature: {temperature}K\n"
                    "319.0 1.0E-22\n323.0 2.0E-22\n326.0 1.0E-22\n"
                )
                files.append(filename)
            cross_sections.append(
                dict(
                    species=species,
                    format="bogumil",
                    files=files,
                    temperatures_k=[200, 350],
                    included_cases=["UV"],
                    temperature_policy="clip",
                    pressure_policy="fixed laboratory spectrum",
                )
            )
        namespace["native_solar"] = namespace["read_solar_spectrum_csv"](directory / "solar.csv")
        scenario.inputs = FiveBandInputs.model_validate(
            {
                "mipas": {"file": directory / "reference.atm", "species": []},
                "cross_sections": cross_sections,
                "era5": directory / "meteorology.nc",
                "solar_csv": directory / "solar.csv",
                "coordinates": ["time", "pressure", "latitude", "longitude"],
                "backend": "classic",
                "valid_time": "2026-01-01T00:00:00+00:00",
                "latitude_deg": 45.0,
                "longitude_deg": 5.0,
                "pressure_interfaces_pa": [100000, 80000, 60000, 40000, 20000],
                "spectroscopy": spectroscopy,
            }
        )
        namespace["inputs"] = scenario.inputs
        from orsia_data import (
            SpectroscopyCollection,
            read_cross_section_measurements,
            read_hitran_lines,
            read_partition_table,
        )
        from orsia_radiative import LineSpectroscopy
        from scipy.constants import atomic_mass

        references = {}
        for gas in scenario.inputs.spectroscopy:
            database = read_hitran_lines(
                gas.lines,
                molecule_id=gas.molecule_id,
                isotopologue_id=gas.isotopologue_id,
                interval_cm1=(1e7 / high - 26, 1e7 / low + 26),
            )
            partition = read_partition_table(
                gas.partition,
                molecule_id=gas.molecule_id,
                isotopologue_id=gas.isotopologue_id,
            )
            references[(gas.species, gas.isotopologue_id)] = LineSpectroscopy(
                gas.species,
                (database,),
                partition,
                gas.mass_u * atomic_mass,
            )
        namespace["spectroscopy"] = SpectroscopyCollection(
            references,
            {
                (entry.species, "all"): read_cross_section_measurements(
                    tuple(entry.files),
                    species=entry.species,
                    isotopologue="all",
                    format=entry.format,
                    interval_nm=(320, 325),
                )
                for entry in scenario.inputs.cross_sections
            },
        )
        exec(cells["atmosphere"], namespace)
        exec(cells["water-continuum"], namespace)
        # Execute every preparation cell in this case, preserving notebook order.
        for cell_id, source_code in cells.items():
            if cell_id.startswith(prefix + "-") and cell_id != prefix + "-compare":
                cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
                if cell["cell_type"] == "code":
                    exec(source_code, namespace)
        grid = namespace[f"{prefix}_grid"]
        scene = namespace[f"{prefix}_scene"]
        optics = namespace[f"{prefix}_optics"]
        source = namespace[f"{prefix}_source"]
        shared = support.sasktran2_inputs(
            scene,
            optics,
            source,
            mu0=namespace["MU0"],
            muv=namespace["MUV"],
            emissivity=scenario.physics.emissivity,
            surface_temperature_k=scenario.physics.surface_temperature_k,
        )
        assert shared["absorption_depth"].shape == (4, 5)
        assert np.all(shared["absorption_depth"] >= 0)
        species = {
            "UV": ("h2o", "o3", "so2", "no2"),
            "VNIR": ("h2o", "o2"),
            "SWIR": ("h2o", "co2", "ch4"),
            "MWIR": ("h2o", "co2"),
            "LWIR": ("h2o", "co2", "ch4", "o3", "nh3"),
        }[name]
        components = [namespace[f"{prefix}_{gas}"] for gas in species]
        for component in components[1:]:
            if component.kind == "voigt_lines":
                assert component.line_wing_convention == "truncated"
        if name == "UV":
            assert components[0].line_wing_convention == "truncated"
            assert "uv_h2o_continuum" not in namespace
            with pytest.raises(ValueError, match="stencil"):
                namespace["water_continuum_absorption"](
                    namespace["atmosphere"], grid, namespace["water_continuum_reference"]
                )
        else:
            assert components[0].line_wing_convention == "mt_ckd_local_25"
            continuum = namespace[f"{prefix}_h2o_continuum"]
            assert continuum.grid.same_points(grid)
            assert np.all(continuum.optical_depth > 0)
            components.insert(1, continuum)
        np.testing.assert_array_equal(
            shared["absorption_depth"], sum(component.optical_depth for component in components)
        )
        if shared["thermal"]:
            assert optics.aerosol is None
            assert not any(key.startswith("aerosol_") for key in shared)
            # Both MWIR and LWIR use the same nonblack thermal surface.
            np.testing.assert_allclose(shared["emissivity"], 0.97)
            np.testing.assert_allclose(shared["albedo"], 0.03)
            result = namespace["thermal"].thermal_radiance(
                scene, optics=optics, hemispheric_method="expn"
            )
        else:
            aerosol = optics.aerosol
            assert shared["aerosol_extinction_depth"] is aerosol.optical_depth
            assert shared["aerosol_scattering_depth"] is aerosol.scattering_depth
            assert shared["aerosol_asymmetry"] is aerosol.asymmetry
            expected_column = aod * ((1e7 / grid.points) / 550.0) ** -1.3
            np.testing.assert_allclose(aerosol.optical_depth.sum(axis=0), expected_column)
            np.testing.assert_allclose(aerosol.scattering_depth, 0.95 * aerosol.optical_depth)
            np.testing.assert_array_equal(aerosol.asymmetry, np.full((4, 5), 0.65))
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
            if aod == 0:
                absent = solar.reflected_radiance(
                    scene, source=source, optics=replace(optics, aerosol=None), **controls
                )
                np.testing.assert_array_equal(result.values, absent.values)
        assert result.values.shape == grid.points.shape
        assert np.all(np.isfinite(result.values))
        record = namespace["cached_reference"](
            shared,
            16,
            16,
            runs=scenario.timing_runs,
            interpreter=namespace["REFERENCE_PYTHON"],
            worker=ROOT / "notebooks/support/notebook_sasktran2_layers.py",
            cache_directory=scenario.reference.cache_directory,
            mode=scenario.reference.cache_mode,
        )
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
            sys.executable,
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
        cwd=ROOT,
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


def _write_synthetic_files(directory: Path, *, centre_cm1: float, include_solar: bool) -> None:
    """Invent format fixtures; none of these values are authentic provider data."""
    pressure = np.array([100000.0, 80000.0, 60000.0, 40000.0, 20000.0])
    dimensions = ("time", "pressure", "latitude", "longitude")
    dataset = xr.Dataset(
        {
            "t": (dimensions, np.array([290.0, 280.0, 265.0, 250.0, 230.0])[None, :, None, None]),
            "q": (dimensions, np.full((1, 5, 1, 1), 0.002)),
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


def test_water_continuum_offline_import_and_verification(tmp_path):
    from notebooks.support.prepare_spectroscopy import prepare_water_continuum

    receipt = tmp_path / "retained/manifest.json"
    with pytest.raises(FileNotFoundError, match="--water-continuum"):
        prepare_water_continuum(receipt, native=None)
    native = tmp_path / "synthetic.nc"
    _write_synthetic_continuum(native)
    prepare_water_continuum(receipt, native=native)
    retained, provenance = support.verify_water_continuum(receipt)
    assert retained.read_bytes() == native.read_bytes()
    assert set(provenance) == {str(receipt), str(retained)}
    before = receipt.read_bytes()
    prepare_water_continuum(receipt, native=None)
    prepare_water_continuum(receipt, native=native)
    assert receipt.read_bytes() == before
    changed = tmp_path / "different.nc"
    changed.write_bytes(native.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs"):
        prepare_water_continuum(receipt, native=changed)
    retained.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        support.verify_water_continuum(receipt)


def test_instrument_water_absorption_uses_local_lines_and_one_continuum(tmp_path):
    """Execute real instrument read, line and composition code on a wet column."""
    from orsia_data import SpectroscopyCollection, read_mt_ckd_water
    from orsia_radiative import (
        Line,
        LineSpectroscopy,
        PartitionFunctionTable,
        SpectralLineTable,
        compose_absorption_on_grid,
        evaluate_line_absorption,
        hydrostatic_atmosphere,
        water_continuum_absorption,
        wavenumber_grid,
    )

    from notebooks.support.prepare_spectroscopy import prepare_water_continuum

    notebook = json.loads((ROOT / "notebooks/era5_hitran_instrument_responses.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    native = tmp_path / "synthetic.nc"
    _write_synthetic_continuum(native)
    receipt = tmp_path / "outputs/spectroscopy/mt-ckd-4.3/manifest.json"
    prepare_water_continuum(receipt, native=native)
    grid = wavenumber_grid(np.array([6059.0, 6060.0, 6061.0]))
    pressure = np.array([100000.0, 50000.0, 10000.0])
    atmosphere = hydrostatic_atmosphere(
        pressure,
        np.array([296.0, 270.0]),
        np.array([0.005, 0.001]),
        target_pressure_boundaries_pa=pressure,
        gravity_m_s2=9.80665,
        bottom_height_m=0.0,
        gas_columns_molecules_m2={"CO2": np.full(2, 1e22), "CH4": np.full(2, 1e20)},
        air_broadener="dry_air",
    )
    references = {}
    manifest = {"spectroscopy": []}
    for species, molecule, mass in (("H2O", 1, 18.0), ("CO2", 2, 44.0), ("CH4", 6, 16.0)):
        references[(species, 1)] = LineSpectroscopy(
            species,
            (
                SpectralLineTable(
                    (Line(6060.0, 1e-24, 0.0, 0.07, 0.1, 0.7),), molecule, 1, (6000, 6100)
                ),
            ),
            PartitionFunctionTable(molecule, 1, np.array([200.0, 296.0, 350.0]), np.ones(3)),
            mass * 1.660539e-27,
        )
        manifest["spectroscopy"].append(
            dict(species=species, isotopologue_id=1, pressure_range_pa=(100, 105000))
        )
    namespace: dict = dict(
        ROOT=tmp_path,
        Path=Path,
        atmosphere=atmosphere,
        grid=grid,
        manifest=manifest,
        spectroscopy=SpectroscopyCollection(references, {}),
        verify_water_continuum=support.verify_water_continuum,
        read_mt_ckd_water=read_mt_ckd_water,
        evaluate_line_absorption=evaluate_line_absorption,
        water_continuum_absorption=water_continuum_absorption,
        compose_absorption_on_grid=compose_absorption_on_grid,
        provenance={},
        input_paths=[],
    )
    exec(cells["water-continuum"], namespace)
    for species in ("h2o", "co2", "ch4"):
        exec(cells["absorption-" + species], namespace)
    # The composition is the first statement of the transport cell; execute it unchanged.
    transport = next(
        source for source in cells.values() if source.startswith("absorption = compose")
    )
    first = ast.Module(body=ast.parse(transport).body[:1], type_ignores=[])
    exec(compile(first, "instrument-composition", "exec"), namespace)
    assert namespace["h2o_absorption"].line_wing_convention == "mt_ckd_local_25"
    for species in ("co2", "ch4"):
        assert namespace[f"{species}_absorption"].line_wing_convention == "truncated"
    continuum = namespace["h2o_continuum"]
    assert continuum.grid.same_points(grid)
    assert np.all(continuum.optical_depth > 0)
    np.testing.assert_array_equal(
        namespace["absorption"].optical_depth,
        sum(
            namespace[name].optical_depth
            for name in ("h2o_absorption", "h2o_continuum", "co2_absorption", "ch4_absorption")
        ),
    )
    assert str(receipt) in namespace["provenance"]
    assert receipt in namespace["input_paths"]


def _write_synthetic_continuum(path: Path) -> None:
    """Write invented coefficients on the retained edition's spectral support."""
    from scipy.io import netcdf_file

    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("point", 2003)
        for name, values in (
            ("wavenumbers", np.arange(-20.0, 20001.0, 10.0)),
            ("self_absco_ref", np.full(2003, 1e-27)),
            ("for_absco_ref", np.full(2003, 2e-27)),
            ("self_texp", np.full(2003, 3.0)),
        ):
            dataset.createVariable(name, "d", ("point",))[:] = values
        dataset.createVariable("ref_press", "d", ())[...] = 1013.0
        dataset.createVariable("ref_temp", "d", ())[...] = 296.0


def _settings_namespace() -> dict:
    """Supply canonical settings without requiring authentic ignored input files."""
    settings = json.loads((ROOT / "notebooks/data/five-band.json").read_text())
    settings["inputs"] = dict(
        era5="synthetic.nc",
        solar_csv="synthetic.csv",
        coordinates=("time", "pressure", "latitude", "longitude"),
        backend="classic",
        valid_time="2026-01-01T00:00:00+00:00",
        latitude_deg=45,
        longitude_deg=5,
        pressure_interfaces_pa=[100000, 20000],
        spectroscopy=[],
        mipas=dict(file="synthetic.atm", species=[]),
        cross_sections=[],
    )
    scenario = FiveBandScenario.model_validate(settings)
    scenario.reference.cache_directory = ROOT / "outputs/sasktran2-reference-cache"
    return dict(
        Path=Path, sys=sys, np=np, ROOT=ROOT, load_scenario=Mock(return_value=(scenario, [], {}))
    )
