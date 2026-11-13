"""Check notebook optical-law consumers with synthetic inputs and no providers."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from orsia_radiative import (
    AerosolComponent,
    RayleighComponent,
    angstrom_optical_depth,
    rayleigh_from_air_columns,
    wavenumber_grid,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"


@pytest.mark.parametrize("band", ["uv", "vnir", "swir"])
def test_solar_optics_cells_match_replaced_formulas(band):
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(
        next(cell["source"] for cell in notebook["cells"] if cell["id"] == f"{band}-optics")
    )
    settings = json.loads((ROOT / "notebooks/data/five-band.json").read_text())
    physics = SimpleNamespace(**settings["physics"])
    aerosol = SimpleNamespace(**physics.solar_aerosol)
    window = settings["bands"][band.upper()]
    points = np.linspace(1e7 / window["high_nm"], 1e7 / window["low_nm"], settings["points"])
    grid = wavenumber_grid(points)
    columns = np.array([1e28, 2e28, 0.0])
    fractions = np.array([0.7, 0.2, 0.1])
    rayleigh_call = Mock(wraps=rayleigh_from_air_columns)
    angstrom_call = Mock(wraps=angstrom_optical_depth)
    namespace: dict = {
        "np": np,
        "physics": physics,
        "aerosol": aerosol,
        "atmosphere": SimpleNamespace(dry_air_columns_molecules_m2=columns),
        "aerosol_layer_fraction": fractions,
        "compose_absorption_on_grid": Mock(),
        "atmospheric_optics": Mock(),
        "AerosolComponent": AerosolComponent,
        "rayleigh_from_air_columns": rayleigh_call,
        "angstrom_optical_depth": angstrom_call,
        f"{band}_grid": grid,
    }
    for species in ("h2o", "h2o_continuum", "o3", "so2", "no2", "o2", "co2", "ch4"):
        namespace[f"{band}_{species}"] = object()
    exec(source, namespace)
    rayleigh_call.assert_called_once_with(
        grid=grid,
        dry_air_columns_molecules_m2=columns,
        reference_cross_section_m2=physics.rayleigh_cross_section_m2,
        reference_wavelength_um=physics.rayleigh_wavelength_um,
    )
    angstrom_call.assert_called_once_with(
        grid=grid,
        reference_optical_depth=aerosol.optical_depth,
        reference_wavelength_nm=aerosol.reference_wavelength_nm,
        angstrom_exponent=aerosol.angstrom_exponent,
    )
    # These are the former notebook formulas, retained as independent regression oracles.
    expected_rayleigh = (
        columns[:, None]
        * (
            physics.rayleigh_cross_section_m2
            * (physics.rayleigh_wavelength_um / (1e4 / points)) ** 4
        )[None, :]
    )
    expected_column = aerosol.optical_depth * (
        (1e7 / points) / aerosol.reference_wavelength_nm
    ) ** (-aerosol.angstrom_exponent)
    rayleigh = namespace[f"{band}_rayleigh"]
    result = namespace[f"{band}_aerosol"]
    assert rayleigh.grid is grid and result.grid is grid
    np.testing.assert_allclose(rayleigh.optical_depth, expected_rayleigh, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        result.optical_depth, fractions[:, None] * expected_column, rtol=2e-15
    )
    np.testing.assert_allclose(result.optical_depth.sum(axis=0), expected_column, rtol=2e-15)
    np.testing.assert_allclose(
        result.scattering_depth, aerosol.single_scattering_albedo * result.optical_depth, rtol=2e-15
    )
    np.testing.assert_array_equal(result.asymmetry, np.full((3, len(points)), aerosol.asymmetry))
    namespace["atmospheric_optics"].assert_called_once_with(
        namespace["atmosphere"],
        grid,
        absorption=namespace[f"{band}_absorption"],
        rayleigh=rayleigh,
        aerosol=result,
    )


@pytest.mark.parametrize("band", ["mwir", "lwir"])
def test_thermal_rayleigh_stays_zero(band):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = "".join(next(cell["source"] for cell in cells if cell["id"] == f"{band}-optics"))
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == f"{band}_rayleigh"
            for target in node.targets
        )
    )
    grid = wavenumber_grid(np.array([1000.0, 1100.0]))
    namespace: dict = {
        "np": np,
        "RayleighComponent": RayleighComponent,
        f"{band}_grid": grid,
        f"{band}_absorption": SimpleNamespace(optical_depth=np.ones((3, 2))),
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    np.testing.assert_array_equal(namespace[f"{band}_rayleigh"].optical_depth, np.zeros((3, 2)))


def test_notebook_imports_public_laws_and_states_model_limits():
    notebook = json.loads(NOTEBOOK.read_text())
    imports = ast.parse(
        "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "imports"))
    )
    names = {
        alias.name
        for node in ast.walk(imports)
        if isinstance(node, ast.ImportFrom) and node.module == "orsia_radiative"
        for alias in node.names
    }
    assert {"rayleigh_from_air_columns", "angstrom_optical_depth", "RayleighComponent"} <= names
    prose = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    assert (
        "aerosol-case convergence and aerosol-impact validation have not been performed." in prose
    )
    assert "UV lies outside this support" in prose
    assert "without continuum or extrapolation" in prose
