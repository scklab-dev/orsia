"""Synthetic checks for the enriched notebook's bounded scientific operations."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from orsia_radiative import hydrostatic_atmosphere, wavenumber_grid

ROOT = Path(__file__).resolve().parents[2]
GASES = runpy.run_path(str(ROOT / "notebooks/support/five_band_gases.py"))


def test_profile_conversion_before_layer_average(tmp_path):
    path = tmp_path / "reference.atm"
    path.write_text(
        "3\n*HGT [km]\n0 1 2\n*PRE [hPa]\n1000 700 100\n"
        "*TEM [K]\n290 250 210\n*H2O [ppmv]\n200000 100000 0\n"
        "*O3 [ppmv]\n8 9 10\n*NO2 [ppmv]\n0 0 0\n*END\n"
    )
    # Wet fractions vary with native water; the native dry fraction is constant.
    pressure = np.array([100000.0, 40000.0, 10000.0])
    humidity = np.array([0.02, 0.0])
    columns, report = GASES["project_trace_gases"](path, pressure, humidity, ["O3", "NO2"])
    dry_air = -np.diff(pressure) / 9.80665 * (1 - humidity) / 0.0289647 * 6.02214076e23
    np.testing.assert_allclose(report["O3"]["dry_mol_mol"], 1e-5, rtol=1e-14)
    np.testing.assert_allclose(columns["O3"], dry_air * 1e-5)
    assert report["O3"]["era5_wet_mol_mol"][0] < 1e-5
    np.testing.assert_array_equal(columns["NO2"], 0)
    with pytest.raises(KeyError, match="SO2"):
        GASES["project_trace_gases"](path, pressure, humidity, ["SO2"])
    with pytest.raises(ValueError, match="coverage"):
        GASES["project_trace_gases"](path, pressure + 1000, humidity, ["O3"])
    path.write_text(path.read_text().replace("200000 100000 0", "1000000 100000 0"))
    with pytest.raises(ValueError, match="dry-air"):
        GASES["project_trace_gases"](path, pressure, humidity, ["O3"])


def test_spectral_then_temperature_interpolation_and_clipping():
    # Different native axes at two temperatures; sigma(lambda,T) is bilinear.
    axes = [np.array([319.0, 322.0, 326.0]), np.array([318.0, 321.0, 327.0])]
    temperatures = np.array([200.0, 300.0])
    spectra = [(axis, (axis + temp) * 1e-25) for axis, temp in zip(axes, temperatures, strict=True)]
    target = np.array([325.0, 322.0, 320.0])  # increasing wavenumbers imply decreasing nm
    actual = np.array([190.0, 200.0, 250.0, 300.0, 310.0])
    sigma, effective = GASES["interpolate_uv"](
        spectra, temperatures, actual, target, temperature_policy="clip"
    )
    np.testing.assert_array_equal(effective, [200, 200, 250, 300, 300])
    np.testing.assert_allclose(sigma, (target[None, :] + effective[:, None]) * 1e-25, atol=0)
    assert not np.shares_memory(effective, actual)
    with pytest.raises(ValueError, match="temperature outside"):
        GASES["interpolate_uv"](spectra, temperatures, actual, target, temperature_policy="error")
    with pytest.raises(ValueError, match="wavelength outside"):
        GASES["interpolate_uv"](
            spectra, temperatures, actual, target + 20, temperature_policy="clip"
        )
    with pytest.raises(ValueError, match="temperature policy"):
        GASES["interpolate_uv"](spectra, temperatures, actual, target, temperature_policy="guess")
    spectra[0][1][0] = -1e-25
    with pytest.raises(ValueError, match="nonnegative"):
        GASES["interpolate_uv"](spectra, temperatures, actual, target, temperature_policy="clip")


@pytest.mark.parametrize("invalid", ["temperatures", "layers", "wavelength", "axis", "shape"])
def test_uv_invalid_support(invalid):
    temps = np.array([200.0, 300.0])
    layers = np.array([250.0])
    target = np.array([320.0, 325.0])
    spectra = [(target.copy(), np.ones(2)) for _ in temps]
    if invalid == "temperatures":
        temps[:] = 300
    elif invalid == "layers":
        layers[0] = np.nan
    elif invalid == "wavelength":
        target[0] = 0
    elif invalid == "axis":
        spectra[0][0][:] = 320
    else:
        spectra[0] = (target, np.ones(3))
    with pytest.raises(ValueError):
        GASES["interpolate_uv"](spectra, temps, layers, target, temperature_policy="clip")


def test_native_cross_section_headers_units_and_temperature_order(tmp_path):
    axis = np.array([318.0, 319.0, 320.0, 322.0, 325.0, 326.0, 327.0])
    # The source ordering is 293 -> 193 K; returned rows follow ascending T.
    table = np.column_stack([axis, np.tile(np.arange(11.0, 0.0, -1.0), (len(axis), 1)) * 1e-20])
    path = tmp_path / "ozone.dat"
    np.savetxt(
        path,
        table,
        header="vacuum wavelength (nm)\ncm^2/molecule\n"
        + " ".join(f"@{t}K" for t in range(293, 192, -10)),
        fmt="%.9e",
    )
    entry = dict(format="serdyuchenko", files=[path.name], temperatures_k=list(range(193, 294, 10)))
    rows = GASES["read_uv_spectra"](tmp_path, entry)
    assert len(rows) == 11
    np.testing.assert_allclose(rows[0][1], 1e-24, atol=0)
    np.testing.assert_allclose(rows[-1][1], 11e-24, atol=0)
    with pytest.raises(ValueError, match="temperature columns"):
        GASES["read_uv_spectra"](tmp_path, {**entry, "temperatures_k": list(range(190, 291, 10))})
    # Actual provider ASCII wavelength is decimal rather than scientific notation.
    raw = path.read_text().replace("vacuum", "air")
    path.write_text(raw)
    with pytest.raises(ValueError, match="vacuum"):
        GASES["read_uv_spectra"](tmp_path, entry)


def test_bogumil_native_temperature_validation(tmp_path):
    path = tmp_path / "so2.dat"
    path.write_text(
        "1. column title: wavelength (vacuum) nm\n2. column unit: cm2/molec\n"
        "Temperature: 203K\n319.0 1.0E-20\n326.0 2.0E-20\n"
    )
    entry = dict(format="bogumil", files=[path.name], temperatures_k=[203])
    rows = GASES["read_uv_spectra"](tmp_path, entry)
    np.testing.assert_allclose(rows[0][1], [1e-24, 2e-24], atol=0)
    with pytest.raises(ValueError, match="temperature must match"):
        GASES["read_uv_spectra"](tmp_path, {**entry, "temperatures_k": [293]})


def test_uv_components_apply_columns_and_report_effective_temperature(tmp_path):
    for temperature, scale in ((200, 1), (300, 3)):
        (tmp_path / f"{temperature}.dat").write_text(
            "wavelength (vacuum) nm; cross section cm2/molec\n"
            f"Temperature: {temperature}K\n"
            f"319.0 {scale}.0E-20\n320.0 {scale}.0E-20\n"
            f"325.0 {2 * scale}.0E-20\n326.0 {2 * scale}.0E-20\n"
        )
    entry = dict(
        species="O3",
        format="bogumil",
        temperatures_k=[200, 300],
        files=["200.dat", "300.dat"],
        included_cases=["UV"],
        temperature_policy="clip",
        pressure_policy="fixed laboratory spectrum",
    )
    pressure = np.array([100000.0, 50000.0, 10000.0])
    columns = np.array([1e22, 3e22])
    atmosphere = hydrostatic_atmosphere(
        pressure,
        np.array([190.0, 310.0]),
        np.zeros(2),
        target_pressure_boundaries_pa=pressure,
        gravity_m_s2=9.80665,
        bottom_height_m=0,
        gas_columns_molecules_m2={"O3": columns},
        air_broadener="dry_air",
    )
    grid = wavenumber_grid(np.array([1e7 / 325, 1e7 / 320]))
    components, report = GASES["uv_components"](tmp_path, [entry], atmosphere, grid)
    np.testing.assert_allclose(
        components[0].optical_depth,
        np.array([[2e-24, 1e-24], [6e-24, 3e-24]]) * columns[:, None],
        atol=0,
    )
    assert report["O3"]["held_layers"] == [0, 1]
    assert report["O3"]["effective_temperature_k"] == [200, 300]


def test_manifest_optional_enrichment_keeps_baseline_notebook_api():
    notebook = json.loads((ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    assert "five-band-enriched-inputs" in cells["config"]
    assert "verify_inputs(MANIFEST)" in cells["config"]
    assert "(320, 325, False)" in cells["config"]
    assert "(9950, 10050, True)" in cells["config"]
    assert "gas_helpers['uv_components']" in cells["materialize"]
    assert "gas_helpers['project_trace_gases']" in cells["atmosphere"]
    assert "points=None" in cells["materialize"]
