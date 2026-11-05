"""Aerosol profile, phase normalization and bounded native-adapter regressions."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import eval_legendre

from notebooks.support.five_band_scenario import ExponentialAerosol
from notebooks.support.notebook_sasktran2_layers import scattering_legendre_coefficients

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"


@pytest.mark.parametrize(
    "edges", [[0, 100, 1000, 8000], [2300, 2310, 2500, 2900], [10, 10.001, 10.002]]
)
def test_exponential_profile_integrates_actual_interfaces(edges):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    code = "".join(next(cell["source"] for cell in cells if cell["id"] == "atmosphere"))
    code = code[code.index("# Integrate the exponential aerosol profile") :]
    heights = np.array(edges, dtype=float)
    namespace: dict = dict(
        np=np,
        physics=SimpleNamespace(solar_aerosol=SimpleNamespace(scale_height_m=1000.0)),
        atmosphere=SimpleNamespace(height_boundaries_m=heights),
        show_atmosphere=Mock(),
        temperature_k=None,
        specific_humidity=None,
    )
    exec(code, namespace)
    fractions = namespace["aerosol_layer_fraction"]
    assert np.all(fractions > 0)
    np.testing.assert_allclose(fractions.sum(), 1.0, rtol=2e-15)
    # Independent quadrature on every interval, including truncated/offset columns.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    integrals = []
    for bottom, top in zip(heights[:-1], heights[1:], strict=True):
        z = (top + bottom) / 2 + (top - bottom) * nodes / 2
        integrals.append((top - bottom) / 2 * np.sum(weights * np.exp(-(z - heights[0]) / 1000)))
    expected = np.array(integrals) / sum(integrals)
    np.testing.assert_allclose(fractions, expected, rtol=2e-13)


@pytest.mark.parametrize("rayleigh,aerosol", [(0.2, 0.0), (0.0, 0.3), (0.2, 0.3), (0.0, 0.0)])
def test_scattering_coefficients_normalization_and_weighting(rayleigh, aerosol):
    g = np.array([[0.65, -0.3], [0.0, 0.8]])
    coefficients = scattering_legendre_coefficients(
        np.full_like(g, rayleigh), np.full_like(g, aerosol), g, moments=32
    )
    np.testing.assert_array_equal(coefficients[0], np.ones_like(g))
    if rayleigh + aerosol == 0:
        assert np.all(coefficients[2] == 0.5)
        assert np.count_nonzero(coefficients) == 2 * g.size
        return
    nodes, weights = np.polynomial.legendre.leggauss(256)
    mu = nodes[:, None, None]
    hg = (1 - g**2) / (1 + g**2 - 2 * g * mu) ** 1.5
    phase = (rayleigh * 0.75 * (1 + mu**2) + aerosol * hg) / (rayleigh + aerosol)
    expected = np.array(
        [
            (2 * order + 1)
            / 2
            * np.sum(weights[:, None, None] * phase * eval_legendre(order, mu), axis=0)
            for order in range(32)
        ]
    )
    np.testing.assert_allclose(coefficients, expected, rtol=1e-10, atol=2e-12)
    np.testing.assert_allclose(coefficients[1] / 3, aerosol * g / (rayleigh + aerosol))


@pytest.mark.parametrize(
    "field,value",
    [
        ("optical_depth", -0.1),
        ("reference_wavelength_nm", 0),
        ("angstrom_exponent", -1),
        ("scale_height_m", 0),
        ("single_scattering_albedo", 1.1),
        ("asymmetry", 0.81),
        ("optical_depth", float("nan")),
        ("scale_height_m", float("inf")),
    ],
)
def test_aerosol_settings_reject_nonphysical_values(field, value):
    settings = json.loads((ROOT / "notebooks/data/five-band.json").read_text())["physics"][
        "solar_aerosol"
    ]
    settings[field] = value
    with pytest.raises(ValidationError):
        ExponentialAerosol.model_validate(settings)


def test_native_hg_coefficients_and_zero_aerosol_compatibility(tmp_path):
    # Keep native backend GIL effects isolated, as in the notebook reference worker.
    code = r"""
import json, sys
from pathlib import Path
import numpy as np
import sasktran2 as sk
from notebooks.support.notebook_sasktran2_layers import calculate, scattering_legendre_coefficients
config = sk.Config()
config.num_stokes = 1
config.num_streams = 32
config.num_singlescatter_moments = 32
config.num_threads = 1
geometry = sk.Geometry1D(
    0.8, 0.0, 6372000.0, np.array([0., 1000., 2000.]),
    sk.InterpolationMethod.LinearInterpolation, sk.GeometryType.PlaneParallel,
)
atmosphere = sk.Atmosphere(
    geometry, config, wavelengths_nm=np.array([550., 600.]), calculate_derivatives=False,
)
hg = sk.optical.HenyeyGreenstein.from_parameters(
    np.array([550., 600.]), np.ones(2), np.full(2, 0.95), np.full(2, 0.65), max_num_moments=32,
)
quantities = hg.atmosphere_quantities(atmosphere)
expected = scattering_legendre_coefficients(
    np.zeros((3, 2)), np.ones((3, 2)), np.full((3, 2), 0.65), moments=32,
)
np.testing.assert_allclose(quantities.leg_coeff, expected, rtol=1e-14, atol=1e-14)
arrays = dict(
    height_m=np.array([0., 1000., 2000.]), temperature_k=np.array([280., 270.]),
    wavenumbers_cm1=1e7/np.array([600., 550.]), absorption_depth=np.full((2, 2), 0.03),
    rayleigh_depth=np.full((2, 2), 0.02), thermal=False, mu0=0.8, muv=0.7,
    irradiance=np.ones(2), albedo=np.full(2, 0.2), emissivity=np.full(2, 0.8),
    surface_temperature_k=280.,
)
path = Path(sys.argv[1]) / "shared.npz"
np.savez(path, **arrays)
absent = calculate(path, cells=4, streams=32, runs=1)
arrays.update(
    aerosol_extinction_depth=np.zeros((2, 2)), aerosol_scattering_depth=np.zeros((2, 2)),
    aerosol_asymmetry=np.full((2, 2), 0.65),
)
np.savez(path, **arrays)
zero = calculate(path, cells=4, streams=32, runs=1)
np.testing.assert_array_equal(absent["radiance"], zero["radiance"])
np.testing.assert_array_equal(absent["los_optical_depth"], zero["los_optical_depth"])
arrays["aerosol_extinction_depth"][:] = 0.05
arrays["aerosol_scattering_depth"][:] = 0.0475
np.savez(path, **arrays)
aerosol = calculate(path, cells=4, streams=32, runs=1)
np.testing.assert_allclose(aerosol["los_optical_depth"], np.full(2, 0.2/0.7), rtol=1e-5)
assert np.all(np.isfinite(aerosol["radiance"]))
arrays["thermal"] = True
np.savez(path, **arrays)
try:
    calculate(path, cells=4, streams=32, runs=1)
except ValueError as error:
    assert "thermal control" in str(error)
else:
    raise AssertionError("thermal aerosols accepted")
print(json.dumps(dict(
    python=sys.version, gil_enabled=sys._is_gil_enabled(), native_hg_moments="matched",
    zero_aerosol="identical", aerosol_los="matched",
)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout.splitlines()[-1])
    assert "3.14" in report["python"]
    assert isinstance(report["gil_enabled"], bool)
    print(report)
