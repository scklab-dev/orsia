"""Required dependencies and deferred imports preserve the free-threaded runtime."""

import subprocess
import sys
from importlib.metadata import metadata


def test_orbital_libraries_expose_explicit_backend_extra():
    for name in ("orsia-geometry", "orsia-data"):
        package = metadata(name)
        requirements = package.get_all("Requires-Dist") or []
        assert any(
            "satkit==0.24.1" in r and "extra == 'orbital'" in r.replace('"', "'")
            for r in requirements
        ), requirements
        assert "satkit==0.24.1" not in requirements
        assert "orbital" in (package.get_all("Provides-Extra") or [])


def test_core_native_imports_preserve_free_threading():
    # Fresh state attributes a GIL transition to its import, even after collection
    # or another test has loaded an optional reader in the calling process.
    code = """
import importlib
import os
import sys
from importlib.metadata import version

assert os.environ.get("PYTHON_GIL") != "0" and sys._xoptions.get("gil") != "0", (
    "Native-import GIL effects must remain observable; remove forced-off overrides"
)
for name in (
    "orsia_geometry", "orsia_geometry.frames", "orsia_geometry.los", "orsia_geometry.attitude",
    "orsia_numerics.execution", "orsia_uncertainty", "orsia_radiative", "orsia_dispersion",
    "orsia_scene",
    "orsia_radiative.irradiance", "orsia_radiative.radiance",
    "orsia_radiative.transfer.native.transfer_native",
    "orsia_instruments", "orsia_instruments.photodetection.coefficients",
    "orsia_instruments.photodetection.frames",
    "orsia_instruments.acquisition.quadrature", "orsia_inverse",
    "orsia_data", "orsia_data.surface.sentinel2",
):
    importlib.import_module(name)
    assert not sys._is_gil_enabled(), f"Importing {name} enabled the GIL"
from datetime import UTC, datetime
import numpy as np
from orsia_data import read_orbital_elements, select_orbital_elements
from orsia_geometry import compute_lines_of_sight
from orsia_geometry.orbits import SGP4MeanElements
assert SGP4MeanElements.__module__ == "orsia_geometry.orbits"
assert callable(read_orbital_elements)
epoch = datetime(2026, 1, 1, tzinfo=UTC)
elements = SGP4MeanElements(
    norad_id=25544, epoch=epoch, eccentricity=0.001, inclination_rad=0.2,
    raan_rad=0.3, argument_of_perigee_rad=0.4, mean_anomaly_rad=0.5,
    mean_motion_rad_s=0.001, bstar_per_earth_radius=1e-5,
)
assert select_orbital_elements(
    [elements], object_ids=(25544,), element_epoch_interval_utc=(epoch, epoch),
) == [elements]
origins, directions = compute_lines_of_sight(
    np.array([0.0]), epoch=datetime(2026, 1, 1, tzinfo=UTC), validity_s=(0, 0),
    trajectory=lambda t: (np.zeros((len(t), 3)), np.zeros((len(t), 3))),
    platform_attitude=lambda t: np.tile(np.eye(3), (len(t), 1, 1)),
    instrument_attitude=lambda t: np.tile(np.eye(3), (len(t), 1, 1)),
    directions_i=np.array([[0.0, 0.0, 1.0]]), mount_offset_p_m=np.zeros(3),
)
assert origins.shape == directions.shape == (1, 1, 3)
assert not sys._is_gil_enabled()
deferred = {
    "cfgrib", "eccodes", "h5netcdf", "h5py", "netCDF4", "cdsapi",
    "rasterio", "pyproj", "hapi", "sasktran2", "satkit",
}
assert not (deferred & sys.modules.keys()), deferred & sys.modules.keys()
# The development environment includes the orbital extra; base imports remain deferred.
importlib.import_module("satkit")
assert version("satkit") == "0.24.1"
assert not sys._is_gil_enabled(), "Importing orbital SatKit enabled the GIL"
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
