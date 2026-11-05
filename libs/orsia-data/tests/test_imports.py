"""Public data imports defer optional readers until callers use them."""

import subprocess
import sys


def test_public_import_defers_optional_backends():
    # pytest collection may already have imported these readers; use fresh state.
    code = """
import sys
import orsia_data

optional = {
    "cfgrib", "eccodes", "h5netcdf", "h5py", "netCDF4", "cdsapi",
    "rasterio", "pyproj", "hapi", "sasktran2",
}
assert not (optional & sys.modules.keys()), optional & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
