"""Observe the real atmospheric reader in a naturally free-threaded process."""

import subprocess
import sys
from pathlib import Path


def test_classic_atmospheric_read_preserves_free_threading(tmp_path: Path) -> None:
    code = """
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file
from orsia_data import read_netcdf_profiles

assert os.environ.get('PYTHON_GIL') is None and 'gil' not in sys._xoptions
assert not sys._is_gil_enabled()
path = Path(sys.argv[1])
with netcdf_file(path, 'w') as dataset:
    for name, values, units in (
        ('time', [0.0], 'hours since 2025-01-01'),
        ('level', [1000.0, 500.0], 'hPa'),
        ('latitude', [0.0], 'degrees_north'),
        ('longitude', [0.0], 'degrees_east'),
    ):
        dataset.createDimension(name, len(values))
        variable = dataset.createVariable(name, 'f8', (name,))
        variable.units = units
        variable[:] = values
    field = dataset.createVariable('t', 'f8', ('time', 'level', 'latitude', 'longitude'))
    field.units = 'K'
    field[:] = np.array([280.0, 250.0]).reshape(1, 2, 1, 1)
levels, fields = read_netcdf_profiles(
    path, variables={'t':'K'}, coordinates=('time','level','latitude','longitude'),
    vertical_units='hPa', valid_time=datetime(2025, 1, 1, tzinfo=UTC),
    latitude_deg=0.0, longitude_deg=0.0,
)
np.testing.assert_array_equal(levels, [1000.0, 500.0])
np.testing.assert_array_equal(fields['t'], [280.0, 250.0])
assert not sys._is_gil_enabled(), 'Atmospheric read enabled the GIL'
assert not {'xarray', 'cftime', 'netCDF4'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "column.nc")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
