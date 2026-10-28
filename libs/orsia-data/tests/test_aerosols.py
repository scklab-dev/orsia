"""CAMS native-column quantity isolation and CF replay."""

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
from orsia_data import read_cams_aod


def test_cams_aod_native_netcdf(tmp_path):
    path = tmp_path / "cams.nc"
    dataset = xr.Dataset(
        {
            "aod550": (
                ("time", "latitude", "longitude"),
                np.array([[[0.1, 0.3], [0.3, 0.5]]]),
                {"units": "1"},
            )
        },
        coords={
            "time": ("time", [0.0], {"units": "hours since 2024-01-01 00:00:00"}),
            "latitude": ("latitude", [40.0, 41.0], {"units": "degrees_north"}),
            "longitude": ("longitude", [0.0, 1.0], {"units": "degrees_east"}),
        },
    )
    dataset.to_netcdf(path, engine="scipy")
    kwargs = {
        "coordinates": ("time", "latitude", "longitude"),
        "valid_time": datetime(2024, 1, 1, tzinfo=UTC),
        "latitude_deg": 40.5,
        "longitude_deg": 0.5,
    }
    result = read_cams_aod(path, wavelengths_nm=(550,), **kwargs)
    assert result[550] == pytest.approx(0.3)
    for waves in ((), (550, 550), (123,)):
        with pytest.raises(ValueError):
            read_cams_aod(path, wavelengths_nm=waves, **kwargs)
    dataset["aod550"].values[:] = -1
    dataset.to_netcdf(path, engine="scipy")
    with pytest.raises(ValueError, match="negative"):
        read_cams_aod(path, wavelengths_nm=(550,), **kwargs)
