"""GRIB selection/lifetime tests with a controlled ecCodes boundary."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_data import read_gfs_profiles
from orsia_data.atmosphere.grib import _scan_order

CYCLE = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("alternate", [False, True])
def test_cfgrib_consecutive_j_correction(alternate: bool) -> None:
    raw = np.array([100, 101, 108, 109, 116, 117.0])
    if alternate:
        raw[2:4] = raw[2:4][::-1]
    cfgrib_values = raw.reshape(2, 3).copy()
    if alternate:
        cfgrib_values[1] = cfgrib_values[1, ::-1]
    field = xr.DataArray(
        cfgrib_values,
        dims=("latitude", "longitude"),
        attrs={
            "GRIB_jPointsAreConsecutive": 1,
            "GRIB_alternativeRowScanning": alternate,
        },
    )
    assert_allclose(_scan_order(field).values, [[100, 108, 116], [101, 109, 117]])
    assert_allclose(field.values, cfgrib_values)  # Correction does not mutate the decoded input.


@pytest.mark.parametrize(
    "native_longitude,negative,expected",
    [
        ([350.0, 0, 10], False, [350.0, 360, 370]),
        ([10.0, 0, 350], True, [10.0, 0, -10]),
        ([0.0, 270], False, [0.0, 270]),
        ([270.0, 0], True, [270.0, 0]),
        ([0.0, 180], False, [0.0, 180]),
        ([180.0, 0], True, [180.0, 0]),
    ],
)
def test_cfgrib_longitude_keeps_native_support(
    native_longitude: list[float], negative: bool, expected: list[float]
) -> None:
    values = np.arange(2 * len(native_longitude)).reshape(2, -1)
    decoded_longitude = np.sort(native_longitude)
    field = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"longitude": decoded_longitude},
        attrs={
            "GRIB_iScansNegatively": int(negative),
            "GRIB_longitudeOfFirstGridPointInDegrees": native_longitude[0],
        },
    )
    corrected = _scan_order(field)
    np.testing.assert_array_equal(corrected.longitude.values, expected)
    np.testing.assert_array_equal(corrected.values, values)
    np.testing.assert_array_equal(field.longitude.values, decoded_longitude)


def message(level: int, **changes: object) -> dict[str, Any]:
    return {
        "shortName": "t",
        "typeOfLevel": "isobaricInhPa",
        "level": level,
        "dataDate": 20250101,
        "dataTime": 0,
        "validityDate": 20250101,
        "validityTime": 300,
        "edition": 2,
        "gridType": "regular_ll",
        "stepType": "instant",
        "units": "K",
        "latitudes": np.array([10, 10, 0, 0]),
        "longitudes": np.array([0, 180, 0, 180]),
        "values": np.array([100, 108, 101, 109]) + level,
        "missingValue": 9999,
        "Ni": 2,
        "Nj": 2,
        "numberOfPoints": 4,
        "md5GridSection": "regular-grid",
        **changes,
    }


def backend(monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]) -> list[object]:
    handles = iter(messages)
    released: list[object] = []
    monkeypatch.setitem(
        sys.modules,
        "eccodes",
        SimpleNamespace(
            codes_grib_new_from_file=lambda stream: next(handles, None),
            codes_get=lambda handle, key: handle[key],
            codes_get_array=lambda handle, key: handle[key],
            codes_is_defined=lambda handle, key: key in handle,
            codes_release=released.append,
        ),
    )

    def open_dataset(path: Path, **kwargs: Any) -> xr.Dataset:
        assert kwargs["engine"] == "cfgrib" and kwargs["chunks"] is None
        options = kwargs["backend_kwargs"]
        assert options["indexpath"] == "" and not options["cache_geo_coords"]
        assert options["values_dtype"] == np.dtype("float64")
        filters = options["filter_by_keys"]
        selected = [
            m
            for m in messages
            if all(m[k] in v if isinstance(v, list) else m[k] == v for k, v in filters.items())
        ]
        fields = []
        for m in selected:
            lat, lon = np.unique(m["latitudes"]), np.unique(m["longitudes"])
            flat = np.searchsorted(lat, m["latitudes"]) * len(lon) + np.searchsorted(
                lon, m["longitudes"]
            )
            if len(np.unique(flat)) != len(lat) * len(lon):
                raise ValueError("invalid generated grid")
            values = np.array(m["values"], dtype=float, copy=True)
            if m.get("bitmapPresent", 0):
                values[values == m["missingValue"]] = np.nan
            grid = np.empty((len(lat), len(lon)))
            grid.flat[flat] = values
            fields.append(
                xr.DataArray(
                    grid.reshape(1, 1, 1, len(lat), len(lon)),
                    dims=("time", "step", "isobaricInhPa", "latitude", "longitude"),
                    coords={
                        "time": [0],
                        "step": [3],
                        "isobaricInhPa": [m["level"]],
                        "latitude": lat,
                        "longitude": lon,
                    },
                )
            )
        return xr.concat(fields, dim="isobaricInhPa").to_dataset(name=filters["shortName"])

    monkeypatch.setattr("orsia_data.atmosphere.grib.xr.open_dataset", open_dataset)
    return released


def read(path: Path) -> dict[str, NDArray[np.float64]]:
    return read_gfs_profiles(
        path,
        variables={"t": "K"},
        pressure_levels_pa=np.array([100000.0, 50000]),
        cycle=CYCLE,
        lead_hours=3,
        latitude_deg=5,
        longitude_deg=90,
    )


def test_grib_identity_and_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "native.grib2"
    path.touch()
    released = backend(
        monkeypatch,
        [
            message(1000),
            message(500),
            message(1000, shortName="q"),
            message(1000, typeOfLevel="surface"),
            message(400),
            message(1000, dataTime=600),
        ],
    )
    assert_allclose(read(path)["t"], [1104.5, 604.5])
    assert len(released) == 6


def test_grib_valid_value_equal_to_preflight_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "native.grib2"
    path.touch()
    backend(monkeypatch, [message(level, values=np.full(4, 9999.0)) for level in (1000, 500)])
    assert_allclose(read(path)["t"], [9999, 9999], rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"units": "degC"},
        {"edition": 1},
        {"gridType": "reduced_ll"},
        {"stepType": "accum"},
        {"perturbationNumber": 0},
        {"numberOfPoints": 3},
    ],
)
def test_grib_rejection_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    path = tmp_path / "native.grib2"
    path.touch()
    released = backend(monkeypatch, [message(1000, **changes)])
    with pytest.raises(ValueError):
        read(path)
    assert len(released) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"values": np.array([9999, 108, 101, 109]), "bitmapPresent": 1},
        {"values": np.array([np.nan, 108, 101, 109])},
        {"longitudes": np.array([0, 180, 0, 0])},
        {"latitudes": np.array([10, 10, 0, 1])},
    ],
)
def test_grib_decoded_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    path = tmp_path / "native.grib2"
    path.touch()
    released = backend(monkeypatch, [message(1000, **changes), message(500)])
    with pytest.raises(ValueError):
        read(path)
    assert len(released) == 2


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [message(1000)],
        [message(1000), message(1000)],
        [message(1000, validityTime=600), message(500)],
    ],
)
def test_grib_missing_and_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]
) -> None:
    path = tmp_path / "native.grib2"
    path.touch()
    released = backend(monkeypatch, messages)
    with pytest.raises(ValueError):
        read(path)
    assert len(released) == len(messages)


def test_grib_input_validation(tmp_path: Path) -> None:
    for levels in (
        np.array([]),
        np.array([10000, 20000.0]),
        np.array([10001.0]),
        np.array([-100.0]),
    ):
        with pytest.raises(ValueError):
            read_gfs_profiles(
                tmp_path / "absent",
                variables={"t": "K"},
                pressure_levels_pa=levels,
                cycle=CYCLE,
                lead_hours=3,
                latitude_deg=5,
                longitude_deg=90,
            )
