"""GRIB selection/lifetime tests with a controlled ecCodes boundary."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_data import read_gfs_profiles

CYCLE = datetime(2025, 1, 1, tzinfo=UTC)


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
        "iScansNegatively": 0,
        "alternativeRowScanning": 0,
        "jPointsAreConsecutive": 0,
        "longitudeOfFirstGridPointInDegrees": 0,
        "bitmapPresent": 0,
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
            codes_get_array=lambda handle, key: (
                (handle["values"] != handle["missingValue"]).astype(int)
                if key == "bitmap"
                else handle[key]
            ),
            codes_is_defined=lambda handle, key: key in handle,
            codes_release=released.append,
        ),
    )

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
        {"missingValueManagementUsed": 1},
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
    assert len(released) == 1


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


@pytest.mark.parametrize(
    "native_longitude,negative",
    [
        ([350.0, 0, 10], False),
        ([10.0, 0, 350], True),
        ([0.0, 270], False),
        ([270.0, 0], True),
        ([0.0, 180], False),
        ([180.0, 0], True),
    ],
)
def test_grib_longitude_keeps_native_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_longitude: list[float],
    negative: bool,
) -> None:
    path = tmp_path / "native.grib2"
    path.touch()
    ni = len(native_longitude)
    for index, longitude in enumerate(native_longitude):
        backend(
            monkeypatch,
            [
                message(
                    level,
                    Ni=ni,
                    numberOfPoints=ni * 2,
                    latitudes=np.repeat([10.0, 0], ni),
                    longitudes=np.tile(native_longitude, 2),
                    values=100 + level + np.tile(np.arange(ni) * 8, 2) + np.repeat([0, 1], ni),
                    iScansNegatively=int(negative),
                    longitudeOfFirstGridPointInDegrees=native_longitude[0],
                )
                for level in (1000, 500)
            ],
        )
        fields = read_gfs_profiles(
            path,
            variables={"t": "K"},
            pressure_levels_pa=np.array([100000.0, 50000]),
            cycle=CYCLE,
            lead_hours=3,
            latitude_deg=5,
            longitude_deg=longitude,
        )
        assert_allclose(fields["t"], np.array([1100.5, 600.5]) + index * 8)


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
