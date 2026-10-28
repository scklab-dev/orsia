"""Optional native backends on generated files, never live provider qualification."""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_data import read_gfs_profiles, read_netcdf_profiles

pytestmark = pytest.mark.skipif(
    os.environ.get("ORSIA_TEST_ATMOSPHERE_BACKENDS") != "1",
    reason="run just test-atmosphere-backends for optional native interoperability",
)


def test_native_eccodes_generated_grib(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.grib2"
    _write_grib(path)
    values = _read_grib(path)
    assert_allclose(values, [1104.5, 604.5])
    assert not list(tmp_path.glob("*.idx"))
    assert values.dtype == np.float64 and values.flags.writeable
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("levels", [(1000, 1000, 500), (1000,)])
def test_native_grib_missing_and_duplicate(tmp_path: Path, levels: tuple[int, ...]) -> None:
    path = tmp_path / "synthetic.grib2"
    _write_grib(path, levels=levels)
    with pytest.raises(ValueError, match="duplicate|missing requested"):
        _read_grib(path)
    assert not list(tmp_path.glob("*.idx"))


def test_native_grib_bitmap_exact_node_and_seam(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.grib2"
    _write_grib(
        path,
        changes={"bitmapPresent": 1, "missingValue": 9999.0},
        values=np.array([9999.0, 108, 101, 109]),
    )
    with pytest.raises(ValueError, match="missing/nonfinite"):
        _read_grib(path)
    assert_allclose(_read_grib(path, latitude=0, longitude=10), [1109, 609])
    assert_allclose(_read_grib(path, latitude=0, longitude=185, periodic=True), [1105, 605])
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("name,units", [("t", "K"), ("gh", "gpm")])
def test_native_grib_valid_sentinel_without_bitmap(tmp_path: Path, name: str, units: str) -> None:
    path = tmp_path / "synthetic.grib2"
    _write_grib(path, short_name=name, values=np.array([9999.0, 108, 101, 109]))
    eccodes = pytest.importorskip("eccodes")
    with path.open("rb") as stream:
        handle = eccodes.codes_grib_new_from_file(stream)
        try:
            assert eccodes.codes_get(handle, "bitmapPresent") == 0
            assert eccodes.codes_get(handle, "numberOfMissing") == 0
            assert eccodes.codes_get(handle, "missingValue") == 9999
        finally:
            eccodes.codes_release(handle)
    assert_allclose(
        _read_grib(path, latitude=10, longitude=0, name=name, units=units), [9999, 9999]
    )
    assert_allclose(_read_grib(path, name=name, units=units), [3329.25, 2954.25])


@pytest.mark.parametrize(
    "changes,values",
    [
        ({"alternativeRowScanning": 1}, np.array([100, 108, 109, 101.0])),
        ({"jPointsAreConsecutive": 1}, np.array([100, 101, 108, 109.0])),
        (
            {
                "jPointsAreConsecutive": 1,
                "iScansNegatively": 1,
                "longitudeOfFirstGridPointInDegrees": 10,
                "longitudeOfLastGridPointInDegrees": 0,
            },
            np.array([108, 109, 100, 101.0]),
        ),
        (
            {
                "iScansNegatively": 1,
                "longitudeOfFirstGridPointInDegrees": 10,
                "longitudeOfLastGridPointInDegrees": 0,
            },
            np.array([108, 100, 109, 101.0]),
        ),
        (
            {
                "jScansPositively": 1,
                "latitudeOfFirstGridPointInDegrees": 0,
                "latitudeOfLastGridPointInDegrees": 10,
            },
            np.array([101, 109, 100, 108.0]),
        ),
        (
            {"jPointsAreConsecutive": 1, "alternativeRowScanning": 1},
            np.array([100, 101, 109, 108.0]),
        ),
        (
            {"jPointsAreConsecutive": 1, "Ni": 3, "longitudeOfLastGridPointInDegrees": 20},
            np.array([100, 101, 108, 109, 116, 117.0]),
        ),
    ],
)
def test_native_grib_scan_order(
    tmp_path: Path, changes: dict[str, int | float], values: NDArray[np.float64]
) -> None:
    path = tmp_path / "synthetic.grib2"
    _write_grib(path, changes=changes, values=values)
    assert_allclose(_read_grib(path, latitude=10, longitude=10), [1108, 608])
    assert_allclose(_read_grib(path, latitude=0, longitude=0), [1101, 601])


def test_native_grib_different_grids_and_irrelevant_cycles(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.grib2"
    _write_grib(path, levels=(1000,))
    _write_grib(
        path,
        levels=(500,),
        changes={"longitudeOfLastGridPointInDegrees": 20.0, "iDirectionIncrementInDegrees": 20.0},
        mode="ab",
    )
    _write_grib(path, changes={"step": 6}, mode="ab")
    assert_allclose(_read_grib(path), [1104.5, 602.5])


@pytest.mark.parametrize("negative_scan", [False, True])
def test_native_grib_wrapped_scan(tmp_path: Path, negative_scan: bool) -> None:
    path = tmp_path / "synthetic.grib2"
    values = np.array([[100, 108, 116], [101, 109, 117.0]])
    _write_grib(
        path,
        changes={
            "Ni": 3,
            "iScansNegatively": int(negative_scan),
            "longitudeOfFirstGridPointInDegrees": 10 if negative_scan else 350,
            "longitudeOfLastGridPointInDegrees": 350 if negative_scan else 10,
        },
        values=(values[:, ::-1] if negative_scan else values).ravel(),
    )
    assert_allclose(_read_grib(path, latitude=10, longitude=350), [1100, 600])
    assert_allclose(_read_grib(path, latitude=10, longitude=0), [1108, 608])
    assert_allclose(_read_grib(path, latitude=10, longitude=10), [1116, 616])
    for longitude in (355, -5, 715):
        assert_allclose(_read_grib(path, latitude=10, longitude=longitude), [1104, 604])
    assert_allclose(_read_grib(path, latitude=5, longitude=5), [1112.5, 612.5])
    for longitude in (180, 349, 11):
        with pytest.raises(ValueError, match="outside coordinate coverage"):
            _read_grib(path, longitude=longitude)


def _read_grib(
    path: Path,
    *,
    latitude: float = 5,
    longitude: float = 5,
    periodic: bool = False,
    name: str = "t",
    units: str = "K",
) -> NDArray[np.float64]:
    return read_gfs_profiles(
        path,
        variables={name: units},
        pressure_levels_pa=np.array([100000.0, 50000]),
        cycle=datetime(2025, 1, 1, tzinfo=UTC),
        lead_hours=3,
        latitude_deg=latitude,
        longitude_deg=longitude,
        periodic_longitude=periodic,
    )[name]


def _write_grib(
    path: Path,
    *,
    levels: tuple[int, ...] = (1000, 500),
    short_name: str = "t",
    changes: dict[str, int | float] | None = None,
    values: NDArray[np.float64] | None = None,
    mode: str = "wb",
) -> None:
    eccodes = pytest.importorskip("eccodes")
    with path.open(mode) as stream:
        for level in levels:
            handle = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
            try:
                for key, value in {
                    "Ni": 2,
                    "Nj": 2,
                    "latitudeOfFirstGridPointInDegrees": 10.0,
                    "longitudeOfFirstGridPointInDegrees": 0.0,
                    "latitudeOfLastGridPointInDegrees": 0.0,
                    "longitudeOfLastGridPointInDegrees": 10.0,
                    "iDirectionIncrementInDegrees": 10.0,
                    "jDirectionIncrementInDegrees": 10.0,
                    "dataDate": 20250101,
                    "dataTime": 0,
                    "step": 3,
                    "typeOfLevel": "isobaricInhPa",
                    "level": level,
                    "shortName": short_name,
                    **(changes or {}),
                }.items():
                    eccodes.codes_set(handle, key, value)
                data = np.array([100, 108, 101, 109.0]) if values is None else values.copy()
                data[data != 9999] += level
                eccodes.codes_set_values(handle, data)
                eccodes.codes_write(handle, stream)
            finally:
                eccodes.codes_release(handle)


def test_native_hdf5_generated_file(tmp_path: Path) -> None:
    netcdf = pytest.importorskip("h5netcdf")
    path = tmp_path / "synthetic.nc"
    with netcdf.File(path, "w") as ds:
        for name, values, units in (
            ("time", [0.0], "hours since 2025-01-01"),
            ("level", [1000.0, 500.0], "hPa"),
            ("latitude", [10.0, 0.0], "degrees_north"),
            ("longitude", [0.0, 10.0], "degrees_east"),
        ):
            ds.dimensions[name] = len(values)
            variable = ds.create_variable(name, (name,), dtype="f8")
            variable.attrs["units"] = units
            variable[:] = values
        variable = ds.create_variable("t", ("time", "level", "latitude", "longitude"), dtype="f8")
        variable.attrs["units"] = "K"
        variable[:] = np.array([[[[100, 108], [101, 109]], [[200, 208], [201, 209]]]])
        q = ds.create_variable(
            "q", ("longitude", "time", "level", "latitude"), dtype="i2", fillvalue=-99
        )
        q.attrs.update(
            {
                "units": "1",
                "scale_factor": 0.001,
                "add_offset": 0.01,
                "valid_range": np.array([0, 1], dtype="i2"),
            }
        )
        q[:] = 1
        q[0, 0, 0, 0] = -99
    levels, profiles = read_netcdf_profiles(
        path,
        variables={"t": "K"},
        coordinates=("time", "level", "latitude", "longitude"),
        vertical_units="hPa",
        valid_time=datetime(2025, 1, 1, tzinfo=UTC),
        latitude_deg=5,
        longitude_deg=5,
        backend="hdf5",
    )
    assert_allclose(levels, [1000, 500])
    assert_allclose(profiles["t"], [104.5, 204.5])
    assert not sys._is_gil_enabled()
    options = dict(
        variables={"q": "1"},
        coordinates=("time", "level", "latitude", "longitude"),
        vertical_units="hPa",
        valid_time=datetime(2025, 1, 1, tzinfo=UTC),
        latitude_deg=5.0,
        longitude_deg=5.0,
        backend="hdf5",
    )
    with pytest.raises(ValueError, match="missing/nonfinite"):
        read_netcdf_profiles(path, **options)  # type: ignore
    levels, fields = read_netcdf_profiles(path, vertical_interval=(500, 900), **options)  # type: ignore
    assert_allclose(levels, [500])
    assert_allclose(fields["q"], [0.011])
    fields["q"][:] = 99  # Detached even after the HDF5 file has closed.
    with netcdf.File(path, "a") as dataset:
        dataset.variables["q"][0, 0, 1, 0] = 2
    with pytest.raises(ValueError, match="missing/nonfinite"):
        read_netcdf_profiles(path, vertical_interval=(500, 900), **options)  # type: ignore
    assert not sys._is_gil_enabled()
