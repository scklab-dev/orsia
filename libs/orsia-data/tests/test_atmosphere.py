"""Synthetic files/clients only; no external atmospheric dataset downloads."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from orsia_data import (
    download_cams,
    download_era5,
    download_gfs,
    file_sha256,
    gfs_valid_time,
    read_netcdf_profiles,
    read_netcdf_surface_fields,
    read_pressure_level_atmosphere,
)
from orsia_data.atmosphere.sampling import linear_axis_weights
from orsia_data.services import nomads as providers
from orsia_radiative import hydrostatic_atmosphere, pressure_level_layer_means
from scipy.io import netcdf_file

CYCLE = datetime(2025, 1, 1, tzinfo=UTC)


def fixture_file(path: Path) -> None:
    with netcdf_file(path, "w") as ds:
        for name, values, units in (
            ("time", [0, 6], "hours since 2025-01-01 00:00:00"),
            ("level", [1000, 500], "hPa"),
            ("latitude", [10, 0], "degrees_north"),
            ("longitude", [0, 180], "degrees_east"),
        ):
            ds.createDimension(name, 2)
            v = ds.createVariable(name, "f8", (name,))
            v.units = units
            v[:] = values
        v = ds.createVariable("t", "f8", ("longitude", "time", "level", "latitude"))
        v.units = "K"
        x, t, z, y = np.indices((2, 2, 2, 2))
        v[:] = 100 + x * 8 + t * 4 + z * 2 + y
        q = ds.createVariable("q", "i2", ("time", "level", "latitude", "longitude"))
        q.units = "kg kg**-1"
        q.scale_factor = 0.001
        q.add_offset = 0.01
        q.missing_value = -99
        q[:] = 1


def read(
    path: Path, **kwargs: object
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    options: dict[str, Any] = dict(
        variables={"t": "K", "q": "kg kg**-1"},
        coordinates=("time", "level", "latitude", "longitude"),
        vertical_units="hPa",
        valid_time=datetime(2025, 1, 1, 3, tzinfo=UTC),
        latitude_deg=5.0,
        longitude_deg=90.0,
    )
    options.update(kwargs)
    return read_netcdf_profiles(path, **options)  # type: ignore


def test_netcdf_interpolation_packing_and_detached_replay(tmp_path: Path) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    levels, fields = read(path)
    assert_allclose(levels, [1000, 500])
    assert_allclose(fields["t"], [106.5, 108.5])
    assert_allclose(fields["q"], [0.011, 0.011])
    levels[:] = 0
    fields["q"][:] = 9
    assert_allclose(read(path)[0], [1000, 500])
    assert_allclose(
        read(path, periodic_longitude=True, longitude_deg=270.0)[1]["t"], [106.5, 108.5]
    )
    with netcdf_file(path, "a") as ds:
        ds.variables["q"][0, 0, 0, 0] = -99
    with pytest.raises(ValueError, match="missing"):
        read(path)
    assert_allclose(read(path, vertical_interval=(500.0, 900.0))[1]["q"], [0.011])
    # An exact point does not consume a missing adjacent time/grid node.
    assert_allclose(
        read(
            path,
            valid_time=datetime(2025, 1, 1, 6, tzinfo=UTC),
            latitude_deg=0.0,
            longitude_deg=180.0,
        )[1]["q"],
        [0.011, 0.011],
    )


@pytest.mark.parametrize(
    "options",
    [
        {"variables": {}},
        {"variables": {"missing": "K"}},
        {"variables": {"t": "degC"}},
        {"vertical_units": "Pa"},
        {"coordinates": ("time", "time", "latitude", "longitude")},
        {"longitude_deg": 190.0},
        {"latitude_deg": 91.0},
        {"latitude_deg": -1.0},
        {"longitude_deg": np.nan},
        {"valid_time": datetime(2025, 1, 1)},
        {"valid_time": datetime(2024, 1, 1, tzinfo=UTC)},
        {"backend": "bad"},
        {"vertical_interval": (800, 500)},
        {"vertical_interval": (np.nan, 500)},
        {"vertical_interval": (1, 10)},
    ],
)
def test_netcdf_support_failures(tmp_path: Path, options: dict[str, object]) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    with pytest.raises(ValueError):
        read(path, **options)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("calendar", "360_day"),
        ("units", "fortnights since 2025-01-01"),
        ("units", "hours"),
    ],
)
def test_bad_time_metadata(tmp_path: Path, attribute: str, value: str) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    with netcdf_file(path, "a") as ds:
        setattr(ds.variables["time"], attribute, value)
    with pytest.raises(ValueError):
        read(path)


def test_sampling_strictness() -> None:
    assert linear_axis_weights(np.array([3, 2, 1.0]), 2, periodic=False) == [(1, 1)]
    assert linear_axis_weights(np.array([1.0]), 1, periodic=False) == [(0, 1)]
    for axis, point, periodic in (
        (np.array([]), 0, False),
        (np.array([0, 1, 0.0]), 0, False),
        (np.array([0, 0.0]), 0, False),
        (np.array([0, np.nan]), 0, False),
        (np.array([0.0]), 1, False),
        (np.array([0.0]), 1, True),
        (np.array([0, 360.0]), 1, True),
    ):
        with pytest.raises(ValueError):
            linear_axis_weights(axis, point, periodic=periodic)


def test_surface_and_local_materialization(tmp_path: Path) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    with netcdf_file(path, "a") as ds:
        surface = ds.createVariable("sp", "f8", ("time", "latitude", "longitude"))
        surface.units = "Pa"
        surface[:] = 100000
    context = read_netcdf_surface_fields(
        path,
        variables={"sp": "Pa"},
        coordinates=("time", "latitude", "longitude"),
        valid_time=CYCLE,
        latitude_deg=5,
        longitude_deg=90,
    )
    assert context == {"sp": 100000}
    pressure, fields = read(path)
    pressure *= 100  # Explicit native hPa -> Pa conversion.
    boundaries = np.array([context["sp"], 75000, 50000.0])
    atmosphere = hydrostatic_atmosphere(
        boundaries,
        pressure_level_layer_means(pressure, fields["t"], boundaries),
        pressure_level_layer_means(pressure, fields["q"], boundaries),
        target_pressure_boundaries_pa=boundaries,
        gravity_m_s2=10,
        bottom_height_m=0,
        gas_columns_molecules_m2={},
        air_broadener="dry_air",
    )
    assert_allclose(atmosphere.temperature_k, [107, 108])
    assert len(atmosphere.absorbers) == 1


def local_atmosphere(path: Path, **overrides: Any) -> Any:
    options: dict[str, Any] = dict(
        temperature_variable="t",
        specific_humidity_variable="q",
        specific_humidity_units="kg kg**-1",
        coordinates=("time", "level", "latitude", "longitude"),
        pressure_units="hPa",
        valid_time=datetime(2025, 1, 1, 3, tzinfo=UTC),
        latitude_deg=5,
        longitude_deg=90,
        surface_pressure_pa=100000.0,
        pressure_boundaries_pa=np.array([100000.0, 75000, 50000]),
        gravity_m_s2=10.0,
        bottom_height_m=0.0,
        gas_columns_molecules_m2={},
        air_broadener="dry_air",
    )
    options.update(overrides)
    return read_pressure_level_atmosphere(path, **options)


def test_direct_atmosphere_replay(tmp_path: Path) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    atmosphere = local_atmosphere(path)
    assert_allclose(atmosphere.temperature_k, [107, 108])
    with netcdf_file(path, "a") as ds:
        ds.variables["level"][:] = [500, 1000]
        ds.variables["q"][:] = 1
    assert_allclose(local_atmosphere(path).temperature_k, [108, 107])
    with pytest.raises(ValueError):
        local_atmosphere(
            path, surface_pressure_pa=90000, pressure_boundaries_pa=np.array([80000.0, 50000])
        )
    with netcdf_file(path, "a") as ds:
        ds.variables["q"][:] = 2000  # scale=0.001: q>1, cannot average it away.
    with pytest.raises(ValueError, match="physical"):
        local_atmosphere(path)


@pytest.mark.parametrize(
    "options",
    [
        {"temperature_variable": "q"},
        {"specific_humidity_units": "%"},
        {"pressure_units": "mb"},
        {"surface_pressure_pa": np.nan},
        {"surface_pressure_pa": 0},
        {"pressure_boundaries_pa": np.array([100000.0])},
        {"surface_pressure_pa": 90000},
    ],
)
def test_direct_atmosphere_invalid_selection(tmp_path: Path, options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        local_atmosphere(tmp_path / "absent", **options)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("valid_range", [2, 3]),
        ("valid_min", 2),
        ("valid_max", 0),
        ("valid_range", [1]),
        ("valid_range", [3, 2]),
        ("valid_min", np.nan),
        ("_Unsigned", "true"),
    ],
)
def test_cf_validity_checks(tmp_path: Path, attribute: str, value: object) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    with netcdf_file(path, "a") as ds:
        setattr(ds.variables["q"], attribute, value)
    with pytest.raises(ValueError):
        read(path)


def test_xarray_bounded_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    decode = xr.decode_cf
    decoded_shapes = []

    def record(dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        if "field" in dataset and dataset["field"].attrs.get("units") in ("K", "kg kg**-1"):
            decoded_shapes.append(dataset["field"].shape)
        return decode(dataset, **kwargs)

    monkeypatch.setattr(xr, "decode_cf", record)
    levels, fields = read(path, vertical_interval=(500, 900))
    assert_allclose(levels, [500])
    assert_allclose(fields["q"], [0.011])
    assert decoded_shapes == [(1,)] * 16  # Two fields, eight stencil nodes, one retained level.


@pytest.mark.parametrize(
    "units,values",
    [
        ("seconds since 2025-01-01T01:00:00+01:00", [0, 21600]),
        ("minutes since 2025-01-01", [0, 360]),
        ("days since 2025-01-01", [0, 0.25]),
    ],
)
def test_xarray_cf_time_decoding(tmp_path: Path, units: str, values: list[float]) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    with netcdf_file(path, "a") as dataset:
        dataset.variables["time"].units = units
        dataset.variables["time"][:] = values
    assert_allclose(read(path)[1]["t"], [106.5, 108.5])


def test_xarray_packed_validity_before_scaling(tmp_path: Path) -> None:
    path = tmp_path / "native.nc"
    fixture_file(path)
    with netcdf_file(path, "a") as dataset:
        dataset.variables["q"].valid_range = np.array([0, 1], dtype=np.int16)
        dataset.variables["q"]._FillValue = np.int16(-99)
    assert_allclose(read(path)[1]["q"], [0.011, 0.011])
    with netcdf_file(path, "a") as dataset:
        dataset.variables["q"][0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="missing/nonfinite"):
        read(path)


def test_explicit_provider_requests_and_no_clobber(tmp_path: Path) -> None:
    calls = []
    request: dict[str, object] = {"variable": ["temperature"], "date": "2025-01-01"}

    def retrieve(product: str, payload: dict[str, object], target: str) -> None:
        calls.append((product, payload.copy()))
        payload["date"] = "mutated"
        Path(target).write_bytes(b"native")

    target = tmp_path / "era5.nc"
    assert (
        download_era5(
            product="reanalysis-era5-pressure-levels",
            request=request,
            retrieve=retrieve,
            destination=target,
        )
        == target
    )
    assert request["date"] == "2025-01-01"
    assert calls[0][0] == "reanalysis-era5-pressure-levels"
    assert len(file_sha256(target)) == 64
    with pytest.raises(FileExistsError):
        download_era5(
            product="reanalysis-era5-pressure-levels",
            request=request,
            retrieve=retrieve,
            destination=target,
        )
    assert len(calls) == 1
    for product in (
        "cams-global-greenhouse-gas-forecasts",
        "cams-global-atmospheric-composition-forecasts",
    ):
        # Native product separation, not a generic 'CAMS' label.
        download_cams(
            product=product,
            request=request,
            retrieve=retrieve,  # type: ignore
            destination=tmp_path / product,
            expected_sha256=file_sha256(target),
        )
    assert [c[0] for c in calls[1:]] == [
        "cams-global-greenhouse-gas-forecasts",
        "cams-global-atmospheric-composition-forecasts",
    ]


@pytest.mark.parametrize(
    "behavior", ["raise", "empty", "absent", "oversize", "digest", "race", "symlink"]
)
def test_sdk_failure_cleanup(tmp_path: Path, behavior: str) -> None:
    destination = tmp_path / "result"

    def retrieve(product: str, request: dict[str, object], target: str) -> None:
        path = Path(target)
        if behavior == "absent":
            return
        if behavior == "symlink":
            path.symlink_to(destination)
            return
        path.write_bytes(b"" if behavior == "empty" else b"native")
        if behavior == "raise":
            raise RuntimeError("SDK failed")
        if behavior == "race":
            destination.write_bytes(b"other writer")

    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        download_cams(
            product="cams-global-greenhouse-gas-forecasts",
            request={"date": "x"},
            retrieve=retrieve,
            destination=destination,
            max_bytes=1 if behavior == "oversize" else 20,
            expected_sha256="0" * 64 if behavior == "digest" else None,
        )
    assert list(tmp_path.iterdir()) == ([destination] if behavior == "race" else [])
    if behavior == "race":
        assert destination.read_bytes() == b"other writer"


def test_invalid_provider_selection(tmp_path: Path) -> None:
    def forbidden(*args: object) -> None:
        pytest.fail("unexpected network call")

    with pytest.raises(ValueError):
        download_cams(
            product="cams",  # type: ignore[bad-argument-type]
            request={"x": 1},
            retrieve=forbidden,  # type: ignore
            destination=tmp_path / "a",
        )
    with pytest.raises(ValueError):
        download_era5(
            product="era5",  # type: ignore[bad-argument-type]
            request={"x": 1},
            retrieve=forbidden,  # type: ignore
            destination=tmp_path / "a",
        )
    cases: tuple[tuple[dict[str, object], int, str | None], ...] = (
        ({}, 20, None),
        ({"x": 1}, 0, None),
        ({"x": 1}, 20, "bad"),
    )
    for request, size, digest in cases:
        with pytest.raises(ValueError):
            download_cams(
                product="cams-global-greenhouse-gas-forecasts",
                request=request,
                retrieve=forbidden,
                destination=tmp_path / "a",
                max_bytes=size,
                expected_sha256=digest,
            )


def test_gfs_url_cycle_lead_and_area(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def transport(url: str, destination: Path, **kwargs: object) -> Path:
        calls.append((url, kwargs))
        return destination

    monkeypatch.setattr(providers, "download_file", transport)
    target = tmp_path / "gfs.grib2"
    assert (
        download_gfs(
            cycle=CYCLE,
            lead_hours=3,
            variables=("TMP", "SPFH"),
            levels=("1000 mb", "surface"),
            area=(0, 20, -5, 10),
            destination=target,
        )
        == target
    )
    query = parse_qs(urlsplit(calls[0][0]).query)
    assert query["file"] == ["gfs.t00z.pgrb2.0p25.f003"]
    assert query["dir"] == ["/gfs.20250101/00/atmos"]
    assert query["lev_1000 mb"] == ["on"]
    assert query["var_SPFH"] == ["on"]
    assert gfs_valid_time(CYCLE, 3) == datetime(2025, 1, 1, 3, tzinfo=UTC)
    for cycle, lead in (
        (datetime(2025, 1, 1), 0),
        (CYCLE.replace(hour=1), 0),
        (CYCLE, -1),
        (CYCLE, 385),
        (CYCLE, 121),
    ):
        with pytest.raises(ValueError):
            gfs_valid_time(cycle, lead)
    for area, variables, levels in (
        ((20, 0, 0, 10), ("TMP",), ("surface",)),
        ((0, 20, 0, 10), (), ("surface",)),
        ((0, 20, 0, 10), ("TMP",), ()),
    ):
        with pytest.raises(ValueError):
            download_gfs(
                cycle=CYCLE,
                lead_hours=0,
                variables=variables,
                levels=levels,
                area=area,
                destination=target,
            )
