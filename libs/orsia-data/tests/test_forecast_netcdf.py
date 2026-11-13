"""Native one-cycle layouts and bounded surface-grid decoding, offline only."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from orsia_data.atmosphere.netcdf import (
    read_netcdf_profiles,
    read_netcdf_surface_fields,
    read_netcdf_surface_grid,
)
from scipy.io import netcdf_file

CYCLE = "forecast_reference_time"
PERIOD = "forecast_period"
VALID = datetime(2025, 1, 1, 1, tzinfo=UTC)


def forecast_file(path: Path, *, reordered: bool, singleton: bool = False, cycles: int = 1) -> None:
    """Write packed fields with independently known space/time/level contributions."""
    n = 1 if singleton else 2
    with netcdf_file(path, "w") as ds:
        for name, size in {
            CYCLE: cycles,
            PERIOD: 3,
            "level": n,
            "latitude": n,
            "longitude": n,
        }.items():
            ds.createDimension(name, size)
        for name, values, units in (
            ("level", [1000, 500][:n], "hPa"),
            ("latitude", [10, 0][:n], "degrees_north"),
            ("longitude", [20, 30][:n], "degrees_east"),
        ):
            v = ds.createVariable(name, "f8", (name,))
            v[:] = values
            v.units = units
        dims = (PERIOD, CYCLE) if reordered else (CYCLE, PERIOD)
        times = np.tile([[0.0, 2.0, 4.0]], (cycles, 1))
        time = ds.createVariable("valid_time", "f8", dims)
        time[:] = times.T if reordered else times
        time.units = "hours since 2025-01-01"
        time.calendar = "proleptic_gregorian"
        for name, canonical, units in (
            ("aod", (CYCLE, PERIOD, "latitude", "longitude"), "~"),
            ("t", (CYCLE, PERIOD, "level", "latitude", "longitude"), "K"),
        ):
            shape = tuple(ds.dimensions[d] for d in canonical)
            index = np.indices(shape)
            raw = np.ones(shape, dtype=np.int64)
            for i in range(len(shape)):
                raw += (10**i) * index[i]
            dims = tuple(reversed(canonical)) if reordered else canonical
            raw = raw.transpose(tuple(canonical.index(d) for d in dims))
            v = ds.createVariable(name, "i2", dims)
            v[:] = raw
            v.units = units
            v.scale_factor = 0.01
            v.add_offset = 0.1
            v.valid_range = [0, 30000]


def grid(path: Path, **kwargs: object) -> tuple:
    return read_netcdf_surface_grid(
        path,
        variables={"aod": ("1", "~", "dimensionless")},
        coordinates=("valid_time", "latitude", "longitude"),
        valid_time=VALID,
        backend="classic",
        forecast_dimensions=(CYCLE, PERIOD),
        **kwargs,
    )


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("singleton", [False, True])
def test_forecast_profiles_grid_and_point_parity(
    tmp_path: Path, reordered: bool, singleton: bool
) -> None:
    path = tmp_path / "native.nc"
    forecast_file(path, reordered=reordered, singleton=singleton)
    before = path.read_bytes()
    lon, lat, fields = grid(path)
    for iy, y in enumerate(lat):
        for ix, x in enumerate(lon):
            point = read_netcdf_surface_fields(
                path,
                variables={"aod": "~"},
                coordinates=("valid_time", "latitude", "longitude"),
                valid_time=VALID,
                latitude_deg=float(y),
                longitude_deg=float(x),
                backend="classic",
                forecast_dimensions=(CYCLE, PERIOD),
            )
            expected = (1 + 5 + 100 * iy + 1000 * ix) * 0.01 + 0.1
            assert point["aod"] == pytest.approx(expected)
            assert fields["aod"][iy, ix] == point["aod"]
            levels, profiles = read_netcdf_profiles(
                path,
                variables={"t": "K"},
                coordinates=("valid_time", "level", "latitude", "longitude"),
                vertical_units="hPa",
                valid_time=VALID,
                latitude_deg=float(y),
                longitude_deg=float(x),
                backend="classic",
                forecast_dimensions=(CYCLE, PERIOD),
            )
            expected = (1 + 5 + 100 * np.arange(levels.size) + 1000 * iy + 10000 * ix) * 0.01 + 0.1
            np.testing.assert_allclose(profiles["t"], expected)
    assert path.read_bytes() == before
    for values in (lon, lat, fields["aod"]):
        assert values.flags.owndata and values.flags.writeable
    fields["aod"][:] = -1
    assert (grid(path)[2]["aod"] >= 0).all()


@pytest.mark.parametrize("fault", ["time_dims", "cycles", "extra", "duplicate", "missing", "units"])
def test_forecast_grid_rejects_invalid_layout_and_data(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "native.nc"
    forecast_file(path, reordered=False, cycles=2 if fault == "cycles" else 1)
    with netcdf_file(path, "a") as ds:
        if fault == "time_dims":
            ds.createDimension("other", 2)
            # Select unrelated coordinate names rather than infer forecast axes.
            v = ds.createVariable("wrong_time", "f8", ("other", PERIOD))
            v[:] = 0
            v.units = "hours since 2025-01-01"
        elif fault == "extra":
            ds.createDimension("extra", 1)
            v = ds.createVariable("bad", "f8", ("extra", PERIOD, "latitude", "longitude"))
            v[:] = 1
            v.units = "~"
        elif fault == "duplicate":
            ds.variables["latitude"][:] = [10, 10]
        elif fault == "missing":
            ds.variables["aod"][0, 0, -1, -1] = -99
        elif fault == "units":
            ds.variables["aod"].units = "m"
    coordinates = ("wrong_time" if fault == "time_dims" else "valid_time", "latitude", "longitude")
    variables: dict[str, str | tuple[str, ...]] = {"bad" if fault == "extra" else "aod": "~"}
    with pytest.raises(ValueError):
        read_netcdf_surface_grid(
            path,
            variables=variables,
            coordinates=coordinates,
            valid_time=VALID,
            backend="classic",
            forecast_dimensions=(CYCLE, PERIOD),
        )


def test_grid_one_open_and_selected_planes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy.io
    from scipy.io import netcdf_variable

    path = tmp_path / "native.nc"
    forecast_file(path, reordered=True)
    opens = []
    reads = []
    getitem = netcdf_variable.__getitem__

    def opening(*args: object, **kwargs: object) -> object:
        opens.append(args[0])
        return netcdf_file(*args, **kwargs)

    def reading(self: netcdf_variable, key: object) -> object:
        if self.dimensions == ("longitude", "latitude", PERIOD, CYCLE):
            reads.append(key)
        return getitem(self, key)

    monkeypatch.setattr(scipy.io, "netcdf_file", opening)
    monkeypatch.setattr(netcdf_variable, "__getitem__", reading)
    grid(path)
    assert len(opens) == 1
    assert reads == [(slice(None), slice(None), 0, 0), (slice(None), slice(None), 1, 0)]


@pytest.mark.parametrize("forecast", [None, (CYCLE, PERIOD)])
@pytest.mark.parametrize("cycle_kind", ["absent", "scalar", "singleton", "bad_coordinate"])
def test_one_dimensional_surface_grid(
    tmp_path: Path, forecast: tuple[str, str] | None, cycle_kind: str
) -> None:
    path = tmp_path / "surface.nc"
    with netcdf_file(path, "w") as ds:
        for name, values, units in (
            ("time", [0, 2], "hours since 2025-01-01"),
            ("latitude", [10], "degrees_north"),
            ("longitude", [30, 20], "degrees_east"),
        ):
            ds.createDimension(name, len(values))
            v = ds.createVariable(name, "f8", (name,))
            v[:] = values
            v.units = units
        if cycle_kind != "absent":
            dims = ()
            if cycle_kind in ("singleton", "bad_coordinate"):
                ds.createDimension(CYCLE, 1)
                dims = (CYCLE,) if cycle_kind == "singleton" else ("longitude",)
            ds.createVariable(CYCLE, "f8", dims)[...] = 0
        v = ds.createVariable("aod", "f8", ("longitude", "time", "latitude"))
        v[:] = np.array([[[1], [3]], [[2], [4]]])
        v.units = "1"
    options: dict[str, Any] = dict(
        variables={"aod": "1"},
        coordinates=("time", "latitude", "longitude"),
        valid_time=VALID,
        forecast_dimensions=forecast,
    )
    if forecast is not None and cycle_kind == "bad_coordinate":
        with pytest.raises(ValueError, match="cycle coordinate"):
            read_netcdf_surface_grid(path, **options)
    else:
        lon, lat, fields = read_netcdf_surface_grid(path, **options)
        np.testing.assert_array_equal(lon, [30, 20])
        np.testing.assert_array_equal(lat, [10])
        np.testing.assert_array_equal(fields["aod"], [[2, 3]])


@pytest.mark.parametrize(
    "fault",
    [
        "strict",
        "same_dims",
        "empty_variables",
        "utc",
        "backend",
        "missing_field",
        "empty_units",
        "nonfinite_axis",
        "spatial_2d",
        "cycle_axis",
        "time_3d",
    ],
)
def test_grid_contract_errors(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "native.nc"
    forecast_file(path, reordered=False)
    options: dict[str, Any] = dict(
        variables={"aod": "~"},
        coordinates=("valid_time", "latitude", "longitude"),
        valid_time=VALID,
        forecast_dimensions=(CYCLE, PERIOD),
    )
    if fault == "strict":
        options["forecast_dimensions"] = None
    elif fault == "same_dims":
        options["forecast_dimensions"] = (CYCLE, CYCLE)
    elif fault == "empty_variables":
        options["variables"] = {}
    elif fault == "utc":
        options["valid_time"] = datetime(2025, 1, 1)
    elif fault == "backend":
        options["backend"] = "invalid"
    elif fault == "missing_field":
        options["variables"] = {"missing": "~"}
    elif fault == "empty_units":
        options["variables"] = {"aod": ()}
    else:
        with netcdf_file(path, "a") as ds:
            if fault == "nonfinite_axis":
                ds.variables["longitude"][:] = [20, np.nan]
            elif fault == "spatial_2d":
                ds.createVariable("bad_lat", "f8", (CYCLE, "latitude"))[:] = 10
                options["coordinates"] = ("valid_time", "bad_lat", "longitude")
            elif fault == "cycle_axis":
                ds.createVariable("bad_time", "f8", (CYCLE,))[:] = 0
                options["coordinates"] = ("bad_time", "latitude", "longitude")
            elif fault == "time_3d":
                ds.createVariable("bad_time", "f8", (CYCLE, PERIOD, "latitude"))[:] = 0
                options["coordinates"] = ("bad_time", "latitude", "longitude")
    with pytest.raises(ValueError):
        read_netcdf_surface_grid(path, **options)
