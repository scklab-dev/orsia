"""Offline rectilinear CF NetCDF extraction; provider conventions stay explicit.

No acquisition, units guessed from names, nearest-time fallback or vertical
remapping. Reader buffers are fresh writable float64 arrays, detached from the
file. Native file metadata and file identity remain caller-owned provenance.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .sampling import linear_axis_weights


def read_netcdf_profiles(
    path: Path,
    *,
    variables: dict[str, str],
    coordinates: tuple[str, str, str, str],
    vertical_units: str,
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    backend: Literal["classic", "hdf5"] = "classic",
    periodic_longitude: bool = False,
    vertical_interval: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Linearly sample native vertical profiles at a UTC instant and location.

    variables maps native names to REQUIRED native units (e.g. {'t': 'K',
    'q': 'kg kg**-1'}). No unit conversions or species-name inference. coordinates
    names the 1D (time, vertical, latitude, longitude) coordinate variables.
    All fields must use exactly those four dimensions, in any order. Additional
    axes (ensemble, expver, forecast cycle/lead) must be selected upstream, not
    silently squeezed. Vertical order/values/units are retained, not interpreted
    as pressure interfaces. Returned pair is (native_levels, {native_name: values}).
    Optional vertical_interval=(min,max) selects inclusive native coordinate
    support BEFORE field-validity checks (e.g. exclude below-terrain knots).

    Horizontal and time interpolation is linear; exact nodes use only that node.
    Axes may ascend or descend; singleton axes permit only exact selection. No
    extrapolation or missing-data fill. periodic_longitude=True explicitly permits
    a 360-degree seam; caller must establish full global coverage, since a regional
    subset cannot establish it from its coordinates alone. Time supports CF seconds,
    minutes, hours or days since an ISO epoch with standard/gregorian calendar.
    Forecast cycle/lead must be retained separately, not inferred from valid time.

    xarray handles named dimensions and CF decoding. classic explicitly selects
    its SciPy NetCDF CDF-1/2 engine; hdf5 selects the optional h5netcdf engine.
    Native backends run synchronously, with caller-owned serialization and
    execution policy; disabled GIL is not a backend thread-safety guarantee.
    Missing fields, units, support or decoded values raise ValueError. Source file
    must not be modified concurrently. No raw provider files are downloaded here.

    Args:
        path: Local classic NetCDF or HDF5 file.
        variables: Native field name to required native unit string.
        coordinates: Native (time, vertical, latitude, longitude) variable names.
        vertical_units: Required unit string of the native vertical coordinate.
        valid_time: UTC instant within the native time support.
        latitude_deg: Sampling latitude in degrees north.
        longitude_deg: Sampling longitude in degrees east.
        backend: Select the classic SciPy or optional h5netcdf engine.
        periodic_longitude: Enable a seam for caller-established global coverage.
        vertical_interval: Optional inclusive (minimum, maximum) native level range.

    Returns:
        Fresh writable native-level vector and native-name profile vectors, all
        float64 and ordered along the retained native vertical coordinate.

    Raises:
        ValueError: If requested fields, units, coordinates, support or sampled
            decoded values are invalid.
    """
    return _read_column(
        path,
        variables=variables,
        coordinates=coordinates,
        vertical_units=vertical_units,
        valid_time=valid_time,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        backend=backend,
        periodic_longitude=periodic_longitude,
        vertical_interval=vertical_interval,
    )


def read_netcdf_surface_fields(
    path: Path,
    *,
    variables: dict[str, str],
    coordinates: tuple[str, str, str],
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    backend: Literal["classic", "hdf5"] = "classic",
    periodic_longitude: bool = False,
) -> dict[str, float]:
    """Sample explicit (time, latitude, longitude) surface/context fields.

    Return native-name to native-unit scalar values. Units, interpolation,
    missing-data, backend and ownership semantics are read_netcdf_profiles's.
    Supports e.g. surface pressure, 10 m winds, boundary-layer height, surface
    temperature and screening fields WITHOUT interpreting them as plume winds,
    radiative surface parameters or cloud/ice optical properties.

    Args:
        path: Local classic NetCDF or HDF5 file.
        variables: Native field name to required native unit string.
        coordinates: Native (time, latitude, longitude) variable names.
        valid_time: UTC instant within the native time support.
        latitude_deg: Sampling latitude in degrees north.
        longitude_deg: Sampling longitude in degrees east.
        backend: Select the classic SciPy or optional h5netcdf engine.
        periodic_longitude: Enable a seam for caller-established global coverage.

    Returns:
        Native field names mapped to sampled finite scalar values in native units.

    Raises:
        ValueError: If requested fields, units, coordinates, support or sampled
            decoded values are invalid.
    """
    _, fields = _read_column(
        path,
        variables=variables,
        coordinates=coordinates,
        vertical_units=None,
        valid_time=valid_time,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        backend=backend,
        periodic_longitude=periodic_longitude,
    )
    return {name: float(values[0]) for name, values in fields.items()}


def _read_column(
    path: Path,
    *,
    variables: dict[str, str],
    coordinates: tuple[str, ...],
    vertical_units: str | None,
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    backend: Literal["classic", "hdf5"],
    periodic_longitude: bool,
    vertical_interval: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    if not variables or valid_time.utcoffset() != timedelta(0):
        raise ValueError("explicit variables and UTC valid time are required")
    if not np.isfinite(latitude_deg) or not -90 <= latitude_deg <= 90:
        raise ValueError("latitude must be finite and within [-90, 90]")
    if not np.isfinite(longitude_deg):
        raise ValueError("longitude must be finite")
    if vertical_interval is not None and (
        not np.all(np.isfinite(vertical_interval)) or vertical_interval[0] > vertical_interval[1]
    ):
        raise ValueError("vertical interval must be finite and increasing")
    if backend not in ("classic", "hdf5"):
        raise ValueError("unknown NetCDF backend")
    # Keep raw packed validity metadata until after the bounded selection. xarray
    # handles CF fill/packing and dates; it does not enforce CF valid_* bounds.
    with xr.open_dataset(
        path,
        engine="scipy" if backend == "classic" else "h5netcdf",
        decode_cf=False,
        chunks=None,
        cache=False,
        create_default_indexes=False,
    ) as dataset:
        names = (*coordinates, *variables)
        if any(name not in dataset.variables for name in names):
            raise ValueError("missing requested NetCDF variable or coordinate")
        coords = [dataset[name] for name in coordinates]
        if any(c.ndim != 1 for c in coords):
            raise ValueError("coordinate variables must be one-dimensional")
        dimensions = tuple(c.dims[0] for c in coords)
        if len(set(dimensions)) != len(coordinates):
            raise ValueError("coordinates must identify distinct dimensions")
        if vertical_units is None:
            time, latitude, longitude = coords
            native_levels = np.empty(0)
            vertical_mask = None
        else:
            time, level, latitude, longitude = coords
            _units(level, vertical_units)
            native_levels = _values(level)
            if (
                native_levels.ndim != 1
                or not len(native_levels)
                or not np.all(np.isfinite(native_levels))
            ):
                raise ValueError("invalid native vertical coordinate")
            vertical_mask = np.ones(len(native_levels), dtype=bool)
            if vertical_interval is not None:
                vertical_mask = (native_levels >= vertical_interval[0]) & (
                    native_levels <= vertical_interval[1]
                )
                native_levels = native_levels[vertical_mask]
                if not len(native_levels):
                    raise ValueError("vertical interval has no native coordinate support")
        _units(latitude, "degrees_north")
        _units(longitude, "degrees_east")
        selectors = (
            linear_axis_weights(_time_offsets(time, valid_time), 0.0, periodic=False),
            linear_axis_weights(_values(latitude), latitude_deg, periodic=False),
            linear_axis_weights(_values(longitude), longitude_deg, periodic=periodic_longitude),
        )
        profiles = {}
        for name, units in variables.items():
            variable = dataset[name]
            _units(variable, units)
            if len(variable.dims) != len(dimensions) or set(variable.dims) != set(dimensions):
                raise ValueError("field dimensions must match the explicit coordinates")
            # Read only the bracketing columns, not the complete provider cube.
            result = np.zeros(
                len(native_levels) if vertical_units is not None else 1, dtype=np.float64
            )
            for ti, tw in selectors[0]:
                for yi, yw in selectors[1]:
                    for xi, xw in selectors[2]:
                        positions = (
                            (ti, yi, xi) if vertical_units is None else (ti, slice(None), yi, xi)
                        )
                        indices = dict(zip(dimensions, positions, strict=True))
                        if vertical_mask is not None:
                            indices[dimensions[1]] = np.flatnonzero(vertical_mask)
                        values = np.atleast_1d(_values(variable.isel(indices)))
                        if values.shape != result.shape or not np.all(np.isfinite(values)):
                            raise ValueError("missing/nonfinite sampled NetCDF values")
                        result += tw * yw * xw * values
            profiles[name] = result
        return native_levels.copy(), profiles


def _values(variable: xr.DataArray) -> NDArray[np.float64]:
    if _text(variable.attrs.get("_Unsigned", "false")).lower() != "false":
        raise ValueError("unsigned packed reinterpretation is not supported")
    values = np.array(variable.values, dtype=np.float64, copy=True)
    valid_range = variable.attrs.get("valid_range")
    if valid_range is not None:
        bounds = np.asarray(valid_range, dtype=np.float64)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] < bounds[0]:
            raise ValueError("invalid CF valid_range")
        values[(values < bounds[0]) | (values > bounds[1])] = np.nan
    for name, lower in (("valid_min", True), ("valid_max", False)):
        bound = variable.attrs.get(name)
        if bound is not None:
            bound = float(bound)
            if not np.isfinite(bound):
                raise ValueError("invalid CF valid bound")
            values[values < bound if lower else values > bound] = np.nan
    decoded = xr.decode_cf(
        xr.Dataset({"field": xr.Variable(variable.dims, values, attrs=variable.attrs)}),
        decode_times=False,
        decode_timedelta=False,
    )
    return np.array(decoded["field"].values, dtype=np.float64, copy=True)


def _text(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def _units(variable: xr.DataArray, expected: str) -> None:
    if _text(variable.attrs.get("units", "")) != expected:
        raise ValueError("NetCDF variable units do not match the explicit native convention")


def _time_offsets(variable: xr.DataArray, instant: datetime) -> NDArray[np.float64]:
    calendar = _text(variable.attrs.get("calendar", "standard"))
    units = _text(variable.attrs.get("units", ""))
    unit, separator, _ = units.partition(" since ")
    if calendar not in ("standard", "gregorian"):
        raise ValueError("unsupported NetCDF time calendar")
    if not separator or unit not in ("seconds", "minutes", "hours", "days"):
        raise ValueError("unsupported CF time units")
    dates = xr.decode_cf(
        xr.Dataset(
            {
                "time": xr.Variable(
                    variable.dims, _values(variable), attrs={"units": units, "calendar": calendar}
                )
            }
        ),
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=False, time_unit="us"),
    )["time"].values
    if np.any(np.isnat(dates)):
        raise ValueError("invalid native time coordinate")
    target = np.datetime64(instant.replace(tzinfo=None), "us")
    return np.asarray((dates - target) / np.timedelta64(1, "s"), dtype=np.float64)
