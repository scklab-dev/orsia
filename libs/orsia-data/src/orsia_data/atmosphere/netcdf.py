"""Offline rectilinear CF NetCDF extraction; provider conventions stay explicit.

No acquisition, units guessed from names, nearest-time fallback or vertical
remapping. Reader buffers are fresh writable float64 arrays, detached from the
file. Native file metadata and file identity remain caller-owned provenance.
"""

import re
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
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

    classic uses SciPy's NetCDF CDF-1/2 reader; hdf5 uses optional h5netcdf.
    Named dimensions, packed validity and Gregorian dates are decoded locally,
    without importing xarray, cftime or netCDF4. Dates before the Gregorian
    transition (1582-10-15) are unsupported.
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
    with ExitStack() as files:
        if backend == "classic":
            from scipy.io import netcdf_file

            dataset = files.enter_context(netcdf_file(path, "r", mmap=True))
        else:
            import h5netcdf

            dataset = files.enter_context(h5netcdf.File(path, "r"))
        # Successful sampling releases borrowed mmap variables before closure.
        # Failed reads can retain them through exception tracebacks (see guide).
        return _sample_dataset(
            dataset,
            variables,
            coordinates,
            vertical_units,
            valid_time,
            latitude_deg,
            longitude_deg,
            periodic_longitude,
            vertical_interval,
            backend,
        )


def _sample_dataset(
    dataset: Any,
    variables: dict[str, str],
    coordinates: tuple[str, ...],
    vertical_units: str | None,
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    periodic_longitude: bool,
    vertical_interval: tuple[float, float] | None,
    backend: str,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    # Only selected stencil columns are decoded; raw packed validity is
    # applied before scaling. Backends retain ownership until context exit.
    names = (*coordinates, *variables)
    if any(name not in dataset.variables for name in names):
        raise ValueError("missing requested NetCDF variable or coordinate")
    coords = [dataset.variables[name] for name in coordinates]
    if any(len(c.shape) != 1 for c in coords):
        raise ValueError("coordinate variables must be one-dimensional")
    dimensions = tuple(c.dimensions[0] for c in coords)
    if len(set(dimensions)) != len(coordinates):
        raise ValueError("coordinates must identify distinct dimensions")
    if vertical_units is None:
        time, latitude, longitude = coords
        native_levels = np.empty(0)
        vertical_mask = None
    else:
        time, level, latitude, longitude = coords
        _units(level, vertical_units, backend)
        native_levels = _values(level, backend)
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
    _units(latitude, "degrees_north", backend)
    _units(longitude, "degrees_east", backend)
    selectors = (
        linear_axis_weights(_time_offsets(time, valid_time, backend), 0.0, periodic=False),
        linear_axis_weights(_values(latitude, backend), latitude_deg, periodic=False),
        linear_axis_weights(
            _values(longitude, backend), longitude_deg, periodic=periodic_longitude
        ),
    )
    profiles = {}
    for name, units in variables.items():
        variable = dataset.variables[name]
        _units(variable, units, backend)
        if len(variable.dimensions) != len(dimensions) or set(variable.dimensions) != set(
            dimensions
        ):
            raise ValueError("field dimensions must match the explicit coordinates")
        # Read only the bracketing columns, not the complete provider cube.
        result = np.zeros(len(native_levels) if vertical_units is not None else 1, dtype=np.float64)
        for ti, tw in selectors[0]:
            for yi, yw in selectors[1]:
                for xi, xw in selectors[2]:
                    positions = (
                        (ti, yi, xi) if vertical_units is None else (ti, slice(None), yi, xi)
                    )
                    indices = dict(zip(dimensions, positions, strict=True))
                    if vertical_mask is not None:
                        indices[dimensions[1]] = np.flatnonzero(vertical_mask)
                    selection = tuple(indices[dimension] for dimension in variable.dimensions)
                    values = np.atleast_1d(_values(variable, backend, selection))
                    if values.shape != result.shape or not np.all(np.isfinite(values)):
                        raise ValueError("missing/nonfinite sampled NetCDF values")
                    result += tw * yw * xw * values
        profiles[name] = result
    return native_levels.copy(), profiles


def _attribute(variable: Any, name: str, backend: str, default: Any = None) -> Any:
    return (
        getattr(variable, name, default)
        if backend == "classic"
        else variable.attrs.get(name, default)
    )


def _values(variable: Any, backend: str, selection: Any = Ellipsis) -> NDArray[np.float64]:
    if _text(_attribute(variable, "_Unsigned", backend, "false")).lower() != "false":
        raise ValueError("unsigned packed reinterpretation is not supported")
    values = np.array(variable[selection], dtype=np.float64, copy=True)
    for name in ("_FillValue", "missing_value"):
        missing = _attribute(variable, name, backend)
        if missing is not None:
            values[np.isin(values, np.asarray(missing).ravel())] = np.nan
    valid_range = _attribute(variable, "valid_range", backend)
    if valid_range is not None:
        bounds = np.asarray(valid_range, dtype=np.float64)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] < bounds[0]:
            raise ValueError("invalid CF valid_range")
        values[(values < bounds[0]) | (values > bounds[1])] = np.nan
    for name, lower in (("valid_min", True), ("valid_max", False)):
        bound = _attribute(variable, name, backend)
        if bound is not None:
            bound = float(bound)
            if not np.isfinite(bound):
                raise ValueError("invalid CF valid bound")
            values[values < bound if lower else values > bound] = np.nan
    values *= float(_attribute(variable, "scale_factor", backend, 1.0))
    values += float(_attribute(variable, "add_offset", backend, 0.0))
    return values


def _text(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def _units(variable: Any, expected: str, backend: str) -> None:
    if _text(_attribute(variable, "units", backend, "")) != expected:
        raise ValueError("NetCDF variable units do not match the explicit native convention")


def _time_offsets(variable: Any, instant: datetime, backend: str) -> NDArray[np.float64]:
    calendar = _text(_attribute(variable, "calendar", backend, "standard"))
    units = _text(_attribute(variable, "units", backend, ""))
    unit, separator, epoch_text = units.partition(" since ")
    if calendar not in ("standard", "gregorian"):
        raise ValueError("unsupported NetCDF time calendar")
    if not separator or unit not in ("seconds", "minutes", "hours", "days"):
        raise ValueError("unsupported CF time units")
    epoch = datetime.fromisoformat(epoch_text.strip())
    epoch = epoch.replace(tzinfo=UTC) if epoch.tzinfo is None else epoch.astimezone(UTC)
    values = _values(variable, backend)
    seconds = values * {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[unit]
    transition = datetime(1582, 10, 15, tzinfo=UTC)
    if (
        not np.all(np.isfinite(seconds))
        or epoch < transition
        or np.any(seconds < (transition - epoch).total_seconds())
    ):
        raise ValueError("invalid native time coordinate or pre-Gregorian date")
    # datetime retains microseconds. Preserve the ISO epoch's remaining
    # nanosecond digits and native floating offsets without rounding time knots.
    fraction = re.search(r"[T ]\d{2}:?\d{2}:?\d{2}[.,](\d+)", epoch_text)
    remainder = 0.0 if fraction is None else int((fraction[1] + "000000000")[:9]) % 1000 / 1e9
    return np.asarray(seconds + ((epoch - instant).total_seconds() + remainder), dtype=np.float64)
