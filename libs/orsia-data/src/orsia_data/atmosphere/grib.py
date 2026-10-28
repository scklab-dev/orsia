"""Strict offline deterministic pressure-level GRIB2 extraction via xarray/cfgrib."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .providers import gfs_valid_time
from .sampling import linear_axis_weights


def read_gfs_profiles(
    path: Path,
    *,
    variables: dict[str, str],
    pressure_levels_pa: NDArray[np.float64],
    cycle: datetime,
    lead_hours: int,
    latitude_deg: float,
    longitude_deg: float,
    periodic_longitude: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Return bottom-to-top native GFS isobaric profiles, without layer inference.

    variables maps ecCodes shortName to exact required units (e.g. {'t': 'K',
    'q': 'kg kg**-1'}), not NOMADS acquisition selector codes. Levels are unique
    decreasing positive Pa and must correspond to integer hPa levels. All requested
    fields and levels must occur once for the exact cycle AND valid time; duplicate,
    ensemble, accumulated, missing or nonfinite samples are rejected. Only regular
    latitude/longitude GRIB2 fields are supported, with linear interpolation and
    no extrapolation. Regional longitude support follows the native scan across
    0 degrees; equivalent longitudes are accepted without enabling periodicity.
    A periodic seam requires explicit global-coverage policy.

    Requires orsia-data[atmosphere]; importing this module does not import ecCodes.
    Calls are synchronous; caller owns native-backend serialization/execution.
    Returned arrays are fresh writable float64; native handles are always released.
    No network or unit conversion except requested isobaric hPa to Pa identity.
    No archive-coverage, meteorological coherence or thread-safety qualification.

    Args:
        path: Local GRIB2 file containing the selected native isobaric fields.
        variables: ecCodes shortName to exact expected native unit string.
        pressure_levels_pa: Unique positive decreasing pressure levels in Pa,
            each matching an integer native hPa level.
        cycle: UTC initialization instant for the requested forecast.
        lead_hours: Forecast lead time in hours.
        latitude_deg: Sampling latitude in degrees north.
        longitude_deg: Sampling longitude in degrees east.
        periodic_longitude: Permit interpolation over a caller-established global seam.

    Returns:
        Fresh writable float64 profile vectors in requested bottom-to-top level
        order, keyed by native shortName.

    Raises:
        ValueError: If selection, GRIB identities or sampled native support are
            invalid, incomplete or ambiguous.
    """
    valid = gfs_valid_time(cycle, lead_hours)
    levels = pressure_levels_pa
    if (
        not variables
        or levels.ndim != 1
        or not len(levels)
        or not np.all(np.isfinite(levels))
        or np.any(levels <= 0)
        or np.any(np.diff(levels) >= 0)
        or np.any(levels % 100 != 0)
        or not np.isfinite(latitude_deg)
        or not -90 <= latitude_deg <= 90
        or not np.isfinite(longitude_deg)
    ):
        raise ValueError("explicit variables, decreasing hPa levels and finite location required")
    import eccodes  # type: ignore[import-not-found]

    # ecCodes preflight is deliberate: cfgrib coalesces duplicate messages and
    # can hide ensemble/edition distinctions when constructing its hypercube.
    found: set[tuple[str, int]] = set()
    grids: dict[tuple[str, str], list[int]] = {}
    with path.open("rb") as stream:
        while (handle := eccodes.codes_grib_new_from_file(stream)) is not None:
            try:

                def get(key: str) -> Any:
                    return eccodes.codes_get(handle, key)

                name = get("shortName")
                if name not in variables or get("typeOfLevel") != "isobaricInhPa":
                    continue
                pressure = float(get("level")) * 100
                matches = np.flatnonzero(levels == pressure)
                if not len(matches):
                    continue
                if (int(get("dataDate")), int(get("dataTime"))) != (
                    int(cycle.strftime("%Y%m%d")),
                    int(cycle.strftime("%H%M")),
                ) or (int(get("validityDate")), int(get("validityTime"))) != (
                    int(valid.strftime("%Y%m%d")),
                    int(valid.strftime("%H%M")),
                ):
                    continue
                if (
                    get("edition") != 2
                    or get("gridType") != "regular_ll"
                    or get("stepType") != "instant"
                    or get("units") != variables[name]
                    or get("numberOfPoints") != get("Ni") * get("Nj")
                    or eccodes.codes_is_defined(handle, "perturbationNumber")
                ):
                    raise ValueError(
                        "unsupported GRIB grid, edition, units or forecast representation"
                    )
                identity = (name, int(matches[0]))
                if identity in found:
                    raise ValueError("duplicate requested GRIB field/level/cycle/validity")
                found.add(identity)
                grids.setdefault((name, get("md5GridSection")), []).append(identity[1])
            finally:
                eccodes.codes_release(handle)
    if len(found) != len(variables) * len(levels):
        raise ValueError("missing requested GRIB field/level/cycle/validity")
    result = {name: np.empty(len(levels), dtype=np.float64) for name in variables}
    for (name, grid), indices in grids.items():
        # Group requested levels by native grid: ordinary profiles need one
        # dataset per variable, while different grids remain supported. No disk
        # index, global geometry cache or Dask task graph.
        with xr.open_dataset(
            path,
            engine="cfgrib",
            chunks=None,
            cache=False,
            backend_kwargs={
                "indexpath": "",
                "cache_geo_coords": False,
                "errors": "raise",
                "squeeze": False,
                "read_keys": [
                    "jPointsAreConsecutive",
                    "alternativeRowScanning",
                    "iScansNegatively",
                    "longitudeOfFirstGridPointInDegrees",
                ],
                "values_dtype": np.dtype("float64"),
                "filter_by_keys": {
                    "shortName": name,
                    "typeOfLevel": "isobaricInhPa",
                    "level": [int(levels[i] / 100) for i in indices],
                    "md5GridSection": grid,
                    "dataDate": int(cycle.strftime("%Y%m%d")),
                    "dataTime": int(cycle.strftime("%H%M")),
                    "validityDate": int(valid.strftime("%Y%m%d")),
                    "validityTime": int(valid.strftime("%H%M")),
                },
            },
        ) as dataset:
            if len(dataset.data_vars) != 1:
                raise ValueError("ambiguous requested GRIB field")
            field = next(iter(dataset.data_vars.values()))
            if set(field.dims) != {"time", "step", "isobaricInhPa", "latitude", "longitude"}:
                raise ValueError("GRIB field is not a complete rectilinear grid")
            if dataset.sizes["time"] != 1 or dataset.sizes["step"] != 1:
                raise ValueError("ambiguous requested GRIB cycle/validity")
            for index in indices:
                # Packing requires one full horizontal field, never a full
                # multi-level provider cube. Reuse it across stencil nodes.
                column = field.sel(isobaricInhPa=levels[index] / 100).isel(time=0, step=0).load()
                column = _scan_order(column)
                ys = linear_axis_weights(column.latitude.values, latitude_deg, periodic=False)
                longitude = column.longitude.values
                # Choose the equivalent coordinate on the native unwrapped axis,
                # without closing a regional grid's uncovered longitude interval.
                lower = float(np.min(longitude))
                point = (longitude_deg - lower) % 360 + lower
                xs = linear_axis_weights(longitude, point, periodic=periodic_longitude)
                amount = 0.0
                for y, wy in ys:
                    for x, wx in xs:
                        value = float(column.isel(latitude=y, longitude=x).values)
                        # cfgrib represents decoded missing samples as NaN. The
                        # preflight missingValue may also be a valid packed value.
                        if not np.isfinite(value):
                            raise ValueError("missing/nonfinite sampled GRIB values")
                        amount += wy * wx * value
                result[name][index] = amount
    return result


def _scan_order(field: xr.DataArray) -> xr.DataArray:
    # cfgrib 0.9.x assumes latitude-major values and pairs them with sorted
    # longitudes, even for consecutive-j, decreasing-i or wrapped scans.
    if field.attrs.get("GRIB_jPointsAreConsecutive", 0):
        values = field.transpose("latitude", "longitude").values.copy()
        alternate = field.attrs.get("GRIB_alternativeRowScanning", 0)
        if alternate:
            values[1::2] = values[1::2, ::-1]  # Undo cfgrib's row reversal.
        values = values.reshape(field.sizes["longitude"], field.sizes["latitude"]).T.copy()
        if alternate:
            values[:, 1::2] = values[::-1, 1::2]
        field = field.transpose("latitude", "longitude").copy(data=values)
    longitude = field.longitude.values % 360
    first = field.attrs.get("GRIB_longitudeOfFirstGridPointInDegrees", longitude[0])
    order = np.argsort(longitude)
    if field.attrs.get("GRIB_iScansNegatively", 0):
        order = order[::-1]
    start = int(np.argmin(np.abs((longitude[order] - first + 180) % 360 - 180)))
    native_longitude = np.roll(longitude[order], -start)
    # Unwrap in the declared scan direction, not by shortest angular distance:
    # even a regular grid can have spacing greater than 180 degrees. Sorting
    # modulo-360 coordinates would invent coverage across the regional gap.
    if field.attrs.get("GRIB_iScansNegatively", 0):
        native_longitude = first - (first - native_longitude) % 360
    else:
        native_longitude = first + (native_longitude - first) % 360
    if np.array_equal(native_longitude, field.longitude.values):
        return field
    return field.assign_coords(longitude=native_longitude)
