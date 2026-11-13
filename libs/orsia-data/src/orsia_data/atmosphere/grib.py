"""Strict offline deterministic pressure-level GRIB2 extraction via ecCodes."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from orsia_data.atmosphere.sampling import linear_axis_weights
from orsia_data.services.nomads import gfs_valid_time


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
    A periodic seam requires explicit global-coverage policy. Bitmap missing
    samples are supported; declared in-band missing-value management is rejected.

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

    # Select and sample each native message directly, without coalescing
    # duplicates or importing a general dataset/calendar stack.
    found: set[tuple[str, int]] = set()
    result = {name: np.empty(len(levels), dtype=np.float64) for name in variables}
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
                if eccodes.codes_is_defined(handle, "missingValueManagementUsed") and get(
                    "missingValueManagementUsed"
                ):
                    raise ValueError("unsupported GRIB in-band missing-value management")
                identity = (name, int(matches[0]))
                if identity in found:
                    raise ValueError("duplicate requested GRIB field/level/cycle/validity")
                found.add(identity)
                result[name][identity[1]] = _sample_message(
                    eccodes, handle, latitude_deg, longitude_deg, periodic_longitude
                )
            finally:
                eccodes.codes_release(handle)
    if len(found) != len(variables) * len(levels):
        raise ValueError("missing requested GRIB field/level/cycle/validity")
    return result


def _sample_message(
    eccodes: Any, handle: Any, latitude_deg: float, longitude_deg: float, periodic: bool
) -> float:
    # ecCodes supplies native scan coordinates; its coordinate arrays omit
    # alternating-row reversal. Apply that reversal to the index map below.
    # Each message owns one horizontal field, without a multi-level cube.
    latitude = np.asarray(eccodes.codes_get_array(handle, "latitudes"), dtype=np.float64)
    longitude = np.asarray(eccodes.codes_get_array(handle, "longitudes"), dtype=np.float64)
    values = np.asarray(eccodes.codes_get_array(handle, "values"), dtype=np.float64)
    ni, nj = int(eccodes.codes_get(handle, "Ni")), int(eccodes.codes_get(handle, "Nj"))
    if (
        latitude.shape != (ni * nj,)
        or longitude.shape != latitude.shape
        or values.shape != latitude.shape
        or not np.all(np.isfinite(latitude))
        or not np.all(np.isfinite(longitude))
    ):
        raise ValueError("GRIB field is not a complete rectilinear grid")
    first = float(eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees"))
    negative = eccodes.codes_get(handle, "iScansNegatively")
    longitude = first - (first - longitude) % 360 if negative else first + (longitude - first) % 360
    lat_axis, lon_axis = np.unique(latitude), np.unique(longitude)
    flat = np.searchsorted(lat_axis, latitude) * ni + np.searchsorted(lon_axis, longitude)
    if len(lat_axis) != nj or len(lon_axis) != ni or len(np.unique(flat)) != ni * nj:
        raise ValueError("GRIB field is not a complete rectilinear grid")
    if eccodes.codes_get(handle, "alternativeRowScanning"):
        groups = (ni, nj) if eccodes.codes_get(handle, "jPointsAreConsecutive") else (nj, ni)
        flat = flat.reshape(groups).copy()
        flat[1::2] = flat[1::2, ::-1]
        flat = flat.ravel()
    field = np.empty((nj, ni), dtype=np.float64)
    field.flat[flat] = values
    if eccodes.codes_get(handle, "bitmapPresent"):
        bitmap = np.asarray(eccodes.codes_get_array(handle, "bitmap"))
        if bitmap.shape != values.shape:
            raise ValueError("GRIB bitmap does not match the native grid")
        field.flat[flat[bitmap == 0]] = np.nan
    ys = linear_axis_weights(lat_axis, latitude_deg, periodic=False)
    # Preserve regional support across 0 degrees without inventing a global seam.
    point = (longitude_deg - lon_axis[0]) % 360 + lon_axis[0]
    xs = linear_axis_weights(lon_axis, point, periodic=periodic)
    amount = 0.0
    for y, wy in ys:
        for x, wx in xs:
            value = field[y, x]
            if not np.isfinite(value):
                raise ValueError("missing/nonfinite sampled GRIB values")
            amount += wy * wx * value
    return float(amount)
