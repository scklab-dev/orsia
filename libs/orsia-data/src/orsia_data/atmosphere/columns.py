"""Offline pressure-level meteorology to the existing radiative Atmosphere."""

from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import (
    Atmosphere,
    hydrostatic_atmosphere,
    pressure_level_layer_means,
)
from orsia_radiative.atmosphere import AirBroadener

from .netcdf import read_netcdf_profiles


def read_pressure_level_atmosphere(
    path: Path,
    *,
    temperature_variable: str,
    specific_humidity_variable: str,
    specific_humidity_units: str,
    coordinates: tuple[str, str, str, str],
    pressure_units: Literal["Pa", "hPa"],
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    surface_pressure_pa: float,
    pressure_boundaries_pa: NDArray[np.float64],
    gravity_m_s2: float,
    bottom_height_m: float,
    gas_columns_molecules_m2: Mapping[str, NDArray[np.float64]],
    air_broadener: AirBroadener,
    backend: Literal["classic", "hdf5"] = "classic",
    periodic_longitude: bool = False,
) -> Atmosphere:
    """Decode pressure-level NetCDF meteorology and materialise selected layers.

    Temperature must be native K; q must be explicitly declared dimensionless
    kg water/kg total moist air (accepted native units 'kg kg-1', 'kg kg**-1',
    or '1'). This is not RH, g/kg or dry-air mixing-ratio conversion. Native
    point pressure coordinates must use Pa/hPa as selected, ascend or descend,
    and uniquely bracket the caller's bottom-to-top pressure interfaces [Pa].

    Remove native pressure knots below surface_pressure_pa, which the caller
    must obtain for this location/time. Target bottom cannot lie below terrain.
    Remaining knots must bracket ALL interfaces; there is no inferred surface
    knot, extrapolation or top/interface construction. If pressure-level data
    do not bracket the surface, obtain additional model/surface information or
    choose an explicitly partial column. Gas columns are caller-materialised
    molecules/m2 on the TARGET layers, not native point fractions: combining
    differently gridded composition requires explicit conversion/remapping first.

    Reconstruction is piecewise linear in pressure into layer means, followed
    by the public constant-gravity hydrostatic calculation; see those numerical
    functions for height/molar-mass approximations. No acquisition, reference gas,
    covariance or aerosol optics. Return owned Atmosphere copies without
    retaining caller buffers or provider identity. Native reader resource and
    interpolation semantics are read_netcdf_profiles's.

    Args:
        path: Local pressure-level meteorology NetCDF file.
        temperature_variable: Native temperature field name, with units K.
        specific_humidity_variable: Native specific humidity field name.
        specific_humidity_units: Exact declared native mass-fraction unit string.
        coordinates: Native (time, pressure, latitude, longitude) variable names.
        pressure_units: Native pressure coordinate units, Pa or hPa.
        valid_time: UTC sampling instant.
        latitude_deg: Sampling latitude in degrees north.
        longitude_deg: Sampling longitude in degrees east.
        surface_pressure_pa: Surface pressure in Pa at the selected location and time.
        pressure_boundaries_pa: Target bottom-to-top layer interfaces in Pa.
        gravity_m_s2: Constant gravitational acceleration in m/s^2.
        bottom_height_m: Altitude of the target bottom interface in metres.
        gas_columns_molecules_m2: Gas columns on target layers, molecules/m^2.
        air_broadener: Broadening convention passed to hydrostatic materialization.
        backend: Classic NetCDF or HDF5 engine.
        periodic_longitude: Permit a seam for established global longitude coverage.

    Returns:
        Atmosphere with owned target-layer buffers derived from native pressure
        meteorology and caller-supplied gas columns.

    Raises:
        ValueError: If native selection or target physical pressure, humidity or
            temperature support is invalid.
    """
    if temperature_variable == specific_humidity_variable:
        raise ValueError("temperature and specific humidity must be distinct native fields")
    if specific_humidity_units not in ("kg kg-1", "kg kg**-1", "1"):
        raise ValueError("specific humidity must declare dimensionless SI mass fraction units")
    if pressure_units not in ("Pa", "hPa"):
        raise ValueError("pressure-level coordinates must explicitly use Pa or hPa")
    if not isfinite(surface_pressure_pa) or surface_pressure_pa <= 0:
        raise ValueError("surface pressure must be finite and positive")
    if pressure_boundaries_pa.ndim != 1 or len(pressure_boundaries_pa) < 2:
        raise ValueError("at least two target pressure interfaces are required")
    if pressure_boundaries_pa[0] > surface_pressure_pa:
        raise ValueError("target bottom lies below terrain")
    native_pressure, fields = read_netcdf_profiles(
        path,
        variables={temperature_variable: "K", specific_humidity_variable: specific_humidity_units},
        coordinates=coordinates,
        vertical_units=pressure_units,
        valid_time=valid_time,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        backend=backend,
        periodic_longitude=periodic_longitude,
        vertical_interval=(0.0, surface_pressure_pa / (100 if pressure_units == "hPa" else 1)),
    )
    native_pressure *= 100 if pressure_units == "hPa" else 1
    temperature, humidity = fields[temperature_variable], fields[specific_humidity_variable]
    if np.all(np.diff(native_pressure) > 0):
        native_pressure, temperature, humidity = (
            native_pressure[::-1],
            temperature[::-1],
            humidity[::-1],
        )
    if np.any(np.diff(native_pressure) >= 0):
        raise ValueError("native pressure levels must be strictly monotone")
    above_terrain = native_pressure <= surface_pressure_pa
    native_pressure = native_pressure[above_terrain]
    if (
        np.any(temperature[above_terrain] <= 0)
        or np.any(humidity[above_terrain] < 0)
        or np.any(humidity[above_terrain] >= 1)
    ):
        raise ValueError("native temperature/humidity is outside physical support")
    return hydrostatic_atmosphere(
        pressure_boundaries_pa,
        pressure_level_layer_means(
            native_pressure, temperature[above_terrain], pressure_boundaries_pa
        ),
        pressure_level_layer_means(
            native_pressure, humidity[above_terrain], pressure_boundaries_pa
        ),
        target_pressure_boundaries_pa=pressure_boundaries_pa,
        gravity_m_s2=gravity_m_s2,
        bottom_height_m=bottom_height_m,
        gas_columns_molecules_m2=gas_columns_molecules_m2,
        air_broadener=air_broadener,
    )
