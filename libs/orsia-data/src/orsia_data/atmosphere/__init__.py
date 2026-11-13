"""Explicit atmospheric acquisition and offline readers; imports never download."""

from orsia_data.acquisition import file_sha256
from orsia_data.atmosphere.columns import read_pressure_level_atmosphere
from orsia_data.atmosphere.grib import read_gfs_profiles
from orsia_data.atmosphere.netcdf import (
    read_netcdf_profiles,
    read_netcdf_surface_fields,
    read_netcdf_surface_grid,
)
from orsia_data.atmosphere.rfm import read_rfm_atmosphere
from orsia_data.services.ads import download_cams
from orsia_data.services.cds import download_era5
from orsia_data.services.nomads import download_gfs, gfs_valid_time

__all__ = [
    "download_cams",
    "download_era5",
    "download_gfs",
    "file_sha256",
    "gfs_valid_time",
    "read_gfs_profiles",
    "read_rfm_atmosphere",
    "read_pressure_level_atmosphere",
    "read_netcdf_profiles",
    "read_netcdf_surface_fields",
    "read_netcdf_surface_grid",
]
