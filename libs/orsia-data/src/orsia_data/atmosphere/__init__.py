"""Explicit atmospheric acquisition and offline readers; imports never download."""

from ..acquisition import file_sha256
from ..services.ads import download_cams
from ..services.cds import download_era5
from ..services.nomads import download_gfs, gfs_valid_time
from .columns import read_pressure_level_atmosphere
from .grib import read_gfs_profiles
from .netcdf import read_netcdf_profiles, read_netcdf_surface_fields
from .rfm import read_rfm_atmosphere

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
]
