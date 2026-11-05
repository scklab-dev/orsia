"""Explicit acquisition and local scientific readers; imports never download."""

from .acquisition import download_bearer_file, download_file, retrieve_file
from .aerosols import read_cams_aod
from .atmosphere import (
    download_cams,
    download_era5,
    download_gfs,
    file_sha256,
    gfs_valid_time,
    read_gfs_profiles,
    read_netcdf_profiles,
    read_netcdf_surface_fields,
    read_pressure_level_atmosphere,
    read_rfm_atmosphere,
)
from .geospatial import (
    RasterBand,
    RasterGrid,
    crop_raster,
    iter_raster_windows,
    mosaic_rasters,
    read_raster_band,
    read_raster_grid,
    reproject_raster,
    sample_raster,
    window_from_bounds,
    window_grid,
)
from .orbits import parse_orbital_elements, read_orbital_elements, select_orbital_elements
from .services.cdse.odata import download_cdse, search_sentinel2_l2a
from .services.copernicus_dem import download_copernicus_dem
from .services.earthdata.cmr import download_earthdata, earthdata_links, search_earthdata
from .services.mipas import download_mipas_profile
from .solar import fetch_solar_spectrum, read_solar_spectrum_csv
from .spectroscopy import (
    acquire_hitran_references,
    download_mt_ckd_water,
    fetch_hitran_lines,
    fetch_mt_ckd_water,
    read_aer_lines,
    read_cross_section_table,
    read_hitran_lines,
    read_mt_ckd_water,
    read_partition_table,
)
from .spectroscopy.collections import (
    SpectroscopyCollection,
    import_spectroscopy_collection,
    read_line_spectroscopy_collection,
    read_spectroscopy_collection,
)
from .spectroscopy.measurements import read_cross_section_measurements
from .surface import read_aster_ged, read_ecostress_l2t, read_sentinel2_l2a, read_sentinel2_metadata
from .terrain import read_copernicus_dem

__all__ = [
    "SpectroscopyCollection",
    "import_spectroscopy_collection",
    "read_spectroscopy_collection",
    "read_line_spectroscopy_collection",
    "read_cross_section_measurements",
    "acquire_hitran_references",
    "parse_orbital_elements",
    "read_orbital_elements",
    "select_orbital_elements",
    "iter_raster_windows",
    "read_raster_grid",
    "window_from_bounds",
    "download_bearer_file",
    "RasterBand",
    "RasterGrid",
    "crop_raster",
    "mosaic_rasters",
    "read_raster_band",
    "reproject_raster",
    "sample_raster",
    "window_grid",
    "download_cdse",
    "download_earthdata",
    "earthdata_links",
    "retrieve_file",
    "search_earthdata",
    "search_sentinel2_l2a",
    "read_aster_ged",
    "read_ecostress_l2t",
    "read_sentinel2_l2a",
    "read_sentinel2_metadata",
    "download_copernicus_dem",
    "read_copernicus_dem",
    "read_cams_aod",
    "download_cams",
    "download_era5",
    "download_gfs",
    "file_sha256",
    "gfs_valid_time",
    "read_gfs_profiles",
    "read_netcdf_profiles",
    "read_netcdf_surface_fields",
    "read_pressure_level_atmosphere",
    "read_rfm_atmosphere",
    "download_mipas_profile",
    "download_mt_ckd_water",
    "fetch_mt_ckd_water",
    "fetch_solar_spectrum",
    "fetch_hitran_lines",
    "download_file",
    "read_solar_spectrum_csv",
    "read_aer_lines",
    "read_hitran_lines",
    "read_partition_table",
    "read_mt_ckd_water",
    "read_cross_section_table",
]
