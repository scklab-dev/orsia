"""Read native orthometric terrain heights for explicit geographic grids."""

from pathlib import Path

from .geospatial import RasterBand, Window, read_raster_band
from .services.copernicus_dem import download_copernicus_dem as download_copernicus_dem


def read_copernicus_dem(path: Path, *, window: Window | None = None) -> RasterBand:
    """Read native Copernicus DSM heights [m above EGM2008], including vegetation.

    Product identity/datum is asserted by choosing this reader, not inferred from
    elevation values. Requires geographic WGS84. Does not convert to ellipsoidal
    height, bare earth or radiative terrain; absent support remains invalid.

    Args:
        path: Existing local Copernicus DSM raster path.
        window: Optional pixel window selecting a subset of native samples.

    Returns:
        Raster band with orthometric heights in metres above EGM2008, native
        WGS84 grid and validity mask.

    Raises:
        ValueError: The raster coordinate reference system is not WGS84 or
            the requested band fails local raster validation.
    """
    band = read_raster_band(path, units="m", window=window)
    if band.grid.crs.to_epsg() != 4326:
        raise ValueError("native Copernicus DEM requires horizontal WGS84")
    return band
