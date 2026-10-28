"""Explicit CAMS total-column AOD extraction, not aerosol optical reconstruction."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from .atmosphere.netcdf import read_netcdf_surface_fields


def read_cams_aod(
    path: Path,
    *,
    wavelengths_nm: tuple[int, ...],
    coordinates: tuple[str, str, str],
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    backend: Literal["classic", "hdf5"] = "classic",
    periodic_longitude: bool = False,
) -> dict[int, float]:
    """Sample CAMS composition total aerosol optical depths at native wavelengths.

    Supports NetCDF aod469/aod550/aod670/aod865/aod1240, units '1', on explicit
    time/latitude/longitude axes. Returns wavelength [nm] -> dimensionless AOD.
    Acquisition uses atmosphere.download_cams with the composition product.
    This is neither CAMS GHG data nor aerosol mixing ratio/SSA/phase function.
    No spectral extrapolation, vertical distribution or mass-to-optics conversion.
    CF packing, finite support and linear space/time sampling follow the common
    NetCDF reader; extra forecast-cycle/lead axes must be selected upstream.

    Args:
        path: Local CAMS composition NetCDF file.
        wavelengths_nm: Distinct selected native AOD wavelengths in nanometres.
        coordinates: Native (time, latitude, longitude) coordinate variable names.
        valid_time: UTC sampling instant.
        latitude_deg: Sampling latitude in degrees north.
        longitude_deg: Sampling longitude in degrees east.
        backend: Classic NetCDF or HDF5 reader engine.
        periodic_longitude: Permit a global longitude seam when coverage is established.

    Returns:
        Sampled nonnegative total-column optical depth by native wavelength in nm.

    Raises:
        ValueError: If wavelength selection, sampled AOD, units or sampling support
            are invalid.
    """
    if (
        not wavelengths_nm
        or len(set(wavelengths_nm)) != len(wavelengths_nm)
        or any(w not in (469, 550, 670, 865, 1240) for w in wavelengths_nm)
    ):
        raise ValueError("select distinct supported native CAMS AOD wavelengths")
    fields = read_netcdf_surface_fields(
        path,
        variables={f"aod{w}": "1" for w in wavelengths_nm},
        coordinates=coordinates,
        valid_time=valid_time,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        backend=backend,
        periodic_longitude=periodic_longitude,
    )
    if any(value < 0 for value in fields.values()):
        raise ValueError("negative sampled CAMS AOD")
    return {w: fields[f"aod{w}"] for w in wavelengths_nm}
