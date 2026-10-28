"""Explicit NOMADS GFS subset transport, with no reader or archive fallback."""

from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from ..acquisition import download_file


def gfs_valid_time(cycle: datetime, lead_hours: int) -> datetime:
    """Calculate forecast validity for an explicitly selected GFS UTC cycle.

    Args:
        cycle: Exact 00/06/12/18 UTC model-cycle timestamp.
        lead_hours: Hourly forecast lead to 120h; three-hourly thereafter to 384h.

    Returns:
        UTC datetime of the selected forecast sample.

    Raises:
        ValueError: Cycle or lead lies outside the supported NOMADS selection.
    """
    if (
        cycle.utcoffset() != timedelta(0)
        or cycle.hour not in (0, 6, 12, 18)
        or cycle.minute
        or cycle.second
        or cycle.microsecond
    ):
        raise ValueError("GFS cycle must be an exact UTC 00/06/12/18 hour")
    if lead_hours < 0 or lead_hours > 384 or (lead_hours > 120 and lead_hours % 3):
        raise ValueError("unsupported GFS forecast lead")
    return cycle + timedelta(hours=lead_hours)


def download_gfs(
    *,
    cycle: datetime,
    lead_hours: int,
    variables: tuple[str, ...],
    levels: tuple[str, ...],
    area: tuple[float, float, float, float],
    destination: Path,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> Path:
    """Download native 0.25-degree GRIB2 with explicit fields, cycle and lead.

    area=(west,east,south,north), degrees, 0<=west<east<=360 without wrapping.
    Variables/levels are native selection codes. No archive fallback or decoding.
    Experimental endpoint; successful HTTP bytes still require local validation.
    download_file supplies byte bounds, timeout and atomic no-clobber publication.

    Args:
        cycle: Exact UTC model cycle at 00, 06, 12 or 18 hours.
        lead_hours: Forecast lead in hours within supported GFS cadence.
        variables: Nonempty tuple of native NOMADS variable selection codes.
        levels: Nonempty tuple of native NOMADS level selection codes.
        area: (west,east,south,north) geographic degrees without longitude wrap.
        destination: New path to receive the downloaded GRIB2 bytes.
        timeout_s: Per-network-operation timeout in seconds.
        max_bytes: Maximum response size in bytes.
        expected_sha256: Optional expected hexadecimal content digest.

    Returns:
        Destination path after bounded transfer and optional digest verification.

    Raises:
        ValueError: Cycle/lead, area, variables or levels fail selection checks.
        FileExistsError: Destination path already exists.
    """
    gfs_valid_time(cycle, lead_hours)
    west, east, south, north = area
    if not (0 <= west < east <= 360 and -90 <= south < north <= 90):
        raise ValueError("GFS area must be a non-wrapping geographic rectangle")
    if not variables or not levels or any(not item for item in (*variables, *levels)):
        raise ValueError("explicit nonempty GFS variables and levels are required")
    query = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead_hours:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "subregion": "",
        "leftlon": str(west),
        "rightlon": str(east),
        "bottomlat": str(south),
        "toplat": str(north),
    }
    query.update({f"var_{name}": "on" for name in variables})
    query.update({f"lev_{name}": "on" for name in levels})
    return download_file(
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(query),
        destination,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
    )
