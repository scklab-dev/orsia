"""Explicit LISIRD HSRS acquisition; no numerical adaptation or implicit cache."""

import shutil
from pathlib import Path
from typing import Literal

from orsia_radiative import SolarIrradianceSpectrum

from ..acquisition import download_file
from ..solar.irradiance import read_solar_spectrum_csv


def fetch_solar_spectrum(
    *,
    product: Literal["tsis1_hsrs", "tsis1_hsrs_p025nm"],
    edition: Literal["current"],
    directory: Path,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> SolarIrradianceSpectrum:
    """Fetch a native LISIRD CSV and return its physical wavelength tabulation.

    Experimental provider adapter: the live LASP schema/edition is not qualified.
    Only explicit ``edition="current"`` is supported; this is a mutable service,
    not a pinned release or a promise that the SCKLab label 'v2' is selectable.
    Product is required, with no implicit smoothing/resolution choice. The CSV
    must satisfy ``read_solar_spectrum_csv``; unsupported schemas fail closed.

    Create a new, exclusively owned directory containing ``<product>.csv``.
    Existing directories (even empty ones) fail without network access. On any
    transport/decode failure remove the new directory. Do not modify it during
    this call. Reading the retained CSV later is an explicit offline operation.
    Bounds/digest have the ``download_file`` meaning. No resampling, convolution,
    distance correction, uncertainty interpretation or provider metadata record.

    Args:
        product: Requested native HSRS CSV product identifier.
        edition: Explicit ``"current"`` mutable service selection.
        directory: New directory to hold the fetched CSV.
        timeout_s: Transport timeout passed to the downloader, in seconds.
        max_bytes: Maximum accepted response size in bytes.
        expected_sha256: Optional hexadecimal digest for transfer verification.

    Returns:
        Solar irradiance spectrum decoded from the retained native CSV.

    Raises:
        FileExistsError: If the destination directory already exists.
        ValueError: If selection, transfer verification or CSV decoding fails.
    """
    if product not in ("tsis1_hsrs", "tsis1_hsrs_p025nm"):
        raise ValueError("unsupported LISIRD solar product")
    if edition != "current":
        raise ValueError("LISIRD adapter supports only the explicit current service")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        path = directory / f"{product}.csv"
        download_file(
            f"https://lasp.colorado.edu/lisird/latis/dap/{product}.csv",
            path,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )
        return read_solar_spectrum_csv(path)
    except BaseException:
        shutil.rmtree(directory)
        raise
