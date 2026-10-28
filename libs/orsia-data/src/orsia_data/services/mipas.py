"""Explicit acquisition of Oxford's MIPAS 2001 reference atmosphere files."""

from pathlib import Path
from typing import Literal

from ..acquisition import download_file

_FILES = {
    "midlatitude_day": "day.atm",
    "midlatitude_night": "ngt.atm",
    "polar_winter": "win.atm",
    "polar_summer": "sum.atm",
    "equatorial_day": "equ.atm",
}


def download_mipas_profile(
    *,
    edition: Literal["2001"] = "2001",
    profile: Literal[
        "midlatitude_day", "midlatitude_night", "polar_winter", "polar_summer", "equatorial_day"
    ] = "equatorial_day",
    destination: Path,
    expected_sha256: str | None = None,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
) -> Path:
    """Download the selected MIPAS 2001 reference profile to an explicit path.

    Select one of Oxford's five 121-level, 0--120 km molecular atmospheres.
    Edition denotes the historical collection; source URLs are mutable. Retain
    native bytes and an independently recorded digest for reproducible replay.
    Original concentrations and file comments are transferred verbatim.

    This explicit network operation returns destination. Use the separate local
    read_rfm_atmosphere for decoding. Bounds, lowercase SHA-256 verification,
    network timeout, atomic publication and exceptions follow download_file.
    Existing files/symlinks raise FileExistsError and remain intact; failed
    transfers clean their temporary files. Parent directories may remain.
    Unsupported selections raise ValueError before filesystem/network access.

    Args:
        edition: Historical MIPAS reference collection, currently "2001".
        profile: Selected latitude/season/daytime profile within that edition.
        destination: New local path for the original atmosphere-file bytes.
        expected_sha256: Optional expected lowercase hexadecimal digest.
        timeout_s: Network-operation timeout in seconds.
        max_bytes: Maximum transfer size in bytes.

    Returns:
        Destination path after successful bounded transfer and digest validation.

    Raises:
        ValueError: Collection/profile selection or transfer bounds are invalid.
        FileExistsError: Destination path already exists.
    """
    if edition != "2001":
        raise ValueError("MIPAS reference acquisition supports edition '2001'")
    if profile not in _FILES:
        raise ValueError("unsupported MIPAS 2001 reference profile")
    return download_file(
        f"https://eodg.atm.ox.ac.uk/RFM/atm/{_FILES[profile]}",
        destination,
        expected_sha256=expected_sha256,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
    )
