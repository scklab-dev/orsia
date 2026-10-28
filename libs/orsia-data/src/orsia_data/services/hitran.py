"""HITRAN acquisition through HAPI; execution policy belongs to the caller."""

import shutil
from pathlib import Path
from typing import Literal

from orsia_radiative import LineDatabase

from ..spectroscopy.lines import read_hitran_lines


def fetch_hitran_lines(
    *,
    molecule_id: int,
    isotopologue_id: int,
    interval_cm1: tuple[float, float],
    edition: Literal["current"],
    directory: Path,
) -> LineDatabase:
    """Use HAPI fetch(M, I, min, max), retaining native lines.data/lines.header.

    One HITRAN molecule/local isotopologue per call; combine multiple returned
    databases explicitly. Interval is inclusive line-centre coverage [cm-1],
    not later wing coverage. Intensities retain HITRAN abundance weighting.
    Only explicit ``edition="current"`` is supported: HAPI fetch has no edition
    selector, so this does not claim to pin HITRAN2020/2024 or an archival release.

    Requires the optional ``orsia-data[hitran]`` extra. Imports never load HAPI.
    HAPI is imported on acquisition and runs synchronously in the caller's
    process/thread. Its db_begin/fetch operations mutate shared module-wide
    database/cache state, which is not restored on success or failure. Separate
    directories do not make concurrent HAPI calls safe, including calls outside
    Orsia. The caller owns serialization, isolation and cancellation policy;
    Orsia supplies no lock, executor or total deadline. HAPI transfer has no byte
    bound here. Missing HAPI raises ImportError; upstream exceptions propagate.
    A successful empty table is valid. Disabled GIL is not thread-safety evidence.

    Directory must not exist and is exclusively owned for this call. Failures
    remove it, including partial HAPI files. Do not modify it concurrently. The
    local reader remains independently usable without HAPI or network access.
    No partition evaluation, line materialisation, registry or manifest is added.

    Args:
        molecule_id: Positive HITRAN molecule identifier passed to HAPI.
        isotopologue_id: Local isotopologue identifier passed to HAPI.
        interval_cm1: Inclusive requested line-centre range in cm^-1.
        edition: Explicit ``"current"`` service selection.
        directory: New directory for the native HAPI header and data files.

    Returns:
        Sorted line database for the selected identity and declared coverage.

    Raises:
        FileExistsError: If the destination directory already exists.
        ImportError: If optional HAPI support is unavailable after directory creation.
        ValueError: If the selection, native output or parsed records are invalid.
    """
    if edition != "current":
        raise ValueError("HAPI fetch supports only the explicit current service")
    # Validate physical support before acquiring resources or contacting HAPI.
    LineDatabase((), molecule_id, isotopologue_id, interval_cm1)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        import hapi  # type: ignore[import-not-found]

        hapi.db_begin(str(directory.resolve()))
        hapi.fetch("lines", molecule_id, isotopologue_id, *interval_cm1)
        # A data file alone is not evidence of a successful HAPI fetch.
        if not (directory / "lines.header").is_file():
            raise ValueError("HAPI acquisition did not produce its native header")
        path = directory / "lines.data"
        database = read_hitran_lines(
            path,
            molecule_id=molecule_id,
            isotopologue_id=isotopologue_id,
            interval_cm1=interval_cm1,
        )
        with path.open(encoding="ascii") as stream:
            row_count = sum(bool(row.strip()) for row in stream)
        if row_count != len(database.lines):
            raise ValueError("HAPI returned lines outside the requested identity or interval")
        return database
    except BaseException:
        shutil.rmtree(directory)
        raise
