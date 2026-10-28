"""Read a local native solar irradiance table into a physical spectrum."""

from __future__ import annotations

import csv
from math import isfinite
from pathlib import Path

import numpy as np
from orsia_radiative import SolarIrradianceSpectrum

_WAVELENGTH = "wavelength (nm)"
_IRRADIANCE = "irradiance (W/m^2/nm)"
_UNCERTAINTY = "uncertainty (W/m^2/nm)"
_BANDWIDTH = "bandwidth (nm)"


def read_solar_spectrum_csv(path: Path) -> SolarIrradianceSpectrum:
    """Read a local CSV; never download or infer missing values.

    The supported schemas are exactly the native two-column table, that table
    plus uncertainty, or that table plus uncertainty and bandwidth. Comment
    lines beginning with ``#`` and empty rows are ignored.

    Args:
        path: Local UTF-8 CSV with wavelength in nm and irradiance in W/m2/nm,
            optionally followed by uncertainty and bandwidth columns.

    Returns:
        Spectrum of ordered physical wavelength and irradiance samples with
        optional uncertainty in W/m2/nm and bandwidth in nm.

    Raises:
        ValueError: Table schema, numerical samples or spectral ordering is invalid.
        OSError: The local CSV cannot be opened or read.
    """
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [
                row
                for row in csv.reader(stream, strict=True)
                if row and any(cell.strip() for cell in row) and not row[0].lstrip().startswith("#")
            ]
    except (UnicodeError, csv.Error) as error:
        raise ValueError("could not read solar irradiance CSV") from error
    if not rows:
        raise ValueError("solar irradiance CSV is empty")
    header = tuple(cell.strip() for cell in rows[0])
    allowed = {
        (_WAVELENGTH, _IRRADIANCE),
        (_WAVELENGTH, _IRRADIANCE, _UNCERTAINTY),
        (_WAVELENGTH, _IRRADIANCE, _UNCERTAINTY, _BANDWIDTH),
    }
    if header not in allowed:
        raise ValueError("unsupported solar irradiance CSV columns")
    if len(rows) == 1:
        raise ValueError("solar irradiance CSV contains no data rows")
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError("solar irradiance CSV rows do not match the header")
    columns = [[_number(cell, header[index]) for index, cell in enumerate(row)] for row in rows[1:]]
    values = np.asarray(columns, dtype=np.float64).T
    return SolarIrradianceSpectrum(
        values[0],
        values[1],
        values[2] if len(header) >= 3 else None,
        values[3] if len(header) == 4 else None,
    )


def _number(value: str, name: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as error:
        raise ValueError(f"solar irradiance {name} must contain numbers") from error
    if not isfinite(parsed):
        raise ValueError(f"solar irradiance {name} must contain finite numbers")
    return parsed
