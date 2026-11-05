"""Decode native Serdyuchenko and Bogumil laboratory absorption spectra offline."""

import re
from pathlib import Path
from typing import Literal

import numpy as np
from orsia_radiative.spectroscopy.measurements import CrossSectionMeasurements, CrossSectionSpectrum


def read_cross_section_measurements(
    paths: tuple[Path, ...],
    *,
    species: str,
    isotopologue: str,
    format: Literal["serdyuchenko", "bogumil"],
    interval_nm: tuple[float, float],
    resolution: str | None = None,
) -> CrossSectionMeasurements:
    """Read laboratory spectra on caller-selected native vacuum-wavelength support.

    Headers must declare vacuum nm and cm2/molecule. Convert amplitudes by 1e-4
    to m2/molecule. Retain native interpolation brackets around the requested interval
    before checking nonnegative cross sections. Negative samples outside retained
    support are excluded. Decode declared Bogumil total pressure in mbar to Pa;
    absent pressure remains None. Declared concatenation cuts split support.
    Serdyuchenko columns declare eleven temperatures; Bogumil files declare one
    temperature each. Files are sorted by temperature and retain their own axes.

    Args:
        paths: One Serdyuchenko file or one or more Bogumil files.
        species: Whole-species absorber identity.
        isotopologue: Isotope convention, such as 'all' for natural abundance.
        format: Explicit native text format.
        interval_nm: Positive closed retained vacuum-wavelength interval in nm.
        resolution: Source resolution description if supplied.

    Returns:
        Immutable numerical measurements with native per-spectrum axes and known T.

    Raises:
        OSError: A local file cannot be read.
        ValueError: Header, temperature identity, retained support or values are invalid.
    """
    lo, hi = interval_nm
    if not np.all(np.isfinite(interval_nm)) or lo <= 0 or hi <= lo or not paths:
        raise ValueError("a positive increasing native wavelength interval and files are required")
    if format not in ("serdyuchenko", "bogumil"):
        raise ValueError("unsupported native cross-section format")
    spectra = []
    for path in paths:
        lines = path.read_text().splitlines()
        header = "\n".join(lines[:45]).lower()
        unit = re.search(r"1\.\s*column unit:\s*([^\n]+)", header)
        has_nm = (
            unit[1].strip() == "nm"
            if unit is not None
            else re.search(r"vacuum[^\n]*\bnm\b", header) is not None
        )
        if (
            "vacuum" not in header
            or not has_nm
            or not ("cm2/molec" in header or "cm^2/molecule" in header)
        ):
            raise ValueError("cross-section header must declare vacuum nm and cm2/molecule")
        identity = re.search(r"\b([a-z][a-z0-9]*)\s+absorption\s+cross", header)
        if identity is not None and identity[1].upper() != species.upper():
            raise ValueError("native cross-section species conflicts with requested identity")
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if re.match(r"^\s*\d+\.\d+(?:[Ee][+-]?\d+)?\s+[+-]?\d+\.\d+[Ee][+-]?\d+", line)
            ),
            None,
        )
        if start is None:
            raise ValueError("native cross-section file has no numerical rows")
        table = np.loadtxt(lines[start:])
        if (
            table.ndim != 2
            or not np.all(np.isfinite(table[:, 0]))
            or np.any(np.diff(table[:, 0]) <= 0)
        ):
            raise ValueError("native wavelengths must be finite and increasing")
        axis = table[:, 0]
        if axis[0] > lo or axis[-1] < hi:
            raise ValueError("native spectrum must bracket the requested interval")
        table = table[
            max(0, int(np.searchsorted(axis, lo)) - 1) : int(np.searchsorted(axis, hi)) + 1
        ]
        if format == "serdyuchenko":
            temperatures = [int(t) for t in re.findall(r"@(\d+)k", header)]
            if (
                len(paths) != 1
                or table.shape[1] != 12
                or temperatures != list(range(293, 192, -10))
            ):
                raise ValueError("Serdyuchenko temperature columns must match the native header")
        else:
            match = re.search(r"temperature:\s*(\d+)k", header)
            if table.shape[1] != 2 or match is None:
                raise ValueError("Bogumil temperature must match the native header")
            temperatures = [int(match[1])]
        pressure = re.search(r"total pressure:\s*([\d.]+)\s*mbar", header)
        pressure_pa = float(pressure[1]) * 100 if pressure is not None else None
        cut_match = re.search(r"cut and concatenated at ([^\n]+)", header)
        cuts = (
            [] if cut_match is None else [float(x) for x in re.findall(r"\d+\.\d+", cut_match[1])]
        )
        boundaries = [
            0,
            *sorted(
                {
                    int(np.searchsorted(table[:, 0], cut))
                    for cut in cuts
                    if table[0, 0] < cut < table[-1, 0]
                }
            ),
            table.shape[0],
        ]
        support = []
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            if right - left < 2:
                raise ValueError("retained concatenation segments need two samples each")
            support.append((float(table[left, 0]), float(table[right - 1, 0])))
        for index, temperature in enumerate(temperatures):
            spectra.append(
                CrossSectionSpectrum(
                    table[:, 0],
                    table[:, index + 1] * 1e-4,
                    float(temperature),
                    pressure_pa,
                    tuple(support),
                    resolution,
                )
            )
    return CrossSectionMeasurements(
        species, isotopologue, tuple(sorted(spectra, key=lambda spectrum: spectrum.temperature_k))
    )
