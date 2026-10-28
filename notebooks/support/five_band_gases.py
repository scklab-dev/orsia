"""Local-file trace-gas assembly for the five-band demonstration.

Functions return owned arrays. Spectral interpolation uses native vacuum nm;
optical depths use the common increasing cm-1 grid and bottom-to-top layers.
"""

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from orsia_data import read_rfm_atmosphere
from orsia_radiative import (
    CrossSectionField,
    gas_columns_from_fraction,
    pressure_level_layer_means,
    table_absorption,
)


def project_trace_gases(path, interfaces, humidity, species):
    """Project MIPAS dry mole fractions linearly in pressure to ERA5 layers.

    Native x_g / (1-x_H2O) defines dry mol/mol before pressure averaging.
    ERA5 humidity sets dry-air columns; gas amount = mean(x_dry)*N_dry.
    Returned diagnostic wet fractions use ERA5 water and total-air columns.
    All target interfaces must fall inside the native pressure support.
    """
    _, pressure, _, gases = read_rfm_atmosphere(path)
    water = gases["H2O"]
    if np.any(water >= 1):
        raise ValueError("MIPAS water must leave positive dry-air fraction")
    dry_air = gas_columns_from_fraction(
        interfaces,
        humidity,
        np.ones_like(humidity),
        convention="dry_air_mole",
        gravity_m_s2=9.80665,
    )
    total_air = gas_columns_from_fraction(
        interfaces,
        humidity,
        np.ones_like(humidity),
        convention="total_air_mole",
        gravity_m_s2=9.80665,
    )
    columns, fractions = {}, {}
    for name in species:
        native_dry = gases[name] / (1 - water)
        dry = pressure_level_layer_means(pressure, native_dry, interfaces)
        columns[name] = gas_columns_from_fraction(
            interfaces,
            humidity,
            dry,
            convention="dry_air_mole",
            gravity_m_s2=9.80665,
        )
        fractions[name] = dict(
            dry_mol_mol=dry.tolist(),
            era5_wet_mol_mol=(dry * dry_air / total_air).tolist(),
            column_molecules_m2=columns[name].tolist(),
        )
    return columns, fractions


def read_uv_spectra(base: Path, entry: dict):
    """Decode selected native Bremen files, retaining 319-326 nm plus brackets.

    Each file explicitly declares vacuum wavelengths and cm2/molecule. Cross
    sections are multiplied by 1e-4; the coordinate is a sample axis, so that
    conversion leaves cross-section amplitudes unchanged. Source rows outside
    the retained band are irrelevant to this bounded demonstration.
    """
    spectra = []
    for index, filename in enumerate(entry["files"]):
        lines = (base / filename).read_text().splitlines()
        header = "\n".join(lines[:45]).lower()
        if "vacuum" not in header or not ("cm2/molec" in header or "cm^2/molecule" in header):
            raise ValueError("cross-section header must declare vacuum nm and cm2/molecule")
        start = next(
            i
            for i, line in enumerate(lines)
            if re.match(r"^\s*\d+\.\d+(?:[Ee][+-]?\d+)?\s+[+-]?\d+\.\d+[Ee][+-]?\d+", line)
        )
        table = np.loadtxt(lines[start:])
        if not np.all(np.isfinite(table)) or np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError("cross-section wavelength rows must be finite and increasing")
        axis = table[:, 0]
        left = max(0, np.searchsorted(axis, 319.0) - 1)
        right = np.searchsorted(axis, 326.0) + 1
        table = table[left:right]
        if table[0, 0] > 320 or table[-1, 0] < 325:
            raise ValueError("cross-section file must bracket the UV window")
        if entry["format"] == "serdyuchenko":
            if table.shape[1] != 12:
                raise ValueError("Serdyuchenko table requires wavelength plus 11 temperatures")
            expected = list(range(193, 294, 10))
            if (
                len(entry["files"]) != 1
                or entry["temperatures_k"] != expected
                or [int(t) for t in re.findall(r"@(\d+)k", header)] != expected[::-1]
            ):
                raise ValueError("Serdyuchenko temperature columns must match the native header")
            spectra.extend((table[:, 0].copy(), row * 1e-4) for row in table[:, 1:][:, ::-1].T)
        elif entry["format"] == "bogumil":
            if table.shape[1] != 2:
                raise ValueError("Bogumil table requires wavelength and cross section")
            match = re.search(r"temperature:\s*(\d+)k", header)
            if (
                match is None
                or index >= len(entry["temperatures_k"])
                or int(match[1]) != entry["temperatures_k"][index]
            ):
                raise ValueError("Bogumil temperature must match the native header")
            spectra.append((table[:, 0].copy(), table[:, 1] * 1e-4))
        else:
            raise ValueError("unsupported cross-section format")
    if len(spectra) != len(entry["temperatures_k"]):
        raise ValueError("temperature and spectrum counts must agree")
    return spectra


def interpolate_uv(
    spectra: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    temperatures_k: NDArray[np.float64],
    layer_temperature_k: NDArray[np.float64],
    wavelength_nm: NDArray[np.float64],
    *,
    temperature_policy: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linear native-wavelength then linear-temperature interpolation [m2/molecule].

    Each temperature spectrum may have its own strictly increasing nm axis.
    ``clip`` holds the nearest laboratory temperature outside its range;
    ``error`` requires bracketing. Return layer-by-point values and effective T.
    The caller retains the resulting temperature discrepancy as diagnostics.
    """
    if (
        temperatures_k.ndim != 1
        or temperatures_k.size < 2
        or np.any(~np.isfinite(temperatures_k))
        or np.any(temperatures_k <= 0)
        or np.any(np.diff(temperatures_k) <= 0)
        or len(spectra) != temperatures_k.size
    ):
        raise ValueError("temperatures must be positive, increasing and align with spectra")
    if (
        layer_temperature_k.ndim != 1
        or layer_temperature_k.size == 0
        or np.any(~np.isfinite(layer_temperature_k))
        or np.any(layer_temperature_k <= 0)
        or wavelength_nm.ndim != 1
        or wavelength_nm.size == 0
        or np.any(~np.isfinite(wavelength_nm))
        or np.any(wavelength_nm <= 0)
    ):
        raise ValueError(
            "layer temperatures and target wavelengths must be positive finite vectors"
        )
    if temperature_policy not in ("clip", "error"):
        raise ValueError("temperature policy must be clip or error")
    effective = np.clip(layer_temperature_k, temperatures_k[0], temperatures_k[-1])
    if temperature_policy == "error" and np.any(effective != layer_temperature_k):
        raise ValueError("layer temperature outside laboratory support")
    rows = []
    for wavelength, sigma in spectra:
        if (
            wavelength.ndim != 1
            or wavelength.size < 2
            or sigma.shape != wavelength.shape
            or np.any(~np.isfinite(wavelength))
            or np.any(np.diff(wavelength) <= 0)
            or np.any(~np.isfinite(sigma))
            or np.any(sigma < 0)
        ):
            raise ValueError(
                "spectra require increasing finite axes and nonnegative cross sections"
            )
        if np.any(wavelength_nm < wavelength[0]) or np.any(wavelength_nm > wavelength[-1]):
            raise ValueError("target wavelength outside laboratory support")
        rows.append(np.interp(wavelength_nm, wavelength, sigma))
    rows = np.stack(rows)
    left = np.clip(
        np.searchsorted(temperatures_k, effective, side="right") - 1, 0, temperatures_k.size - 2
    )
    weight = (
        (effective - temperatures_k[left]) / (temperatures_k[left + 1] - temperatures_k[left])
    )[:, None]
    return (1 - weight) * rows[left] + weight * rows[left + 1], effective


def uv_components(base, entries, atmosphere, grid):
    """Materialize UV fields at ERA5 T/p with explicit temperature-hold diagnostics."""
    components, diagnostics = [], {}
    for entry in entries:
        if "UV" not in entry["included_cases"]:
            continue
        values, effective = interpolate_uv(
            read_uv_spectra(base, entry),
            np.array(entry["temperatures_k"], dtype=float),
            atmosphere.temperature_k,
            1e7 / grid.points,
            temperature_policy=entry["temperature_policy"],
        )
        field = CrossSectionField(
            entry["species"],
            "all",
            "cross_section",
            grid,
            atmosphere.temperature_k,
            atmosphere.pressure_pa,
            values,
        )
        components.append(table_absorption(atmosphere, grid, field))
        diagnostics[entry["species"]] = dict(
            actual_temperature_k=atmosphere.temperature_k.tolist(),
            effective_temperature_k=effective.tolist(),
            held_layers=np.flatnonzero(effective != atmosphere.temperature_k).tolist(),
            temperature_policy=entry["temperature_policy"],
            pressure_policy=entry["pressure_policy"],
        )
    return components, diagnostics
