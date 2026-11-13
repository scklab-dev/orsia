"""Local-file trace-gas assembly for the five-band demonstration.

Functions return owned arrays. Spectral interpolation uses native vacuum nm;
optical depths use the common increasing cm-1 grid and bottom-to-top layers.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from orsia_data import read_cross_section_measurements, read_rfm_atmosphere
from orsia_radiative import (
    CrossSectionMeasurements,
    CrossSectionSpectrum,
    LayerCrossSections,
    gas_columns_from_fraction,
    interpolate_cross_section_measurements,
    pressure_level_layer_means,
    table_absorption,
)
from orsia_radiative.spectral.grid import wavenumber_grid


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
    """Adapt the retained notebook entry to the owning native Data reader."""
    reference = read_cross_section_measurements(
        tuple(base / filename for filename in entry["files"]),
        species=entry.get("species", "UV"),
        isotopologue="all",
        format=entry["format"],
        interval_nm=(319.0, 326.0),
    )
    temperatures = [s.temperature_k for s in reference.spectra]
    if temperatures != entry["temperatures_k"]:
        raise ValueError("temperature columns or temperature must match the native header")
    return [
        (s.wavelength_nm.copy(), s.cross_sections_m2_per_molecule.copy()) for s in reference.spectra
    ]


def interpolate_uv(
    spectra: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    temperatures_k: NDArray[np.float64],
    layer_temperature_k: NDArray[np.float64],
    wavelength_nm: NDArray[np.float64],
    *,
    temperature_policy: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Adapt the historical clip spelling to the owning Radiative interpolation API."""
    if temperature_policy not in ("clip", "error"):
        raise ValueError("temperature policy must be clip or error")
    if temperatures_k.ndim != 1 or temperatures_k.size < 2 or len(spectra) != temperatures_k.size:
        raise ValueError("temperatures must align with spectra")
    reference = CrossSectionMeasurements(
        "UV",
        "all",
        tuple(
            CrossSectionSpectrum(
                axis, sigma, float(temperature), None, ((float(axis[0]), float(axis[-1])),)
            )
            for (axis, sigma), temperature in zip(spectra, temperatures_k, strict=True)
        ),
    )
    values, diagnostic = interpolate_cross_section_measurements(
        reference,
        layer_temperature_k,
        wavelength_nm,
        spectral_interpolation="linear_native_wavelength",
        temperature_interpolation="linear",
        temperature_outside="hold_nearest" if temperature_policy == "clip" else "error",
        pressure_model="independent",
    )
    return values, diagnostic.effective_k.copy()


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
        field = LayerCrossSections(
            entry["species"],
            "all",
            "cross_section",
            grid,
            atmosphere.temperature_k,
            atmosphere.pressure_pa,
            values,
            height_edges_m=atmosphere.height_boundaries_m,
        )
        components.append(table_absorption(atmosphere, wavenumber_grid(grid), field))
        diagnostics[entry["species"]] = dict(
            actual_temperature_k=atmosphere.temperature_k.tolist(),
            effective_temperature_k=effective.tolist(),
            held_layers=np.flatnonzero(effective != atmosphere.temperature_k).tolist(),
            temperature_policy=entry["temperature_policy"],
            pressure_policy=entry["pressure_policy"],
        )
    return components, diagnostics
