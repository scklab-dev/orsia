"""Analytic hydrostatic conversions and conservation, not real-product validation."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_radiative import (
    Atmosphere,
    gas_columns_from_fraction,
    geometric_height,
    hydrostatic_atmosphere,
    pressure_level_layer_means,
    remap_layer_columns,
    specific_humidity,
)
from orsia_radiative.atmosphere.hydrostatic import (
    AVOGADRO_MOL1,
    GAS_CONSTANT_J_MOL_K,
    MOLAR_MASS_DRY_AIR_KG_MOL,
    MOLAR_MASS_WATER_KG_MOL,
)
from orsia_radiative.atmosphere.layers import (
    validate_pressure_boundaries,
    validate_vertical_values,
)

P = np.array([100000.0, 60000.0, 20000.0])
Q = np.array([0.01, 0.02])
T = np.array([280.0, 240.0])


@pytest.mark.parametrize(
    "amounts,expected",
    [
        ([40000.0, 0.0], [10000.0, 20000.0, 0.0]),
        ([0.0, 40000.0], [0.0, 20000.0, 20000.0]),
        ([40000.0, 40000.0], [10000.0, 40000.0, 20000.0]),
    ],
)
def test_remap_overlap_and_ownership(amounts: list[float], expected: list[float]) -> None:
    source = P.copy()
    target = np.array([90000.0, 80000.0, 40000.0, 20000.0])
    columns = np.array(amounts)
    original_target = target.copy()
    for values in (source, target, columns):
        values.flags.writeable = False
    remapped = remap_layer_columns(source, columns, target)
    assert_allclose(remapped, expected)
    assert remapped.flags.writeable
    for values in (source, target, columns):
        assert not np.shares_memory(remapped, values)
    remapped[:] = 0
    assert_allclose(source, P)
    assert_allclose(target, original_target)
    assert_allclose(columns, amounts)


def test_vertical_coordinate_validation() -> None:
    validate_pressure_boundaries(np.array([100.0, 0.0]), "interfaces")
    validate_vertical_values(np.array([-1.0, 0.0]), 2, "values")
    for values in (np.ones((2, 1)), np.array([np.inf]), np.array([np.nan]), np.ones(3)):
        with pytest.raises(ValueError, match="values must be a finite vector"):
            validate_vertical_values(values, 2, "values")
    for values in (np.array([]), np.array([1.0]), np.array([1.0, 1.0]), np.array([1.0, -1.0])):
        with pytest.raises(ValueError, match="interfaces must have at least two"):
            validate_pressure_boundaries(values, "interfaces")


def test_documented_atmosphere_example() -> None:
    document = Path(__file__).resolve().parents[3] / "docs" / "atmospheric-data.md"
    snippet = document.read_text().split("```python\n")[1].split("```")[0]
    namespace: dict[str, Any] = {}
    exec(compile(snippet, str(document), "exec"), namespace)
    atmosphere = namespace["atmosphere"]
    assert isinstance(atmosphere, Atmosphere)
    assert_allclose(atmosphere.absorbers[1].columns_molecules_m2.sum(), namespace["ch4"].sum())


def test_analytic_air_gas_columns_and_ownership() -> None:
    gas = gas_columns_from_fraction(
        P, Q, np.full(2, 1e-6), convention="dry_air_mole", gravity_m_s2=10
    )
    p = P.copy()
    atmosphere = hydrostatic_atmosphere(
        p,
        T,
        Q,
        target_pressure_boundaries_pa=p,
        gravity_m_s2=10,
        bottom_height_m=100,
        gas_columns_molecules_m2={"CH4": gas},
        air_broadener="dry_air",
    )
    mass = np.array([4000.0, 4000.0])
    dry = mass * (1 - Q) / MOLAR_MASS_DRY_AIR_KG_MOL * AVOGADRO_MOL1
    water = mass * Q / MOLAR_MASS_WATER_KG_MOL * AVOGADRO_MOL1
    assert_allclose(atmosphere.dry_air_columns_molecules_m2, dry)
    assert_allclose(atmosphere.water_vapour_columns_molecules_m2, water)
    assert_allclose(atmosphere.total_air_columns_molecules_m2, dry + water)
    assert_allclose(atmosphere.absorbers[0].columns_molecules_m2, water)
    assert_allclose(atmosphere.absorbers[1].columns_molecules_m2, dry * 1e-6)
    r = GAS_CONSTANT_J_MOL_K * ((1 - Q) / MOLAR_MASS_DRY_AIR_KG_MOL + Q / MOLAR_MASS_WATER_KG_MOL)
    assert_allclose(np.diff(atmosphere.height_boundaries_m), r * T / 10 * np.log(P[:-1] / P[1:]))
    p[:] = 1
    gas[:] = 0
    assert_allclose(atmosphere.pressure_boundaries_pa, P)
    assert atmosphere.temperature_k.flags.writeable
    assert_allclose(T, [280, 240])


@pytest.mark.parametrize("convention", ["total_air_mole", "total_air_mass", "dry_air_mass"])
def test_fraction_conventions(convention: str) -> None:
    # Explicitly typed alternatives avoid treating a native mass fraction as molar.
    if convention == "total_air_mole":
        out = gas_columns_from_fraction(
            P, Q, np.full(2, 1e-6), convention="total_air_mole", gravity_m_s2=10
        )
        expected = (
            4000
            * AVOGADRO_MOL1
            * ((1 - Q) / MOLAR_MASS_DRY_AIR_KG_MOL + Q / MOLAR_MASS_WATER_KG_MOL)
            * 1e-6
        )
    elif convention == "dry_air_mass":
        out = gas_columns_from_fraction(
            P,
            Q,
            np.full(2, 1e-6),
            convention="dry_air_mass",
            gravity_m_s2=10,
            molar_mass_kg_mol=0.044,
        )
        expected = 4000 * (1 - Q) * AVOGADRO_MOL1 * 1e-6 / 0.044
    else:
        out = gas_columns_from_fraction(
            P,
            Q,
            np.full(2, 1e-6),
            convention="total_air_mass",
            gravity_m_s2=10,
            molar_mass_kg_mol=0.044,
        )
        expected = np.full(2, 4000 * AVOGADRO_MOL1 * 1e-6 / 0.044)
    assert_allclose(out, expected)


def test_conservative_split_merge_and_partial_terrain_support() -> None:
    target = np.array([100000.0, 80000.0, 60000.0, 20000.0])
    columns = np.array([2.0, 8.0])
    split = remap_layer_columns(P, columns, target)
    assert_allclose(split, [1, 1, 8])
    assert_allclose(remap_layer_columns(target, split, P), columns)
    assert_allclose(remap_layer_columns(P, columns, np.array([90000.0, 20000.0])), [9.5])
    gas = gas_columns_from_fraction(
        P, Q, np.full(2, 1e-6), convention="dry_air_mole", gravity_m_s2=10
    )
    merged = hydrostatic_atmosphere(
        P,
        T,
        Q,
        target_pressure_boundaries_pa=np.array([100000.0, 20000.0]),
        gravity_m_s2=10,
        bottom_height_m=0,
        gas_columns_molecules_m2={"CH4": gas},
        air_broadener="total_air",
    )
    assert_allclose(merged.temperature_k, [260])
    assert_allclose(
        merged.dry_air_columns_molecules_m2.sum(),
        (4000 * (1 - Q) * AVOGADRO_MOL1 / MOLAR_MASS_DRY_AIR_KG_MOL).sum(),
    )
    assert_allclose(
        merged.water_vapour_columns_molecules_m2.sum(),
        (4000 * Q * AVOGADRO_MOL1 / MOLAR_MASS_WATER_KG_MOL).sum(),
    )
    assert_allclose(merged.absorbers[1].columns_molecules_m2.sum(), gas.sum())


def test_pressure_reconstruction_and_humidity() -> None:
    assert_allclose(
        pressure_level_layer_means(P, 2 * P + 3, np.array([90000.0, 30000.0])), [120003]
    )
    assert_allclose(pressure_level_layer_means(P, np.array([0, 8, 0.0]), P), [4, 4])
    assert_allclose(specific_humidity(np.array([0, 1.0])), [0, 0.5])
    assert_allclose(
        geometric_height(np.array([0, 10000.0]), reference_gravity_m_s2=10, earth_radius_m=6371000),
        [0, 6371000 * 1000 / 6370000],
    )
    for gravity, radius in ((0, 6371000), (10, 0), (np.nan, 1), (10, np.nan), (10, 500)):
        with pytest.raises(ValueError):
            geometric_height(
                np.array([10000.0]), reference_gravity_m_s2=gravity, earth_radius_m=radius
            )


@pytest.mark.parametrize(
    "p,q,g",
    [
        ([1], [0], 10),
        ([100, 200, 0], [0, 0], 10),
        ([100, 0, -1], [0, 0], 10),
        ([100, 50, 1], [0, 1], 10),
        ([100, 50, 1], [0, -1], 10),
        ([100, 50, 1], [0, 0], 0),
        ([100, 50, 1], [0, 0], np.nan),
        ([100, 50, 1], [np.nan, 0], 10),
        ([100, 50, 1], [0], 10),
    ],
)
def test_invalid_hydrostatic_inputs(p: list[float], q: list[float], g: float) -> None:
    with pytest.raises(ValueError):
        gas_columns_from_fraction(
            np.array(p), np.array(q), np.zeros(2), convention="dry_air_mole", gravity_m_s2=g
        )


def test_other_invalid_inputs() -> None:
    for target in (np.array([110000.0, 20000]), np.array([100000.0, 10000])):
        with pytest.raises(ValueError, match="coverage"):
            remap_layer_columns(P, np.ones(2), target)
        with pytest.raises(ValueError, match="coverage"):
            pressure_level_layer_means(P, np.ones(3), target)
    with pytest.raises(ValueError):
        remap_layer_columns(P, np.array([-1, 1.0]), P)
    with pytest.raises(ValueError):
        specific_humidity(np.array([-1.0]))
    for fraction, molar_mass in ((-1, 0.044), (2, 0.044), (0.99, 0.044), (0.01, None), (0.01, 0)):
        with pytest.raises(ValueError):
            gas_columns_from_fraction(
                P,
                Q,
                np.full(2, fraction),
                convention="total_air_mass",
                gravity_m_s2=10,
                molar_mass_kg_mol=molar_mass,
            )
    for temperature, target, gases in (
        (np.array([-1, 1.0]), P, {}),
        (T, np.array([100000.0, 0]), {}),
        (T, P, {"H2O": np.ones(2)}),
        (T, P, {"CH4": -np.ones(2)}),
        (T, P, {"CH4": np.full(2, 1e40)}),
    ):
        # For zero model top, provide source coverage explicitly.
        source = np.array([100000.0, 60000.0, 0]) if target[-1] == 0 else P
        with pytest.raises(ValueError):
            hydrostatic_atmosphere(
                source,
                temperature,
                Q,
                target_pressure_boundaries_pa=target,
                gravity_m_s2=10,
                bottom_height_m=0,
                gas_columns_molecules_m2=gases,
                air_broadener="dry_air",
            )
