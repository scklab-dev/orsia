"""Level interpolation, explicit surface continuation and gas convention checks."""

import numpy as np
import pytest
from orsia_radiative.atmosphere import LevelAtmosphere
from orsia_radiative.atmosphere.profiles import (
    convert_gas_fraction,
    resample_level_atmosphere,
    resample_pressure_profile,
)


def test_pressure_coordinates_and_continuation():
    p = np.array([1000.0, 500.0, 100.0])
    for method, values in (("log_pressure", np.log(p)), ("pressure", p)):
        target = np.array([750.0, 100.0, 1200.0])
        actual = resample_pressure_profile(p, values, target, method=method, extend_surface=True)
        np.testing.assert_allclose(actual, np.log(target) if method == "log_pressure" else target)
    for targets in ([1200.0], [50.0], [-1.0]):
        with pytest.raises(ValueError):
            resample_pressure_profile(p, p, np.array(targets), method="log_pressure")
    for pressure, values in ((p[::-1], p), (p, p[:2]), (np.array([0.0, 0.0]), p[:2])):
        with pytest.raises(ValueError):
            resample_pressure_profile(pressure, values, p, method="pressure")


def test_fraction_roundtrip_and_water_conversion():
    q = np.array([0.0, 0.01, 0.1])
    ratio = np.full(3, 400e-6)
    conventions = ("dry_air_mole", "total_air_mole", "dry_air_mass", "total_air_mass")
    for source in conventions:
        for target in conventions:
            converted = convert_gas_fraction(
                ratio,
                source=source,
                target=target,
                specific_humidity_kg_kg=q,
                molar_mass_kg_mol=0.044,
            )
            inverse = convert_gas_fraction(
                converted,
                source=target,
                target=source,
                specific_humidity_kg_kg=q,
                molar_mass_kg_mol=0.044,
            )
            np.testing.assert_allclose(inverse, ratio, rtol=5e-16)
    water = convert_gas_fraction(
        q,
        source="total_air_mass",
        target="dry_air_mole",
        specific_humidity_kg_kg=q,
        molar_mass_kg_mol=0.01801528,
    )
    np.testing.assert_allclose(water, q / (1 - q) * 0.0289647 / 0.01801528)
    with pytest.raises(ValueError):
        convert_gas_fraction(
            q, source="total_air_mass", target="dry_air_mole", specific_humidity_kg_kg=q
        )
    with pytest.raises(ValueError):
        convert_gas_fraction(
            q, source="dry_air_mole", target="total_air_mole", specific_humidity_kg_kg=np.ones(3)
        )


def test_surface_uses_native_profiles_and_preserves_level_contract():
    z = np.array([0.0, 1000.0, 3000.0])
    p = 1e5 * np.exp(-z / 8000)
    t = 290 - 0.006 * z
    q = 0.01 - z * 1e-6
    source = LevelAtmosphere(z, p, t, q, ("CO2",), np.full((3, 1), 400e-6))
    target = 1e5 * np.exp(-np.array([500.0, 2000.0, 3000.0]) / 8000)
    result = resample_level_atmosphere(
        source,
        target,
        surface_altitude_m=-100.0,
        method="log_pressure",
        surface_continuation="linear_altitude",
    )
    np.testing.assert_allclose(result.level_altitude_m, [-100.0, 500.0, 2000.0, 3000.0], atol=1e-10)
    np.testing.assert_allclose(result.level_temperature_k, 290 - 0.006 * result.level_altitude_m)
    np.testing.assert_allclose(
        result.level_specific_humidity_kg_kg, 0.01 - result.level_altitude_m * 1e-6
    )
    np.testing.assert_allclose(result.species_vmr, 400e-6)
    assert not result.level_temperature_k.flags.writeable
    with pytest.raises(ValueError):
        resample_level_atmosphere(source, target, surface_altitude_m=-100.0, method="log_pressure")
    with pytest.raises(ValueError):
        resample_level_atmosphere(
            source, target[:2], surface_altitude_m=2500.0, method="log_pressure"
        )
    with pytest.raises(ValueError):
        resample_level_atmosphere(source, target, surface_altitude_m=3000.0, method="log_pressure")
