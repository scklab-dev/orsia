from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import pytest
from orsia_radiative import (
    LayerCrossSections,
    LevelAtmosphere,
    SpectralGrid,
    WaterContinuumCoefficients,
    column_absorption_optical_depth,
    integrate_atmospheric_columns,
)
from orsia_radiative.atmosphere.levels import (
    log_linear_density_integral,
    moist_air_number_densities,
)


@pytest.fixture
def column() -> LevelAtmosphere:
    z = np.array([0.0, 100.0, 200.0, 300.0])
    return LevelAtmosphere(
        level_altitude_m=z,
        level_pressure_pa=1e5 * np.exp(-z / 8000),
        level_temperature_k=np.full(4, 280.0),
        level_specific_humidity_kg_kg=np.zeros(4),
        species=("CO2", "CH4"),
        species_vmr=np.array([[0.0004] * 4, [0.0] * 4]).T,
    )


def test_isothermal_exponential_column_and_indexing(column: LevelAtmosphere) -> None:
    expected = 8000 * 1e5 / (1.380649e-23 * 280) * (np.exp(-50 / 8000) - np.exp(-300 / 8000))
    saved = replace(column)
    path = integrate_atmospheric_columns(column, surface_altitude_m=50.0)
    np.testing.assert_allclose(path.layer_dry_air_column_molecules_m2.sum(), expected, rtol=2e-12)
    np.testing.assert_allclose(
        path.layer_species_column_molecules_m2[:, 0].sum(), expected * 0.0004, rtol=2e-12
    )
    assert np.all(path.layer_species_column_molecules_m2[:, 1] == 0)
    assert np.all(path.layer_water_self_density_product_molecules2_m5 == 0)
    assert path.partial_layer_index == 0
    assert path.first_complete_layer_index == 1
    np.testing.assert_allclose(path.surface_pressure_pa, 1e5 * np.exp(-50 / 8000), rtol=1e-15)
    for i, altitude in enumerate(column.level_altitude_m[:-1]):
        exact = integrate_atmospheric_columns(column, surface_altitude_m=float(altitude) + 0.5e-6)
        assert exact.surface_altitude_m == altitude
        assert exact.first_complete_layer_index == i
        assert exact.partial_layer_index is None
        assert exact.partial_h2o_self_density_product == 0
        assert np.all(exact.layer_dry_air_column_molecules_m2[:i] == 0)
    for item in fields(column):
        np.testing.assert_array_equal(getattr(column, item.name), getattr(saved, item.name))
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(
            pool.map(
                lambda _: integrate_atmospheric_columns(column, surface_altitude_m=50.0), range(3)
            )
        )
    for output in outputs:
        np.testing.assert_array_equal(
            output.layer_species_column_molecules_m2, path.layer_species_column_molecules_m2
        )
        assert not np.shares_memory(
            output.layer_species_column_molecules_m2, path.layer_species_column_molecules_m2
        )


def test_endpoint_water_products_and_depth(column: LevelAtmosphere) -> None:
    column = replace(column, level_specific_humidity_kg_kg=np.full(4, 0.01))
    path = integrate_atmospheric_columns(column, surface_altitude_m=150.0)
    pressure = np.r_[1e5 * np.exp(-150 / 8000), column.level_pressure_pa[2:]]
    total = pressure / (1.380649e-23 * 280)
    water_fraction = (0.01 / 0.01801528) / (0.01 / 0.01801528 + 0.99 / 0.0289647)
    water = total * water_fraction
    foreign = total - water
    dz = np.array([50.0, 100.0])
    self_integral = dz * (water[:-1] ** 2 + water[:-1] * water[1:] + water[1:] ** 2) / 3
    foreign_integral = (
        dz
        * (
            2 * water[:-1] * foreign[:-1]
            + water[:-1] * foreign[1:]
            + water[1:] * foreign[:-1]
            + 2 * water[1:] * foreign[1:]
        )
        / 6
    )
    np.testing.assert_allclose(
        path.layer_water_self_density_product_molecules2_m5,
        np.r_[0.0, self_integral],
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        path.layer_water_foreign_density_product_molecules2_m5,
        np.r_[0.0, foreign_integral],
        rtol=2e-15,
    )
    assert (
        path.partial_h2o_self_density_product
        == path.layer_water_self_density_product_molecules2_m5[1]
    )
    assert (
        path.partial_h2o_foreign_density_product
        == path.layer_water_foreign_density_product_molecules2_m5[1]
    )
    sections = np.full((2, 3, 5), 1e-25)
    own, other = np.full((3, 5), 1e-54), np.full((3, 5), 1e-55)
    tables, continuum, spectral = coefficient_inputs(column, sections, own, other)
    depth = column_absorption_optical_depth(path, tables, spectral=spectral, continuum=continuum)
    expected = (
        path.layer_species_column_molecules_m2.sum() * 1e-25
        + self_integral.sum() * 1e-54
        + foreign_integral.sum() * 1e-55
    )
    np.testing.assert_allclose(depth, np.full(5, expected), rtol=2e-15)
    assert depth.flags.writeable


def test_density_integral_limits_and_broadcast() -> None:
    lower = np.array([[1.0 + 5e-6, 0.0], [2.0, 3.0]])
    upper = np.ones((2, 2))
    dz = np.array([10.0, 20.0])
    integral = log_linear_density_integral(lower, upper, dz)
    expected = np.array([[0.5 * (2 + 5e-6) * 10, 10], [10 / np.log(2), 40 / np.log(3)]])
    np.testing.assert_allclose(integral, expected, rtol=2e-16)
    a = np.array([1.0, 0.0])
    np.testing.assert_array_equal(log_linear_density_integral(a, a, dz), a * dz)
    np.testing.assert_array_equal(log_linear_density_integral(a, a, np.zeros(2)), 0)
    p, t, q = np.array([1e5, 5e4]), np.array([280.0, 250.0]), np.array([0.0, 0.01])
    total, dry, water = moist_air_number_densities(p, t, q)
    np.testing.assert_allclose(total, dry + water, rtol=2e-16)
    assert water[0] == 0 and dry[0] == total[0]


@pytest.mark.parametrize("altitude", [-2e-6, 300.0, 299.9999995, np.nan])
def test_invalid_terrain(column: LevelAtmosphere, altitude: float) -> None:
    with pytest.raises(ValueError, match="surface"):
        integrate_atmospheric_columns(column, surface_altitude_m=altitude)


@pytest.mark.parametrize(
    "change",
    [
        {"level_altitude_m": np.array([0.0])},
        {"level_temperature_k": np.array([280.0])},
        {"species_vmr": np.zeros((2, 3))},
        {"species_vmr": np.zeros((0, 4))},
        {"level_pressure_pa": np.arange(4.0)},
        {"level_specific_humidity_kg_kg": np.ones(4)},
        {"species_vmr": np.full((4, 2), -1.0)},
        {"level_altitude_m": np.array([0.0, 100.0, np.nan, 300.0])},
    ],
)
def test_invalid_columns(column: LevelAtmosphere, change: dict) -> None:
    with pytest.raises(ValueError):
        integrate_atmospheric_columns(
            replace(column, **cast(dict[str, Any], change)), surface_altitude_m=0.0
        )


@pytest.mark.parametrize("name", ["pressure", "temperature", "humidity"])
def test_invalid_level_densities(name: str) -> None:
    values = dict(
        pressure=np.array([1e5]), temperature=np.array([280.0]), humidity=np.array([0.01])
    )
    values[name][:] = -1
    with pytest.raises(ValueError):
        moist_air_number_densities(**values)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_density_integral(bad: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        log_linear_density_integral(np.array([bad]), np.ones(1), np.ones(1))


@pytest.mark.parametrize(
    "change", ["sections_shape", "self_shape", "foreign_shape", "negative", "nonfinite"]
)
def test_invalid_optical_depth(column: LevelAtmosphere, change: str) -> None:
    path = integrate_atmospheric_columns(column, surface_altitude_m=0.0)
    sections, own, foreign = np.ones((2, 3, 5)), np.ones((3, 5)), np.ones((3, 5))
    if change == "sections_shape":
        sections = sections[:, :-1]
    elif change == "self_shape":
        own = own[:-1]
    elif change == "foreign_shape":
        foreign = foreign[:-1]
    elif change == "negative":
        sections[0, 0, 0] = -1
    else:
        own[0, 0] = np.nan
    with pytest.raises(ValueError):
        tables, continuum, spectral = coefficient_inputs(column, sections, own, foreign)
        column_absorption_optical_depth(path, tables, spectral=spectral, continuum=continuum)


def coefficient_inputs(column, sections, own, foreign):
    """Declare synthetic fixed coefficients at explicitly selected lower-level conditions."""
    spectral = SpectralGrid("wavelength_nm", np.arange(5.0) + 500)
    tables = tuple(
        LayerCrossSections(
            species,
            "all",
            "cross_section",
            spectral,
            column.level_temperature_k[:-1],
            column.level_pressure_pa[:-1],
            values,
            height_edges_m=column.level_altitude_m,
        )
        for species, values in zip(column.species, sections, strict=True)
    )
    continuum = WaterContinuumCoefficients(
        spectral,
        column.level_altitude_m,
        column.level_temperature_k[:-1],
        column.level_pressure_pa[:-1],
        own,
        foreign,
        "mt_ckd_local_25",
    )
    return tables, continuum, spectral


def test_optional_zero_based_indices_and_species_permutation(column):
    top = integrate_atmospheric_columns(column, surface_altitude_m=250.0)
    assert top.partial_layer_index == 2 and top.first_complete_layer_index is None
    permuted = replace(
        column, species=column.species[::-1], species_vmr=column.species_vmr[:, ::-1]
    )
    other = integrate_atmospheric_columns(permuted, surface_altitude_m=250.0)
    np.testing.assert_array_equal(
        top.layer_species_column_molecules_m2[:, ::-1], other.layer_species_column_molecules_m2
    )
    empty = replace(column, species=(), species_vmr=np.empty((4, 0)))
    air = integrate_atmospheric_columns(empty, surface_altitude_m=0.0)
    assert air.layer_species_column_molecules_m2.shape == (3, 0)
    for array in (column.species_vmr, top.layer_species_column_molecules_m2, top.height_edges_m):
        with pytest.raises(ValueError):
            array.setflags(write=True)
