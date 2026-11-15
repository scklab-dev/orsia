"""Explicit water coefficients, column supports and nonoverlapping composition."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_radiative import (
    Absorber,
    LayerAtmosphere,
    LayerCrossSections,
    LevelAtmosphere,
    WaterContinuumReference,
    column_absorption_optical_depth,
    compose_absorption_on_grid,
    convert_spectral_grid,
    evaluate_table_absorption,
    integrate_atmospheric_columns,
    water_continuum_absorption,
    water_continuum_coefficients,
    wavenumber_grid,
)
from orsia_radiative.spectroscopy.components import validate_absorption_freshness


def inputs():
    grid = wavenumber_grid(np.array([1.0, 7.0, 15.0, 27.0, 45.0]))
    knots = np.arange(-10.0, 61.0, 10.0)
    reference = WaterContinuumReference(
        knots,
        np.full(8, 1e-25),
        np.full(8, 1e-26),
        np.full(8, 2.0),
        101325.0,
        296.0,
        (250.0, 310.0),
    )
    water = np.array([2e26, 1e26])
    atmosphere = LayerAtmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([1e5, 9e4, 8e4]),
        np.array([95000.0, 85000.0]),
        np.array([280.0, 290.0]),
        np.array([2e28, 1e28]),
        np.array([2e28, 1e28]) - water,
        water,
        (Absorber("H2O", water, "dry_air"),),
    )
    return grid, reference, atmosphere


def test_separate_density_pair_coefficients_match_retained_homogeneous_law():
    grid, reference, atmosphere = inputs()
    coefficients = water_continuum_coefficients(
        grid,
        reference,
        height_edges_m=atmosphere.height_boundaries_m,
        temperature_k=atmosphere.temperature_k,
        pressure_pa=atmosphere.pressure_pa,
    )
    water = atmosphere.water_vapour_columns_molecules_m2
    dz = np.diff(atmosphere.height_boundaries_m)
    expected = water_continuum_absorption(atmosphere, grid, reference).optical_depth
    actual = (water * water / dz)[:, None] * coefficients.self_coefficients_m5_molecule2 + (
        water * (atmosphere.total_air_columns_molecules_m2 - water) / dz
    )[:, None] * coefficients.foreign_coefficients_m5_molecule2
    assert_allclose(actual, expected, rtol=5e-15)
    reversed_coefficients = water_continuum_coefficients(
        convert_spectral_grid(grid),
        reference,
        height_edges_m=atmosphere.height_boundaries_m,
        temperature_k=atmosphere.temperature_k,
        pressure_pa=atmosphere.pressure_pa,
    )
    assert_allclose(
        reversed_coefficients.self_coefficients_m5_molecule2,
        coefficients.self_coefficients_m5_molecule2[:, ::-1],
        rtol=5e-15,
    )
    assert_allclose(
        reversed_coefficients.foreign_coefficients_m5_molecule2,
        coefficients.foreign_coefficients_m5_molecule2[:, ::-1],
        rtol=5e-15,
    )
    for name in (
        "height_edges_m",
        "temperature_k",
        "pressure_pa",
        "self_coefficients_m5_molecule2",
        "foreign_coefficients_m5_molecule2",
    ):
        values = getattr(coefficients, name)
        with pytest.raises(ValueError):
            values.setflags(write=True)
    assert coefficients.line_wing_convention == "mt_ckd_local_25"
    for changes in (
        {"temperature_k": np.ones((2, 1))},
        {"temperature_k": np.full(2, np.nan)},
        {"pressure_pa": np.zeros(2)},
        {"height_edges_m": np.array([0.0, 0.0, 2000.0])},
        {"self_coefficients_m5_molecule2": np.ones((1, 5))},
        {"foreign_coefficients_m5_molecule2": -np.ones((2, 5))},
        {"line_wing_convention": "unknown"},
    ):
        with pytest.raises(ValueError):
            replace(coefficients, **cast(dict[str, Any], changes))


def test_total_water_tables_require_explicit_nonoverlap_and_matching_wings():
    grid, reference, atmosphere = inputs()
    table = LayerCrossSections(
        "H2O",
        "all",
        "cross_section",
        grid,
        atmosphere.temperature_k,
        atmosphere.pressure_pa,
        np.full((2, 5), 1e-27),
        height_edges_m=atmosphere.height_boundaries_m,
    )
    continuum = water_continuum_absorption(atmosphere, grid, reference)
    for flag in (None, True):
        component = evaluate_table_absorption(
            atmosphere, replace(table, includes_water_continuum=flag)
        )
        with pytest.raises(ValueError, match="explicitly exclude"):
            compose_absorption_on_grid(atmosphere, grid, (component, continuum))
    with pytest.raises(ValueError, match="matching local line"):
        compose_absorption_on_grid(
            atmosphere,
            grid,
            (
                evaluate_table_absorption(
                    atmosphere, replace(table, includes_water_continuum=False)
                ),
                continuum,
            ),
        )
    paired = replace(table, includes_water_continuum=False, line_wing_convention="mt_ckd_local_25")
    component = evaluate_table_absorption(atmosphere, paired)
    assert_array_equal(
        compose_absorption_on_grid(atmosphere, grid, (component, continuum)).optical_depth,
        component.optical_depth + continuum.optical_depth,
    )
    complete = evaluate_table_absorption(atmosphere, replace(table, includes_water_continuum=True))
    changed = replace(
        atmosphere,
        total_air_columns_molecules_m2=atmosphere.total_air_columns_molecules_m2 * 1.01,
        dry_air_columns_molecules_m2=atmosphere.total_air_columns_molecules_m2 * 1.01
        - atmosphere.water_vapour_columns_molecules_m2,
    )
    with pytest.raises(ValueError, match="stale"):
        validate_absorption_freshness(changed, complete, reference_atmosphere=atmosphere)


def test_column_contraction_requires_actual_declared_axes_and_pairing():
    grid, reference, atmosphere = inputs()
    levels = LevelAtmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([1e5, 9e4, 8e4]),
        np.full(3, 280.0),
        np.full(3, 0.01),
        ("H2O",),
        np.full((3, 1), 0.01),
    )
    columns = integrate_atmospheric_columns(levels, surface_altitude_m=500.0)
    table = LayerCrossSections(
        "H2O",
        "all",
        "cross_section",
        grid,
        atmosphere.temperature_k,
        atmosphere.pressure_pa,
        np.ones((2, 5)) * 1e-27,
        height_edges_m=levels.level_altitude_m,
        line_wing_convention="mt_ckd_local_25",
        includes_water_continuum=False,
    )
    continuum = water_continuum_coefficients(
        grid,
        reference,
        height_edges_m=table.height_edges_m,
        temperature_k=table.temperature_k,
        pressure_pa=table.pressure_pa,
    )
    depth = column_absorption_optical_depth(columns, (table,), spectral=grid, continuum=continuum)
    expected = (
        columns.layer_species_column_molecules_m2[:, 0] @ table.cross_sections_m2_per_molecule
        + columns.layer_water_self_density_product_molecules2_m5
        @ continuum.self_coefficients_m5_molecule2
        + columns.layer_water_foreign_density_product_molecules2_m5
        @ continuum.foreign_coefficients_m5_molecule2
    )
    assert_allclose(depth, expected, rtol=5e-15)
    assert depth.flags.writeable
    assert_array_equal(column_absorption_optical_depth(columns, (), spectral=grid), 0)
    for tables, water in (
        ((table, table), None),
        ((replace(table, species="CH4"),), None),
        ((replace(table, height_edges_m=table.height_edges_m + 1),), None),
        ((replace(table, kind="continuum_excluding_lines"),), continuum),
        ((table,), replace(continuum, pressure_pa=continuum.pressure_pa + 1)),
        ((), replace(continuum, height_edges_m=continuum.height_edges_m + 1)),
        ((), replace(continuum, grid=wavenumber_grid(grid.points + 0.1))),
    ):
        with pytest.raises(ValueError):
            column_absorption_optical_depth(columns, tables, spectral=grid, continuum=water)
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        column_absorption_optical_depth(
            columns,
            (replace(table, cross_sections_m2_per_molecule=np.full((2, 5), 1e300)),),
            spectral=grid,
        )


def test_material_integral_snapshot_invariants():
    levels = LevelAtmosphere(
        np.array([0.0, 100.0, 200.0]),
        np.array([1e5, 9e4, 8e4]),
        np.full(3, 280.0),
        np.zeros(3),
        ("CH4",),
        np.full((3, 1), 1e-6),
    )
    material = integrate_atmospheric_columns(levels, surface_altitude_m=100.0)
    assert material.partial_layer_index is None and material.first_complete_layer_index == 1
    changes: dict[str, Any]
    for changes in (
        {"height_edges_m": np.array([0.0, 0.0, 200.0])},
        {"surface_pressure_pa": 0.0},
        {"surface_altitude_m": 200.0},
        {"species": ("",)},
        {"layer_dry_air_column_molecules_m2": np.ones(2)},
        {"layer_species_column_molecules_m2": np.ones((2, 2))},
    ):
        with pytest.raises(ValueError):
            replace(material, **changes)
    assert material.partial_h2o_foreign_density_product == 0
