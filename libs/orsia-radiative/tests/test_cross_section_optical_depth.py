"""Fixed-cross-section array contracts and spectroscopy adapter regressions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import orsia_radiative as radiative
import pytest
from orsia_radiative.optics import absorption, absorption_native, cross_sections
from orsia_radiative.optics.cross_sections import (
    cross_section_optical_depth,
    cross_section_optical_depth_tangent,
)
from test_absorption_solar_tangents import layout_copy
from test_reusable_spectroscopy import atmosphere, measurement


@pytest.mark.parametrize("k", [0, 1, 4])
@pytest.mark.parametrize("layout", ["F", "reversed", "unaligned"])
def test_direct_tangent_and_ownership(k, layout):
    rng = np.random.default_rng(732)
    sigma = rng.uniform(0, 2, (3, 5))
    sigma[0, 0] = 0
    columns = np.array([0.0, 2.0, 3.0])
    directions = rng.normal(size=(3, k))
    arrays = [layout_copy(a, layout) for a in (sigma, columns, directions)]
    snapshots = [a.tobytes() for a in arrays]
    s, n, dn = arrays
    depth = cross_section_optical_depth(s, n)
    value, tangent = cross_section_optical_depth_tangent(s, n, dn)
    np.testing.assert_array_equal(depth, sigma * columns[:, None])
    np.testing.assert_array_equal(value, depth)
    np.testing.assert_array_equal(tangent, sigma[..., None] * directions[:, None, :])
    outputs = (depth, value, tangent)
    for i, output in enumerate(outputs):
        assert output.dtype == np.float64
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        assert all(not np.shares_memory(output, a) for a in (*arrays, *outputs[:i]))
    assert [a.tobytes() for a in arrays] == snapshots


def test_centered_column_differences():
    rng = np.random.default_rng(53)
    sigma = rng.uniform(0, 1, (2, 4))
    columns = np.array([2.0, 3.0])
    directions = rng.normal(size=(2, 3))
    _, tangent = cross_section_optical_depth_tangent(sigma, columns, directions)
    h = 1e-4
    for d in range(3):
        finite_difference = (
            cross_section_optical_depth(sigma, columns + h * directions[:, d])
            - cross_section_optical_depth(sigma, columns - h * directions[:, d])
        ) / (2 * h)
        np.testing.assert_allclose(finite_difference, tangent[..., d], rtol=2e-10, atol=2e-12)


def test_canonical_names_root_exports_and_keyword_calls():
    for module in (radiative, absorption, absorption_native, cross_sections):
        assert not any(name.startswith("fixed_cross_section_") for name in dir(module))
        assert not any("adjoint" in name for name in dir(module))
    for name in (
        "cross_section_optical_depth",
        "cross_section_optical_depth_tangent",
    ):
        assert getattr(radiative, name) is globals()[name]
        assert name in radiative.__all__
        assert callable(getattr(absorption_native, name))
    sigma, columns = np.ones((2, 3)), np.ones(2)
    value, _ = cross_section_optical_depth_tangent(
        cross_sections_m2=sigma,
        columns_molecules_m2=columns,
        columns_tangent_molecules_m2=np.ones((2, 1)),
    )
    np.testing.assert_array_equal(
        value,
        cross_section_optical_depth(cross_sections_m2=sigma, columns_molecules_m2=columns),
    )


@pytest.mark.parametrize(
    "index,bad",
    [
        (0, np.zeros((0, 3))),
        (0, np.zeros((2, 0))),
        (0, np.zeros(2)),
        (0, np.ones((2, 3), dtype=np.float32)),
        (0, np.ones((2, 3), dtype=">f8")),
        (0, np.full((2, 3), -1.0)),
        (0, np.full((2, 3), np.nan)),
        (0, np.full((2, 3), np.inf)),
        (1, np.zeros(3)),
        (1, np.ones((2, 1))),
        (1, np.ones(2, dtype=np.float32)),
        (1, np.array([np.inf, 1.0])),
        (1, np.array([np.nan, 1.0])),
        (1, np.array([-1.0, 1.0])),
    ],
)
def test_direct_invalid_inputs(index, bad):
    arrays = [np.ones((2, 3)), np.zeros(2)]
    arrays[index] = bad
    with pytest.raises(ValueError):
        cross_section_optical_depth(*arrays)


def test_direct_overflow_and_zero_column_limits():
    sigma = np.full((2, 3), 1e308)
    with pytest.raises(ValueError, match="overflow"):
        cross_section_optical_depth(sigma, np.full(2, 2.0))
    np.testing.assert_array_equal(cross_section_optical_depth(sigma, np.zeros(2)), 0.0)
    np.testing.assert_array_equal(
        cross_section_optical_depth(np.zeros((2, 3)), np.full(2, 1e308)), 0.0
    )


def test_concurrent_direct_calls():
    sigma, columns = np.full((3, 7), 0.2), np.arange(3, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: cross_section_optical_depth(sigma, columns), range(4)))
    for output in outputs:
        np.testing.assert_array_equal(output, sigma * columns[:, None])
    assert not np.shares_memory(outputs[0], outputs[1])


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_table_and_measurement_adapters_preserve_supported_storage(dtype):
    atm = atmosphere()
    amounts = np.array([2, 3], dtype=dtype)
    gas = next(gas for gas in atm.absorbers if gas.species == "CO2")
    atm = replace(atm, absorbers=(replace(gas, columns_molecules_m2=amounts),))
    grid = radiative.wavenumber_grid(np.array([1e7 / 325.0, 1e7 / 320.0]))
    sigma = np.array([[1, 2], [3, 4]], dtype=dtype)
    table = radiative.LayerCrossSections(
        "CO2",
        "all",
        "cross_section",
        grid,
        atm.temperature_k,
        atm.pressure_pa,
        sigma,
        height_edges_m=atm.height_boundaries_m,
    )
    prepared = replace(table, grid=grid, cross_sections_m2_per_molecule=sigma)
    before = prepared.cross_sections_m2_per_molecule.copy()
    for component in (
        radiative.evaluate_table_absorption(atm, prepared),
        radiative.table_absorption(atm, radiative.wavenumber_grid(grid), table),
    ):
        np.testing.assert_array_equal(component.optical_depth, sigma * amounts[:, None])
        assert component.grid is grid and component.species == "CO2"
        assert component.isotopologue == "all" and component.kind == "cross_section"
        assert component.optical_depth.dtype == np.float64
        assert component.optical_depth.flags.owndata and component.optical_depth.flags.writeable
        assert not np.shares_memory(
            component.optical_depth, prepared.cross_sections_m2_per_molecule
        )
    np.testing.assert_array_equal(prepared.cross_sections_m2_per_molecule, before)
    reference = measurement()
    values, expected_diagnostics = radiative.interpolate_cross_section_measurements(
        reference,
        atm.temperature_k,
        1e7 / grid.points,
        spectral_interpolation="linear_native_wavelength",
        temperature_interpolation="linear",
        temperature_outside="hold_nearest",
        pressure_model="independent",
    )
    component, diagnostics = radiative.measurement_absorption(
        atm,
        grid,
        reference,
        spectral_interpolation="linear_native_wavelength",
        temperature_interpolation="linear",
        temperature_outside="hold_nearest",
        pressure_model="independent",
    )
    np.testing.assert_array_equal(component.optical_depth, values * amounts[:, None])
    assert component.grid is grid and component.optical_depth.dtype == np.float64
    np.testing.assert_array_equal(diagnostics.effective_k, expected_diagnostics.effective_k)
    np.testing.assert_array_equal(diagnostics.held_layers, expected_diagnostics.held_layers)
    np.testing.assert_array_equal(atm.absorbers[0].columns_molecules_m2, amounts)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_mutated_prepared_cross_sections_rejected_even_with_zero_columns(bad):
    atm = atmosphere()
    gas = next(gas for gas in atm.absorbers if gas.species == "CO2")
    atm = replace(atm, absorbers=(replace(gas, columns_molecules_m2=np.zeros(2)),))
    grid = radiative.wavenumber_grid(np.array([1000.0, 1001.0]))
    table = radiative.LayerCrossSections(
        "CO2",
        "all",
        "cross_section",
        grid,
        atm.temperature_k,
        atm.pressure_pa,
        np.ones((2, 2)),
        height_edges_m=atm.height_boundaries_m,
    )
    prepared = radiative.resample_layer_cross_sections(radiative.wavenumber_grid(grid), table)
    with pytest.raises(ValueError, match="read-only"):
        prepared.cross_sections_m2_per_molecule[0, 0] = bad
    invalid = prepared.cross_sections_m2_per_molecule.copy()
    invalid[0, 0] = bad
    with pytest.raises(ValueError, match="cross section"):
        replace(prepared, cross_sections_m2_per_molecule=invalid)
