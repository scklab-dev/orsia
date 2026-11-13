"""Layer interfaces, immutable coefficient ownership and reciprocal resampling."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_radiative import (
    Absorber,
    LayerAtmosphere,
    LayerCrossSections,
    SpectralDomain,
    SpectralGrid,
    convert_spectral_grid,
    evaluate_table_absorption,
    resample_layer_cross_sections,
    wavenumber_grid,
)
from orsia_radiative.spectral.interpolation import spectral_interpolation_weights


def table(grid=None, values=None):
    spectral = wavenumber_grid(np.array([999.13, 1000.37, 1002.89])) if grid is None else grid
    coefficients = np.array([[1.0, 3.0, 2.0]]) if values is None else values
    return LayerCrossSections(
        "CH4",
        "all",
        "cross_section",
        spectral,
        np.array([290.0]),
        np.array([95000.0]),
        coefficients,
        height_edges_m=np.array([0.0, 1000.0]),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_owned_coefficient_dtype_and_source_lifetime(dtype):
    values = np.array([[1, 3, 2]], dtype=dtype)
    inputs = table(values=values)
    sampled = resample_layer_cross_sections(inputs.grid, inputs)
    assert sampled.cross_sections_m2_per_molecule.dtype == np.float64
    assert inputs.cross_sections_m2_per_molecule.dtype == dtype
    assert not hasattr(sampled, "inputs")
    values.fill(99)
    assert_array_equal(inputs.cross_sections_m2_per_molecule, [[1, 3, 2]])
    for name in (
        "temperature_k",
        "pressure_pa",
        "height_edges_m",
        "cross_sections_m2_per_molecule",
    ):
        a, b = getattr(inputs, name), getattr(sampled, name)
        assert not np.shares_memory(a, b)
        assert not a.flags.writeable and not b.flags.writeable
        with pytest.raises(ValueError):
            b.setflags(write=True)


def test_singleton_values_remain_a_valid_table_but_cannot_extrapolate():
    inputs = table(grid=wavenumber_grid(np.array([1000.0])), values=np.array([[2.0]]))
    sampled = resample_layer_cross_sections(inputs.grid, inputs)
    assert_array_equal(sampled.cross_sections_m2_per_molecule, [[2.0]])
    assert_array_equal(
        resample_layer_cross_sections(
            convert_spectral_grid(inputs.grid), inputs
        ).cross_sections_m2_per_molecule,
        [[2.0]],
    )
    with pytest.raises(ValueError, match="extrapolation"):
        resample_layer_cross_sections(wavenumber_grid(np.array([1000.1])), inputs)


def test_reciprocal_nodes_are_exact_and_interpolation_uses_source_coordinate():
    inputs = table()
    wavelength = convert_spectral_grid(inputs.grid)
    sampled = resample_layer_cross_sections(wavelength, inputs)
    assert sampled.grid is wavelength
    assert_array_equal(
        sampled.cross_sections_m2_per_molecule, inputs.cross_sections_m2_per_molecule[:, ::-1]
    )
    for index in range(wavelength.points.size):
        singleton = SpectralGrid("wavelength_nm", wavelength.points[index : index + 1])
        result = resample_layer_cross_sections(singleton, inputs)
        assert_array_equal(
            result.cross_sections_m2_per_molecule,
            sampled.cross_sections_m2_per_molecule[:, index : index + 1],
        )
    with pytest.raises(ValueError, match="wavenumber"):
        _ = sampled.wavenumbers_cm1
    target = SpectralGrid("wavelength_nm", np.array([1e7 / 1001.5, 1e7 / 999.5]))
    expected = np.interp(
        1e7 / target.points, inputs.grid.points, inputs.cross_sections_m2_per_molecule[0]
    )
    assert_allclose(
        resample_layer_cross_sections(target, inputs).cross_sections_m2_per_molecule[0], expected
    )
    wavelength_table = table(
        SpectralGrid("wavelength_nm", np.array([500.0, 1000.0])), np.array([[1.0, 2.0]])
    )
    result = resample_layer_cross_sections(wavenumber_grid(np.array([15000.0])), wavelength_table)
    assert_allclose(result.cross_sections_m2_per_molecule, [[4 / 3]])


def test_wavelength_stencils_reject_gaps_and_mismatched_coordinates():
    grid = SpectralGrid(
        "wavelength_nm",
        np.array([500.0, 501.0, 510.0, 511.0]),
        SpectralDomain("wavelength_nm", ((500.0, 501.0), (510.0, 511.0))),
    )
    inputs = table(grid, np.ones((1, 4)))
    for wavelength in (499.0, 505.0, 512.0):
        target = SpectralGrid("wavelength_nm", np.array([wavelength]))
        with pytest.raises(ValueError, match="gap|extrapolation"):
            resample_layer_cross_sections(target, inputs)
        with pytest.raises(ValueError, match="gap|extrapolation"):
            resample_layer_cross_sections(convert_spectral_grid(target), inputs)
    with pytest.raises(ValueError, match="coordinates"):
        spectral_interpolation_weights(grid, convert_spectral_grid(grid))


def test_exact_vertical_support_and_conditions_precede_column_application():
    inputs = table()
    atmosphere = LayerAtmosphere(
        np.array([0.0, 1000.0]),
        np.array([1e5, 9e4]),
        np.array([95000.0]),
        np.array([290.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
        (Absorber("CH4", np.array([3.0]), "dry_air"),),
    )
    assert_array_equal(
        evaluate_table_absorption(atmosphere, inputs).optical_depth, [[3.0, 9.0, 6.0]]
    )
    for changes in (
        {"height_edges_m": inputs.height_edges_m + 1e-8},
        {"temperature_k": inputs.temperature_k + 1e-8},
        {"pressure_pa": inputs.pressure_pa + 1e-8},
    ):
        with pytest.raises(ValueError, match="match atmosphere exactly"):
            evaluate_table_absorption(atmosphere, replace(inputs, **cast(dict[str, Any], changes)))
    for edges in (np.array([0.0]), np.array([1.0, 0.0]), np.array([0.0, np.inf])):
        with pytest.raises(ValueError, match="interfaces"):
            replace(inputs, height_edges_m=edges)


def test_resampling_preserves_known_gaps_for_later_use():
    domain = SpectralDomain("wavelength_nm", ((500.0, 501.0), (510.0, 511.0)))
    grid = SpectralGrid("wavelength_nm", np.array([500.0, 501.0, 510.0, 511.0]), domain)
    inputs = table(grid, np.ones((1, 4)))
    requested = SpectralGrid("wavelength_nm", grid.points)
    sampled = resample_layer_cross_sections(requested, inputs)
    assert sampled.grid.domain == domain
    with pytest.raises(ValueError, match="gap"):
        resample_layer_cross_sections(SpectralGrid("wavelength_nm", np.array([505.0])), sampled)
    expanded = SpectralGrid(
        "wavelength_nm", grid.points, SpectralDomain("wavelength_nm", ((500.0, 511.0),))
    )
    with pytest.raises(ValueError, match="coverage"):
        resample_layer_cross_sections(expanded, inputs)
    narrower = SpectralGrid(
        "wavelength_nm", np.array([500.0]), SpectralDomain("wavelength_nm", ((500.0, 500.0),))
    )
    assert resample_layer_cross_sections(narrower, inputs).grid is narrower
    reciprocal = convert_spectral_grid(requested)
    assert (
        resample_layer_cross_sections(reciprocal, inputs).grid.domain
        == convert_spectral_grid(grid).domain
    )
