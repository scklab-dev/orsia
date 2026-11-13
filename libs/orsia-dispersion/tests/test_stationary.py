"""Stationary physics through the common finite-support mass/moment contract."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import (
    StationaryGaussianPlume,
    concentration,
    layer_columns,
    pixel_layer_moments,
)
from scipy.special import ndtr
from test_fields import layered_cells


@pytest.mark.parametrize("temperature", [280.0, 300.0, 600.0, 1000.0])
def test_reflected_profile_briggs_rise_and_layer_partition(temperature: float) -> None:
    model = StationaryGaussianPlume(
        emission_rate_kg_s=1.0,
        wind_speed_m_s=5.0,
        source_regularization_m=20.0,
        stack_height_m=100.0,
        stack_inside_diameter_m=2.0,
        stack_exit_velocity_m_s=20.0,
        stack_exit_temperature_k=temperature,
        ambient_temperature_k=290.0,
    )
    east = np.array([0.0, 100.0, 1000.0, 3000.0])
    xy = np.stack((east, np.zeros_like(east)), axis=-1)
    edges = np.array([0.0, 90.0, 110.0, 200.0, 5000.0])
    distance = np.maximum(east, 20.0)
    buoyancy = 9.80665 * 20 * 2**2 * max(temperature - 290, 0) / (4 * temperature)
    final = (21.425 * buoyancy**0.75 if buoyancy < 55 else 38.71 * buoyancy**0.6) / 5
    centre = 100 + np.minimum(1.6 * buoyancy ** (1 / 3) * distance ** (2 / 3) / 5, final)
    sigma_y = 0.08 * distance / np.sqrt(1 + 0.0001 * distance)
    sigma_z = np.maximum(0.06 * distance / np.sqrt(1 + 0.0015 * distance), 1)
    expected = np.zeros((len(east), len(edges) - 1))
    for sign in (-1, 1):
        expected += np.diff(ndtr((edges[None, :] - sign * centre[:, None]) / sigma_z[:, None]))
    expected /= 5 * np.sqrt(2 * np.pi) * sigma_y[:, None]
    columns = layer_columns(model, xy, edges)
    assert_allclose(columns, expected, rtol=1e-12, atol=1e-15)
    assert_allclose(columns.sum(-1), 1 / (5 * np.sqrt(2 * np.pi) * sigma_y), rtol=1e-12)
    low = layer_columns(model, xy, edges[:2])
    assert np.all(low.sum(-1) < columns.sum(-1))  # Finite tops never renormalize.
    removed = replace(model, removal_timescale_s=100.0)
    assert_allclose(
        layer_columns(removed, xy, edges),
        columns * np.exp(-np.maximum(east - 20, 0) / 500)[:, None],
        atol=1e-15,
    )
    assert_array_equal(layer_columns(replace(model, emission_rate_kg_s=0), xy, edges), 0)
    assert_array_equal(concentration(model, np.array([[-1.0, 0.0, 100.0]])), 0)


def test_stationary_clipped_mass_and_first_moment_against_volume_quadrature() -> None:
    model = StationaryGaussianPlume(
        emission_rate_kg_s=1.0,
        wind_speed_m_s=5.0,
        source_regularization_m=20.0,
        stack_height_m=10.0,
        stack_inside_diameter_m=4.0,
        stack_exit_velocity_m_s=20.0,
        stack_exit_temperature_k=290.0,
        ambient_temperature_k=290.0,
    )
    lower, upper = np.array([[100.0, -10.0]]), np.array([[150.0, 20.0]])
    edges = np.array([0.0, 10.0, 30.0])
    terrain = np.array([7.0])
    result = pixel_layer_moments(model, layered_cells(lower, upper, edges, terrain))
    nodes, weights = np.polynomial.legendre.leggauss(32)
    for k, (a, b) in enumerate(((7.0, 10.0), (10.0, 30.0))):
        lo, hi = np.r_[lower[0], a], np.r_[upper[0], b]
        axes = (lo[:, None] + hi[:, None]) / 2 + (hi - lo)[:, None] / 2 * nodes
        x, y, z = np.meshgrid(*axes, indexing="ij")
        sy = 0.08 * x / np.sqrt(1 + 0.0001 * x)
        sz = np.maximum(0.06 * x / np.sqrt(1 + 0.0015 * x), 2)
        density = (
            np.exp(-0.5 * (y / sy) ** 2)
            / (5 * 2 * np.pi * sy * sz)
            * (np.exp(-0.5 * ((z - 10) / sz) ** 2) + np.exp(-0.5 * ((z + 10) / sz) ** 2))
        )
        w = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        w = w * np.prod((hi - lo) / 2) / 1500
        assert_allclose(result.columns_kg_m2[0, k], np.sum(density * w), rtol=1e-10)
        assert_allclose(result.height_moments_kg_m[0, k], np.sum(z * density * w), rtol=1e-10)
    whole = pixel_layer_moments(model, layered_cells(lower, upper, edges[[0, -1]], terrain))
    assert_allclose(result.columns_kg_m2.sum(-1), whole.columns_kg_m2[:, 0], rtol=1e-12)
    assert_allclose(result.height_moments_kg_m.sum(-1), whole.height_moments_kg_m[:, 0], rtol=1e-12)
