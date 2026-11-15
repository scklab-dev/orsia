"""Physical integration shared by direct and tangent photoelectron calculations."""

from math import fsum

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_instruments.photodetection.photoelectrons import (
    integrate_photoelectron_tangents,
    integrate_photoelectrons,
)


@pytest.mark.parametrize("shape", [(3, 7), (3, 4, 7), (3, 2, 4, 7), (0, 7), (3, 0)])
def test_direct_integration_axes_and_input_ownership(shape):
    rng = np.random.default_rng(42)
    coefficients = rng.uniform(0, 2, shape)
    radiance = rng.uniform(0, 3, shape)
    before = coefficients.copy(), radiance.copy()
    expected = np.array(
        [fsum((k * light).flat) for k, light in zip(coefficients, radiance, strict=True)]
    )
    values = integrate_photoelectrons(coefficients, radiance)
    assert values.shape == (shape[0],)
    assert values.dtype == np.float64
    assert_allclose(values, expected, rtol=2e-15, atol=0)
    assert not np.shares_memory(values, coefficients)
    assert not np.shares_memory(values, radiance)
    assert_array_equal(coefficients, before[0])
    assert_array_equal(radiance, before[1])


@pytest.mark.parametrize("directions", [0, 1, 3])
@pytest.mark.parametrize("with_response_tangent", [False, True])
def test_tangent_means_match_direct_on_noncontiguous_arrays(directions, with_response_tangent):
    rng = np.random.default_rng(42)
    coefficients = rng.uniform(0, 2, (3, 7, 4)).transpose(0, 2, 1)
    radiance = rng.uniform(0, 3, (3, 7, 4)).transpose(0, 2, 1)
    seeds = rng.normal(size=(*coefficients.shape, directions))
    response_seeds = rng.normal(size=seeds.shape) if with_response_tangent else None
    values, derivative = integrate_photoelectron_tangents(
        coefficients, radiance, seeds, response_seeds
    )
    assert_array_equal(values, integrate_photoelectrons(coefficients, radiance))
    assert_allclose(values, np.einsum("nqk,nqk->n", radiance, coefficients), rtol=2e-15)
    expected = np.sum(coefficients[..., None] * seeds, axis=(1, 2))
    if response_seeds is not None:
        expected += np.sum(response_seeds * radiance[..., None], axis=(1, 2))
    assert_array_equal(derivative, expected)
    assert derivative.shape == (3, directions)


@pytest.mark.parametrize(
    "coefficients,radiance",
    [
        (np.array(1.0), np.array(1.0)),
        (np.ones(3), np.ones(3)),
        (np.ones((2, 3)), np.ones(3)),
        (np.ones((2, 3)), np.ones((2, 1))),
        (-np.ones((2, 3)), np.ones((2, 3))),
        (np.ones((2, 3)), -np.ones((2, 3))),
        (np.full((2, 3), np.nan), np.ones((2, 3))),
        (np.full((2, 3), np.inf), np.ones((2, 3))),
        (np.ones((2, 3)), np.full((2, 3), np.nan)),
        (np.ones((2, 3)), np.full((2, 3), np.inf)),
    ],
)
def test_invalid_physical_inputs_rejected_by_both_paths(coefficients, radiance):
    with pytest.raises(ValueError):
        integrate_photoelectrons(coefficients, radiance)
    with pytest.raises(ValueError):
        integrate_photoelectron_tangents(coefficients, radiance, np.zeros((*coefficients.shape, 1)))


@pytest.mark.parametrize("radiance", [1.0, 2.0])
def test_nonfinite_products_or_sums_rejected(radiance):
    coefficients = np.full((2, 3), np.finfo(np.float64).max)
    light = np.full((2, 3), radiance)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite numerical domain"):
            integrate_photoelectrons(coefficients, light)
        with pytest.raises(ValueError, match="finite numerical domain"):
            integrate_photoelectron_tangents(coefficients, light, np.zeros((2, 3, 0)))


def test_zero_response_and_zero_radiance():
    zeros, ones = np.zeros((2, 3)), np.ones((2, 3))
    assert_array_equal(integrate_photoelectrons(zeros, ones), np.zeros(2))
    assert_array_equal(integrate_photoelectrons(ones, zeros), np.zeros(2))
