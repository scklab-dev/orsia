"""Independent references and limits for the adopted SciPy-backed methods."""

from math import exp, expm1

import numpy as np
import pytest
from orsia_radiative.transfer import formal, orders, thermal
from scipy.integrate import quad


def test_exponential_weights_cover_vacuum_thin_and_opaque_limits() -> None:
    attenuation = np.array([0.0, 1e-20, 1e-10, 0.01, 1.0, 700.0, 1e300])
    expected = np.ones_like(attenuation)
    np.divide(-np.expm1(-attenuation), attenuation, out=expected, where=attenuation != 0)
    np.testing.assert_allclose(formal._constant_source_weight(attenuation), expected, rtol=3e-16)
    a = np.array([0.0, -1e-20, -1.0, -700.0, -1e300])
    b = np.array([-1e-20, 0.0, -1.0 - 1e-12, -710.0, -1e300])
    actual = formal.integrated_exponential(a, b)
    reference = np.array(
        [
            quad(lambda t, x=x, y=y: exp((1 - t) * x + t * y), 0, 1, epsabs=1e-14)[0]
            for x, y in zip(a, b, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-14, atol=0)
    np.testing.assert_array_equal(actual, formal.integrated_exponential(b, a))
    assert actual.flags.writeable and not np.shares_memory(actual, a)


@pytest.mark.parametrize("order", [2, 8, 32, 64])
def test_angular_quadrature_normalization_and_hemisphere_moments(order: int) -> None:
    directions, weights = orders.angular_grid(order)
    mu = directions[:, 2]
    assert np.all(weights > 0) and np.all(np.abs(mu) < 1)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, rtol=3e-16)
    for power in range(min(2 * order, 8)):
        expected = 1 / (power + 1) if power % 2 == 0 else 0
        assert np.sum(weights * mu**power) == pytest.approx(expected, abs=3e-15)


@pytest.mark.parametrize("below", [0.0, 1e-12, 0.1, 10.0, 100.0])
def test_thermal_e3_differences_do_not_cancel_thin_layers(below: float) -> None:
    depths = np.array([0.0, 1e-20, 1e-16, 1e-8, 1e-3, 0.1, 5.0])
    actual = thermal._e3_difference(np.full(depths.shape, below), depths)
    for value, depth in zip(actual[1:], depths[1:], strict=True):
        # Independent angular integral scaled by d and exp(-a). It never
        # subtracts E3 values or integrates E2 in optical-depth coordinates.
        reference = (
            depth
            * exp(-below)
            * quad(
                lambda mu, depth=depth: (
                    mu * exp(-below * (1 / mu - 1)) * -expm1(-depth / mu) / depth
                ),
                0,
                1,
                epsabs=2e-13,
                epsrel=2e-13,
            )[0]
        )
        assert value > 0
        assert value == pytest.approx(reference, rel=3e-12, abs=0)
    assert actual[0] == 0
    np.testing.assert_array_equal(
        thermal._e3_difference(np.array([1000.0, np.inf, 0.0]), np.array([1.0, 1.0, 1e308])),
        [0.0, 0.0, 0.5],
    )
