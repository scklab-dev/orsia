"""Independent optical-depth integration controls for thin thermal layers."""

import numpy as np
from orsia_radiative.transfer.thermal import _e3_difference
from scipy.integrate import quad
from scipy.special import expn


def test_e3_intervals_against_adaptive_depth_integral():
    # Includes the zero-boundary logarithmic derivative, switch neighbourhood,
    # intervals below the spacing of a, and near-opaque representable values.
    below, depth = np.meshgrid(
        np.array([0.0, 1e-16, 1e-8, 1e-3, 0.1, 1.0, 10.0, 100.0, 700.0]),
        np.logspace(-18, 2, 41),
    )
    below, depth = below.ravel(), depth.ravel()
    expected = np.array(
        [
            d
            * expn(2, a)
            * quad(
                lambda u, a=a, d=d: expn(2, a + d * u) / expn(2, a),
                0,
                1,
                epsabs=2e-13,
                epsrel=2e-13,
            )[0]
            for a, d in zip(below, depth, strict=True)
        ]
    )
    actual = _e3_difference(below, depth)
    # Near underflow SciPy's E2/E3 accuracy is absolute, not relative.
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-300)
    assert np.all(actual >= 0)


def test_e3_zero_and_opaque_intervals():
    actual = _e3_difference(
        np.array([0.0, 1.0, 1e308, np.inf]),
        np.array([0.0, 0.0, 1e308, 1.0]),
    )
    np.testing.assert_array_equal(actual, np.zeros(4))
