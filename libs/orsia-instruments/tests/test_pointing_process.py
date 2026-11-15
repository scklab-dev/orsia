"""Stationary covariance of the explicit alternative-law pointing process."""

import numpy as np
import pytest
from orsia_instruments.pointing import smooth_pointing_process


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "student_t"])
def test_smooth_pointing_state_space_covariance(distribution: str) -> None:
    covariance = np.array([[1.0, 0.3], [0.3, 2.0]]) * 1e-12
    rate, period = 3.0, 0.1
    draws = smooth_pointing_process(
        rate,
        covariance,
        period,
        100000,
        np.random.default_rng(42),
        distribution=distribution,
        degrees=7.0,
    )
    np.testing.assert_allclose(np.cov(draws.T), covariance, rtol=0.08, atol=1e-14)
    lag = 5
    expected = (1 + rate * period * lag) * np.exp(-rate * period * lag)
    assert np.corrcoef(draws[:-lag, 0], draws[lag:, 0])[0, 1] == pytest.approx(expected, abs=0.025)
