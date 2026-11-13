"""Independent exposure laws, caller-owned band composition and local derivatives."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_radiative.transfer.extinction import (
    exposure_mean_radiance,
    exposure_mean_radiance_tangent,
    linearize_exposure_mean_radiance,
)


@pytest.mark.parametrize("mode", ["exposure_mean", "mean_air_mass"])
def test_caller_coordinates_and_band_integration(mode):
    wave = np.array([1600.0, 1601.0, 1602.0, 1603.0])
    solar = np.array([1.0, 1.3, 1.1, 0.9])
    views = np.array([0.0, 20.0, 35.0])
    weights = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 3.0, 2.0]])
    ranges = [np.array([0, 1]), np.array([1, 2])]
    tau = np.array([0.18, 0.21, 0.25, 0.22])
    absorber = np.array([0.04, 0.06, 0.08, 0.05])
    mu = np.cos(np.deg2rad(30.0))
    masses = 1 / mu + 1 / np.cos(np.deg2rad(views))
    x = (wave - 1601.5) / 1.5

    def evaluate(q):
        # Parameter binding and spectral band integration belong to this caller.
        depth = tau + np.expm1(q[0]) * absorber
        source = mu / np.pi * solar * (1 + q[1] * x)
        ds = np.stack((np.zeros_like(source), mu / np.pi * solar * x), axis=-1)
        dt = np.stack((np.exp(q[0]) * absorber, np.zeros_like(source)), axis=-1)
        values, derivatives = [], []
        for indices, band in zip(ranges, weights, strict=True):
            selected = masses[indices] if mode == "exposure_mean" else masses[indices].mean()[None]
            value, tangent = exposure_mean_radiance_tangent(
                source,
                depth,
                selected,
                np.ones(selected.size),
                source_radiance_tangent=ds,
                optical_depth_tangent=dt,
            )
            assert_allclose(
                value,
                exposure_mean_radiance(source, depth, selected, np.ones(selected.size)),
                rtol=2e-15,
            )
            values.append(band @ value / band.sum())
            derivatives.append(band @ tangent / band.sum())
        return np.array(values), np.array(derivatives)

    state = np.array([0.12, -0.06])
    radiance, jac = evaluate(state)
    expected = []
    for indices, band in zip(ranges, weights, strict=True):
        selected = masses[indices] if mode == "exposure_mean" else masses[indices].mean()[None]
        value = (
            mu
            / np.pi
            * solar
            * (1 + state[1] * x)
            * np.exp(-selected[:, None] * (tau + np.expm1(state[0]) * absorber)).mean(axis=0)
        )
        expected.append(band @ value / band.sum())
    assert_allclose(radiance, expected, rtol=2e-15)
    for direction in np.eye(2):
        numeric = (
            evaluate(state + 1e-6 * direction)[0] - evaluate(state - 1e-6 * direction)[0]
        ) / 2e-6
        assert_allclose(jac @ direction, numeric, rtol=1e-8, atol=1e-11)


def test_log_tangent_reference_slope_and_off_reference_difference():
    source, tau, masses, weights = (
        np.array([1.0, 0.0, 3.0]),
        np.full(3, 0.2),
        np.array([2.0, 4.0]),
        np.array([1.0, 3.0]),
    )
    value, intercept, effective = linearize_exposure_mean_radiance(source, tau, masses, weights)
    expected = source * (np.exp(-masses[:, None] * tau) * (weights / weights.sum())[:, None]).sum(
        axis=0
    )
    assert_allclose(value, expected)
    assert_allclose(np.exp(intercept - effective * tau), value)
    assert np.isneginf(intercept[1])
    finite_difference = (
        exposure_mean_radiance(source, tau + 1e-6, masses, weights)
        - exposure_mean_radiance(source, tau - 1e-6, masses, weights)
    ) / 2e-6
    assert_allclose(-effective * value, finite_difference, atol=1e-10)
    off = tau + 0.6
    exact = exposure_mean_radiance(source, off, masses, weights)
    local = np.exp(intercept - effective * off)
    assert np.max(np.abs(exact - local)) > 0.01
    mean_mass = exposure_mean_radiance(
        source, off, np.array([weights @ masses / weights.sum()]), np.ones(1)
    )
    assert np.max(np.abs(exact - mean_mass)) > 0.01


def test_weights_dark_samples_opaque_limit_and_concurrent_outputs():
    source, tau = np.array([0.0, 2.0, 3.0]), np.array([0.0, 0.1, 1e308])
    masses, weights = np.array([2.0, 4.0, 9.0]), np.array([1e308, 1e308, 0.0])
    baseline = exposure_mean_radiance(source, tau, masses, weights)
    assert_allclose(baseline, [0.0, np.exp(-0.2) + np.exp(-0.4), 0.0])
    value, logsource, effective = linearize_exposure_mean_radiance(source, tau, masses, weights)
    assert_array_equal(value, baseline)
    assert effective[-1] == 0 and np.isneginf(logsource[-1])
    value, tangent = exposure_mean_radiance_tangent(
        source,
        tau,
        masses,
        weights,
        source_radiance_tangent=np.ones((3, 1)),
        optical_depth_tangent=np.zeros((3, 1)),
    )
    assert tangent[0, 0] == 1
    empty = exposure_mean_radiance_tangent(
        source,
        tau,
        masses,
        weights,
        source_radiance_tangent=np.empty((3, 0)),
        optical_depth_tangent=np.empty((3, 0)),
    )
    assert empty[1].shape == (3, 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(
            pool.map(lambda _: exposure_mean_radiance(source, tau, masses, weights), range(4))
        )
    for result in outputs:
        assert_array_equal(result, baseline)
        assert not np.shares_memory(result, baseline)
    assert_array_equal(source, [0.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "which,bad",
    [
        (0, np.array([-1.0])),
        (0, np.array([np.nan])),
        (0, np.empty(0)),
        (1, np.array([-1.0])),
        (1, np.ones(2)),
        (2, np.array([0.0])),
        (2, np.array([np.inf])),
        (2, np.empty(0)),
        (3, np.array([0.0])),
        (3, np.array([-1.0])),
        (3, np.ones(2)),
    ],
)
def test_invalid_physical_inputs(which, bad):
    args = [np.ones(1), np.ones(1), np.ones(1), np.ones(1)]
    args[which] = bad
    with pytest.raises(ValueError):
        exposure_mean_radiance(*args)


@pytest.mark.parametrize("seed", [np.ones(1), np.ones((2, 1)), np.full((1, 1), np.nan)])
def test_invalid_tangent(seed):
    with pytest.raises(ValueError, match="tangent"):
        exposure_mean_radiance_tangent(
            *[np.ones(1)] * 4, source_radiance_tangent=seed, optical_depth_tangent=np.zeros((1, 1))
        )


def test_tangent_overflow_fails():
    with pytest.raises(ValueError, match="arithmetic"):
        exposure_mean_radiance_tangent(
            np.array([1e308]),
            np.zeros(1),
            np.array([2.0]),
            np.ones(1),
            source_radiance_tangent=np.zeros((1, 1)),
            optical_depth_tangent=np.full((1, 1), 1e308),
        )


def test_tangent_retains_underflowed_transmission_and_signed_cancellation():
    source = np.array([0.0, 1e308, 1e308])
    depth = np.full(3, 1000.0)
    ds = np.array([[1e308], [0.0], [1e308]])
    dt = np.array([[0.0], [1e308], [1.0]])
    values, tangent = exposure_mean_radiance_tangent(
        source,
        depth,
        np.ones(1),
        np.ones(1),
        source_radiance_tangent=ds,
        optical_depth_tangent=dt,
    )
    np.testing.assert_allclose(
        tangent[:, 0],
        [np.exp(np.log(1e308) - 1000), -np.exp(2 * np.log(1e308) - 1000), 0.0],
        rtol=3e-13,
        atol=0,
    )
    assert values[0] == 0 and values[1] > 0


def test_extreme_weight_ratio_retains_representable_source_product():
    value = exposure_mean_radiance(
        np.array([1e308]), np.ones(1), np.array([2000.0, 1.0]), np.array([1e308, 1e-300])
    )
    assert_allclose(value, [1e-300 * np.exp(-1.0)], rtol=3e-13, atol=0)
