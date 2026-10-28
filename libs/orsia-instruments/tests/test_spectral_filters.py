"""Peak, width, edge and invalid-input contracts of supplied transmission laws."""

import numpy as np
import pytest
from orsia_instruments import (
    gaussian_filter,
    longpass_filter,
    lorentzian_filter,
    notch_filter,
    rectangular_filter,
    shortpass_filter,
    super_gaussian_filter,
)


@pytest.mark.parametrize("law", [lorentzian_filter, super_gaussian_filter])
def test_peak_fwhm_broadcast_and_ownership(law):
    centres = np.array([[2e-6], [3e-6]])
    wave = centres + np.array([-1e-8, 0, 1e-8])
    original = wave.copy()
    result = law(wave, centres, 2e-8, peak=0.8)
    # Subtracting micrometre centres to obtain nanometre offsets costs precision.
    np.testing.assert_allclose(result, [[0.4, 0.8, 0.4]] * 2, rtol=5e-14)
    np.testing.assert_array_equal(wave, original)
    assert result.flags.writeable and not np.shares_memory(wave, result)
    assert law(2e-6, 2e-6, 2e-8, peak=0) == 0


def test_profile_definitions_and_gaussian_limit():
    wave = np.linspace(1e-6, 3e-6, 31)
    np.testing.assert_allclose(
        lorentzian_filter(wave, 2e-6, 2e-7, peak=0.7),
        0.7 / (1 + 4 * ((wave - 2e-6) / 2e-7) ** 2),
    )
    np.testing.assert_allclose(
        super_gaussian_filter(wave, 2e-6, 2e-7, order=1),
        gaussian_filter(wave, 2e-6, 2e-7 / np.sqrt(8 * np.log(2))),
        rtol=6e-14,
    )
    for order in (1.5, 3, 8):
        np.testing.assert_allclose(
            super_gaussian_filter(wave, 2e-6, 2e-7, order=order),
            np.exp(-np.log(2) * np.abs((wave - 2e-6) / 1e-7) ** (2 * order)),
        )


def test_edges_levels_complement_and_transition():
    # 10%, 50%, 90% relative to 0.1--0.9, not absolute transmission.
    wave = np.array([1.9e-6, 2e-6, 2.1e-6])
    long = longpass_filter(wave, 2e-6, 2e-7, 0.9, 0.1)
    short = shortpass_filter(wave, 2e-6, 2e-7, 0.9, 0.1)
    np.testing.assert_allclose(long, [0.18, 0.5, 0.82])
    np.testing.assert_allclose(short, [0.82, 0.5, 0.18])
    np.testing.assert_allclose(long + short, 1)
    np.testing.assert_array_equal(longpass_filter(wave, 2e-6, 2e-7, 0.4, 0.4), 0.4)


def test_rectangular_closed_edges_and_notch_depth():
    wave = np.array([1e-6, 1.5e-6, 2e-6, 2.5e-6, 3e-6])
    np.testing.assert_array_equal(
        rectangular_filter(wave, 1.5e-6, 2.5e-6, 0.7), [0, 0.7, 0.7, 0.7, 0]
    )
    notch = notch_filter(wave, 2e-6, 1e-6, depth=0.75, pass_transmission=0.8)
    np.testing.assert_allclose(notch[1:4], [0.5, 0.2, 0.5])
    np.testing.assert_array_equal(notch_filter(wave, 2e-6, 1e-6, depth=0), 1)
    np.testing.assert_allclose(
        notch_filter(wave, 2e-6, 1e-6) + super_gaussian_filter(wave, 2e-6, 1e-6), 1
    )


@pytest.mark.parametrize(
    "law",
    [lorentzian_filter, super_gaussian_filter, longpass_filter, shortpass_filter, notch_filter],
)
def test_extreme_tails_are_finite_without_overflow_warnings(law):
    with np.errstate(all="raise", under="ignore"):
        result = law(np.array([1e-300, 2e-6, 1e300]), 2e-6, 1e-300)
    assert np.isfinite(result).all()
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.parametrize(
    "law",
    [
        lorentzian_filter,
        super_gaussian_filter,
        longpass_filter,
        shortpass_filter,
        notch_filter,
        rectangular_filter,
    ],
)
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_invalid_positive_coordinates(law, value, index):
    args = [2e-6, 1e-6, 3e-6]
    args[index] = value
    with pytest.raises(ValueError):
        law(*args)


@pytest.mark.parametrize(
    "law,kwargs",
    [
        (lorentzian_filter, {"peak": 1.1}),
        (super_gaussian_filter, {"peak": -0.1}),
        (super_gaussian_filter, {"order": 0.9}),
        (super_gaussian_filter, {"order": np.inf}),
        (notch_filter, {"depth": -0.1}),
        (notch_filter, {"depth": 1.1}),
        (notch_filter, {"pass_transmission": 1.1}),
        (longpass_filter, {"stop_transmission": 0.8, "pass_transmission": 0.7}),
        (shortpass_filter, {"stop_transmission": -0.1}),
        (longpass_filter, {"pass_transmission": np.nan}),
        (rectangular_filter, {"peak": np.inf}),
    ],
)
def test_invalid_profile_levels(law, kwargs):
    with pytest.raises(ValueError):
        law(2e-6, 1e-6, 3e-6, **kwargs)


@pytest.mark.parametrize("upper", [1e-6, 0.9e-6])
def test_rectangular_rejects_unordered_bounds(upper):
    with pytest.raises(ValueError, match="strictly ordered"):
        rectangular_filter(2e-6, 1e-6, upper)
