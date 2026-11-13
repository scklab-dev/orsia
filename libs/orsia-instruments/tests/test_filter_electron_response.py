"""Per-read filter response, retained numerical laws and explicit coaddition maps."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_instruments import ReadSchedule
from orsia_instruments.optics.filters import FilterPassband, tabulated_transmission
from orsia_instruments.optics.pixels import square_pixel_etendue
from orsia_instruments.response.spectral import (
    FilterElectronResponse,
    SpectralElectronResponse,
    spectral_electrons,
)
from orsia_radiative import SpectralDomain, SpectralGrid


def grid(wave=None):
    points = np.array([496.0, 498.0, 499.0, 501.0, 504.0, 506.0, 509.0]) if wave is None else wave
    return SpectralGrid(
        "wavelength_nm", points, SpectralDomain("wavelength_nm", ((points[0], points[-1]),))
    )


def schedule(settings=("a", "a", "b", "b", "b"), duration=0.02):
    # Explicit synthetic independent simultaneous reads; identities remain distinct.
    n = len(settings)
    return ReadSchedule(
        tuple(f"r{i}" for i in range(n)),
        tuple(f"p{i}" for i in range(n)),
        settings,
        ("direct",) * n,
        np.column_stack((np.zeros(n), np.broadcast_to(duration, (n,)))),
    )


def passbands(profile: Literal["lorentzian", "supergaussian"] = "lorentzian", exponent=None):
    return {
        "a": FilterPassband(500.0, 2.0, profile, 0.9, exponent),
        "b": FilterPassband(504.0, 3.0, profile, 0.9, exponent),
    }


def preparation(spectral=None, reads=None, filters=None, **changes):
    spectral = grid() if spectral is None else spectral
    wave = spectral.points if spectral.coordinate == "wavelength_nm" else 1e7 / spectral.points
    options: dict[str, Any] = dict(
        accepted_etendue_m2_sr=square_pixel_etendue(15e-6, 2.5, 3.0),
        quantum_efficiency=0.7,
        transmission=0.8 * ((wave >= 498) & (wave <= 506)),
    )
    return FilterElectronResponse(
        spectral,
        schedule() if reads is None else reads,
        passbands() if filters is None else filters,
        **(options | changes),
    )


def test_retained_per_nm_formula_and_explicit_native_coaddition():
    response = preparation()
    nominal = response.coefficients()
    wave = response.grid.points
    quadrature = np.array([1.0, 1.5, 1.5, 2.5, 2.5, 2.5, 1.5])
    profile = 1 / (
        1
        + (2 * (wave[None, :] - np.array([500.0, 504.0])[:, None]) / np.array([2.0, 3.0])[:, None])
        ** 2
    )
    assert_allclose(response.profile_areas_nm(), profile @ quadrature, rtol=1e-13)
    photons = 0.7 * wave * 1e-9 / (6.62607015e-34 * 299792458.0)
    photons *= (wave >= 498) & (wave <= 506)
    scale = 0.8 * 0.9 * (15e-6) ** 2 * np.pi / (4 * 3**2) * 0.02
    one_read = profile * quadrature * photons * scale
    assert_allclose(nominal.coefficients, one_read[[0, 0, 1, 1, 1]], rtol=1e-13)
    assert np.all(nominal.coefficients[:, [0, -1]] == 0)
    assert np.all(nominal.coefficients[:, [1, -2]] > 0)
    assert nominal.reads is response.reads and nominal.grid is response.grid
    assert not nominal.coefficients.flags.writeable
    with pytest.raises(ValueError):
        nominal.coefficients.setflags(write=True)
    accumulation = np.array(
        [[name == key for name in nominal.reads.setting_ids] for key in response.filter_ids],
        dtype=float,
    )
    assert_allclose(
        accumulation @ nominal.coefficients, one_read * np.array([2, 3])[:, None], rtol=1e-13
    )
    signal = spectral_electrons(nominal, np.ones(wave.size))
    summed = accumulation @ signal.total
    mean_map = accumulation / accumulation.sum(axis=1, keepdims=True)
    assert_allclose(mean_map @ signal.total, summed / [2, 3])
    # Each actual read contributes its own shot and read-noise variance.
    covariance = np.diag(signal.total + 9.0)
    assert_allclose(
        np.diag(accumulation @ covariance @ accumulation.T), summed + np.array([2, 3]) * 9
    )
    assert_allclose(
        np.diag(mean_map @ covariance @ mean_map.T),
        (summed + np.array([2, 3]) * 9) / np.array([2, 3]) ** 2,
    )
    doubled = preparation(reads=schedule(duration=0.04))
    assert_array_equal(doubled.coefficients().coefficients, 2 * nominal.coefficients)
    limited = preparation(accepted_etendue_m2_sr=square_pixel_etendue(15e-6, 6.0, 3.0))
    assert_array_equal(limited.coefficients().coefficients, nominal.coefficients / 4)
    assert_array_equal(preparation(transmission=0).coefficients().coefficients, 0)


@pytest.mark.parametrize("exponent", [1.5, 2.0, 3.5, 4.0])
def test_profiles_area_and_fixed_area_directions(exponent):
    response = preparation(
        grid(np.linspace(496.0, 509.0, 100)), filters=passbands("supergaussian", exponent)
    )
    centres, widths = np.array([500.0, 504.0]), np.array([2.0, 3.0])
    nominal, area = response.coefficients(), response.profile_areas_nm()
    unscaled = response.coefficients(centres_nm=centres + 0.1)
    trial_area = response.profile_areas_nm(centres_nm=centres + 0.1)
    trial = response.coefficients(centres_nm=centres + 0.1, fixed_area_nm=area)
    ratios = (area / trial_area)[[0, 0, 1, 1, 1]]
    assert_allclose(
        trial.coefficients, unscaled.coefficients * ratios[:, None], rtol=8 * np.finfo(float).eps
    )
    base, dc, dw = response.fixed_area_filter_sensitivities(0.01)
    assert_array_equal(base.coefficients, nominal.coefficients)
    for actual, key, values in ((dc, "centres_nm", centres), (dw, "fwhm_nm", widths)):
        plus = response.coefficients(**{key: values + 0.01}, fixed_area_nm=area).coefficients
        minus = response.coefficients(**{key: values - 0.01}, fixed_area_nm=area).coefficients
        assert_array_equal(actual, (plus - minus) / 0.02)
    assert dc.flags.writeable and dw.flags.writeable


def test_read_order_duration_and_independent_factors():
    reads = schedule(("b", "a", "b"), np.array([0.01, 0.02, 0.04]))
    etendue = square_pixel_etendue(15e-6, 2.5, 3.0)
    model = preparation(
        reads=reads,
        quantum_efficiency=np.array([[0.3], [0.5], [0.7]]),
        accepted_etendue_m2_sr=etendue * np.array([1.0, 2.0, 3.0]),
    )
    assert model.filter_ids == ("b", "a")
    assert [p.centre_nm for p in model.passbands] == [504.0, 500.0]
    response = model.coefficients()
    assert response.reads.sample_ids == ("r0", "r1", "r2")
    assert_allclose(
        response.coefficients[2], response.coefficients[0] * 3 * 4 * 0.7 / 0.3, rtol=3e-15
    )
    source = np.broadcast_to(np.arange(1.0, 8.0), (3, 7)).copy()
    signal = spectral_electrons(response, source)
    assert_allclose(signal.total, np.einsum("rk,rk->r", response.coefficients, source), rtol=2e-15)


def test_qe_sampled_once_snapshots_and_two_worker_reuse():
    wave = np.linspace(496.0, 509.0, 80)
    spectral = grid(wave)
    qe_wave, qe_values = np.array([490.0, 510.0]) * 1e-9, np.array([0.2, 0.8])
    calls = []

    def qe(indices, wavelengths):
        calls.append(1)
        values = tabulated_transmission(
            wavelengths, nodes_m=qe_wave, fractions=qe_values, outside="error"
        )
        return np.broadcast_to(values, (len(indices), len(values)))

    filters = passbands()
    transmission = np.full(80, 0.8)
    response = preparation(
        spectral, filters=filters, quantum_efficiency=qe, transmission=transmission
    )
    before = response.fixed_area_filter_sensitivities()
    assert len(calls) == 1
    wave[:] = 1
    qe_values[:] = 0
    qe_wave[:] = 1
    transmission[:] = 0
    filters["a"] = replace(filters["a"], centre_nm=1000.0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: response.fixed_area_filter_sensitivities(), range(4)))
    for current in outputs:
        assert_array_equal(current[0].coefficients, before[0].coefficients)
        assert not np.shares_memory(current[0].coefficients, before[0].coefficients)
        for actual, expected in zip(current[1:], before[1:], strict=True):
            assert_array_equal(actual, expected)
            assert not np.shares_memory(actual, expected)
    assert len(calls) == 1


def test_coordinate_conversion_preserves_point_profile_area():
    from orsia_radiative.spectral.grid import convert_spectral_grid

    nm = grid(np.linspace(496.0, 509.0, 101))
    model = preparation(nm)
    cm = preparation(convert_spectral_grid(nm))
    assert_allclose(cm.profile_areas_nm(), model.profile_areas_nm(), rtol=1e-13)
    # Each coordinate uses its own trapezoid measure. Refine to compare integrals.
    errors = []
    for n in (65, 129, 257):
        nm = grid(np.linspace(496.0, 509.0, n))
        a = preparation(nm, transmission=1).coefficients()
        b = preparation(convert_spectral_grid(nm), transmission=1).coefficients()
        la = np.ones(n)
        lb = nm.points[::-1] ** 2 / 1e7
        errors.append(
            np.max(np.abs(spectral_electrons(a, la).total / spectral_electrons(b, lb).total - 1))
        )
    assert errors[-1] < errors[0] and errors[-1] < 1e-7


@pytest.mark.parametrize(
    "change",
    [
        {"centre_nm": 0.0},
        {"fwhm_nm": np.nan},
        {"profile": "unknown"},
        {"peak_transmission": 1.1},
        {"profile": "supergaussian"},
        {"exponent": 2.0},
    ],
)
def test_invalid_passbands(change):
    with pytest.raises(ValueError):
        replace(passbands()["a"], **change)


@pytest.mark.parametrize("exponent", [0.0, -1.0, np.nan, np.inf])
def test_invalid_supergaussian_exponents(exponent):
    with pytest.raises(ValueError):
        FilterPassband(500.0, 2.0, "supergaussian", exponent=exponent)


@pytest.mark.parametrize(
    "change",
    [
        {"quantum_efficiency": -1.0},
        {"quantum_efficiency": np.nan},
        {"transmission": 1.1},
        {"accepted_etendue_m2_sr": -1.0},
        {"accepted_etendue_m2_sr": [1.0, 2.0]},
        {"quantum_efficiency": np.ones(2)},
    ],
)
def test_invalid_factor_inputs(change):
    with pytest.raises(ValueError):
        preparation(**change)


def test_missing_filter_qe_shape_and_qe_support():
    with pytest.raises(ValueError, match="setting"):
        preparation(filters={})
    with pytest.raises(ValueError, match="QE callback"):
        preparation(quantum_efficiency=lambda ids, wave: np.ones(1))

    def qe(ids, wave):
        return np.broadcast_to(
            tabulated_transmission(
                wave,
                nodes_m=np.array([500.0, 510.0]) * 1e-9,
                fractions=np.array([0.2, 0.8]),
                outside="error",
            ),
            (len(ids), len(wave)),
        )

    with pytest.raises(ValueError):
        preparation(quantum_efficiency=qe)
    with pytest.raises(ValueError, match="coverage"):
        preparation(SpectralGrid("wavelength_nm", np.array([500.0, 501.0])))
    with pytest.raises(ValueError, match="at least two"):
        preparation(
            SpectralGrid(
                "wavelength_nm",
                np.array([500.0]),
                SpectralDomain("wavelength_nm", ((500.0, 500.0),)),
            )
        )


@pytest.mark.parametrize("step", [0.0, -0.1, 1e-20, 2.0, np.nan, np.inf])
def test_invalid_difference_steps(step):
    with pytest.raises(ValueError, match="step"):
        preparation().fixed_area_filter_sensitivities(step)


@pytest.mark.parametrize(
    "trial",
    [
        {"centres_nm": [500.0]},
        {"centres_nm": [0.0, 500.0]},
        {"fwhm_nm": [0.0, 2.0]},
        {"fixed_area_nm": [1.0]},
        {"fixed_area_nm": [1.0, -1.0]},
        {"fixed_area_nm": [np.nan, 1.0]},
    ],
)
def test_invalid_trials(trial):
    with pytest.raises(ValueError):
        preparation().coefficients(**trial)


def test_zero_sampled_area_rejects_normalization():
    model = preparation(filters=passbands("supergaussian", 4.0))
    with pytest.raises(ValueError, match="sampled trial areas"):
        model.coefficients(centres_nm=[1000.0, 1005.0], fixed_area_nm=[1.0, 1.0])


def test_response_value_domains_and_square_pixel_inputs():
    nominal = preparation().coefficients()
    for values in (
        np.zeros((2, 2)),
        -nominal.coefficients,
        np.full(nominal.coefficients.shape, np.nan),
    ):
        with pytest.raises(ValueError):
            replace(nominal, coefficients=values)
    for values in (np.ones(2), -np.ones(7), np.full(7, np.nan)):
        with pytest.raises(ValueError):
            spectral_electrons(nominal, values)
    for inputs in ((0.0, 2.0, 3.0), (1.0, 0.0, 3.0), (1.0, 2.0, np.inf), (1e-300, 2.0, 3.0)):
        with pytest.raises(ValueError):
            square_pixel_etendue(*inputs)
    single = SpectralElectronResponse(
        nominal.reads, SpectralGrid("wavelength_nm", np.array([500.0])), np.ones((5, 1))
    )
    assert_array_equal(spectral_electrons(single, np.array([2.0])).total, 2)
