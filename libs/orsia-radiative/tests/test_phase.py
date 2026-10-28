"""Shared phase laws: physical support, broadcasting, ownership and SOS dispatch."""

from math import pi

import numpy as np
import pytest
from orsia_radiative import henyey_greenstein_phase, rayleigh_phase
from orsia_radiative.optics import phases as optical
from orsia_radiative.transfer import orders


def test_transfer_reuses_the_optical_evaluators() -> None:
    assert orders.henyey_greenstein_phase is optical.henyey_greenstein_phase
    assert orders.rayleigh_phase is optical.rayleigh_phase


@pytest.mark.parametrize("g", [-0.8, -0.3, 0.0, 0.6, 0.8])
def test_hg_scalar_and_array_match_independent_formula(g: float) -> None:
    cosines = np.array([-1.0, -0.4, 0.0, 0.7, 1.0])
    expected = np.array([(1 - g * g) / (1 + g * g - 2 * g * c) ** 1.5 for c in cosines])
    values = henyey_greenstein_phase(g, cosines)
    np.testing.assert_allclose(values, expected, rtol=3e-14, atol=0)
    for cosine, value in zip(cosines, expected, strict=True):
        scalar = henyey_greenstein_phase(g, float(cosine))
        assert type(scalar) is float
        assert scalar == pytest.approx(value, rel=3e-14)
    if g == 0:
        np.testing.assert_array_equal(values, np.ones_like(cosines))


def test_hg_broadcasts_layer_point_asymmetry_over_angles_without_aliasing() -> None:
    g = np.array([[-0.8, 0.0], [0.2, 0.8]])
    cosine = np.array([-1.0, 0.0, 1.0])
    original_g, original_cosine = g.copy(), cosine.copy()
    g.flags.writeable = False
    cosine.flags.writeable = False
    values = henyey_greenstein_phase(g[..., None], cosine)
    expected = np.empty((2, 2, 3))
    for layer in range(2):
        for point in range(2):
            for angle, c in enumerate(cosine):
                asymmetry = float(g[layer, point])
                expected[layer, point, angle] = (1 - asymmetry * asymmetry) / (
                    1 + asymmetry * asymmetry - 2 * asymmetry * float(c)
                ) ** 1.5
    np.testing.assert_allclose(values, expected, rtol=3e-14, atol=0)
    assert values.dtype == np.float64
    assert values.flags.writeable
    assert not np.shares_memory(values, g)
    assert not np.shares_memory(values, cosine)
    values[:] = -10
    np.testing.assert_array_equal(g, original_g)
    np.testing.assert_array_equal(cosine, original_cosine)
    np.testing.assert_allclose(henyey_greenstein_phase(g, 0.5), (1 - g**2) / (1 + g**2 - g) ** 1.5)


def test_rayleigh_scalar_array_symmetry_and_ownership() -> None:
    cosine = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    original = cosine.copy()
    cosine.flags.writeable = False
    values = rayleigh_phase(cosine)
    np.testing.assert_allclose(values, [1.5, 0.87, 0.75, 0.87, 1.5], rtol=1e-15)
    np.testing.assert_array_equal(values, values[::-1])
    for c, value in zip(cosine, values, strict=True):
        scalar = rayleigh_phase(float(c))
        assert type(scalar) is float
        assert scalar == pytest.approx(value, rel=1e-15)
    assert values.dtype == np.float64
    assert values.flags.writeable
    assert not np.shares_memory(values, cosine)
    values[:] = -10
    np.testing.assert_array_equal(cosine, original)


def test_zero_dimensional_and_empty_results() -> None:
    for result in (
        henyey_greenstein_phase(np.array(0.2), np.array(0.5)),
        henyey_greenstein_phase(np.array(0.2), 0.5),
        henyey_greenstein_phase(0.2, np.array(0.5)),
        rayleigh_phase(np.array(0.5)),
    ):
        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert result.dtype == np.float64
        assert result.flags.writeable
    empty = np.empty((0, 2))
    assert henyey_greenstein_phase(0.2, empty).shape == (0, 2)
    assert rayleigh_phase(empty).shape == (0, 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -0.81, 0.81])
def test_hg_rejects_invalid_scalar_and_array_asymmetry(bad: float) -> None:
    with pytest.raises(ValueError, match="HG asymmetry"):
        henyey_greenstein_phase(bad, 0.0)
    with pytest.raises(ValueError, match="HG asymmetry"):
        henyey_greenstein_phase(np.array([0.0, bad]), np.array([0.0, 0.5]))


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), -float("inf"), np.nextafter(-1.0, -2.0), np.nextafter(1.0, 2.0)],
)
def test_phase_laws_reject_invalid_cosines_without_clipping(bad: float) -> None:
    with pytest.raises(ValueError, match="scattering-angle cosine"):
        henyey_greenstein_phase(0.2, bad)
    with pytest.raises(ValueError, match="scattering-angle cosine"):
        henyey_greenstein_phase(0.2, np.array([0.0, bad]))
    with pytest.raises(ValueError, match="scattering-angle cosine"):
        rayleigh_phase(bad)
    with pytest.raises(ValueError, match="scattering-angle cosine"):
        rayleigh_phase(np.array([0.0, bad]))


def test_hg_incompatible_shapes_raise_broadcasting_error() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        henyey_greenstein_phase(np.zeros(2), np.zeros(3))


def test_phase_laws_are_nonnegative_and_have_four_pi_integral() -> None:
    cosine, weights = np.polynomial.legendre.leggauss(200)
    for g in [-0.8, 0.0, 0.8]:
        phase = henyey_greenstein_phase(g, cosine)
        assert np.all(phase >= 0)
        assert 2 * pi * float(weights @ phase) == pytest.approx(4 * pi, rel=3e-13)
    phase = rayleigh_phase(cosine)
    assert np.all(phase >= 0)
    assert 2 * pi * float(weights @ phase) == pytest.approx(4 * pi, rel=3e-13)


@pytest.mark.parametrize("g", [None, -0.8, 0.0, 0.8])
def test_sos_dispatch_preserves_geometric_clipping(g: float | None) -> None:
    cosine = np.array([np.nextafter(-1.0, -2.0), -0.5, np.nextafter(1.0, 2.0)])
    original = cosine.copy()
    bounded = np.clip(cosine, -1, 1)
    expected = rayleigh_phase(bounded) if g is None else henyey_greenstein_phase(g, bounded)
    np.testing.assert_array_equal(orders.phase_function(cosine, g), expected)
    np.testing.assert_array_equal(cosine, original)


def test_sos_dispatch_uses_optical_hg_bounds() -> None:
    with pytest.raises(ValueError, match="HG asymmetry"):
        orders.phase_function(np.array([0.0]), 0.81)
