"""Analytic limits and buffer ownership of formal transport operators."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_radiative.transfer.formal import (
    apply_propagation,
    integrated_exponential,
    propagate,
    propagation_coefficients,
)
from scipy.integrate import quad


def test_exponential_path_integral_matches_independent_integrals_and_is_symmetric() -> None:
    a = np.array([0.0, -0.1, -0.5, -1000.0])
    b = np.array([0.0, -0.1 - 1e-12, -0.2, 0.0])
    original_a, original_b = a.copy(), b.copy()
    expected = [
        quad(lambda t, x, y: np.exp(x + (y - x) * t), 0, 1, args=(x, y))[0]
        for x, y in zip(a, b, strict=True)
    ]
    result = integrated_exponential(a, b)
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(result, integrated_exponential(b, a), rtol=1e-13, atol=0)
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)
    assert not np.shares_memory(result, a) and not np.shares_memory(result, b)


def test_formal_absorption_transports_bottom_boundary_without_reflecting_it() -> None:
    depth = np.array([0.1, 0.4, 0.3])
    mu = np.array([-0.7, 0.5, 0.9])
    source = np.zeros((3, 3))
    centres, exits = propagate(source, depth, mu, bottom_radiance=2.0)
    np.testing.assert_array_equal(centres[:, 0], 0.0)
    assert exits[0] == 0.0
    np.testing.assert_allclose(exits[1:], 2 * np.exp(-depth.sum() / mu[1:]), rtol=1e-14)
    below = np.concatenate(([0.0], np.cumsum(depth)[:-1]))
    np.testing.assert_allclose(
        centres[:, 1:], 2 * np.exp(-(below + depth / 2)[:, None] / mu[1:]), rtol=1e-14
    )
    for output in (centres, exits):
        for original in (source, depth, mu):
            assert not np.shares_memory(output, original)
    np.testing.assert_array_equal(source, np.zeros((3, 3)))


def test_direct_beam_forcing_matches_upward_and_downward_continuous_ray_integrals() -> None:
    tau, mu0 = 0.3, 0.8
    mu = np.array([0.5, -0.6])
    forcing = np.array([[0.02, 0.04]])
    _, exits = propagate(forcing, np.array([tau]), mu, solar_mu=mu0)
    expected = []
    for index, direction in enumerate(mu):
        expected.append(
            quad(
                lambda t, intensity, ray_mu: (
                    intensity
                    / abs(ray_mu)
                    * np.exp(
                        -tau * (1 - t) / mu0 - tau * (1 - t if ray_mu > 0 else t) / abs(ray_mu)
                    )
                ),
                0,
                1,
                args=(forcing[0, index], direction),
            )[0]
        )
    np.testing.assert_allclose(exits, expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("solar_mu", [None, 0.6])
@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4, 5], [3, 0, 4, 1, 5, 2]])
def test_prepared_transport_matches_independent_multicell_ray_integrals(
    solar_mu: float | None, order: list[int]
) -> None:
    depth = np.array([0.0, 1e-14, 1e-8, 0.03, 0.8, 10.0])
    mu = np.array([-1.0, -0.6, -0.6 - 1e-12, 0.05, 0.6, 1.0])[order]
    source = np.arange(1.0, depth.size * mu.size + 1).reshape(depth.size, mu.size) / 100
    coefficients = propagation_coefficients(depth, mu, solar_mu=solar_mu)
    result = apply_propagation(
        source, coefficients, bottom_radiance=0.7, solar_source=solar_mu is not None
    )
    expected = _continuous_reference(source, depth, mu, 0.7, solar_mu)
    for actual, reference in zip(result, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=1e-15)
    np.testing.assert_array_equal(
        apply_propagation(
            source,
            coefficients,
            bottom_radiance=0.7,
            solar_source=solar_mu is not None,
            return_centres=False,
        )[1],
        result[1],
    )


@pytest.mark.parametrize("solar_mu", [None, 0.8])
@pytest.mark.parametrize("tau", [0.0, 1e-18, 1e-10, 0.2, 1000.0])
def test_constant_source_limits_include_vacuum_weak_and_opaque_cells(
    solar_mu: float | None, tau: float
) -> None:
    # For zero extinction the integrated vertical source need not be zero.
    mu = np.array([-0.8, 0.5])
    source = np.array([[0.3, 0.7]])
    result = propagate(source, np.array([tau]), mu, solar_mu=solar_mu)
    expected = _continuous_reference(source, np.array([tau]), mu, 0.0, solar_mu)
    for actual, reference in zip(result, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=5e-13, atol=1e-15)


@pytest.mark.parametrize("mu", [np.array([-0.5, -0.8]), np.array([0.8]), np.array([])])
def test_one_or_empty_hemisphere_and_empty_column(mu: NDArray[np.float64]) -> None:
    for depth in (np.array([0.1, 0.4]), np.array([])):
        source = np.zeros((depth.size, mu.size))
        coefficients = propagation_coefficients(depth, mu)
        centres, exits = apply_propagation(source, coefficients, bottom_radiance=1.2)
        assert centres.shape == source.shape
        np.testing.assert_allclose(exits, np.where(mu > 0, 1.2 * np.exp(-depth.sum() / mu), 0.0))
        no_centres, exits_only = apply_propagation(
            source, coefficients, bottom_radiance=1.2, return_centres=False
        )
        assert no_centres is None
        np.testing.assert_array_equal(exits_only, exits)


def test_preparation_owns_readonly_arrays_and_reuse_has_no_mutable_state() -> None:
    depth = np.array([0.0, 1e-10, 0.4])
    mu = np.array([0.5, -0.8, 0.9, -0.3])
    source = np.arange(12.0).reshape(3, 4) / 10
    saved_source = source.copy()
    coefficients = propagation_coefficients(depth, mu, solar_mu=0.8)
    arrays = (
        coefficients.mu,
        coefficients.transmission,
        coefficients.centre_transmission,
        coefficients.source_weight,
        coefficients.centre_source_weight,
        coefficients.solar_source_weight,
        coefficients.solar_centre_source_weight,
    )
    for array in arrays:
        assert array is not None and not array.flags.writeable
        assert not np.shares_memory(array, depth) and not np.shares_memory(array, mu)
    snapshots = [array.copy() for array in arrays if array is not None]
    # Reusing coefficients uses their own grid, not subsequently changed inputs.
    expected = apply_propagation(source, coefficients, solar_source=True, bottom_radiance=2.0)
    depth[:] = 2.0
    mu[:] = 0.1

    def evaluate(_: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return apply_propagation(source, coefficients, solar_source=True, bottom_radiance=2.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, range(4)))
    for result in results:
        for output, reference in zip(result, expected, strict=True):
            np.testing.assert_array_equal(output, reference)
            assert not np.shares_memory(output, reference)
            assert not np.shares_memory(output, source)
    np.testing.assert_array_equal(source, saved_source)
    for array, saved in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, saved)


def test_invalid_prepared_source_support_or_missing_solar_weights_fail() -> None:
    coefficients = propagation_coefficients(np.array([0.1]), np.array([0.5]))
    with pytest.raises(ValueError, match="solar_mu"):
        apply_propagation(np.ones((1, 1)), coefficients, solar_source=True)
    with pytest.raises(ValueError, match="grid"):
        apply_propagation(np.ones((2, 1)), coefficients)


def test_constant_preparation_avoids_general_integrator_and_depth_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orsia_radiative.transfer import formal

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("constant sources need neither general integration nor a prefix sum")

    monkeypatch.setattr(formal, "integrated_exponential", unexpected)
    monkeypatch.setattr(formal.np, "cumsum", unexpected)
    propagate(np.ones((2, 2)), np.array([0.0, 1e-8]), np.array([-0.5, 0.5]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solar_preparation_integrates_full_grid(
    monkeypatch: pytest.MonkeyPatch, dtype: type[np.float32] | type[np.float64]
) -> None:
    from orsia_radiative.transfer import formal

    shapes: list[tuple[int, ...]] = []

    def measured[Real: np.float32 | np.float64](
        a: NDArray[Real], b: NDArray[Real]
    ) -> NDArray[Real]:
        shapes.append(a.shape)
        assert b.shape == a.shape
        assert a.dtype == b.dtype == dtype
        return integrated_exponential(a, b)

    monkeypatch.setattr(formal, "integrated_exponential", measured)
    depth = np.full(513, 0.01, dtype=dtype)
    mu = np.concatenate(
        (
            np.array([-0.8, -np.nextafter(0.8, 0)], dtype=dtype),
            np.linspace(0.1, 1, 255, dtype=dtype),
        )
    )
    coefficients = propagation_coefficients(depth, mu, solar_mu=0.8)
    assert shapes == [(513, 257), (513, 257)]
    assert coefficients.solar_source_weight is not None
    assert coefficients.solar_centre_source_weight is not None
    assert coefficients.solar_source_weight.dtype == dtype
    assert coefficients.solar_centre_source_weight.dtype == dtype
    for cell in (0, 256, 512):
        above = float(depth[cell + 1 :].sum(dtype=np.float64))
        for direction_index in (0, 1, 128, 256):
            ray_mu = float(mu[direction_index])
            for fraction, weight in (
                (1.0, coefficients.solar_source_weight),
                (0.5, coefficients.solar_centre_source_weight),
            ):
                expected = _cell_reference(
                    1.0, float(depth[cell]), above, abs(ray_mu), ray_mu > 0, 0.8, 0.0, fraction
                )
                # Float32 depth prefixes accumulate rounding over up to 512 cells.
                np.testing.assert_allclose(
                    weight[cell, direction_index],
                    expected,
                    rtol=1e-4 if dtype == np.float32 else 2e-12,
                    atol=0,
                )


def _continuous_reference(
    source: NDArray[np.float64],
    depth: NDArray[np.float64],
    mu: NDArray[np.float64],
    boundary: float,
    solar_mu: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate along each travelled ray independently, including half cells."""
    centres = np.empty_like(source)
    exits = np.zeros(mu.size)
    for direction_index, signed_mu in enumerate(mu):
        ray_mu = abs(signed_mu)
        upward = signed_mu > 0
        running = boundary if upward else 0.0
        cells = range(depth.size) if upward else range(depth.size - 1, -1, -1)
        for cell in cells:
            above = sum(depth[cell + 1 :])
            centres[cell, direction_index] = _cell_reference(
                source[cell, direction_index],
                depth[cell],
                above,
                ray_mu,
                upward,
                solar_mu,
                running,
                0.5,
            )
            running = _cell_reference(
                source[cell, direction_index],
                depth[cell],
                above,
                ray_mu,
                upward,
                solar_mu,
                running,
                1.0,
            )
        exits[direction_index] = running
    return centres, exits


def _cell_reference(
    source: float,
    depth: float,
    above: float,
    ray_mu: float,
    upward: bool,
    solar_mu: float | None,
    running: float,
    fraction: float,
) -> float:
    def integrand(t: float) -> float:
        beam = 1.0
        if solar_mu is not None:
            beam = np.exp(-(above + depth * (1 - t if upward else t)) / solar_mu)
        return source / ray_mu * beam * np.exp(-depth * (fraction - t) / ray_mu)

    return (
        running * np.exp(-depth * fraction / ray_mu)
        + quad(integrand, 0.0, fraction, epsabs=1e-14, epsrel=1e-13)[0]
    )
