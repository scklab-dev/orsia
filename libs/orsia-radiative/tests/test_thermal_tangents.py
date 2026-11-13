"""Bounded thermal derivative, endpoint, ownership and concurrency evidence."""

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from orsia_numerics.tangents import jacobian
from orsia_radiative import Absorption, AerosolComponent, AtmosphericOptics, wavenumber_grid
from orsia_radiative.radiometry import planck_wavenumber
from orsia_radiative.transfer import thermal
from orsia_radiative.transfer.thermal import (
    schwarzschild_terms,
    schwarzschild_terms_tangent,
    thermal_radiance,
    thermal_radiance_tangent,
)
from scipy import constants
from scipy.integrate import quad
from scipy.special import expn
from test_absorption_solar_tangents import layout_copy
from test_thermal import case


def seeds(layers, points, k):
    rng = np.random.default_rng(791)
    return {
        "optical_depth_tangent": rng.uniform(-0.1, 0.1, (layers, points, k)),
        "layer_temperature_tangent": rng.uniform(-10, 10, (layers, k)),
        "surface_temperature_tangent": rng.uniform(-10, 10, k),
        "emissivity_tangent": rng.uniform(-0.1, 0.1, (points, k)),
    }


def perturb(scene, optics, directions, d, step):
    depth = optics.absorption.optical_depth + step * directions["optical_depth_tangent"][..., d]
    atmosphere = replace(
        scene.atmosphere,
        temperature_k=scene.atmosphere.temperature_k
        + step * directions["layer_temperature_tangent"][..., d],
    )
    surface = replace(
        scene.surface,
        temperature_k=scene.surface.temperature_k
        + step * directions["surface_temperature_tangent"][d],
        emissivity=scene.surface.emissivity + step * directions["emissivity_tangent"][..., d],
    )
    return replace(scene, atmosphere=atmosphere, surface=surface), replace(
        optics, absorption=Absorption(optics.grid, depth)
    )


@pytest.mark.parametrize("method", ["quadrature", "expn"])
@pytest.mark.parametrize("k", [0, 1, 3, 4])
def test_value_parity_ownership_and_linearity(method, k):
    scene, optics = case(
        np.array([[0.0, 1e-15, 0.2, 100.0], [0.0, 1e-12, 3.0, 900.0]]), emissivity=0.7
    )
    directions = seeds(2, 4, k)
    values, tangents = schwarzschild_terms_tangent(
        scene, optics=optics, hemispheric_method=method, **directions
    )
    reference = schwarzschild_terms(scene, optics=optics, hemispheric_method=method)
    total, dt = thermal_radiance_tangent(
        scene, optics=optics, hemispheric_method=method, **directions
    )
    np.testing.assert_array_equal(total.values, sum(values.values()))
    np.testing.assert_array_equal(dt, sum(tangents.values()))
    for name in values:
        np.testing.assert_array_equal(values[name], reference[name])
        assert tangents[name].shape == (4, k)
    for array in (*values.values(), *tangents.values(), dt):
        assert array.dtype == np.float64 and array.flags.owndata and array.flags.writeable
        assert array.flags.c_contiguous
        assert all(not np.shares_memory(array, a) for a in directions.values())
    matrix = np.arange(k * 2.0).reshape(k, 2) / 10
    mixed = {name: array @ matrix for name, array in directions.items()}
    _, mixed_dt = thermal_radiance_tangent(
        scene, optics=optics, angular_order=32, hemispheric_method=method, **mixed
    )
    np.testing.assert_allclose(mixed_dt, dt @ matrix, rtol=3e-13, atol=1e-16)


@pytest.mark.parametrize("method", ["quadrature", "expn"])
def test_component_multistep_finite_difference_convergence(method):
    scene, optics = case(np.array([[0.2, 0.5, 2.0], [0.7, 0.1, 3.0]]), emissivity=0.4, view=43.0)
    directions = seeds(2, 3, 4)
    _, tangents = schwarzschild_terms_tangent(
        scene, optics=optics, hemispheric_method=method, **directions
    )
    errors = []
    for step in (0.1, 0.025, 0.00625):
        error = 0.0
        for d in range(4):
            plus_scene, plus_optics = perturb(scene, optics, directions, d, step)
            minus_scene, minus_optics = perturb(scene, optics, directions, d, -step)
            plus = schwarzschild_terms(plus_scene, optics=plus_optics, hemispheric_method=method)
            minus = schwarzschild_terms(minus_scene, optics=minus_optics, hemispheric_method=method)
            for name in tangents:
                difference = (plus[name] - minus[name]) / (2 * step)
                error = max(error, float(np.max(np.abs(difference - tangents[name][:, d]))))
        errors.append(error)
    assert errors[1] < errors[0] / 12 and errors[2] < errors[1] / 12
    assert errors[-1] < 2e-8


@pytest.mark.parametrize("method", ["quadrature", "expn"])
@pytest.mark.parametrize("emissivity", [0.0, 1.0])
@pytest.mark.parametrize("depth", [0.0, 1e-15, 0.2, 1000.0])
def test_boundary_inward_differences(method, emissivity, depth):
    scene, optics = case(np.full((1, 3), depth), emissivity=emissivity)
    directions = {
        "optical_depth_tangent": np.ones((1, 3, 1)),
        "layer_temperature_tangent": np.ones((1, 1)),
        "surface_temperature_tangent": np.ones(1),
        "emissivity_tangent": np.full((3, 1), 1.0 if emissivity == 0 else -1.0),
    }
    value, derivative = thermal_radiance_tangent(
        scene, optics=optics, hemispheric_method=method, **directions
    )
    errors = []
    for step in (1e-4, 1e-5, 1e-6):
        shifted_scene, shifted_optics = perturb(scene, optics, directions, 0, step)
        shifted = thermal_radiance(shifted_scene, optics=shifted_optics, hemispheric_method=method)
        errors.append(np.max(np.abs((shifted.values - value.values) / step - derivative[:, 0])))
    assert errors[-1] < 3e-6
    assert errors[-1] < errors[0] / 8


def test_fixed_optics_planck_and_separate_optical_temperature_seeds():
    scene, optics = case(np.full((1, 3), 0.4), emissivity=1.0)
    directions = {
        "optical_depth_tangent": np.zeros((1, 3, 2)),
        "layer_temperature_tangent": np.array([[1.0, 0.0]]),
        "surface_temperature_tangent": np.array([0.0, 1.0]),
        "emissivity_tangent": np.zeros((3, 2)),
    }
    _, dt = thermal_radiance_tangent(scene, optics=optics, **directions)
    step = 0.001
    grid = optics.wavenumbers_cm1
    for d, temperature in enumerate((280.0, 300.0)):
        db = (
            planck_wavenumber(grid, temperature + step)
            - planck_wavenumber(grid, temperature - step)
        ) / (2 * step)
        weight = 1 - np.exp(-0.4) if d == 0 else np.exp(-0.4)
        np.testing.assert_allclose(dt[:, d], db * weight, rtol=1e-9)
    optical_only = {name: np.zeros_like(array) for name, array in directions.items()}
    optical_only["optical_depth_tangent"][:] = 0.02
    _, optical_dt = thermal_radiance_tangent(
        scene, optics=optics, angular_order=32, hemispheric_method="quadrature", **optical_only
    )
    combined = {name: array + optical_only[name] for name, array in directions.items()}
    _, combined_dt = thermal_radiance_tangent(
        scene, optics=optics, angular_order=32, hemispheric_method="quadrature", **combined
    )
    np.testing.assert_allclose(combined_dt, dt + optical_dt, rtol=1e-14)


@pytest.mark.parametrize("layout", ["F", "unaligned", "reversed"])
def test_layout_and_concurrent_calls(layout):
    scene, optics = case(np.full((2, 5), 0.3), emissivity=0.8)
    directions = seeds(2, 5, 4)
    expected = thermal_radiance_tangent(
        scene, optics=optics, hemispheric_method="expn", **directions
    )
    supplied = {name: layout_copy(array, layout) for name, array in directions.items()}

    def evaluate(_):
        return thermal_radiance_tangent(
            scene, optics=optics, angular_order=32, hemispheric_method="expn", **supplied
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, range(6)))
    for value, tangent in results:
        np.testing.assert_array_equal(value.values, expected[0].values)
        np.testing.assert_array_equal(tangent, expected[1])
    assert not sys._is_gil_enabled()


def test_absorbing_aerosol_and_vacuum_optical_roles():
    depth = np.full((2, 3), 0.2)
    scene, optics = case(depth, emissivity=0.5)
    directions = seeds(2, 3, 1)
    expected = thermal_radiance_tangent(scene, optics=optics, **directions)
    aerosol = AerosolComponent(optics.grid, depth, np.zeros_like(depth), np.zeros_like(depth))
    for mixed in (
        replace(optics, absorption=None, aerosol=aerosol),
        replace(
            optics,
            absorption=Absorption(optics.grid, depth / 2),
            aerosol=replace(aerosol, optical_depth=depth / 2),
        ),
    ):
        value, tangent = thermal_radiance_tangent(scene, optics=mixed, **directions)
        np.testing.assert_array_equal(value.values, expected[0].values)
        np.testing.assert_array_equal(tangent, expected[1])
    value, tangent = thermal_radiance_tangent(
        scene, optics=replace(optics, absorption=None), **directions
    )
    assert np.all(np.isfinite(value.values)) and np.all(np.isfinite(tangent))


@pytest.mark.parametrize("field", list(seeds(2, 3, 1)))
@pytest.mark.parametrize("failure", ["dtype", "shape", "nonfinite"])
def test_invalid_tangents(field, failure):
    scene, optics = case(np.full((2, 3), 0.2))
    directions = seeds(2, 3, 1)
    array = directions[field]
    if failure == "dtype":
        directions[field] = array.astype(np.float32)
    elif failure == "shape":
        directions[field] = np.zeros(array.shape + (1,))
    else:
        array.flat[0] = np.nan
    with pytest.raises(ValueError, match="tangent"):
        thermal_radiance_tangent(scene, optics=optics, **directions)


def test_dense_spectral_and_band_jacobian():
    scene, optics = case(np.full((2, 5), 0.3), emissivity=0.6)

    def evaluate(x, v):
        changed = replace(scene, surface=replace(scene.surface, temperature_k=x[0]))
        value, tangent = thermal_radiance_tangent(
            changed,
            optics=optics,
            optical_depth_tangent=np.zeros((2, 5, v.shape[1])),
            layer_temperature_tangent=np.zeros((2, v.shape[1])),
            surface_temperature_tangent=v[0],
            emissivity_tangent=np.zeros((5, v.shape[1])),
        )
        return value.values, tangent

    value, derivatives = jacobian(evaluate, np.array([300.0]))
    grid = optics.wavenumbers_cm1
    band = np.trapezoid(value, grid)
    band_jacobian = np.trapezoid(derivatives, grid, axis=0)
    assert band > 0 and band_jacobian.shape == (1,) and band_jacobian[0] > 0


def test_total_tangent_rejects_component_sum_overflow(monkeypatch):
    scene, optics = case(np.full((1, 3), 0.2))
    names = ("surface_emission", "atmospheric_emission", "reflected_atmospheric_emission")
    monkeypatch.setattr(
        thermal,
        "schwarzschild_terms_tangent",
        lambda *args, **kwargs: (
            {name: np.ones(3) for name in names},
            {name: np.full((3, 1), np.finfo(np.float64).max) for name in names},
        ),
    )
    with pytest.raises(ValueError, match="summed thermal"):
        thermal_radiance_tangent(scene, optics=optics, **seeds(1, 3, 1))


@pytest.mark.parametrize(
    "nu,temperature",
    [
        (1000.0, 300.0),
        (1e-150, 1e100),
        (1e100, 1e100),
        (1e-50, 100 * constants.h * constants.c / constants.k * 1e-50 / 400),
        (1000.0, 2.0),
        (1e8, 3.0),
    ],
)
def test_planck_derivative_regimes_against_decimal(nu, temperature):
    scene, _ = case(np.zeros((1, 1)))
    grid = wavenumber_grid(np.array([nu]))
    scene = replace(scene, surface=replace(scene.surface, grid=grid, temperature_k=temperature))
    optics = AtmosphericOptics(grid, None, None, None)
    value, derivative = thermal_radiance_tangent(
        scene,
        optics=optics,
        optical_depth_tangent=np.zeros((1, 1, 1)),
        layer_temperature_tangent=np.zeros((1, 1)),
        surface_temperature_tangent=np.ones(1),
        emissivity_tangent=np.zeros((1, 1)),
    )
    with localcontext() as context:
        context.prec = 90
        x = (
            Decimal.from_float(100 * constants.h * constants.c / constants.k)
            * Decimal.from_float(nu)
            / Decimal.from_float(temperature)
        )
        rj = Decimal.from_float(2 * constants.k * constants.c * 1e6)
        # In the deep Wien tail the derivative is below binary64 support.
        if x > 10000:
            expected = 0.0
        elif x < Decimal("1e-40"):
            expected = float(rj * Decimal.from_float(nu) ** 2)
        else:
            attenuation = (-x).exp()
            expected = float(
                rj * Decimal.from_float(nu) ** 2 * x**2 * attenuation / (1 - attenuation) ** 2
            )
    np.testing.assert_allclose(derivative[0, 0], expected, rtol=5e-12, atol=5e-324)
    if nu == 1e-50:
        assert value.values[0] == 0 and derivative[0, 0] > 0


@pytest.mark.parametrize(
    "below,depth",
    [
        (0.0, 1e-20),
        (1e-20, 1e-20),
        (1e-8, 1e-12),
        (1e-300, 1e-302),
        (1e-320, 1e-322),
        (0.1, 1e-20),
        (0.1, 0.001),
        (0.1, 0.2),
        (2.0, 0.1),
        (700.0, 1e-8),
    ],
)
def test_analytic_interval_derivative_against_independent_integral(below, depth):
    scene, optics = case(np.array([[below], [depth]]), np.array([1e-100, 280.0]), emissivity=0.0)
    values, tangents = schwarzschild_terms_tangent(
        scene,
        optics=optics,
        hemispheric_method="expn",
        optical_depth_tangent=np.array([[[1.0]], [[0.0]]]),
        layer_temperature_tangent=np.zeros((2, 1)),
        surface_temperature_tangent=np.zeros(1),
        emissivity_tangent=np.zeros((1, 1)),
    )
    # Differentiate only the lower absorbing layer. Its Planck source underflows
    # to zero; the upper source probes integral E1 independently of E2 subtraction.
    if 0 < below < np.finfo(np.float64).tiny:
        # E1(x) = -gamma-log(x)+O(x); integrate log1p on scaled coordinates
        # to avoid quantizing quadrature nodes on a subnormal optical grid.
        average_e1 = (
            -np.euler_gamma
            - np.log(below)
            - quad(lambda s: np.log1p((depth / below) * s), 0, 1, epsabs=1e-14)[0]
        )
    else:
        average_e1 = quad(lambda s: expn(1, below + depth * s), 0, 1, epsabs=1e-300, epsrel=2e-12)[
            0
        ]
    b = planck_wavenumber(optics.wavenumbers_cm1, 280.0)[0]
    expected = (
        -2 * b * (depth * average_e1) * np.exp(-(below + depth))
        - values["reflected_atmospheric_emission"][0]
    )
    np.testing.assert_allclose(
        tangents["reflected_atmospheric_emission"][0, 0], expected, rtol=3e-11, atol=5e-324
    )
