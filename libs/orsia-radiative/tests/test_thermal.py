"""Bounded LTE analytic and independent-ray checks, not domain qualification."""

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import cos, exp, expm1, radians
from typing import Literal, cast

import numpy as np
import pytest
import radiance_terms_support as terms_support
from numpy.typing import NDArray
from orsia_radiative import (
    Absorber,
    Absorption,
    AerosolComponent,
    Atmosphere,
    AtmosphericOptics,
    CrossSectionField,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    atmospheric_optics,
    compose_absorption,
    compose_absorption_on_grid,
    compose_atmospheric_optics,
    solar_source,
    table_absorption,
    wavenumber_grid,
)
from orsia_radiative.radiometry import planck_wavenumber
from orsia_radiative.transfer.formal import schwarzschild
from orsia_radiative.transfer.thermal import _e3_difference
from scipy.integrate import quad
from scipy.special import expn


def case(
    depths: NDArray[np.float64],
    temperatures: NDArray[np.float64] | None = None,
    emissivity: float = 1.0,
    view: float = 0.0,
) -> tuple[RadiativeScene, AtmosphericOptics]:
    n, points = depths.shape
    grid = np.linspace(500.0, 1500.0, points)
    boundaries = np.linspace(100000.0, 10000.0, n + 1)
    atmosphere = Atmosphere(
        np.arange(n + 1, dtype=np.float64) * 1000,
        boundaries,
        (boundaries[1:] + boundaries[:-1]) / 2,
        np.full(n, 280.0) if temperatures is None else temperatures,
        np.full(n, 2e29),
        np.full(n, 1.9e29),
        np.full(n, 1e28),
        (Absorber("X", np.full(n, 1e25), "dry_air"),),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(
            wavenumber_grid(grid), emissivity=np.full(points, emissivity), temperature_k=300.0
        ),
        ObservationGeometry(view),
    )
    absorption = Absorption(
        wavenumber_grid(grid),
        depths,
    )
    return scene, atmospheric_optics(atmosphere, wavenumber_grid(grid), absorption=absorption)


def component(result: terms_support.RadianceDecomposition, name: str) -> NDArray[np.float64]:
    return next(c.values for c in result.components if c.name == name)


@pytest.mark.parametrize("view", [0.0, 60.0, 89.999])
@pytest.mark.parametrize("emissivity", [0.0, 0.4, 1.0])
def test_vacuum_and_result_contract(view: float, emissivity: float) -> None:
    scene, optics = case(np.zeros((2, 3)), emissivity=emissivity, view=view)
    result = terms_support.schwarzschild_radiance(scene, optics=optics)
    expected = emissivity * planck_wavenumber(optics.wavenumbers_cm1, 300.0)
    np.testing.assert_array_equal(result.values, expected)
    assert result.wavenumbers_cm1 is optics.wavenumbers_cm1
    assert result.values.flags.writeable
    np.testing.assert_array_equal(component(result, "atmospheric_emission"), np.zeros(3))
    np.testing.assert_array_equal(component(result, "reflected_atmospheric_emission"), np.zeros(3))


@pytest.mark.parametrize("view", [0.0, 35.0, 80.0])
def test_isothermal_slab_and_nonblack_boundary_against_exponential_integral(view: float) -> None:
    depths = np.array([[0.05, 0.4, 2.0]])
    scene, optics = case(depths, emissivity=0.3, view=view)
    result = terms_support.schwarzschild_radiance(scene, optics=optics, angular_order=96)
    grid = optics.wavenumbers_cm1
    b_atm = planck_wavenumber(grid, 280.0)
    b_surface = planck_wavenumber(grid, 300.0)
    mu = cos(radians(view))
    transmission = np.exp(-depths[0] / mu)
    # Integral_0^1 mu*exp(-tau/mu) dmu = E_3(tau).
    down_over_pi = b_atm * (1 - 2 * expn(3, depths[0]))
    reflected = 0.7 * down_over_pi * transmission
    np.testing.assert_allclose(
        component(result, "surface_emission"), 0.3 * b_surface * transmission
    )
    np.testing.assert_allclose(
        component(result, "atmospheric_emission"), b_atm * -np.expm1(-depths[0] / mu), rtol=3e-14
    )
    np.testing.assert_allclose(
        component(result, "reflected_atmospheric_emission"), reflected, rtol=3e-11
    )
    np.testing.assert_allclose(
        result.values, 0.3 * b_surface * transmission + b_atm * (1 - transmission) + reflected
    )


def test_stratified_nonblack_surface_against_independent_adaptive_ray_integrals() -> None:
    depths = np.array([[0.02, 0.3, 0.7], [0.4, 0.05, 0.2], [0.1, 1.0, 0.08]])
    temperatures = np.array([310.0, 270.0, 230.0])
    scene, optics = case(depths, temperatures, emissivity=0.6, view=52.0)
    result = terms_support.schwarzschild_radiance(scene, optics=optics, angular_order=128)
    grid = optics.wavenumbers_cm1
    b = planck_wavenumber(grid, temperatures)
    mu = cos(radians(52.0))
    direct, reflected = [], []
    for point in range(grid.size):
        tau = depths[:, point]
        # Sum layer source integrals in vertical optical-depth coordinate, rather
        # than propagating with the implementation's layer recurrence.
        volume = 0.0
        downward = 0.0
        for layer in range(3):
            above = float(np.sum(tau[layer + 1 :]))
            below = float(np.sum(tau[:layer]))
            volume += (
                b[layer, point]
                * quad(
                    lambda t, above=above: exp(-(above + t) / mu) / mu,
                    0,
                    float(tau[layer]),
                    epsabs=1e-13,
                )[0]
            )
            downward += (
                2
                * b[layer, point]
                * quad(
                    lambda direction, below=below, depth=float(tau[layer]): (
                        direction * exp(-below / direction) * -expm1(-depth / direction)
                    ),
                    0,
                    1,
                    epsabs=1e-13,
                )[0]
            )
        direct.append(volume)
        reflected.append(0.4 * downward * exp(-float(np.sum(tau)) / mu))
    np.testing.assert_allclose(component(result, "atmospheric_emission"), direct, rtol=3e-14)
    np.testing.assert_allclose(
        component(result, "reflected_atmospheric_emission"), reflected, rtol=1e-10
    )


def test_black_isothermal_equilibrium_opaque_thin_and_quadrature_refinement() -> None:
    scene, optics = case(np.array([[0.0, 1e-14, 1000.0]]), np.array([300.0]))
    result = terms_support.schwarzschild_radiance(scene, optics=optics)
    np.testing.assert_allclose(
        result.values, planck_wavenumber(optics.wavenumbers_cm1, 300.0), rtol=3e-15
    )
    assert component(result, "atmospheric_emission")[1] > 0
    assert component(result, "surface_emission")[-1] == 0
    np.testing.assert_allclose(component(result, "atmospheric_emission")[-1], result.values[-1])
    scene, optics = case(np.array([[0.03, 0.3, 3.0]]), emissivity=0.0)
    assert optics.absorption is not None
    reference = (
        planck_wavenumber(optics.wavenumbers_cm1, 280.0)
        * (1 - 2 * expn(3, optics.absorption.optical_depth[0]))
        * np.exp(-optics.absorption.optical_depth[0])
    )
    errors = [
        np.max(
            np.abs(
                component(
                    terms_support.schwarzschild_radiance(
                        scene,
                        optics=optics,
                        angular_order=order,
                    ),
                    "reflected_atmospheric_emission",
                )
                - reference
            )
        )
        for order in (4, 16, 64)
    ]
    assert errors[2] < errors[1] < errors[0]


def test_relayering_absorbing_aerosol_zero_rayleigh_and_explicit_surface_reflectance() -> None:
    scene, optics = case(np.array([[0.2, 0.7, 1.0]]), emissivity=0.7)
    assert optics.absorption is not None
    original = terms_support.schwarzschild_radiance(scene, optics=optics)
    split_scene, split_optics = case(
        np.tile(np.array([[0.2, 0.7, 1.0]]) / 5, (5, 1)), emissivity=0.7
    )
    split = terms_support.schwarzschild_radiance(split_scene, optics=split_optics)
    np.testing.assert_allclose(split.values, original.values, rtol=3e-14)
    grid = optics.wavenumbers_cm1
    aerosol = AerosolComponent(
        wavenumber_grid(grid),
        optics.absorption.optical_depth,
        np.zeros((1, 3)),
        np.full((1, 3), 0.8),
    )
    rayleigh = RayleighComponent(wavenumber_grid(grid), np.zeros((1, 3)))
    aerosol_optics = atmospheric_optics(
        scene.atmosphere, wavenumber_grid(grid), aerosol=aerosol, rayleigh=rayleigh
    )
    result = terms_support.schwarzschild_radiance(scene, optics=aerosol_optics)
    np.testing.assert_array_equal(result.values, original.values)
    mixed = atmospheric_optics(
        scene.atmosphere,
        wavenumber_grid(grid),
        absorption=Absorption(wavenumber_grid(grid), optics.absorption.optical_depth / 2),
        aerosol=replace(aerosol, optical_depth=aerosol.optical_depth / 2),
    )
    np.testing.assert_array_equal(
        terms_support.schwarzschild_radiance(scene, optics=mixed).values, original.values
    )
    both = replace(scene, surface=replace(scene.surface, reflectance=np.full(3, 0.3)))
    terms_support.assert_decomposition_equal(
        terms_support.schwarzschild_radiance(both, optics=optics), original
    )


def test_explicit_grid_spectroscopy_composition_matches_solar_and_tracks_freshness() -> None:
    scene, _ = case(np.zeros((2, 3)))
    grid = scene.surface.wavenumbers_cm1
    table = CrossSectionField(
        "X",
        "1",
        "cross_section",
        wavenumber_grid(grid),
        scene.atmosphere.temperature_k,
        scene.atmosphere.pressure_pa,
        np.full((2, 3), 1e-26),
    )
    gas = table_absorption(scene.atmosphere, grid, table)
    total = compose_absorption_on_grid(scene.atmosphere, wavenumber_grid(grid), (gas,))
    source = solar_source(
        SpectralIrradiance(wavenumber_grid(grid), np.ones(3)), direction=(20.0, 0.0)
    )
    solar_total = compose_absorption(scene.atmosphere, source, (gas,))
    assert total.grid == solar_total.grid
    np.testing.assert_array_equal(total.optical_depth, solar_total.optical_depth)
    optics = atmospheric_optics(scene.atmosphere, wavenumber_grid(grid), absorption=total)
    solar_optics = compose_atmospheric_optics(scene.atmosphere, source, absorption=total)
    assert optics.grid == solar_optics.grid
    assert optics.absorption is solar_optics.absorption is total
    assert optics.rayleigh is solar_optics.rayleigh is None
    assert optics.aerosol is solar_optics.aerosol is None
    assert optics.wavenumbers_cm1 is total.wavenumbers_cm1 is grid
    assert not hasattr(total, "reference_atmosphere")
    result = terms_support.schwarzschild_radiance(scene, optics=optics)
    assert result.values.shape == grid.shape
    changed = replace(scene.atmosphere, temperature_k=np.full(2, 281.0))
    with pytest.raises(ValueError, match="stale"):
        compose_absorption_on_grid(
            changed, wavenumber_grid(grid), (gas,), reference_atmospheres=(scene.atmosphere,)
        )
    with pytest.raises(ValueError, match="axis"):
        compose_absorption_on_grid(scene.atmosphere, wavenumber_grid(grid + 0.01), (gas,))
    with pytest.raises(ValueError, match="duplicate"):
        compose_absorption_on_grid(scene.atmosphere, wavenumber_grid(grid), (gas, gas))
    mutable = grid.copy()
    empty = compose_absorption_on_grid(scene.atmosphere, wavenumber_grid(mutable), ())
    transparent = atmospheric_optics(scene.atmosphere, wavenumber_grid(mutable))
    mutable[:] = 9000
    np.testing.assert_array_equal(empty.wavenumbers_cm1, grid)
    np.testing.assert_array_equal(empty.optical_depth, np.zeros((2, 3)))
    np.testing.assert_array_equal(
        terms_support.schwarzschild_radiance(
            scene,
            optics=transparent,
        ).values,
        planck_wavenumber(grid, 300.0),
    )


def test_invalid_surface_axes_atmosphere_and_scattering_are_rejected() -> None:
    scene, optics = case(np.ones((2, 3)))
    grid = optics.wavenumbers_cm1
    for bad_scene, match in (
        (replace(scene, surface=replace(scene.surface, grid=wavenumber_grid(grid + 0.01))), "axes"),
        (
            replace(scene, surface=SurfaceState(wavenumber_grid(grid), reflectance=np.ones(3))),
            "emissivity",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            terms_support.schwarzschild_radiance(bad_scene, optics=optics)
    for scattering in (
        RayleighComponent(wavenumber_grid(grid), np.full((2, 3), 1e-30)),
        AerosolComponent(
            wavenumber_grid(grid), np.ones((2, 3)), np.full((2, 3), 1e-30), np.zeros((2, 3))
        ),
    ):
        bad_optics = atmospheric_optics(
            scene.atmosphere,
            wavenumber_grid(grid),
            rayleigh=scattering if isinstance(scattering, RayleighComponent) else None,
            aerosol=scattering if isinstance(scattering, AerosolComponent) else None,
        )
        with pytest.raises(ValueError, match="scattering"):
            terms_support.schwarzschild_radiance(scene, optics=bad_optics)


@pytest.mark.parametrize("order", [True, False, 0, 1, -1, 2.0, np.int64(4)])
def test_invalid_angular_order(order: int) -> None:
    scene, optics = case(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="angular_order"):
        terms_support.schwarzschild_radiance(scene, optics=optics, angular_order=order)


def test_concurrent_determinism_and_no_input_mutation() -> None:
    scene, optics = case(np.array([[0.1, 0.2], [0.05, 0.3]]), emissivity=0.4)
    assert optics.absorption is not None
    snapshots = optics.absorption.optical_depth.copy(), scene.atmosphere.temperature_k.copy()
    expected = terms_support.schwarzschild_radiance(scene, optics=optics)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda _: terms_support.schwarzschild_radiance(scene, optics=optics), range(6))
        )
    for result in results:
        terms_support.assert_decomposition_equal(result, expected)
    np.testing.assert_array_equal(optics.absorption.optical_depth, snapshots[0])
    np.testing.assert_array_equal(scene.atmosphere.temperature_k, snapshots[1])
    assert all(not np.shares_memory(result.values, expected.values) for result in results)


@pytest.mark.parametrize("mu", [-0.3, 0.7])
def test_formal_schwarzschild_signed_ray_limits_ownership_and_overflow(mu: float) -> None:
    b = np.array([[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]])
    tau = np.array([[0.0, 1e-16, 1e308], [0.0, 2e-16, 1e308]])
    boundary = np.array([2.0, 2.0, 2.0])
    saved = b.copy(), tau.copy(), boundary.copy()
    with np.errstate(all="raise"):
        result = schwarzschild(b, tau, mu, boundary)
    assert result[0] == 2.0
    assert result[-1] == (9.0 if mu > 0 else 5.0)
    assert result.flags.writeable and not np.shares_memory(result, boundary)
    for actual, original in zip((b, tau, boundary), saved, strict=True):
        np.testing.assert_array_equal(actual, original)
    modest = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    result = schwarzschild(b, modest, mu, boundary)
    expected = boundary.copy()
    for layer in [0, 1] if mu > 0 else [1, 0]:
        t = np.exp(-modest[layer] / abs(mu))
        expected = expected * t + b[layer] * (1 - t)
    np.testing.assert_allclose(result, expected, rtol=2e-15)


@pytest.mark.parametrize("mu", [0.0, float("nan"), float("inf"), 1.01, -1.01])
def test_invalid_formal_direction(mu: float) -> None:
    with pytest.raises(ValueError, match="cosine"):
        schwarzschild(np.ones((1, 2)), np.ones((1, 2)), mu, np.ones(2))


@pytest.mark.parametrize("which", ["source", "depth", "boundary"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_formal_values(which: str, bad: float) -> None:
    b, tau, boundary = np.ones((1, 2)), np.ones((1, 2)), np.ones(2)
    {"source": b, "depth": tau, "boundary": boundary}[which].flat[0] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        schwarzschild(b, tau, 1.0, boundary)


@pytest.mark.parametrize(
    "b,tau,boundary",
    [
        (np.ones(2), np.ones(2), np.ones(2)),
        (np.ones((1, 2)), np.ones((2, 2)), np.ones(2)),
        (np.ones((1, 2)), np.ones((1, 2)), np.ones(1)),
        (np.ones((0, 2)), np.ones((0, 2)), np.ones(2)),
    ],
)
def test_invalid_formal_support(
    b: NDArray[np.float64], tau: NDArray[np.float64], boundary: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="support"):
        schwarzschild(b, tau, 1.0, boundary)


def test_thermal_expn_complete_call_stratified_and_concurrent() -> None:
    scene, optics = case(
        np.array([[1e-18, 0.02, 0.3], [1e-18, 0.5, 0.1], [0.0, 0.07, 2.0]]),
        np.array([310.0, 260.0, 220.0]),
        emissivity=0.4,
    )

    def evaluate():
        return terms_support.schwarzschild_radiance(scene, optics=optics, hemispheric_method="expn")

    result = evaluate()
    reference = terms_support.schwarzschild_radiance(scene, optics=optics, angular_order=256)
    np.testing.assert_allclose(result.values, reference.values, rtol=3e-13)
    assert component(result, "reflected_atmospheric_emission")[0] > 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for parallel in pool.map(lambda _: evaluate(), range(6)):
            np.testing.assert_array_equal(parallel.values, result.values)
    with pytest.raises(ValueError, match="hemispheric_method"):
        terms_support.schwarzschild_radiance(
            scene, optics=optics, hemispheric_method=cast(Literal["quadrature", "expn"], "bad")
        )
    terms_support.assert_decomposition_equal(
        result,
        terms_support.schwarzschild_radiance(
            scene, optics=optics, hemispheric_method="expn", angular_order=2
        ),
    )
    assert not sys._is_gil_enabled()


def test_thermal_expn_single_slab_thin_equilibrium_and_layer_subdivision() -> None:
    scene, optics = case(np.array([[1e-20, 0.01, 10.0]]), np.array([300.0]), emissivity=0.0)
    result = terms_support.schwarzschild_radiance(scene, optics=optics, hemispheric_method="expn")
    b = planck_wavenumber(optics.wavenumbers_cm1, 300.0)
    assert component(result, "reflected_atmospheric_emission")[0] == pytest.approx(2e-20 * b[0])
    split_scene, split_optics = case(
        np.repeat(np.array([[1e-20, 0.01, 10.0]]) / 5, 5, axis=0),
        np.full(5, 300.0),
        emissivity=0.0,
    )
    split = terms_support.schwarzschild_radiance(
        split_scene, optics=split_optics, hemispheric_method="expn"
    )
    np.testing.assert_allclose(split.values, result.values, rtol=3e-14)


def e3_ray_reference(below: float, depth: float) -> float:
    """Adaptive angular integral, scaled before integration to resolve tiny layers."""
    if depth == 0 or below >= 745:
        return 0.0
    scale = min(depth, 1.0)
    integral, error = quad(
        lambda mu: mu * exp(-below * (1 / mu - 1)) * (-expm1(-depth / mu) / scale),
        0,
        1,
        epsabs=0,
        epsrel=5e-14,
        limit=300,
    )
    assert error < 3e-13 * max(integral, 1e-3)
    return exp(-below) * (scale * integral)


def test_e3_difference_thin_origin_opaque_and_method_boundaries() -> None:
    # Independent ray-coordinate integration across all analytic branches;
    # nextafter brackets the midpoint, near-origin and origin boundaries.
    pairs = [
        (a, d)
        for a in (0.0, 1e-100, 1e-12, 0.01, 0.49, 0.5, 1.0, 10.0, 100.0, 700.0, 710.0)
        for d in (0.0, 1e-100, 1e-16, 1e-7, 0.001, 0.1, 1.0, 10.0, 1000.0)
    ]
    for a in (1e-12, 0.1, 1.0, 20.0):
        boundary = 0.02 * min(a, 1)
        pairs.extend(
            (a, d) for d in (np.nextafter(boundary, 0), boundary, np.nextafter(boundary, 1))
        )
    pairs.extend((0.1, d) for d in (np.nextafter(0.4, 0), 0.4, np.nextafter(0.4, 1)))
    pairs.extend((0.0, d) for d in (np.nextafter(1.0, 0), 1.0, np.nextafter(1.0, 2)))
    pairs.extend([(1e308, 1e308), (float("inf"), 1.0), (740.0, 0.1)])
    a, d = np.array(pairs).T.copy()
    saved = a.copy(), d.copy()
    actual = _e3_difference(a, d)
    expected = np.array([e3_ray_reference(float(x), float(y)) for x, y in pairs])
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-320)
    assert np.all(actual >= 0) and np.all(np.isfinite(actual))
    assert actual.flags.writeable and not np.shares_memory(actual, d)
    np.testing.assert_array_equal(a, saved[0])
    np.testing.assert_array_equal(d, saved[1])


def test_e3_difference_wholly_separated_and_inactive_batches() -> None:
    a = np.array([[0.0, 1.0, 5.0], [700.0, 710.0, 1000.0]])
    d = np.full_like(a, 0.75)
    expected = np.array(
        [e3_ray_reference(float(x), float(y)) for x, y in zip(a.flat, d.flat, strict=True)]
    )
    np.testing.assert_allclose(_e3_difference(a, d).ravel(), expected, rtol=3e-12, atol=1e-320)
    np.testing.assert_array_equal(_e3_difference(a, np.zeros_like(d)), np.zeros_like(d))
    np.testing.assert_array_equal(
        _e3_difference(np.full_like(a, float("inf")), d), np.zeros_like(d)
    )


@pytest.mark.parametrize("emissivity", [0.0, 0.97, 1.0])
def test_expn_radiance_independent_rays_mixed_optical_regimes(emissivity: float) -> None:
    depths = np.array(
        [
            [1e-20, 1e-5, 0.1, 0.5, 10.0, 1000.0],
            [2e-20, 1e-8, 0.002, 0.01, 0.001, 1e-16],
            [0.0, 0.02, 0.3, 2.0, 4.0, 1.0],
        ]
    )
    temperatures = np.array([305.0, 270.0, 220.0])
    scene, optics = case(depths, temperatures, emissivity=emissivity, view=48.0)
    result = terms_support.schwarzschild_radiance(scene, optics=optics, hemispheric_method="expn")
    b = planck_wavenumber(optics.wavenumbers_cm1, temperatures)
    mu = cos(radians(48.0))
    below = np.zeros(depths.shape[1])
    down = below.copy()
    direct = below.copy()
    for layer, tau in enumerate(depths):
        down += (
            2
            * b[layer]
            * np.array(
                [e3_ray_reference(float(a), float(d)) for a, d in zip(below, tau, strict=True)]
            )
        )
        above = np.sum(depths[layer + 1 :], axis=0)
        direct += b[layer] * np.exp(-above / mu) * -np.expm1(-tau / mu)
        below += tau
    transmission = np.exp(-below / mu)
    reflected = (1 - emissivity) * down * transmission
    expected = emissivity * planck_wavenumber(optics.wavenumbers_cm1, 300.0) * transmission
    expected += direct + reflected
    np.testing.assert_allclose(result.values, expected, rtol=3e-13)
    np.testing.assert_allclose(
        component(result, "reflected_atmospheric_emission"), reflected, rtol=3e-12
    )
