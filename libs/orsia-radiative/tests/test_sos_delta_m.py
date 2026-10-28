"""Delta-M algebra, discrete normalization and opt-in residual-order contracts.

These synthetic checks do not qualify continuous transfer or corrected radiances.
The independent cell-grid ray oracle is exercised in test_sos_orders.py.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import cos, pi, radians
from unittest.mock import patch

import numpy as np
import orders_reference as orders
import pytest
import radiance_terms_support as terms_support
from numpy.typing import NDArray
from orsia_radiative import (
    Absorption,
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SolarSource,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    henyey_greenstein_phase,
    solar_source,
    wavenumber_grid,
)
from orsia_radiative.transfer import orders as production_orders
from scipy.special import eval_legendre


@pytest.mark.parametrize("max_order", [0, 1, 4])
@pytest.mark.parametrize("rho", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("case", ["absent", "mixed", "non_forward", "zero_source"])
def test_explicitly_disabled_delta_m_preserves_complete_result(
    max_order: int, rho: float, case: str
) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    scene = replace(scene, surface=replace(scene.surface, reflectance=np.full(2, rho)))
    if case == "non_forward":
        aerosol = replace(aerosol, asymmetry=np.array([[-0.8, 0.0], [-0.2, -0.1]]))
    elif case == "zero_source":
        source = replace(source, irradiance=replace(source.irradiance, values=np.zeros(2)))
    gas = Absorption(wavenumber_grid(source.irradiance.wavenumbers_cm1), np.full((2, 2), 0.02))
    optics = compose_atmospheric_optics(
        scene.atmosphere,
        source,
        absorption=gas,
        rayleigh=None if case == "absent" else rayleigh,
        aerosol=None if case == "absent" else aerosol,
    )
    options = {"max_order": max_order, "angular_order": 3, "layer_subdivisions": 2}
    default = terms_support.reflected_radiance(scene, source=source, optics=optics, **options)
    disabled = terms_support.reflected_radiance(
        scene, source=source, optics=optics, delta_m_moments=None, **options
    )
    # Check every order, support and geometry, not only total radiance.
    # Enabling the option must be an explicit choice.
    terms_support.assert_decomposition_equal(disabled, default)
    assert all(c.name == f"atmospheric_order_{n}" for n, c in enumerate(disabled.components))


@pytest.mark.parametrize("moments", [None, 4])
def test_direct_order_owns_only_transformed_work_buffers(moments: int | None) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    baseline, extinction = orders._direct_order(scene, source, optics, None)
    physical = extinction.copy()
    saved = tuple(array.copy() for array in (aerosol.optical_depth, aerosol.scattering_depth))
    zero_order, extinction = orders._direct_order(scene, source, optics, moments)
    assert zero_order.flags.writeable and extinction.flags.writeable
    assert not np.shares_memory(zero_order, baseline)
    assert not np.shares_memory(zero_order, extinction)
    if moments is None:
        np.testing.assert_array_equal(zero_order, baseline)
        np.testing.assert_array_equal(extinction, physical)
    else:
        removed = np.maximum(aerosol.asymmetry, 0.0) ** moments * aerosol.scattering_depth
        np.testing.assert_array_equal(extinction, physical - removed)
    zero_order.fill(0)
    extinction.fill(0)
    for array, snapshot in zip(
        (aerosol.optical_depth, aerosol.scattering_depth), saved, strict=True
    ):
        np.testing.assert_array_equal(array, snapshot)


@pytest.mark.parametrize("g", [0.1, 0.4, 0.8])
@pytest.mark.parametrize("moments", [2, 3, 4, 8, 16])
def test_residual_phase_retains_normalized_moments(g: float, moments: int) -> None:
    cosine, weights = np.polynomial.legendre.leggauss(48)
    phase = orders._transport_phase(cosine, g, moments)
    fraction = g**moments
    for degree in range(moments + 1):
        actual = np.sum(weights * phase * eval_legendre(degree, cosine)) / 2
        expected = (g**degree - fraction) / (1 - fraction) if degree < moments else 0.0
        assert actual == pytest.approx(expected, abs=8e-14)
    assert phase.flags.writeable and not np.shares_memory(phase, cosine)
    # A transformed first moment does not define another HG law.
    if moments >= 4 and fraction > 1e-14:
        first = (g - fraction) / (1 - fraction)
        second = (g**2 - fraction) / (1 - fraction)
        assert second != pytest.approx(first**2, rel=0, abs=2e-16)


@pytest.mark.parametrize("angular_order", [2, 3, 4])
def test_discrete_forward_phase_normalization_without_renormalizing(angular_order: int) -> None:
    directions, weights = orders.angular_grid(angular_order)
    # Include an arbitrary non-quadrature outgoing direction.
    outgoing = np.vstack((directions, [0.36, 0.48, 0.8]))
    cosines = directions @ outgoing.T
    for moments in range(2, 2 * angular_order + 1):
        phase = orders._transport_phase(cosines, 0.8, moments)
        np.testing.assert_allclose(weights @ phase, 1, rtol=0, atol=8e-15)
        np.testing.assert_allclose(phase[:, :-1] @ weights, 1, rtol=0, atol=8e-15)


@pytest.mark.parametrize("g", [None, -0.8, -0.2, 0.0])
def test_non_forward_phase_is_exactly_unchanged(g: float | None) -> None:
    cosine = np.linspace(-1, 1, 61)
    np.testing.assert_array_equal(
        orders._transport_phase(cosine, g, 4), orders.phase_function(cosine, g)
    )
    assert orders._forward_fraction(g, 4) == 0


@pytest.mark.parametrize("max_order", [0, 1, 4])
def test_transformed_transport_keeps_absorption_and_physical_values(max_order: int) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    gas = Absorption(wavenumber_grid(source.irradiance.wavenumbers_cm1), np.full((2, 2), 0.02))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, absorption=gas, rayleigh=rayleigh, aerosol=aerosol
    )
    saved = tuple(
        a.copy() for a in (aerosol.optical_depth, aerosol.scattering_depth, aerosol.asymmetry)
    )
    physical = gas.optical_depth + rayleigh.optical_depth + aerosol.optical_depth
    fraction = np.maximum(aerosol.asymmetry, 0) ** 4
    transformed = physical - fraction * aerosol.scattering_depth
    # Scaling removes exactly equal extinction and scattering depths.
    np.testing.assert_allclose(
        transformed - rayleigh.optical_depth - (1 - fraction) * aerosol.scattering_depth,
        gas.optical_depth + aerosol.optical_depth - aerosol.scattering_depth,
        rtol=2e-15,
        atol=1e-17,
    )
    with patch.object(
        production_orders,
        "_native_spectral_orders",
        wraps=production_orders._native_spectral_orders,
    ) as prepare:
        result = terms_support.reflected_radiance(
            scene,
            source=source,
            optics=optics,
            max_order=max_order,
            angular_order=3,
            layer_subdivisions=2,
            delta_m_moments=4,
        )
    assert prepare.call_count == 2
    for call in prepare.call_args_list:
        np.testing.assert_array_equal(call.args[0][2], aerosol.optical_depth)
    mu0 = cos(radians(source.zenith_deg))
    mu = cos(radians(scene.geometry.view_zenith_deg))
    transmission = np.exp(-transformed.sum(axis=0) * (1 / mu0 + 1 / mu))
    assert scene.surface.reflectance is not None
    clear = source.irradiance.values * scene.surface.reflectance * mu0 / pi
    np.testing.assert_allclose(result.components[0].values, clear * transmission, rtol=2e-15)
    assert [c.name for c in result.components] == [
        f"residual_order_{n}" for n in range(max_order + 1)
    ]
    np.testing.assert_array_equal(
        result.values, np.sum([c.values for c in result.components], axis=0)
    )
    for a, old in zip(
        (aerosol.optical_depth, aerosol.scattering_depth, aerosol.asymmetry), saved, strict=True
    ):
        np.testing.assert_array_equal(a, old)
    assert optics.aerosol is aerosol and optics.rayleigh is rayleigh and optics.absorption is gas


@pytest.mark.parametrize("case", ["absent", "non_forward", "absorbing", "zero_source"])
def test_no_forward_peak_preserves_values_and_limits(case: str) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    if case == "non_forward":
        aerosol = replace(aerosol, asymmetry=np.array([[-0.8, 0.0], [-0.2, -0.1]]))
    elif case == "absorbing":
        aerosol = replace(aerosol, scattering_depth=np.zeros((2, 2)))
    elif case == "zero_source":
        source = replace(source, irradiance=replace(source.irradiance, values=np.zeros(2)))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=None if case == "absent" else aerosol
    )
    plain = terms_support.reflected_radiance(
        scene, source=source, optics=optics, max_order=3, angular_order=3, layer_subdivisions=2
    )
    scaled = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=optics,
        max_order=3,
        angular_order=3,
        layer_subdivisions=2,
        delta_m_moments=4,
    )
    np.testing.assert_array_equal(scaled.values, plain.values)
    for a, b in zip(scaled.components, plain.components, strict=True):
        np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("moments", [-1, 0, 1, 7])
def test_moment_count_must_be_resolved_by_quadrature(moments: int) -> None:
    scene, source, rayleigh, _ = _inputs()
    with pytest.raises(ValueError, match="2 <= K <= 2"):
        terms_support.reflected_radiance(
            scene,
            source=source,
            optics=compose_atmospheric_optics(scene.atmosphere, source, rayleigh=rayleigh),
            max_order=0,
            angular_order=3,
            delta_m_moments=moments,
        )


def test_signed_phase_is_not_clipped_and_negative_order_is_rejected() -> None:
    cosine = np.array([-1.0, 1.0])
    phase = orders._transport_phase(cosine, 0.8, 2)
    assert phase[0] < 0
    np.testing.assert_allclose(phase, 1 + 3 * (0.8 - 0.8**2) / (1 - 0.8**2) * cosine)
    scene, source, _, aerosol = _inputs()
    scene = replace(
        scene,
        surface=replace(scene.surface, reflectance=np.zeros(2)),
        geometry=ObservationGeometry(0.0),
    )
    source = replace(source, zenith_deg=0.0)
    aerosol = replace(aerosol, asymmetry=np.full((2, 2), 0.8))
    with pytest.raises(ValueError, match="negative order radiance"):
        terms_support.reflected_radiance(
            scene,
            source=source,
            optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=aerosol),
            max_order=1,
            angular_order=2,
            delta_m_moments=2,
        )


@pytest.mark.parametrize("g", [0.4, 1e-100])
def test_cache_distinguishes_truncations_and_returns_non_aliasing_sources(g: float) -> None:
    _, _, _, aerosol = _inputs()
    aerosol = replace(aerosol, asymmetry=np.full((2, 2), g))
    directions, weights = orders.angular_grid(3)
    cosines = directions @ directions.T
    field = np.ones((4, len(weights)))
    snapshot = field.copy()
    cache: dict[orders.PhaseKey, NDArray[np.float64]] = {}
    for moments in (None, 2, 4, 6, 2):
        actual = np.zeros_like(field)
        fresh = np.zeros_like(field)
        orders._scatter(
            field, aerosol, 0, 2, cosines, weights, cache, out=actual, delta_m_moments=moments
        )
        orders._scatter(
            field, aerosol, 0, 2, cosines, weights, {}, out=fresh, delta_m_moments=moments
        )
        np.testing.assert_array_equal(actual, fresh)
        assert not np.shares_memory(actual, field) and actual.flags.writeable
    assert set(cache) == {g, (g, 2), (g, 4), (g, 6)}
    np.testing.assert_array_equal(field, snapshot)


def test_residual_order_prefixes_and_call_local_threaded_results() -> None:
    scene, source, rayleigh, aerosol = _inputs()
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )

    def calculate(n: int):
        return terms_support.reflected_radiance(
            scene,
            source=source,
            optics=optics,
            max_order=n,
            angular_order=3,
            layer_subdivisions=2,
            delta_m_moments=4,
        )

    results = [calculate(n) for n in range(5)]
    for n, result in enumerate(results):
        for a, b in zip(result.components, results[-1].components[: n + 1], strict=True):
            np.testing.assert_array_equal(a.values, b.values)
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(calculate, [4, 4]))
    for result in concurrent:
        terms_support.assert_decomposition_equal(result, results[-1])


def test_high_moment_phase_tends_to_exact_hg_in_a_bounded_case() -> None:
    cosine = np.linspace(-1, 1, 101)
    exact = henyey_greenstein_phase(0.4, cosine)
    errors = [
        np.max(np.abs(orders._transport_phase(cosine, 0.4, k) - exact)) for k in (4, 8, 16, 32)
    ]
    assert all(b < a for a, b in zip(errors[:-1], errors[1:], strict=True))
    # |P_l(c)| <= 1 bounds both the changed retained coefficients and HG tail.
    # The retained-coefficient change grows as K**2 * g**K, not just g**K.
    g, k = 0.4, 32
    fraction = g**k
    bound = fraction * (k**2 / (1 - fraction) + (2 * k + 1) / (1 - g) + 2 * g / (1 - g) ** 2)
    assert errors[-1] <= bound + 2e-14


def _inputs() -> tuple[RadiativeScene, SolarSource, RayleighComponent, AerosolComponent]:
    grid = np.array([12000.0, 12001.0])
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([290.0, 280.0]),
        np.array([2e29, 1.8e29]),
        np.array([1.9e29, 1.7e29]),
        np.array([1e28, 1e28]),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(grid), np.array([0.2, 0.3])),
        ObservationGeometry(20.0),
    )
    source = solar_source(
        SpectralIrradiance(wavenumber_grid(grid), np.array([8.0, 10.0])),
        direction=(30.0, 70.0),
    )
    rayleigh = RayleighComponent(wavenumber_grid(grid), np.array([[0.04, 0.03], [0.01, 0.02]]))
    aerosol = AerosolComponent(
        wavenumber_grid(grid),
        np.array([[0.02, 0.03], [0.06, 0.05]]),
        np.array([[0.01, 0.01], [0.04, 0.03]]),
        np.array([[0.2, -0.8], [0.4, 0.0]]),
    )
    return scene, source, rayleigh, aerosol
