"""Independent continuous-ray integrals and refinement checks for small SOS scenes.

References use adaptive SciPy integrals, not the production angular quadrature,
cell propagation, source-pair accumulation or phase implementation. These cases
are software/numerical evidence only, not external-solver or instrument parity.
"""

from dataclasses import replace
from math import cos, exp, expm1, pi, radians

import numpy as np
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from scipy.integrate import quad
from scipy.special import expn


def _inputs(layers: int = 1, reflectance: float = 0.3) -> tuple[RadiativeScene, SpectralIrradiance]:
    axis = np.array([12000.0], dtype=np.float64)
    atmosphere = Atmosphere(
        1000.0 * np.arange(layers + 1, dtype=np.float64),
        100000.0 - 10000.0 * np.arange(layers + 1, dtype=np.float64),
        95000.0 - 10000.0 * np.arange(layers, dtype=np.float64),
        np.full(layers, 290.0),
        np.full(layers, 2e29),
        np.full(layers, 1.9e29),
        np.full(layers, 1e28),
    )
    return (
        RadiativeScene(
            atmosphere,
            SurfaceState(wavenumber_grid(axis), reflectance=np.array([reflectance])),
            ObservationGeometry(20.0),
        ),
        SpectralIrradiance(wavenumber_grid(axis), np.array([8.0])),
    )


def _value(
    result: terms_support.RadianceDecomposition, name: str, pair: tuple[str, str] | None = None
) -> float:
    return next(c.values[0] for c in result.components if c.name == name)


def _down_beam(t: float, u: float, mu0: float) -> float:
    # Integral of exp(-s/mu0) exp(-(t-s)/u) ds/u, 0 <= s <= t.
    if abs(1 - u / mu0) < 1e-8:
        return exp(-t / mu0) * t / u
    return (exp(-t / mu0) - exp(-t / u)) / (1 - u / mu0)


def _isotropic_mean(t: float, tau: float, mu0: float, incident: float, omega: float) -> float:
    def incoming(u: float) -> float:
        up = exp(-t / mu0) * -expm1(-(tau - t) * (1 / mu0 + 1 / u)) / (1 + u / mu0)
        return up + _down_beam(t, u, mu0)

    return incident * omega / (8 * pi) * quad(incoming, 0, 1, points=[mu0], epsabs=1e-10)[0]


def test_isotropic_first_two_orders_with_repeated_ground_match_continuous_integrals() -> None:
    scene, solar = _inputs()
    tau, omega, rho, incident = 0.3, 0.6, 0.3, 8.0
    mu0, mu = cos(radians(30)), cos(radians(20))
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[tau]]),
        np.array([[tau * omega]]),
        np.zeros((1, 1)),
    )
    result = terms_support.reflected_radiance(
        scene,
        source=solar_source(solar, direction=(30.0, 70.0)),
        optics=compose_atmospheric_optics(
            scene.atmosphere, solar_source(solar, direction=(30.0, 70.0)), aerosol=aerosol
        ),
        angular_order=16,
        layer_subdivisions=128,
    )
    direct_boundary = incident * rho * mu0 / pi * exp(-tau / mu0)

    def volume_mean(t: float) -> float:
        return _isotropic_mean(t, tau, mu0, incident, omega)

    def surface_mean(t: float) -> float:
        return direct_boundary / 2 * float(expn(2, tau - t))

    volume_two = omega / mu * quad(lambda t: exp(-t / mu) * volume_mean(t), 0, tau, epsabs=1e-9)[0]
    reflected_one_boundary = (
        rho
        * incident
        * omega
        / (2 * pi)
        * quad(
            lambda u: u * _down_beam(tau, u, mu0),
            0,
            1,
            points=[mu0],
            epsabs=1e-10,
        )[0]
    )
    before_surface_one = reflected_one_boundary * exp(-tau / mu)
    after_surface_one = (
        omega / mu * quad(lambda t: exp(-t / mu) * surface_mean(t), 0, tau, epsabs=1e-10)[0]
    )
    before_surface_two = (
        2
        * rho
        * omega
        * exp(-tau / mu)
        * quad(
            lambda t: volume_mean(t) * float(expn(2, tau - t)),
            0,
            tau,
            epsabs=1e-9,
        )[0]
    )
    split_surface_two = after_surface_one * reflected_one_boundary / direct_boundary

    def surface_once_mean(t: float) -> float:
        # The isotropic angular transport kernel is E1(|t-s|)/2. Split its
        # integrable logarithmic singularity rather than discretising cell rays.
        below = quad(lambda s: surface_mean(s) * float(expn(1, s - t)), t, tau, epsabs=1e-10)[0]
        above = quad(lambda s: surface_mean(s) * float(expn(1, t - s)), 0, t, epsabs=1e-10)[0]
        return omega / 2 * (below + above)

    after_surface_two = (
        omega
        / mu
        * quad(
            lambda t: exp(-t / mu) * surface_once_mean(t),
            0,
            tau,
            epsabs=1e-9,
        )[0]
    )
    # Additional paths absent in the old one-reflection closure. These integrals
    # reflect every downward order at the same atmospheric count, including
    # sunlight -> ground -> atmosphere -> ground at atmospheric order one.
    return_boundary = (
        2
        * rho
        * omega
        * quad(lambda t: surface_mean(t) * float(expn(2, tau - t)), 0, tau, epsabs=1e-10)[0]
    )
    surface_twice_boundary = (
        2
        * rho
        * omega
        * quad(lambda t: surface_once_mean(t) * float(expn(2, tau - t)), 0, tau, epsabs=1e-9)[0]
    )
    beam_one = (
        incident * omega / (4 * pi * mu) * -expm1(-tau * (1 / mu + 1 / mu0)) / (1 / mu + 1 / mu0)
    )
    attenuation = exp(-tau / mu)
    order_one = beam_one + before_surface_one + after_surface_one + return_boundary * attenuation
    order_two = (
        volume_two
        + before_surface_two
        + split_surface_two
        + after_surface_two
        + after_surface_one * return_boundary / direct_boundary
        + attenuation
        * (
            surface_twice_boundary
            + return_boundary * (reflected_one_boundary + return_boundary) / direct_boundary
        )
    )
    assert _value(result, "atmospheric_order_1") == pytest.approx(order_one, rel=2e-3, abs=2e-8)
    assert _value(result, "atmospheric_order_2") == pytest.approx(order_two, rel=2e-3, abs=2e-8)
    old_one_reflection = beam_one + before_surface_one + after_surface_one
    assert _value(result, "atmospheric_order_1") > old_one_reflection


def test_disjoint_rayleigh_and_isotropic_aerosol_layers_have_distinct_nonzero_cross_paths() -> None:
    scene, solar = _inputs(layers=2, reflectance=0.0)
    lower, upper, omega_a, incident = 0.08, 0.12, 0.5, 8.0
    mu0, mu = cos(radians(30)), cos(radians(20))
    molecular = RayleighComponent(
        wavenumber_grid(solar.wavenumbers_cm1), np.array([[lower], [0.0]])
    )
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.0], [upper]]),
        np.array([[0.0], [upper * omega_a]]),
        np.zeros((2, 1)),
    )
    result = terms_support.reflected_radiance(
        scene,
        source=solar_source(solar, direction=(30.0, 70.0)),
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            solar_source(solar, direction=(30.0, 70.0)),
            rayleigh=molecular,
            aerosol=aerosol,
        ),
        angular_order=16,
        layer_subdivisions=64,
    )

    def rayleigh_azimuth_mean(u: float, v: float) -> float:
        return 0.75 * (1 + u * u * v * v + (1 - u * u) * (1 - v * v) / 2)

    def rayleigh_then_aerosol(u: float) -> float:
        beam_layer = exp(-upper / mu0) * -expm1(-lower * (1 / mu0 + 1 / u)) / (1 + u / mu0)
        escape_layer = quad(lambda t: exp(-(upper - t) / u - t / mu), 0, upper, epsabs=1e-11)[0]
        return rayleigh_azimuth_mean(mu0, u) * beam_layer * escape_layer

    ra = incident * omega_a / (8 * pi * mu) * quad(rayleigh_then_aerosol, 0, 1, epsabs=1e-10)[0]

    def aerosol_then_rayleigh(u: float) -> float:
        escape = exp(-upper / mu) * -expm1(-lower * (1 / mu + 1 / u)) / (1 / mu + 1 / u)
        return rayleigh_azimuth_mean(mu, u) * _down_beam(upper, u, mu0) * escape

    ar = (
        incident
        * omega_a
        / (8 * pi * mu)
        * quad(
            aerosol_then_rayleigh,
            0,
            1,
            points=[mu0],
            epsabs=1e-10,
        )[0]
    )
    # Isolate cross contributions by subtraction at FIXED total extinction;
    # output aggregation does not expose individual constituent sequences.
    # Suppress molecular redistribution by replacing its extinction with
    # artificial absorption in the aerosol record, not a nonconservative Rayleigh.
    # This is a matched-extinction numerical experiment, not new aerosol physics.
    isolated = []
    for rayleigh, hg in (
        (molecular, replace(aerosol, scattering_depth=np.zeros((2, 1)))),
        (None, replace(aerosol, optical_depth=aerosol.optical_depth + molecular.optical_depth)),
    ):
        isolated.append(
            terms_support.reflected_radiance(
                scene,
                source=solar_source(solar, direction=(30.0, 70.0)),
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    solar_source(solar, direction=(30.0, 70.0)),
                    rayleigh=rayleigh,
                    aerosol=hg,
                ),
                angular_order=16,
                layer_subdivisions=64,
            )
        )
    cross = _value(result, "atmospheric_order_2") - sum(
        _value(r, "atmospheric_order_2") for r in isolated
    )
    assert cross == pytest.approx(ra + ar, rel=2e-3, abs=2e-8)
    assert ra > 0 and ar > 0
    assert ra != pytest.approx(ar, rel=0.01)


def test_vertical_refinement_converges_to_independent_isotropic_order_two() -> None:
    scene, solar = _inputs(reflectance=0.0)
    tau, omega = 0.4, 0.5
    mu0, mu = cos(radians(30)), cos(radians(20))
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[tau]]),
        np.array([[tau * omega]]),
        np.zeros((1, 1)),
    )
    reference = (
        omega
        / mu
        * quad(
            lambda t: exp(-t / mu) * _isotropic_mean(t, tau, mu0, 8.0, omega),
            0,
            tau,
            epsabs=1e-9,
        )[0]
    )
    errors = []
    for subdivisions in (4, 16, 64):
        result = terms_support.reflected_radiance(
            scene,
            source=solar_source(solar, direction=(30.0, 70.0)),
            optics=compose_atmospheric_optics(
                scene.atmosphere, solar_source(solar, direction=(30.0, 70.0)), aerosol=aerosol
            ),
            angular_order=16,
            layer_subdivisions=subdivisions,
        )
        errors.append(abs(_value(result, "atmospheric_order_2") - reference))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / reference < 2e-3


@pytest.mark.parametrize("g", [-0.8, 0.8])
def test_forward_and_backward_peaked_scenes_need_and_respond_to_angular_refinement(
    g: float,
) -> None:
    scene, solar = _inputs()
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.2]]),
        np.array([[0.1]]),
        np.array([[g]]),
    )
    values = []
    for order in (4, 8, 16):
        result = terms_support.reflected_radiance(
            scene,
            source=solar_source(solar, direction=(30.0, 70.0)),
            optics=compose_atmospheric_optics(
                scene.atmosphere, solar_source(solar, direction=(30.0, 70.0)), aerosol=aerosol
            ),
            angular_order=order,
            layer_subdivisions=32,
        )
        values.append(result.values[0])
        assert all(c.values[0] >= 0 for c in result.components)
    assert abs(values[2] - values[1]) < abs(values[1] - values[0])
    # This is a refinement trend, not a validated error bound for |g|=0.8.


def test_scattering_order_scaling_and_nonlinear_repeated_surface_response() -> None:
    scene, solar = _inputs()
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.2]]),
        np.array([[0.04]]),
        np.array([[0.3]]),
    )
    source = solar_source(solar, direction=(30.0, 70.0))
    first = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=aerosol),
        angular_order=6,
    )
    doubled = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            source,
            aerosol=replace(aerosol, scattering_depth=np.array([[0.08]])),
        ),
        angular_order=6,
    )
    for n, (a, b) in enumerate(zip(first.components, doubled.components, strict=True)):
        scale = 2**n
        assert b.values[0] == pytest.approx(scale * a.values[0], rel=1e-13)
    brighter = replace(scene, surface=replace(scene.surface, reflectance=np.array([0.6])))
    second_surface = terms_support.reflected_radiance(
        brighter,
        source=source,
        optics=compose_atmospheric_optics(brighter.atmosphere, source, aerosol=aerosol),
        angular_order=6,
    )
    assert second_surface.components[0].values[0] == pytest.approx(
        2 * first.components[0].values[0]
    )
    black = terms_support.reflected_radiance(
        (_scene := replace(scene, surface=replace(scene.surface, reflectance=np.zeros(1)))),
        source=source,
        optics=compose_atmospheric_optics(_scene.atmosphere, source, aerosol=aerosol),
        angular_order=6,
    )
    for a, b, volume in zip(
        first.components[1:], second_surface.components[1:], black.components[1:], strict=True
    ):
        assert b.values[0] - volume.values[0] > 2 * (a.values[0] - volume.values[0])
    thin = replace(aerosol, optical_depth=np.array([[1e-7]]), scattering_depth=np.array([[5e-8]]))
    thicker = replace(
        aerosol, optical_depth=np.array([[2e-7]]), scattering_depth=np.array([[1e-7]])
    )
    black_scene = replace(scene, surface=replace(scene.surface, reflectance=np.zeros(1)))
    one = terms_support.reflected_radiance(
        black_scene,
        source=source,
        optics=compose_atmospheric_optics(black_scene.atmosphere, source, aerosol=thin),
        angular_order=6,
    )
    two = terms_support.reflected_radiance(
        black_scene,
        source=source,
        optics=compose_atmospheric_optics(black_scene.atmosphere, source, aerosol=thicker),
        angular_order=6,
    )
    assert _value(two, "atmospheric_order_1") / _value(one, "atmospheric_order_1") == pytest.approx(
        2, rel=1e-4
    )
    assert _value(two, "atmospheric_order_2") / _value(one, "atmospheric_order_2") == pytest.approx(
        4, rel=1e-4
    )


def test_same_physical_column_relayering_preserves_the_effective_cell_grid() -> None:
    one_layer, solar = _inputs()
    two_layers, _ = _inputs(layers=2)
    first = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.2]]),
        np.array([[0.1]]),
        np.array([[0.3]]),
    )
    split = replace(
        first,
        optical_depth=np.full((2, 1), 0.1),
        scattering_depth=np.full((2, 1), 0.05),
        asymmetry=np.full((2, 1), 0.3),
    )
    source = solar_source(solar, direction=(30.0, 70.0))
    one = terms_support.reflected_radiance(
        one_layer,
        source=source,
        optics=compose_atmospheric_optics(one_layer.atmosphere, source, aerosol=first),
        angular_order=6,
        layer_subdivisions=16,
    )
    two = terms_support.reflected_radiance(
        two_layers,
        source=source,
        optics=compose_atmospheric_optics(two_layers.atmosphere, source, aerosol=split),
        angular_order=6,
        layer_subdivisions=8,
    )
    assert one.values == pytest.approx(two.values, rel=1e-13)
    for a, b in zip(one.components, two.components, strict=True):
        assert a.values == pytest.approx(b.values, rel=1e-13)
