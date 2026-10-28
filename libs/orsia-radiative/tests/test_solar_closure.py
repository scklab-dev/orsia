"""Bounded reflected-solar limits and independent first-order ray integrals."""

from dataclasses import replace
from math import cos, exp, pi, radians, sin

import numpy as np
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    SpectralIrradiance,
    SurfaceState,
    compose_absorption,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from scipy.integrate import quad


def _scene() -> tuple[RadiativeScene, SpectralIrradiance]:
    axis = np.array([900.0, 901.0])
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([285.0]),
        np.array([2.0e29]),
        np.array([1.9e29]),
        np.array([1.0e28]),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(axis), reflectance=np.array([0.4, 0.0])),
        ObservationGeometry(21.0),
    )
    return scene, SpectralIrradiance(wavenumber_grid(axis), np.array([8.0, 8.0]))


def _hg_phase_mean(g: float, first: float, second: float) -> float:
    # Independent adaptive azimuthal integral for the continuous ray reference.
    return quad(
        lambda phi: (
            (1 - g * g)
            / (
                1
                + g * g
                - 2
                * g
                * (first * second + ((1 - first * first) * (1 - second * second)) ** 0.5 * cos(phi))
            )
            ** 1.5
        ),
        0.0,
        2 * pi,
        epsabs=1e-10,
    )[0] / (2 * pi)


@pytest.mark.parametrize("g", [0.5, -0.4])
def test_hg_first_order_matches_independent_ground_coupled_ray_integrals(g: float) -> None:
    scene, solar = _scene()
    axis = solar.wavenumbers_cm1
    absorption = AbsorptionComponent(
        wavenumber_grid(axis), "X", None, "lorentz_lines", np.array([[0.08, 0.08]])
    )
    # A declared gas is required for absorption inputs, independent of transfer.
    atmosphere = replace(
        scene.atmosphere,
        absorbers=(Absorber("X", np.array([1e25]), "dry_air"),),
    )
    scene = replace(scene, atmosphere=atmosphere)
    aerosol = AerosolComponent(
        wavenumber_grid(axis),
        np.array([[0.12, 0.12]]),
        np.array([[0.07, 0.07]]),
        np.array([[g, g]]),
    )
    result = terms_support.order_one(
        scene,
        angular_order=16,
        layer_subdivisions=128,
        source=solar_source(solar, direction=(32.0, 70.0)),
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            solar_source(solar, direction=(32.0, 70.0)),
            absorption=compose_absorption(
                scene.atmosphere, solar_source(solar, direction=(32.0, 70.0)), (absorption,)
            ),
            aerosol=aerosol,
        ),
    )
    mu0 = cos(radians(32.0))
    mu = cos(radians(21.0))
    tau, scatter, rho, incident = 0.2, 0.07, 0.4, 8.0
    phase_cosine = -mu0 * mu + sin(radians(32.0)) * sin(radians(21.0)) * cos(radians(70.0))
    direct = incident * rho * mu0 / pi * exp(-tau / mu0 - tau / mu)
    atmospheric = (
        incident
        * scatter
        / (4 * pi * mu)
        * (1 - g * g)
        / (1 + g * g - 2 * g * phase_cosine) ** 1.5
        * quad(lambda t: exp(-tau * (1 - t) * (1 / mu0 + 1 / mu)), 0, 1)[0]
    )

    def down(u: float) -> float:
        layer = quad(
            lambda t: exp(-tau * (1 - t) / mu0 - tau * t / u),
            0.0,
            1.0,
        )[0]
        return _hg_phase_mean(g, mu0, u) * layer

    def up(u: float) -> float:
        layer = quad(
            lambda t: exp(-tau * t / u - tau * (1 - t) / mu),
            0.0,
            1.0,
        )[0]
        return _hg_phase_mean(g, u, mu) * layer

    expected = (
        direct,
        atmospheric,
        incident * rho / (2 * pi) * exp(-tau / mu) * scatter * quad(down, 0.0, 1.0, epsabs=1e-9)[0],
        incident
        * rho
        * mu0
        / (2 * pi * mu)
        * exp(-tau / mu0)
        * scatter
        * quad(up, 0.0, 1.0, epsabs=1e-9)[0],
    )
    assert result.components[0].values[0] == pytest.approx(direct, rel=1e-14)
    # Integrate sunlight -> ground -> atmosphere -> ground -> observer using
    # independent hemisphere nodes and adaptive azimuth/layer ray integrals.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    repeated_ground = 0.0
    for u, wu in zip((nodes + 1) / 2, weights / 2, strict=True):
        for v, wv in zip((nodes + 1) / 2, weights / 2, strict=True):
            ray = quad(lambda t, u=u, v=v: exp(-tau * t * (1 / u + 1 / v)), 0, 1)[0]
            repeated_ground += wu * wv * _hg_phase_mean(g, u, -v) * ray
    repeated_ground *= direct * rho * scatter
    assert result.components[1].values[0] > sum(expected[1:])
    assert result.components[1].values[0] == pytest.approx(
        sum(expected[1:]) + repeated_ground, rel=5e-4, abs=1e-10
    )
    assert result.values[1] == pytest.approx(atmospheric, rel=1e-13)


def test_zero_illumination_black_surface_and_non_scattering_extinction_limits() -> None:
    scene, solar = _scene()
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.4, 0.4]]),
        np.zeros((1, 2)),
        np.array([[0.5, 0.5]]),
    )
    source = solar_source(solar, direction=(32.0, 70.0))
    transparent = terms_support.order_zero(
        scene, source=source, optics=compose_atmospheric_optics(scene.atmosphere, source)
    )
    unscattered = terms_support.order_zero(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=aerosol),
    )
    with_source = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=aerosol),
    )
    mu0, mu = cos(radians(32)), cos(radians(21))
    assert unscattered.values[0] == pytest.approx(
        transparent.values[0] * exp(-0.4 / mu0 - 0.4 / mu)
    )
    np.testing.assert_array_equal(with_source.values, unscattered.values)
    assert all(np.all(c.values == 0) for c in with_source.components[1:])
    no_light = terms_support.order_one(
        scene,
        source=solar_source(replace(solar, values=np.zeros(2)), direction=(32.0, 70.0)),
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            solar_source(replace(solar, values=np.zeros(2)), direction=(32.0, 70.0)),
            aerosol=aerosol,
        ),
    )
    np.testing.assert_array_equal(no_light.values, np.zeros(2))
    assert transparent.values[1] == 0  # Black surface; no atmospheric scattering.
    assert all(np.isfinite(no_light.values))


def test_positive_scattering_above_black_surface_and_extreme_finite_depth() -> None:
    scene, solar = _scene()
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.1, 700.0]]),
        np.array([[0.05, 0.1]]),
        np.array([[-0.5, 0.5]]),
    )
    result = terms_support.order_one(
        scene,
        source=solar_source(solar, direction=(32.0, 70.0)),
        optics=compose_atmospheric_optics(
            scene.atmosphere, solar_source(solar, direction=(32.0, 70.0)), aerosol=aerosol
        ),
    )
    assert result.components[0].values[1] == 0.0
    assert result.components[1].values[1] >= 0
    assert all(c.values[1] == 0.0 for c in result.components[2:])
    assert all(np.isfinite(result.values))


def test_illumination_and_optical_values_need_no_source_identity() -> None:
    scene, solar = _scene()
    aerosol = AerosolComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([[0.1, 0.1]]),
        np.array([[0.05, 0.05]]),
        np.zeros((1, 2)),
    )
    source = solar_source(solar, direction=(32.0, 70.0))
    for method in (terms_support.order_zero, terms_support.order_one):
        result = method(
            scene,
            source=source,
            optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=aerosol),
        )
        assert np.all(np.isfinite(result.values))
    scene = replace(
        scene,
        atmosphere=replace(
            scene.atmosphere, absorbers=(Absorber("X", np.array([1e25]), "dry_air"),)
        ),
    )
    gas = AbsorptionComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        "X",
        None,
        "lorentz_lines",
        np.array([[0.1, 0.1]]),
    )
    for method in (terms_support.order_zero, terms_support.order_one):
        result = method(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                source,
                absorption=compose_absorption(scene.atmosphere, source, (gas,)),
            ),
        )
        assert np.all(np.isfinite(result.values))
