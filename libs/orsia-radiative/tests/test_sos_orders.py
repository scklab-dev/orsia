"""Fixed-N checks with independent scalar ray integrals on the declared cell grid.

The oracle shares angular nodes and cell-centre collocation only. It uses scalar
adaptive ray integrals, separate phase expressions and an explicit reflected
boundary, not production propagation or redistribution. This is numerical
assembly evidence, not a continuum solution or an external-solver qualification.
"""

from dataclasses import replace
from math import cos, exp, pi, radians, sin

import numpy as np
import pytest
import radiance_terms_support as terms_support
from numpy.typing import NDArray
from orsia_radiative import (
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SolarSource,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from orsia_radiative.transfer.orders import angular_grid
from scipy.integrate import quad
from scipy.special import eval_legendre


def scene_inputs() -> tuple[
    RadiativeScene, SpectralIrradiance, RayleighComponent, AerosolComponent
]:
    """Two-point stratified synthetic optics for independent order checks."""
    axis = np.array([12000.0, 12001.0], dtype=np.float64)
    scene = RadiativeScene(
        Atmosphere(
            np.array([0.0, 1000.0, 2000.0]),
            np.array([100000.0, 90000.0, 80000.0]),
            np.array([95000.0, 85000.0]),
            np.array([290.0, 280.0]),
            np.array([2e29, 1.8e29]),
            np.array([1.9e29, 1.7e29]),
            np.array([1e28, 1e28]),
        ),
        SurfaceState(wavenumber_grid(axis), reflectance=np.array([0.2, 0.0])),
        ObservationGeometry(20.0),
    )
    sun = SpectralIrradiance(wavenumber_grid(axis), np.array([8.0, 10.0]))
    rayleigh = RayleighComponent(wavenumber_grid(axis), np.array([[0.04, 0.03], [0.01, 0.02]]))
    aerosol = AerosolComponent(
        wavenumber_grid(axis),
        np.array([[0.02, 0.03], [0.06, 0.05]]),
        np.array([[0.01, 0.01], [0.04, 0.03]]),
        np.array([[0.2, -0.1], [0.4, 0.0]]),
    )
    return scene, sun, rayleigh, aerosol


@pytest.mark.parametrize("max_order", [0, 1, 2, 5])
@pytest.mark.parametrize("rho", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("delta_m_moments", [None, 2, 3, 4])
def test_all_orders_match_independent_discrete_ray_reference(
    max_order: int, rho: float, delta_m_moments: int | None
) -> None:
    scene, sun, rayleigh, aerosol = scene_inputs()
    scene = replace(scene, surface=replace(scene.surface, reflectance=np.array([rho, rho])))
    constituents = (rayleigh, aerosol)
    source = solar_source(sun, direction=(30.0, 70.0))
    actual = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
        ),
        max_order=max_order,
        angular_order=2,
        layer_subdivisions=2,
        delta_m_moments=delta_m_moments,
    )
    for point in range(2):
        reference = _orders(constituents, point, sun.values[point], rho, max_order, delta_m_moments)
        np.testing.assert_allclose(
            [c.values[point] for c in actual.components], reference, rtol=2e-13, atol=1e-15
        )


def test_fixed_extinction_scattering_scaling_through_order_six() -> None:
    scene, sun, _, aerosol = scene_inputs()
    source = solar_source(sun, direction=(30.0, 70.0))
    base = replace(aerosol, scattering_depth=np.array([[0.004, 0.005], [0.01, 0.008]]))
    doubled = replace(base, scattering_depth=2 * base.scattering_depth)
    results = [
        terms_support.reflected_radiance(
            scene,
            source=source,
            optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=c),
            max_order=6,
            angular_order=3,
            layer_subdivisions=3,
        )
        for c in (base, doubled)
    ]
    for n, (a, b) in enumerate(zip(results[0].components, results[1].components, strict=True)):
        np.testing.assert_allclose(b.values, np.asarray(a.values) * 2**n, rtol=3e-13, atol=1e-18)


def test_prefixes_and_last_order_diagnostics_are_not_tail_bounds() -> None:
    scene, sun, rayleigh, aerosol = scene_inputs()
    source = solar_source(sun, direction=(30.0, 70.0))
    results = [
        terms_support.reflected_radiance(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
            ),
            max_order=n,
            angular_order=3,
            layer_subdivisions=3,
        )
        for n in range(6)
    ]
    for n, result in enumerate(results):
        assert len(result.components) == n + 1
        np.testing.assert_allclose(
            result.values, np.sum([c.values for c in result.components], axis=0)
        )
        expected = np.zeros(2)
        np.divide(
            result.components[-1].values,
            result.values,
            out=expected,
            where=np.asarray(result.values) != 0,
        )
        for a, b in zip(result.components, results[-1].components[: n + 1], strict=True):
            np.testing.assert_allclose(a.values, b.values, rtol=3e-13, atol=1e-18)
        if n:
            np.testing.assert_allclose(
                np.asarray(result.values) - results[n - 1].values,
                result.components[-1].values,
                rtol=1e-9,
                atol=1e-16,
            )


def test_zero_source_has_finite_zero_diagnostics_and_amplitude_scales_each_order() -> None:
    scene, sun, rayleigh, aerosol = scene_inputs()
    source = solar_source(sun, direction=(30.0, 70.0))
    other = replace(source, irradiance=replace(sun, values=2 * sun.values))
    zero = replace(source, irradiance=replace(sun, values=np.zeros(2)))

    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )

    def calculate(source: SolarSource) -> terms_support.RadianceDecomposition:
        return terms_support.reflected_radiance(
            scene,
            source=source,
            optics=optics,
            max_order=4,
            angular_order=3,
            layer_subdivisions=3,
        )

    empty = calculate(zero)
    assert all(np.all(c.values == 0) for c in empty.components)
    first, second = calculate(source), calculate(other)
    np.testing.assert_allclose(second.values, 2 * first.values, rtol=3e-13)
    for original, scaled in zip(first.components, second.components, strict=True):
        np.testing.assert_allclose(scaled.values, 2 * original.values, rtol=3e-13)


def _orders(
    constituents: tuple[RayleighComponent | AerosolComponent, ...],
    point: int,
    incident: float,
    rho: float,
    max_order: int,
    delta_m_moments: int | None = None,
) -> list[float]:
    directions, weights = angular_grid(2)
    mu0 = cos(radians(30))
    view = np.array(
        [sin(radians(20)) * cos(radians(70)), sin(radians(20)) * sin(radians(70)), cos(radians(20))]
    )
    outgoing = np.vstack((directions, view))
    sun = np.array([sin(radians(30)), 0.0, -mu0])
    total_depth = np.zeros(constituents[0].optical_depth.shape[0], dtype=np.float64)
    for component in constituents:
        total_depth += component.optical_depth[:, point]
    if delta_m_moments is not None:
        for c in constituents:
            if isinstance(c, AerosolComponent):
                for layer, g in enumerate(c.asymmetry[:, point]):
                    if g > 0:
                        total_depth[layer] -= g**delta_m_moments * c.scattering_depth[layer, point]
    depth = np.repeat(total_depth / 2, 2)
    # Vertical optical coordinate increases from the ground toward the top.
    edges = np.concatenate(([0.0], np.cumsum(depth)))
    middle = (edges[:-1] + edges[1:]) / 2
    bottom = incident * rho * mu0 / pi * exp(-edges[-1] / mu0)
    field = np.array(
        [[bottom * exp(-z / d[2]) if d[2] > 0 else 0.0 for d in directions] for z in middle]
    )
    values = [bottom * exp(-edges[-1] / view[2])]
    for n in range(1, max_order + 1):
        density = np.zeros((depth.size, outgoing.shape[0]))
        beam = np.zeros_like(density)
        for cell in range(depth.size):
            layer = cell // 2
            for component in constituents:
                g = (
                    None
                    if isinstance(component, RayleighComponent)
                    else float(component.asymmetry[layer, point])
                )
                scattering_depth = (
                    component.optical_depth
                    if isinstance(component, RayleighComponent)
                    else component.scattering_depth
                )
                omega = scattering_depth[layer][point] / (2 * depth[cell])
                if g is not None and g > 0 and delta_m_moments is not None:
                    omega *= 1 - g**delta_m_moments
                for j, out in enumerate(outgoing):
                    density[cell, j] += omega * sum(
                        w * value * _phase(float(incoming @ out), g, delta_m_moments)
                        for w, value, incoming in zip(weights, field[cell], directions, strict=True)
                    )
                    if n == 1:
                        beam[cell, j] += (
                            incident
                            / (4 * pi)
                            * omega
                            * _phase(float(sun @ out), g, delta_m_moments)
                        )
        centres = np.empty_like(field)
        exits = []
        for j, direction in enumerate(outgoing):
            mu = float(direction[2])
            end = edges[-1] if mu > 0 else 0.0
            exits.append(_ray(end, mu, edges, density[:, j], beam[:, j], mu0))
            if j < directions.shape[0]:
                for cell, z in enumerate(middle):
                    centres[cell, j] = _ray(z, mu, edges, density[:, j], beam[:, j], mu0)
        boundary = (
            4
            * rho
            * sum(
                w * -d[2] * v
                for w, d, v in zip(weights, directions, exits[:-1], strict=True)
                if d[2] < 0
            )
        )
        for cell, z in enumerate(middle):
            for j, direction in enumerate(directions):
                if direction[2] > 0:
                    centres[cell, j] += boundary * exp(-z / direction[2])
        values.append(exits[-1] + boundary * exp(-edges[-1] / view[2]))
        field = centres
    return values


def _phase(cosine: float, g: float | None, moments: int | None = None) -> float:
    if g is not None and g > 0 and moments is not None:
        return sum(
            (2 * degree + 1)
            * (g**degree - g**moments)
            / (1 - g**moments)
            * float(eval_legendre(degree, cosine))
            for degree in range(moments)
        )
    return 0.75 * (1 + cosine**2) if g is None else (1 - g**2) / (1 + g**2 - 2 * g * cosine) ** 1.5


def _ray(
    z: float,
    mu: float,
    edges: NDArray[np.float64],
    density: NDArray[np.float64],
    beam: NDArray[np.float64],
    mu0: float,
) -> float:
    total = 0.0
    for cell in range(density.size):
        lower, upper = edges[cell : cell + 2]
        if mu > 0:
            upper = min(upper, z)
        else:
            lower = max(lower, z)
        if upper <= lower:
            continue
        total += quad(
            lambda x, cell=cell: (
                (density[cell] + beam[cell] * exp(-(edges[-1] - x) / mu0))
                * exp(-abs(z - x) / abs(mu))
                / abs(mu)
            ),
            lower,
            upper,
            epsabs=1e-14,
            epsrel=1e-13,
        )[0]
    return total
