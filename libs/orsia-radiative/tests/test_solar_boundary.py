"""Boundary-response reuse versus repeated formal sweeps on the same cell grid."""

from dataclasses import replace
from math import cos, pi, radians, sin
from unittest.mock import patch

import formal_reference
import numpy as np
import orders_reference as reference
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    Absorption,
    RadiativeScene,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
)
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.transfer import formal, orders
from test_sos import scene_inputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("amplitude", [0.0, 1e-12, 0.3, 1e12])
@pytest.mark.parametrize("thick", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_unit_boundary_scaling_matches_recurrence(
    dtype, amplitude: float, thick: bool, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not native:
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    # Include transparent cells, underflow, grazing and interleaved directions.
    depth = np.array([0.0, 0.01, 0.2, 0.0, 0.5] * 12, dtype=dtype)
    if thick:
        depth *= 100
    mu = np.array([0.9, -0.2, 1e-3, -1.0, 0.4], dtype=dtype)
    coefficients = formal.propagation_coefficients(depth, mu)
    zero = np.zeros((depth.size, mu.size), dtype=dtype)
    unit_centres, unit_exits = formal.apply_propagation(zero, coefficients, bottom_radiance=1.0)
    centres, exits = formal.apply_propagation(zero, coefficients, bottom_radiance=amplitude)
    tolerance = 64 * np.finfo(dtype).eps
    # Scaling subnormal unit responses has amplitude-dependent absolute error;
    # SOS falls back to full sweeps there. Formal float32 keeps its recurrence.
    underflow_bound = 64 * float(np.finfo(dtype).smallest_subnormal) * max(1.0, amplitude)
    np.testing.assert_allclose(
        unit_centres * amplitude, centres, rtol=tolerance, atol=underflow_bound
    )
    np.testing.assert_allclose(unit_exits * amplitude, exits, rtol=tolerance, atol=underflow_bound)
    np.testing.assert_array_equal(centres[:, mu < 0], 0)
    np.testing.assert_array_equal(exits[mu < 0], 0)
    assert centres.dtype == dtype and exits.dtype == dtype
    np.testing.assert_array_equal(zero, 0)


@pytest.mark.parametrize("max_order", [0, 1, 4, 16])
@pytest.mark.parametrize("roles", ["rayleigh", "aerosol", "both"])
@pytest.mark.parametrize("reflectance", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("extreme", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_orders_match_repeated_boundary_sweeps(
    max_order: int,
    roles: str,
    reflectance: float,
    extreme: bool,
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native:
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    scene, irradiance, molecular, aerosol = scene_inputs()
    scene = replace(
        scene,
        surface=SurfaceState(irradiance.grid, reflectance=np.full(2, reflectance)),
    )
    if extreme:
        scene = replace(scene, geometry=replace(scene.geometry, view_zenith_deg=89.0))
        irradiance = replace(irradiance, values=irradiance.values * 1e200)
    source = solar_source(irradiance, direction=(30.0, 20.0))
    absorption = np.array([[0.0, 0.4], [0.02, 0.0]])
    if extreme:
        absorption += 100.0
    optics = compose_atmospheric_optics(
        scene.atmosphere,
        source,
        absorption=Absorption(irradiance.grid, absorption),
        rayleigh=molecular if roles != "aerosol" else None,
        aerosol=aerosol if roles != "rayleigh" else None,
    )
    controls = dict(max_order=max_order, angular_order=4, layer_subdivisions=3)
    with patch.object(reference, "_accumulate_orders", _sweep_reference):
        expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    with patch.object(
        orders, "_native_spectral_orders", wraps=orders._native_spectral_orders
    ) as calls:
        actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14, atol=0)
    assert calls.call_count == 1
    assert np.isfinite(actual).all()


def _sweep_reference(
    scene: RadiativeScene,
    source: SolarSource,
    extinction: NDArray[np.float64],
    scattering,
    contributions: NDArray[np.float64],
    *,
    angular_order: int,
    layer_subdivisions: int,
    delta_m_moments: int | None,
) -> None:
    """Pre-reuse discrete calculation: sweep each actual boundary amplitude.

    Shared phase/source kernels isolate this regression to the changed boundary
    calculation; independent physical checks live in test_sos_numerics.py.
    This reference deliberately retains every centre field, including the last.
    """
    max_order = contributions.shape[0] - 1
    mu0 = cos(radians(source.zenith_deg))
    zenith, azimuth = radians(scene.geometry.view_zenith_deg), radians(source.relative_azimuth_deg)
    view = np.array([sin(zenith) * cos(azimuth), sin(zenith) * sin(azimuth), cos(zenith)])
    sun = np.array([sin(radians(source.zenith_deg)), 0.0, -mu0])
    directions, weights = orders.angular_grid(angular_order)
    outgoing = np.vstack((directions, view))
    factors = reference._prepare_rayleigh_factors(directions, outgoing, weights)
    mu, cosines = outgoing[:, 2], directions @ outgoing.T
    reflectance = scene.surface.reflectance
    assert reflectance is not None
    for point in range(contributions.shape[1]):
        depth = np.repeat(extinction[:, point] / layer_subdivisions, layer_subdivisions)
        transport = formal.propagation_coefficients(depth, mu, solar_mu=mu0)
        zero = np.zeros((depth.size, mu.size))
        amplitude = source.irradiance.values[point] * reflectance[point] * mu0 / pi
        amplitude *= np.exp(-depth.sum() / mu0)
        field, _ = formal.apply_propagation(zero, transport, bottom_radiance=float(amplitude))
        for order in range(1, max_order + 1):
            redistributed = np.zeros_like(zero)
            for component in scattering:
                reference._scatter(
                    field[:, :-1],
                    component,
                    point,
                    layer_subdivisions,
                    cosines,
                    weights,
                    {},
                    out=redistributed,
                    rayleigh_factors=factors,
                    delta_m_moments=delta_m_moments,
                )
            centres, exits = formal.apply_propagation(redistributed, transport)
            if order == 1:
                beam = reference._beam_source(
                    scattering,
                    point,
                    source.irradiance.values[point],
                    layer_subdivisions,
                    outgoing @ sun,
                    depth.size,
                    delta_m_moments,
                )
                added_centres, added_exits = formal.apply_propagation(
                    beam, transport, solar_source=True
                )
                centres += added_centres
                exits += added_exits
            boundary = reference._reflected(exits[:-1], mu[:-1], weights, reflectance[point])
            if boundary != 0:
                added_centres, added_exits = formal.apply_propagation(
                    zero, transport, bottom_radiance=boundary
                )
                centres += added_centres
                exits += added_exits
            contributions[order, point] = exits[-1]
            field = centres
