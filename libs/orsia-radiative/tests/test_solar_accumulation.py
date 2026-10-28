"""Bounded Rayleigh accumulation versus the original per-component/sweep recurrence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import pi
from unittest.mock import patch

import formal_reference
import numpy as np
import orders_reference as reference
import pytest
from orsia_radiative import (
    Absorption,
    Atmosphere,
    AtmosphericOptics,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.transfer import formal, orders
from test_solar_boundary import _sweep_reference
from test_sos import scene_inputs


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("max_order", [0, 1, 4, 16])
@pytest.mark.parametrize("reflectance", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("case", ["positive", "zero_layer", "zero_point", "dark", "opaque"])
def test_batched_orders_match_original_sweeps(
    native: bool,
    max_order: int,
    reflectance: float,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native:
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    scene, source, optics = _inputs(case, reflectance)
    controls = dict(max_order=max_order, angular_order=3, layer_subdivisions=2)
    with patch.object(reference, "_accumulate_orders", _sweep_reference):
        expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    with patch.object(
        orders, "_native_spectral_orders", wraps=orders._native_spectral_orders
    ) as batched:
        actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14, atol=0)
    assert actual.dtype == np.float64 and actual.flags.writeable
    assert np.isfinite(actual).all()
    assert batched.call_count == 1


@pytest.mark.parametrize("moments", [None, 4])
@pytest.mark.parametrize("native", [False, True])
def test_rayleigh_delta_m_and_mixed_role_dispatch(
    moments: int | None, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not native:
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
    scene, source, optics = _inputs("positive", 0.3)
    controls = dict(max_order=4, angular_order=4, layer_subdivisions=3, delta_m_moments=moments)
    with patch.object(reference, "_accumulate_orders", _sweep_reference):
        expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    with patch.object(reference, "_scatter", side_effect=AssertionError("generic scatter called")):
        actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    # Two-layer mixed/HG scenes retain their established per-component path.
    small_scene, irradiance, rayleigh, aerosol = scene_inputs()
    small_source = solar_source(irradiance, direction=(30.0, 20.0))
    mixed = compose_atmospheric_optics(
        small_scene.atmosphere, small_source, rayleigh=rayleigh, aerosol=aerosol
    )
    with patch.object(
        reference, "_native_rayleigh_orders", side_effect=AssertionError("Rayleigh-only path")
    ):
        orders.scattering_orders(small_scene, source=small_source, optics=mixed, **controls)


@pytest.mark.parametrize("roles", ["rayleigh", "both"])
@pytest.mark.parametrize("moments", [None, 4])
@pytest.mark.parametrize("amplitude", [0.0, 1e-200, 1.0, 1e200])
def test_vectorized_beam_matches_layer_scalar_formula(
    roles: str, moments: int | None, amplitude: float
) -> None:
    _, _, rayleigh, aerosol = scene_inputs()
    components = (rayleigh,) if roles == "rayleigh" else (rayleigh, aerosol)
    cosines = np.linspace(-1.0, 1.0, 33)
    expected = np.zeros((6, 33), dtype=np.float64)
    for component in components:
        is_rayleigh = isinstance(component, RayleighComponent)
        depths = component.optical_depth if is_rayleigh else component.scattering_depth
        for layer, row in enumerate(depths):
            g = (
                None
                if isinstance(component, RayleighComponent)
                else float(component.asymmetry[layer, 0])
            )
            phase = (
                orders.phase_function(cosines, g)
                if moments is None
                else reference._transport_phase(cosines, g, moments)
            )
            expected[layer * 3 : (layer + 1) * 3] += (
                amplitude
                / (4 * pi)
                * row[0]
                * (1 - reference._forward_fraction(g, moments))
                / 3
                * phase
            )
    actual = reference._beam_source(components, 0, amplitude, 3, cosines, 6, moments)
    np.testing.assert_array_equal(actual, expected)


def test_many_layer_accumulation_is_owned_and_concurrent() -> None:
    scene, source, optics = _inputs("positive", 0.3, layers=11)
    assert optics.rayleigh is not None and scene.surface.reflectance is not None
    arrays = (source.irradiance.values, optics.rayleigh.optical_depth, scene.surface.reflectance)
    snapshots = [array.copy() for array in arrays]

    def calculate(_: int):
        return orders.scattering_orders(
            scene, source=source, optics=optics, max_order=8, angular_order=4, layer_subdivisions=4
        )

    expected = calculate(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(calculate, range(8)))
    for result in results:
        np.testing.assert_array_equal(result, expected)
        assert not np.shares_memory(result, expected)
    for array, snapshot in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)
    results[0][:] = 0
    np.testing.assert_array_equal(calculate(0), expected)


def _inputs(
    case: str, reflectance: float, *, layers: int = 3
) -> tuple[RadiativeScene, SolarSource, AtmosphericOptics]:
    grid = wavenumber_grid(np.array([12000.0, 12001.0, 12002.0]))
    atmosphere = Atmosphere(
        np.arange(layers + 1, dtype=np.float64) * 1000,
        np.linspace(1e5, 2e4, layers + 1, dtype=np.float64),
        np.linspace(9.5e4, 2.5e4, layers, dtype=np.float64),
        np.full(layers, 280.0),
        np.full(layers, 2e29),
        np.full(layers, 1.9e29),
        np.full(layers, 1e28),
        (),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(grid, reflectance=np.full(3, reflectance)),
        ObservationGeometry(20.0),
    )
    irradiance = np.array([8.0, 10.0, 12.0])
    depth = np.linspace(0.02, 0.08, layers)[:, None] * np.array([1.0, 0.8, 1.2])
    absorption = np.full_like(depth, 0.01)
    if case == "zero_layer":
        depth[1] = 0
    elif case == "zero_point":
        depth[:, 1] = 0
    elif case == "dark":
        irradiance[:] = 0
    elif case == "opaque":
        absorption += 100
        irradiance *= 1e200
        scene = replace(scene, geometry=ObservationGeometry(89.0))
    source = solar_source(SpectralIrradiance(grid, irradiance), direction=(30.0, 20.0))
    optics = compose_atmospheric_optics(
        atmosphere,
        source,
        absorption=Absorption(grid, absorption),
        rayleigh=RayleighComponent(grid, depth),
    )
    return scene, source, optics
