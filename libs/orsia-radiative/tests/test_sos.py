"""Typed solar inputs, per-order provenance and deterministic fixed-N SOS contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import pi
from typing import Any, cast
from unittest.mock import patch

import formal_reference
import numpy as np
import orders_reference as reference
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    AerosolComponent,
    AerosolOptics,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    ScatteringProperties,
    SpectralIrradiance,
    SurfaceState,
    aerosol_component,
    compose_absorption,
    compose_atmospheric_optics,
    rayleigh_component,
    solar_source,
    transfer,
    validate_optical_inputs,
    wavenumber_grid,
)
from orsia_radiative.transfer.formal import propagate
from orsia_radiative.transfer.orders import angular_grid, phase_function


def scene_inputs() -> tuple[
    RadiativeScene, SpectralIrradiance, RayleighComponent, AerosolComponent
]:
    axis = np.array([12000.0, 12001.0], dtype=np.float64)
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([290.0, 280.0]),
        np.array([2e29, 1.8e29]),
        np.array([1.9e29, 1.7e29]),
        np.array([1e28, 1e28]),
        (Absorber("X", np.array([1e25, 1e25]), "dry_air"),),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(axis), reflectance=np.array([0.2, 0.0])),
        ObservationGeometry(20.0),
    )
    solar = SpectralIrradiance(wavenumber_grid(axis), np.array([8.0, 10.0]))
    molecular = RayleighComponent(wavenumber_grid(axis), np.array([[0.04, 0.03], [0.01, 0.02]]))
    aerosol = AerosolComponent(
        wavenumber_grid(axis),
        np.array([[0.02, 0.03], [0.06, 0.05]]),
        np.array([[0.01, 0.01], [0.04, 0.03]]),
        np.array([[0.2, -0.1], [0.4, 0.0]]),
    )
    return scene, solar, molecular, aerosol


@pytest.mark.parametrize("calculation", ["sos_1", "sos_2"])
@pytest.mark.parametrize("constituents", ["rayleigh", "hg", "both"])
def test_phase_factorisation_preserves_every_transfer_component(
    calculation: str, constituents: str
) -> None:
    from orsia_radiative.transfer import orders

    scene, sun, molecular, aerosol = scene_inputs()
    aerosol = replace(aerosol, asymmetry=np.array([[-0.8, 0.0], [0.8, -0.2]]))
    source = solar_source(sun, direction=(30.0, 20.0))

    def calculate() -> terms_support.RadianceDecomposition:
        return terms_support.reflected_radiance(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                source,
                rayleigh=None if constituents == "hg" else molecular,
                aerosol=None if constituents == "rayleigh" else aerosol,
            ),
            max_order=1 if calculation == "sos_1" else 2,
            angular_order=3,
            layer_subdivisions=3,
        )

    # Preserve the pre-factorisation numerical expressions as an independent
    # local regression oracle; do not call the shared production phase laws.
    def legacy_hg(asymmetry: Any, cosine: Any) -> Any:
        return (1 - asymmetry**2) / (1 + asymmetry**2 - 2 * asymmetry * cosine) ** 1.5

    def legacy_rayleigh(cosine: Any) -> Any:
        return 0.75 * (1 + cosine**2)

    def legacy_dispatch(cosine: Any, g: float | None) -> Any:
        cosine = np.clip(cosine, -1, 1)
        return legacy_rayleigh(cosine) if g is None else legacy_hg(g, cosine)

    actual = calculate()
    with (
        patch.object(reference, "phase_function", side_effect=legacy_dispatch),
        patch.object(orders, "scattering_orders", reference.scattering_orders),
    ):
        expected = calculate()
    np.testing.assert_allclose(actual.values, expected.values, rtol=3e-13, atol=1e-15)
    assert len(actual.components) == len(expected.components)
    for component, expected_component in zip(actual.components, expected.components, strict=True):
        np.testing.assert_allclose(
            component.values, expected_component.values, rtol=3e-13, atol=1e-15
        )
        assert component.name == expected_component.name
        assert component.role == expected_component.role
    assert actual.grid == expected.grid
    assert actual.geometry == expected.geometry


@pytest.mark.parametrize("max_order", [1, 4, 16])
@pytest.mark.parametrize("constituents", ["rayleigh", "hg", "both"])
def test_native_preparation_preserves_radiance_and_each_order(
    monkeypatch: pytest.MonkeyPatch, max_order: int, constituents: str
) -> None:
    from orsia_radiative.transfer import formal

    scene, sun, molecular, aerosol = scene_inputs()
    source = solar_source(sun, direction=(30.0, 20.0))
    optics = compose_atmospheric_optics(
        scene.atmosphere,
        source,
        rayleigh=None if constituents == "hg" else molecular,
        aerosol=None if constituents == "rayleigh" else aerosol,
    )

    def calculate() -> terms_support.RadianceDecomposition:
        return terms_support.reflected_radiance(
            scene,
            source=source,
            optics=optics,
            max_order=max_order,
            angular_order=8,
            layer_subdivisions=16,
        )

    assert formal._native_prepare is not None
    actual = calculate()
    monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    with patch.object(transfer.solar.orders, "scattering_orders", reference.scattering_orders):
        expected = calculate()
    np.testing.assert_allclose(actual.values, expected.values, rtol=2e-13, atol=1e-15)
    for a, e in zip(actual.components, expected.components, strict=True):
        assert a.name == e.name and a.role == e.role
        np.testing.assert_allclose(a.values, e.values, rtol=2e-13, atol=1e-15)
    assert actual.grid == expected.grid and actual.geometry == expected.geometry


@pytest.mark.parametrize("max_order", [1, 2, 4])
def test_sos_uses_one_full_spectrum_native_call(max_order: int) -> None:
    from orsia_radiative.transfer import orders

    scene, sun, molecular, aerosol = scene_inputs()
    with patch.object(
        orders, "_native_spectral_orders", wraps=orders._native_spectral_orders
    ) as native:
        transfer.solar.reflected_radiance(
            scene,
            source=solar_source(sun, direction=(30.0, 20.0)),
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                solar_source(sun, direction=(30.0, 20.0)),
                rayleigh=molecular,
                aerosol=aerosol,
            ),
            max_order=max_order,
            angular_order=2,
            layer_subdivisions=2,
        )
    assert native.call_count == 1
    assert len(native.call_args.args) == 12


def test_typed_roles_and_explicit_materialisation_share_one_contract() -> None:
    scene, solar, molecular, aerosol = scene_inputs()
    source = solar_source(solar, direction=(30.0, 70.0))
    assert (
        validate_optical_inputs(
            scene.atmosphere,
            source,
            absorption=compose_absorption(scene.atmosphere, source, ()),
            rayleigh=molecular,
            aerosol=aerosol,
        )
        is None
    )
    for rayleigh, hg in ((None, None), (molecular, None), (None, aerosol), (molecular, aerosol)):
        prepared = compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=rayleigh, aerosol=hg
        )
        for method in (
            terms_support.order_zero,
            terms_support.order_one,
            terms_support.reflected_radiance,
        ):
            typed = method(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere, source, rayleigh=rayleigh, aerosol=hg
                ),
            )
            terms_support.assert_decomposition_equal(
                typed, method(scene, source=source, optics=prepared)
            )
    properties = ScatteringProperties(
        wavenumber_grid(solar.wavenumbers_cm1),
        molecular.optical_depth,
        np.tile([1.0, 0.0, 0.1], (2, 2, 1)),
        "source_untruncated",
    )
    optics = AerosolOptics(
        wavenumber_grid(solar.wavenumbers_cm1),
        aerosol.optical_depth,
        np.full((2, 2), 0.5),
        aerosol.asymmetry,
    )
    converted = rayleigh_component(properties)
    assert converted.grid == molecular.grid
    np.testing.assert_array_equal(converted.optical_depth, molecular.optical_depth)
    assert type(rayleigh_component(properties)) is RayleighComponent
    assert type(aerosol_component(optics)) is AerosolComponent
    np.testing.assert_array_equal(
        aerosol_component(optics).scattering_depth, aerosol.optical_depth * 0.5
    )
    with pytest.raises(ValueError, match="untruncated"):
        rayleigh_component(replace(properties, truncation="source_truncated"))


def test_typed_inputs_own_immutable_axes_and_preserve_physical_checks() -> None:
    _, solar, molecular, aerosol = scene_inputs()
    axis = solar.wavenumbers_cm1.copy()
    retained = replace(molecular, grid=wavenumber_grid(axis))
    axis[0] = 100.0
    assert retained.grid == molecular.grid
    np.testing.assert_array_equal(retained.optical_depth, molecular.optical_depth)
    assert not retained.wavenumbers_cm1.flags.writeable
    with pytest.raises(ValueError):
        retained.wavenumbers_cm1.setflags(write=True)
    rayleigh_changes: tuple[dict[str, Any], ...] = (
        {"optical_depth": np.array([[-0.1, 0.1]] * 2)},
        {"optical_depth": np.array([[np.nan, 0.1]] * 2)},
        {"optical_depth": np.full((2, 1), 0.1)},
    )
    for changes in rayleigh_changes:
        with pytest.raises(ValueError):
            replace(cast(Any, molecular), **changes)
    aerosol_changes: tuple[dict[str, Any], ...] = (
        {"scattering_depth": np.full((2, 2), 0.3)},
        {"scattering_depth": np.array([[np.inf, 0.1]] * 2)},
        {"asymmetry": np.array([[0.81, 0.1]] * 2)},
        {"asymmetry": np.empty((0, 2))},
    )
    for changes in aerosol_changes:
        with pytest.raises(ValueError):
            replace(cast(Any, aerosol), **changes)


def test_no_scattering_and_absorbing_aerosol_reduce_to_unscattered() -> None:
    scene, solar, _, aerosol = scene_inputs()
    gas = AbsorptionComponent(
        wavenumber_grid(solar.wavenumbers_cm1),
        "X",
        None,
        "lorentz_lines",
        np.array([[0.1, 0.3], [0.2, 0.1]]),
    )
    absorbing = replace(aerosol, scattering_depth=np.zeros((2, 2)))
    for order in (0, 1, 2, 5):
        for hg in (None, absorbing):
            source = solar_source(solar, direction=(30.0, 70.0))
            result = terms_support.reflected_radiance(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    absorption=compose_absorption(scene.atmosphere, source, (gas,)),
                    aerosol=hg,
                ),
                max_order=order,
                angular_order=4,
            )
            beam = terms_support.order_zero(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    absorption=compose_absorption(scene.atmosphere, source, (gas,)),
                    aerosol=hg,
                ),
            )
            np.testing.assert_array_equal(result.values, beam.values)
            assert all(np.all(c.values == 0) for c in result.components[1:])


def test_rayleigh_ground_coupling_increases_first_order() -> None:
    scene, solar, molecular, aerosol = scene_inputs()
    black = replace(scene, surface=replace(scene.surface, reflectance=np.zeros(2)))
    source = solar_source(solar, direction=(30.0, 70.0))
    black_result = terms_support.order_one(
        black,
        source=source,
        optics=compose_atmospheric_optics(black.atmosphere, source, rayleigh=molecular),
    )
    rayleigh_only = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, rayleigh=molecular),
        max_order=1,
        angular_order=8,
    )
    np.testing.assert_array_equal(black_result.components[0].values, np.zeros(2))
    assert rayleigh_only.components[1].values[0] > black_result.components[1].values[0]
    assert rayleigh_only.components[1].values[1] == black_result.components[1].values[1]


def test_orders_are_aggregated_with_explicit_optics_and_solar_origin() -> None:
    scene, solar, molecular, aerosol = scene_inputs()
    source = solar_source(solar, direction=(30.0, 70.0))
    one = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=molecular, aerosol=aerosol
        ),
        max_order=1,
        angular_order=4,
    )
    two = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=molecular, aerosol=aerosol
        ),
        angular_order=4,
    )
    assert len(two.components) == 3
    assert [c.name for c in two.components] == [f"atmospheric_order_{n}" for n in range(3)]
    for actual, expected in zip(two.components[:2], one.components, strict=True):
        assert actual.values == pytest.approx(expected.values, rel=1e-14)
    second = two.components[2].values
    np.testing.assert_allclose(np.asarray(two.values) - one.values, second, rtol=1e-12)
    assert two.components[0].values[1] == 0
    absent = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, rayleigh=molecular),
        angular_order=4,
    )
    assert len(absent.components) == 3
    zero = terms_support.reflected_radiance(
        scene,
        source=replace(source, irradiance=replace(solar, values=np.zeros(2))),
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            replace(source, irradiance=replace(solar, values=np.zeros(2))),
            rayleigh=molecular,
            aerosol=aerosol,
        ),
        angular_order=4,
    )
    np.testing.assert_array_equal(zero.values, np.zeros(2))
    assert all(np.all(c.values == 0) for c in zero.components)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_order": -1},
        {"angular_order": 1},
        {"layer_subdivisions": 0},
    ],
)
def test_invalid_numerical_controls_fail(changes: dict[str, Any]) -> None:
    from orsia_radiative.transfer import orders

    scene, solar, molecular, _ = scene_inputs()
    with pytest.raises(ValueError):
        terms_support.reflected_radiance(
            scene,
            source=solar_source(solar, direction=(30.0, 70.0)),
            optics=compose_atmospheric_optics(
                scene.atmosphere, solar_source(solar, direction=(30.0, 70.0)), rayleigh=molecular
            ),
            **changes,
        )
    with pytest.raises(ValueError):
        orders.scattering_orders(
            scene,
            source=solar_source(solar, direction=(30.0, 70.0)),
            optics=compose_atmospheric_optics(
                scene.atmosphere, solar_source(solar, direction=(30.0, 70.0)), rayleigh=molecular
            ),
            **changes,
        )


def test_scene_support_and_optical_overflow_checks_are_shared() -> None:
    scene, solar, molecular, aerosol = scene_inputs()
    source = solar_source(solar, direction=(30.0, 70.0))
    for function in (
        terms_support.order_zero,
        terms_support.order_one,
        terms_support.reflected_radiance,
    ):
        with pytest.raises(ValueError, match="axis and layer"):
            function(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    rayleigh=replace(molecular, grid=wavenumber_grid(np.array([12000.0, 12002.0]))),
                ),
            )
        with pytest.raises(ValueError, match="axis and layer"):
            function(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    rayleigh=replace(molecular, optical_depth=molecular.optical_depth[:1]),
                ),
            )
        with pytest.raises(ValueError, match="overflow"):
            function(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    rayleigh=replace(molecular, optical_depth=np.full((2, 2), 1e308)),
                ),
            )
        mixed = replace(
            scene,
            surface=replace(scene.surface, emissivity=np.array([0.8, 1.0]), temperature_k=290),
        )
        np.testing.assert_array_equal(
            function(
                mixed,
                source=source,
                optics=compose_atmospheric_optics(mixed.atmosphere, source),
            ).values,
            function(
                scene,
                source=source,
                optics=compose_atmospheric_optics(scene.atmosphere, source),
            ).values,
        )


def test_full_sphere_measure_and_analytic_phase_convolutions() -> None:
    directions, weights = angular_grid(16)
    assert weights.sum() == pytest.approx(1.0, rel=1e-14)
    np.testing.assert_allclose(weights @ directions, (0.0, 0.0, 0.0), atol=1e-15)
    assert float(4 * pi * np.sum(weights * np.maximum(directions[:, 2], 0))) == pytest.approx(pi)
    first = np.array([0.6, 0.0, -0.8])
    last = np.array([0.0, 0.6, 0.8])
    cosine = float(first @ last)
    for ga, gb in ((0.5, 0.4), (-0.4, 0.5), (None, None)):
        computed = np.sum(
            weights * phase_function(directions @ first, ga) * phase_function(directions @ last, gb)
        )
        if ga is None or gb is None:
            expected = 1 + 0.025 * (3 * cosine**2 - 1)  # 5 * beta_2**2 * P_2
        else:
            expected = float(phase_function(np.array([cosine]), ga * gb)[0])
        assert computed == pytest.approx(expected, rel=2e-5)


def test_formal_transport_passes_between_layers_and_has_no_hidden_surface_reflection() -> None:
    source = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    centres, exiting = propagate(source, np.array([0.1, 0.2]), np.array([0.5, -0.5]))
    assert centres[1, 0] > 0 and centres[0, 1] > 0
    assert exiting[0] > 0 and exiting[1] > 0
    zeros = np.zeros_like(source)
    surface, exits = propagate(
        zeros, np.array([0.1, 0.2]), np.array([0.5, -0.5]), bottom_radiance=1.0
    )
    assert np.all(surface[:, 1] == 0) and exits[1] == 0
    assert exits[0] == pytest.approx(np.exp(-0.6))


def test_calls_are_deterministic_and_inputs_unchanged_under_concurrent_use() -> None:
    scene, solar, molecular, aerosol = scene_inputs()
    copies = [obj.wavenumbers_cm1.copy() for obj in (solar, molecular, aerosol)]

    def evaluate(_: int) -> terms_support.RadianceDecomposition:
        return terms_support.reflected_radiance(
            scene,
            source=solar_source(solar, direction=(30.0, 70.0)),
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                solar_source(solar, direction=(30.0, 70.0)),
                rayleigh=molecular,
                aerosol=aerosol,
            ),
            max_order=4,
            angular_order=3,
            layer_subdivisions=3,
        )

    reference = evaluate(0)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(evaluate, range(4)))
    for result in results:
        terms_support.assert_decomposition_equal(result, reference)
    for obj, snapshot in zip((solar, molecular, aerosol), copies, strict=True):
        np.testing.assert_array_equal(obj.wavenumbers_cm1, snapshot)
