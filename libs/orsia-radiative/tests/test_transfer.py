"""Composition and approximation contracts, separate from frozen physics references."""

from dataclasses import replace
from math import cos, exp, pi, radians
from typing import Any

import numpy as np
import orsia_radiative
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    compose_absorption,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)


def inputs() -> tuple[RadiativeScene, SpectralIrradiance, AerosolComponent]:
    axis = np.array([1000.0, 1001.0], dtype=np.float64)
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([290.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
        (Absorber("X", np.array([1e25]), "dry_air"),),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(axis), reflectance=np.array([0.2, 0.4])),
        ObservationGeometry(10.0),
    )
    solar = SpectralIrradiance(wavenumber_grid(axis), np.array([10.0, 20.0]))
    constituent = AerosolComponent(
        wavenumber_grid(axis),
        np.array([[0.02, 0.03]]),
        np.array([[0.01, 0.02]]),
        np.array([[0.2, 0.4]]),
    )
    return scene, solar, constituent


def test_order_selection_preserves_direct_reflection_and_adds_scattering() -> None:
    scene, solar, constituent = inputs()
    source = solar_source(solar, direction=(30.0, 20.0))
    beam = terms_support.order_zero(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=constituent),
    )
    path = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=constituent),
    )
    terms_support.assert_term_equal(beam.components[0], path.components[0])
    assert len(beam.components) == 1
    assert len(path.components) == 2
    assert np.all(beam.values < path.values)
    mu0, mu = cos(radians(30)), cos(radians(10))
    for i in range(2):
        assert scene.surface.reflectance is not None
        assert beam.values[i] == pytest.approx(
            solar.values[i]
            * scene.surface.reflectance[i]
            * mu0
            / pi
            * exp(-constituent.optical_depth[0][i] * (1 / mu0 + 1 / mu))
        )


def test_mixed_roles_use_common_extinction_without_double_counting() -> None:
    scene, solar, first = inputs()
    molecular = RayleighComponent(wavenumber_grid(first.wavenumbers_cm1), np.array([[0.01, 0.02]]))
    source = solar_source(solar, direction=(30.0, 20.0))
    combined = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=molecular, aerosol=first
        ),
    )
    reference = terms_support.order_zero(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=molecular, aerosol=first
        ),
    )
    terms_support.assert_term_equal(combined.components[0], reference.components[0])
    assert len(combined.components) == 2
    assert [c.name for c in combined.components] == ["atmospheric_order_0", "atmospheric_order_1"]
    standalone = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=first),
    )
    rayleigh_only = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, rayleigh=molecular),
    )
    assert combined.values != pytest.approx(standalone.values + rayleigh_only.values)


def test_transparent_collections_and_zero_scattering_reductions() -> None:
    scene, solar, first = inputs()
    source = solar_source(solar, direction=(30.0, 20.0))
    transparent = terms_support.order_zero(
        scene, source=source, optics=compose_atmospheric_optics(scene.atmosphere, source)
    )
    single = terms_support.order_one(
        scene, source=source, optics=compose_atmospheric_optics(scene.atmosphere, source)
    )
    np.testing.assert_array_equal(transparent.values, single.values)
    no_scattering = replace(first, scattering_depth=np.zeros((1, 2)))
    extinction_only = terms_support.order_zero(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=no_scattering),
    )
    single_extinction = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=no_scattering),
    )
    np.testing.assert_array_equal(single_extinction.values, extinction_only.values)
    assert all(np.all(c.values == 0) for c in single_extinction.components[1:])


@pytest.mark.parametrize(
    "changes",
    [
        {"optical_depth": np.empty((0, 2))},
        {"scattering_depth": np.array([[0.03, 0.02]])},
        {"scattering_depth": np.array([[-0.01, 0.02]])},
        {"scattering_depth": np.array([[np.nan, 0.02]])},
        {"optical_depth": np.array([[np.inf, 0.03]])},
        {"scattering_depth": np.array([[0.01]])},
        {"scattering_depth": np.array([[0.01, 0.02], [0.01, 0.02]])},
        {"asymmetry": np.empty((0, 2))},
        {"asymmetry": np.array([[0.9, 0.2]])},
        {"asymmetry": np.array([[np.nan, 0.2]])},
    ],
)
def test_constituent_numerical_contract(changes: dict[str, Any]) -> None:
    _, _, first = inputs()
    with pytest.raises(ValueError):
        replace(first, **changes)


def test_support_and_total_overflow_rejected_by_both_methods() -> None:
    scene, solar, first = inputs()
    gas = AbsorptionComponent(
        wavenumber_grid(first.wavenumbers_cm1),
        "X",
        None,
        "lorentz_lines",
        np.array([[0.1, 0.2]]),
    )
    source = solar_source(solar, direction=(30.0, 20.0))
    for method in (terms_support.order_zero, terms_support.order_one):
        result = method(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                source,
                absorption=compose_absorption(scene.atmosphere, source, (gas,)),
                aerosol=first,
            ),
        )
        assert np.all(np.isfinite(result.values))
        other_axis = np.array([1000.0, 1002.0])
        with pytest.raises(ValueError, match="axis and layer"):
            method(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    aerosol=replace(first, grid=wavenumber_grid(other_axis)),
                ),
            )
        assert first.asymmetry is not None
        wrong_layers = replace(
            first,
            optical_depth=np.tile(first.optical_depth, (2, 1)),
            scattering_depth=np.tile(first.scattering_depth, (2, 1)),
            asymmetry=np.tile(first.asymmetry, (2, 1)),
        )
        with pytest.raises(ValueError, match="axis and layer"):
            method(
                scene,
                source=source,
                optics=compose_atmospheric_optics(scene.atmosphere, source, aerosol=wrong_layers),
            )
        huge = replace(first, optical_depth=np.full((1, 2), 1e308))
        with pytest.raises(ValueError, match="overflow"):
            method(
                scene,
                source=source,
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    source,
                    rayleigh=RayleighComponent(
                        wavenumber_grid(first.wavenumbers_cm1), huge.optical_depth
                    ),
                    aerosol=huge,
                ),
            )


def test_order_rows_are_additive_and_preserve_inputs() -> None:
    scene, solar, first = inputs()
    snapshot = (scene, solar, first)
    result = terms_support.order_one(
        scene,
        source=solar_source(solar, direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            scene.atmosphere, solar_source(solar, direction=(30.0, 20.0)), aerosol=first
        ),
    )
    assert all(a is b for a, b in zip((scene, solar, first), snapshot, strict=True))
    np.testing.assert_array_equal(result.wavenumbers_cm1, solar.wavenumbers_cm1)
    assert tuple(sum(c.values[i] for c in result.components) for i in range(2)) == pytest.approx(
        result.values
    )


def test_removed_case_specific_entry_points_are_not_exported() -> None:
    for name in (
        "lambertian_solar_radiance",
        "single_rayleigh_solar_radiance",
        "solar_radiance_with_aerosol_scattering",
    ):
        assert name not in orsia_radiative.__all__
        assert not hasattr(orsia_radiative, name)
