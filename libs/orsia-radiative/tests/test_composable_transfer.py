"""Singular source, complete transfer boundary and upstream dependency-aware reuse."""

import importlib.util
import inspect
from dataclasses import fields, replace
from functools import partial
from math import cos, exp, pi, radians
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import orsia_radiative
import pytest
from orsia_radiative import (
    Absorber,
    Absorption,
    AerosolComponent,
    Atmosphere,
    CrossSectionField,
    Line,
    LineSet,
    ObservationGeometry,
    Radiance,
    RadiativeScene,
    RayleighComponent,
    SolarSource,
    SpectralIrradiance,
    SurfaceState,
    compose_absorption,
    compose_atmospheric_optics,
    cross_sections,
    evaluate_table_absorption,
    line_absorption,
    solar_source,
    table_absorption,
    transfer,
    validate_atmospheric_optics,
    wavenumber_grid,
)


def case() -> tuple[RadiativeScene, SpectralIrradiance, RayleighComponent, AerosolComponent]:
    grid = np.array([1000.0, 1001.0])
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([290.0, 280.0]),
        np.array([2e29, 1.8e29]),
        np.array([1.9e29, 1.7e29]),
        np.array([1e28, 1e28]),
        (
            Absorber("X", np.array([1e25, 2e25]), "dry_air"),
            Absorber("Y", np.array([3e25, 4e25]), "total_air"),
        ),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(grid), reflectance=np.array([0.2, 0.3])),
        ObservationGeometry(20.0),
    )
    solar = SpectralIrradiance(wavenumber_grid(grid), np.array([8.0, 10.0]))
    rayleigh = RayleighComponent(wavenumber_grid(grid), np.array([[0.02, 0.03], [0.01, 0.02]]))
    aerosol = AerosolComponent(
        wavenumber_grid(grid),
        np.array([[0.03, 0.02], [0.01, 0.04]]),
        np.array([[0.02, 0.01], [0.01, 0.02]]),
        np.array([[0.2, -0.2], [0.0, 0.4]]),
    )
    return scene, solar, rayleigh, aerosol


def field(scene: RadiativeScene, species: str = "X") -> CrossSectionField:
    return CrossSectionField(
        species,
        "1",
        "cross_section",
        wavenumber_grid(np.array([999.0, 1000.0, 1002.0])),
        scene.atmosphere.temperature_k,
        scene.atmosphere.pressure_pa,
        np.array([[1e-27, 2e-27, 4e-27], [2e-27, 4e-27, 6e-27]]),
    )


def change_gas(scene: RadiativeScene, factor: float, species: str = "X") -> RadiativeScene:
    gases = tuple(
        replace(gas, columns_molecules_m2=factor * gas.columns_molecules_m2)
        if gas.species == species
        else gas
        for gas in scene.atmosphere.absorbers
    )
    return replace(scene, atmosphere=replace(scene.atmosphere, absorbers=gases))


def test_transfer_exposes_complete_solar_boundary() -> None:
    assert transfer.__all__ == ["solar", "thermal"]
    assert transfer.solar.__all__ == ["reflected_radiance"]
    assert not hasattr(transfer.solar, "scattering_orders")
    assert not hasattr(transfer.solar, "allocate_solar_buffers")
    for removed in ("direct_beam", "single_scattering", "single_scattering_terms"):
        assert not hasattr(transfer.solar, removed)
    assert importlib.import_module("orsia_radiative.transfer") is transfer
    assert importlib.import_module("orsia_radiative.transfer.solar") is transfer.solar
    for name in transfer.solar.__all__:
        function = getattr(transfer.solar, name)
        parameters = inspect.signature(function).parameters
        for required in ("scene", "source", "optics"):
            assert parameters[required].default is inspect.Parameter.empty
        assert "surface_coupling" not in parameters
        assert (
            not {"sources", "absorption", "rayleigh", "aerosol", "scattering"} & parameters.keys()
        )
        assert name not in orsia_radiative.__all__ and not hasattr(transfer, name)
    evaluation = inspect.signature(transfer.solar.reflected_radiance).parameters
    assert evaluation["source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert evaluation["optics"].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("ScatteringComponent", "update_optical_inputs", "CollimatedSource", "solar_sos"):
        assert not hasattr(orsia_radiative, name)
    assert importlib.import_module("orsia_radiative.transfer.thermal") is transfer.thermal
    assert importlib.util.find_spec("orsia_radiative.sos") is None


@pytest.mark.parametrize("max_order", [0, 1, 2])
def test_unique_origin_explicit_roles_and_no_repeated_composition(max_order: int) -> None:
    scene, solar, rayleigh, aerosol = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    gas = table_absorption(scene.atmosphere, solar.wavenumbers_cm1, field(scene))
    total = compose_absorption(scene.atmosphere, source, (gas,))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, absorption=total, rayleigh=rayleigh, aerosol=aerosol
    )
    function = partial(transfer.solar.reflected_radiance, max_order=max_order)
    with (
        patch(
            "orsia_radiative.optics.absorption.validate_absorption_freshness",
            side_effect=AssertionError("spectroscopy re-entered by transfer"),
        ),
        patch(
            "orsia_radiative.optics.atmosphere._validate_materialised_inputs",
            side_effect=AssertionError("prepared composition revalidated"),
        ),
    ):
        result = function(scene, source=source, optics=optics)
        repeated = function(scene, source=source, optics=optics)
        assert result.grid == repeated.grid
        assert result.geometry == repeated.geometry
        np.testing.assert_array_equal(result.values, repeated.values)
    assert optics.absorption is total and not hasattr(total, "components")
    assert optics.rayleigh is rayleigh and optics.aerosol is aerosol
    assert {field.name for field in fields(result)} == {"grid", "values", "geometry"}


@pytest.mark.parametrize("method", ["reflected_radiance"])
def test_source_and_boundary_validation(method: str) -> None:
    scene, solar, rayleigh, _ = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    optics = compose_atmospheric_optics(scene.atmosphere, source, rayleigh=rayleigh)
    function = getattr(transfer.solar, method)
    for obsolete in (
        {"sources": (source,)},
        {"absorption": ()},
        {"rayleigh": None},
        {"aerosol": None},
        {"illumination": solar},
        {"scattering": ()},
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            function(scene, source=source, optics=optics, **cast(Any, obsolete))
    with pytest.raises(TypeError, match="optics"):
        function(scene, source=source)
    with pytest.raises(ValueError, match="axes"):
        function(
            scene,
            source=replace(
                source, irradiance=replace(solar, grid=wavenumber_grid(np.array([1000.0, 1002.0])))
            ),
            optics=optics,
        )
    # Values do not retain identity; check a separate reference before reuse.
    equal_scene = replace(scene, atmosphere=replace(scene.atmosphere))
    with pytest.raises(ValueError, match="stale"):
        validate_atmospheric_optics(
            equal_scene, source, optics, reference_atmosphere=scene.atmosphere
        )
    function(equal_scene, source=source, optics=optics)
    emissive = replace(
        scene,
        surface=SurfaceState(
            wavenumber_grid(solar.wavenumbers_cm1),
            emissivity=np.array([0.8, 0.7]),
            temperature_k=290.0,
        ),
    )
    with pytest.raises(ValueError, match="reflectance"):
        function(emissive, source=source, optics=optics)


@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0])
def test_upstream_selective_reuse_matches_full_recomputation(factor: float) -> None:
    scene, solar, rayleigh, aerosol = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    grid = solar.wavenumbers_cm1
    x_table, y_table = field(scene), field(scene, "Y")
    prepared_table = cross_sections(grid, x_table)
    x = evaluate_table_absorption(scene.atmosphere, prepared_table)
    y = table_absorption(scene.atmosphere, grid, y_table)
    old_total = compose_absorption(scene.atmosphere, source, (x, y))
    old_optics = compose_atmospheric_optics(scene.atmosphere, source, absorption=old_total)
    changed = change_gas(scene, factor)
    current_x = evaluate_table_absorption(changed.atmosphere, prepared_table)
    np.testing.assert_allclose(
        current_x.optical_depth, np.array([[0.02, 0.03], [0.08, 0.10]]) * factor, rtol=1e-14
    )
    total = compose_absorption(
        changed.atmosphere,
        source,
        (current_x, y),
        reference_atmospheres=(changed.atmosphere, scene.atmosphere),
    )
    full_total = compose_absorption(
        changed.atmosphere,
        source,
        (
            table_absorption(changed.atmosphere, grid, x_table),
            table_absorption(changed.atmosphere, grid, y_table),
        ),
    )
    assert total.grid == full_total.grid
    np.testing.assert_array_equal(total.optical_depth, full_total.optical_depth)
    updated = compose_atmospheric_optics(
        changed.atmosphere, source, absorption=total, rayleigh=rayleigh, aerosol=aerosol
    )
    full = compose_atmospheric_optics(
        changed.atmosphere, source, absorption=full_total, rayleigh=rayleigh, aerosol=aerosol
    )
    assert updated.rayleigh is rayleigh and updated.aerosol is aerosol
    for max_order in (0, 1, 2):
        function = partial(transfer.solar.reflected_radiance, max_order=max_order)
        reused_result = function(changed, source=source, optics=updated)
        full_result = function(changed, source=source, optics=full)
        assert reused_result.grid == full_result.grid
        assert reused_result.geometry == full_result.geometry
        np.testing.assert_array_equal(reused_result.values, full_result.values)
    with pytest.raises(ValueError, match="stale"):
        validate_atmospheric_optics(
            changed, source, old_optics, reference_atmosphere=scene.atmosphere
        )
    with pytest.raises(ValueError, match="stale absorption"):
        compose_absorption(
            changed.atmosphere,
            source,
            (x, y),
            reference_atmospheres=(scene.atmosphere, scene.atmosphere),
        )
    np.testing.assert_array_equal(
        prepared_table.cross_sections_m2_per_molecule, [[2e-27, 3e-27], [4e-27, 5e-27]]
    )


def test_line_recomputation_is_not_tau_scaling() -> None:
    scene, solar, _, _ = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    scene = change_gas(scene, 1000.0)
    grid = solar.wavenumbers_cm1
    lines = LineSet(
        "X",
        "1",
        (Line(1000.0, 1e-26, 0.0, 0.05, 0.5, 0.7),),
        np.ones(2),
        (250.0, 310.0),
        (80000.0, 100000.0),
    )
    original = line_absorption(scene.atmosphere, grid, lines)
    y = table_absorption(scene.atmosphere, grid, field(scene, "Y"))
    changed = change_gas(scene, 2.0)
    with pytest.raises(ValueError, match="stale absorption"):
        compose_absorption(
            changed.atmosphere,
            source,
            (original, y),
            reference_atmospheres=(scene.atmosphere, scene.atmosphere),
        )
    current = line_absorption(changed.atmosphere, grid, lines)
    assert not np.allclose(current.optical_depth, original.optical_depth * 2, rtol=1e-3, atol=0)
    total = compose_absorption(changed.atmosphere, source, (current, y))
    optics = compose_atmospheric_optics(changed.atmosphere, source, absorption=total)
    result = transfer.solar.reflected_radiance(
        changed, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    mu0, mu = cos(radians(30.0)), cos(radians(20.0))
    for point, rho in enumerate((0.2, 0.3)):
        tau = sum(row[point] for c in (current, y) for row in c.optical_depth)
        assert result.values[point] == pytest.approx(
            solar.values[point] * rho * mu0 / pi * exp(-tau * (1 / mu0 + 1 / mu)), rel=1e-14
        )


def test_table_support_and_owned_snapshots() -> None:
    scene, solar, _, _ = case()
    grid = solar.wavenumbers_cm1.copy()
    prepared = cross_sections(grid, field(scene))
    grid[0] = 900.0
    assert prepared.wavenumbers_cm1[0] == 1000.0
    single = cross_sections(np.array([1000.5]), field(scene))
    np.testing.assert_allclose(single.cross_sections_m2_per_molecule, [[2.5e-27], [4.5e-27]])
    for target in (np.array([998.0]), np.array([1003.0])):
        with pytest.raises(ValueError, match="extrapolation"):
            cross_sections(target, field(scene))
    for changes in (
        {"temperature_k": np.array([291.0, 280.0])},
        {"pressure_pa": np.array([94000.0, 85000.0])},
    ):
        with pytest.raises(ValueError, match="conditions"):
            evaluate_table_absorption(replace(scene.atmosphere, **cast(Any, changes)), prepared)


@pytest.mark.parametrize(
    "dependency", ["temperature", "pressure", "boundaries", "air", "broadener"]
)
def test_reuse_checks_physical_dependencies_not_retrieval_membership(dependency: str) -> None:
    scene, solar, _, _ = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    atmosphere = scene.atmosphere
    grid = solar.wavenumbers_cm1
    lines = LineSet(
        "X",
        "1",
        (Line(1000.0, 1e-26, 0.0, 0.05, 0.5, 0.7),),
        np.ones(2),
        (250.0, 310.0),
        (80000.0, 100000.0),
    )
    x = line_absorption(atmosphere, grid, lines)
    y = table_absorption(atmosphere, grid, field(scene, "Y"))
    changes: dict[str, Any] = {
        "temperature": {"temperature_k": atmosphere.temperature_k + 1.0},
        "pressure": {"pressure_pa": atmosphere.pressure_pa * 0.99},
        "boundaries": {"height_boundaries_m": atmosphere.height_boundaries_m + 10.0},
        "air": {
            "total_air_columns_molecules_m2": atmosphere.total_air_columns_molecules_m2 + 1e27,
            "dry_air_columns_molecules_m2": atmosphere.dry_air_columns_molecules_m2 + 1e27,
        },
        "broadener": {
            "absorbers": (
                replace(atmosphere.absorbers[0], air_broadener="total_air"),
                atmosphere.absorbers[1],
            )
        },
    }
    changed = replace(atmosphere, **changes[dependency])
    with pytest.raises(ValueError, match="stale absorption"):
        compose_absorption(changed, source, (x,), reference_atmospheres=(atmosphere,))
    if dependency in ("temperature", "pressure", "boundaries"):
        # Y's column was not modified, but its physical support was.
        with pytest.raises(ValueError, match="stale absorption"):
            compose_absorption(changed, source, (y,), reference_atmospheres=(atmosphere,))
    else:
        # These fixed-table depths do not depend on the air broadener pool.
        np.testing.assert_array_equal(
            compose_absorption(
                changed, source, (y,), reference_atmospheres=(atmosphere,)
            ).optical_depth,
            y.optical_depth,
        )


def test_reprepare_shares_records_and_surface_view_changes_do_not_invalidate() -> None:
    scene, solar, rayleigh, aerosol = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    total = compose_absorption(
        scene.atmosphere,
        source,
        (table_absorption(scene.atmosphere, solar.wavenumbers_cm1, field(scene)),),
    )
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, absorption=total, rayleigh=rayleigh, aerosol=aerosol
    )
    new_aerosol = replace(aerosol, scattering_depth=np.array([[0.01, 0.01], [0.005, 0.01]]))
    updated = compose_atmospheric_optics(
        scene.atmosphere, source, absorption=total, rayleigh=rayleigh, aerosol=new_aerosol
    )
    assert updated.absorption is total and updated.rayleigh is rayleigh
    assert optics.aerosol is aerosol and updated.aerosol is new_aerosol
    geometry_changed = replace(scene, geometry=ObservationGeometry(10.0))
    assert transfer.solar.reflected_radiance(
        geometry_changed, max_order=0, delta_m_moments=None, source=source, optics=optics
    ).geometry == (geometry_changed.geometry)
    brighter = replace(scene, surface=replace(scene.surface, reflectance=np.array([0.4, 0.6])))
    original = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    doubled = transfer.solar.reflected_radiance(
        brighter, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    np.testing.assert_allclose(doubled.values, 2 * original.values, rtol=1e-14)


def test_external_radiance_requires_only_physical_values_and_support() -> None:
    scene, solar, _, _ = case()
    result = Radiance(
        wavenumber_grid(solar.wavenumbers_cm1),
        np.array([1.0, 2.0]),
        scene.geometry,
    )
    assert {field.name for field in fields(result)} == {"grid", "values", "geometry"}
    source = SolarSource(solar, 30.0, 70.0)
    constructed = solar_source(solar, direction=(30.0, 70.0))
    assert source.irradiance is constructed.irradiance is solar
    assert source.zenith_deg == constructed.zenith_deg == 30.0
    assert source.relative_azimuth_deg == constructed.relative_azimuth_deg == 70.0
    result = transfer.solar.reflected_radiance(
        scene,
        max_order=0,
        delta_m_moments=None,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source),
    )
    assert result.geometry == scene.geometry  # Prepared support owns a geometry snapshot.


def test_aggregate_validation_and_external_supplier_contract() -> None:
    scene, solar, rayleigh, _ = case()
    source = solar_source(solar, direction=(30.0, 70.0))
    grid = solar.wavenumbers_cm1
    external = Absorption(wavenumber_grid(grid), np.ones((2, 2)))
    for changes in (
        {"optical_depth": -np.ones((2, 2))},
        {"optical_depth": np.full((2, 2), np.nan)},
    ):
        with pytest.raises(ValueError):
            replace(external, **cast(Any, changes))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, absorption=external, rayleigh=rayleigh
    )
    assert optics.absorption is external
    for invalid in (
        replace(external, grid=wavenumber_grid(np.array([1000.0, 1002.0]))),
        replace(external, optical_depth=np.ones((1, 2))),
    ):
        with pytest.raises(ValueError, match="axis and layer"):
            replace(optics, absorption=invalid)
    assert {field.name for field in fields(external)} == {"grid", "optical_depth"}
