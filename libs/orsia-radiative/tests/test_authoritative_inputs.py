"""Source-defined sampling, atmosphere ownership and reusable support association."""

import inspect
from dataclasses import fields, replace
from functools import partial
from typing import Any, cast

import numpy as np
import orsia_radiative
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    Absorber,
    AerosolComponent,
    Atmosphere,
    AtmosphericOptics,
    CrossSectionField,
    Line,
    LineSet,
    ObservationGeometry,
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
    transfer,
    validate_atmospheric_optics,
    validate_optical_inputs,
    wavenumber_grid,
)


def inputs() -> tuple[Atmosphere, SolarSource, CrossSectionField]:
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([290.0, 280.0]),
        np.array([2e29, 1.8e29]),
        np.array([1.9e29, 1.7e29]),
        np.full(2, 1e28),
        (Absorber("X", np.array([1e25, 2e25]), "dry_air"),),
    )
    source = solar_source(
        SpectralIrradiance(wavenumber_grid(np.array([1000.0, 1001.0])), np.array([8.0, 10.0])),
        direction=(30.0, 70.0),
    )
    table = CrossSectionField(
        "X",
        "1",
        "cross_section",
        wavenumber_grid(np.array([999.0, 1003.0])),
        atmosphere.temperature_k,
        atmosphere.pressure_pa,
        np.array([[1e-27, 5e-27], [2e-27, 6e-27]]),
    )
    return atmosphere, source, table


def scene(atmosphere: Atmosphere, source: SolarSource) -> RadiativeScene:
    return RadiativeScene(
        atmosphere,
        SurfaceState(
            wavenumber_grid(source.irradiance.wavenumbers_cm1), reflectance=np.array([0.2, 0.3])
        ),
        ObservationGeometry(20.0),
    )


def test_atmospheric_optics_exports_domain_names_without_compatibility_aliases() -> None:
    from orsia_radiative.optics import atmosphere as optical

    for name in ("AtmosphericOptics", "compose_atmospheric_optics", "validate_atmospheric_optics"):
        assert name in orsia_radiative.__all__
        assert getattr(orsia_radiative, name) is getattr(optical, name)
    for name in (
        "PreparedOpticalInputs",
        "prepare_optical_inputs",
        "validate_prepared_optical_inputs",
    ):
        assert name not in orsia_radiative.__all__
        assert not hasattr(orsia_radiative, name)
        assert not hasattr(optical, name)
    atmosphere, source, _ = inputs()
    assert type(compose_atmospheric_optics(atmosphere, source)) is AtmosphericOptics


def test_optical_composition_takes_authorities_not_independent_grid_surface_or_view() -> None:
    atmosphere, source, _ = inputs()
    for function in (compose_atmospheric_optics, validate_optical_inputs):
        parameters = inspect.signature(function).parameters
        assert tuple(parameters) == ("atmosphere", "source", "absorption", "rayleigh", "aerosol")
        assert parameters["source"].default is inspect.Parameter.empty
    optics = compose_atmospheric_optics(atmosphere, source)
    assert not hasattr(optics, "reference_atmosphere")
    assert optics.wavenumbers_cm1 is source.irradiance.wavenumbers_cm1
    assert validate_optical_inputs(atmosphere, source) is None
    assert {field.name for field in fields(optics)} == {
        "grid",
        "absorption",
        "rayleigh",
        "aerosol",
    }
    # A solar source's illumination/direction is not retained in atmospheric optics.
    for obsolete in (
        {"scene": scene(atmosphere, source)},
        {"wavenumbers_cm1": optics.wavenumbers_cm1},
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            compose_atmospheric_optics(atmosphere, source, **cast(Any, obsolete))


def test_composition_requires_source_and_retains_only_physical_values() -> None:
    atmosphere, source, _ = inputs()
    parameters = inspect.signature(compose_absorption).parameters
    assert tuple(parameters) == ("atmosphere", "source", "components", "reference_atmospheres")
    assert parameters["source"].default is inspect.Parameter.empty
    total = compose_absorption(atmosphere=atmosphere, source=source, components=())
    assert not hasattr(total, "reference_atmosphere")
    assert total.wavenumbers_cm1 is source.irradiance.wavenumbers_cm1
    np.testing.assert_array_equal(total.optical_depth, np.zeros((2, 2)))
    assert {field.name for field in fields(total)} == {
        "grid",
        "optical_depth",
    }
    with pytest.raises(TypeError, match="source"):
        cast(Any, compose_absorption)(atmosphere, components=())
    with pytest.raises(TypeError, match="unexpected keyword"):
        compose_absorption(
            atmosphere, source, (), **cast(Any, {"wavenumbers_cm1": total.wavenumbers_cm1})
        )


@pytest.mark.parametrize("change", ["irradiance", "direction", "equal_grid"])
def test_composition_does_not_depend_on_solar_illumination(change: str) -> None:
    atmosphere, source, table = inputs()
    grid = source.irradiance.wavenumbers_cm1
    gas = evaluate_table_absorption(atmosphere, cross_sections(grid, table))
    total = compose_absorption(atmosphere, source, (gas,))
    if change == "irradiance":
        changed = replace(source, irradiance=replace(source.irradiance, values=np.zeros(2)))
    elif change == "direction":
        changed = replace(source, zenith_deg=40.0, relative_azimuth_deg=120.0)
    else:
        changed = replace(
            source, irradiance=replace(source.irradiance, grid=wavenumber_grid(grid.copy()))
        )
        assert changed.irradiance.wavenumbers_cm1 is not grid
    recomposed = compose_absorption(atmosphere, changed, (gas,))
    assert recomposed.grid == total.grid
    np.testing.assert_array_equal(recomposed.optical_depth, total.optical_depth)
    assert recomposed.wavenumbers_cm1 is changed.irradiance.wavenumbers_cm1
    assert not hasattr(recomposed, "input_source_ids")
    assert compose_atmospheric_optics(atmosphere, changed, absorption=total).absorption is total


@pytest.mark.parametrize("grid", [np.array([1000.0, 1001.25]), np.array([1000.0])])
def test_composition_rejects_components_outside_source_support(grid: NDArray[np.float64]) -> None:
    atmosphere, source, table = inputs()
    gas = evaluate_table_absorption(
        atmosphere, cross_sections(source.irradiance.wavenumbers_cm1, table)
    )
    changed = replace(
        source,
        irradiance=replace(
            source.irradiance, grid=wavenumber_grid(grid), values=np.ones(grid.size)
        ),
    )
    with pytest.raises(ValueError, match="axis and layer"):
        compose_absorption(atmosphere, changed, (gas,))
    empty = compose_absorption(atmosphere, changed, ())
    assert empty.wavenumbers_cm1 is changed.irradiance.wavenumbers_cm1
    assert empty.optical_depth.shape == (2, grid.size)


def test_source_coordinate_storage_is_shared_through_spectroscopy_and_transfer() -> None:
    atmosphere, source, table = inputs()
    grid = source.irradiance.wavenumbers_cm1
    prepared_table = cross_sections(grid, table)
    gas = evaluate_table_absorption(atmosphere, prepared_table)
    total = compose_absorption(atmosphere, source, (gas,))
    rayleigh = RayleighComponent(wavenumber_grid(grid), np.full((2, 2), 0.01))
    aerosol = AerosolComponent(
        wavenumber_grid(grid),
        np.full((2, 2), 0.03),
        np.full((2, 2), 0.02),
        np.full((2, 2), 0.2),
    )
    optics = compose_atmospheric_optics(
        atmosphere, source, absorption=total, rayleigh=rayleigh, aerosol=aerosol
    )
    material = scene(atmosphere, source)
    lines = LineSet(
        "X",
        "1",
        (Line(1000.0, 1e-26, 0.0, 0.05, 0.5, 0.7),),
        np.ones(2),
        (250.0, 310.0),
        (80000.0, 100000.0),
    )
    line_component = line_absorption(atmosphere, grid, lines)
    result = transfer.solar.reflected_radiance(
        material, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    for record in (
        prepared_table,
        gas,
        total,
        rayleigh,
        aerosol,
        optics,
        material.surface,
        line_component,
        result,
    ):
        assert record.wavenumbers_cm1 is grid
        assert np.shares_memory(record.wavenumbers_cm1, grid)
        with pytest.raises(ValueError):
            record.wavenumbers_cm1.setflags(write=True)
    assert table.wavenumbers_cm1 is not grid  # native supplier sampling remains distinct
    assert not hasattr(gas, "reference_atmosphere")
    assert not np.shares_memory(total.optical_depth, gas.optical_depth)
    np.testing.assert_allclose(total.optical_depth, [[0.02, 0.03], [0.06, 0.08]], rtol=1e-14)


@pytest.mark.parametrize(
    "kind",
    [
        "writable",
        "readonly_owner",
        "readonly_view",
        "memoryview_mutable",
        "memoryview_bytes",
        "strided_bytes",
    ],
)
def test_only_safely_immutable_contiguous_coordinates_are_shared(kind: str) -> None:
    owner = np.array([1000.0, 1001.0])
    mutable_buffer = bytearray(owner.tobytes())
    if kind == "writable":
        supplied = owner
    elif kind == "readonly_owner":
        owner.setflags(write=False)
        supplied = owner
    elif kind == "readonly_view":
        supplied = owner.view()
        supplied.setflags(write=False)
    elif kind == "memoryview_mutable":
        supplied = np.frombuffer(memoryview(mutable_buffer).toreadonly(), dtype=np.float64)
    elif kind == "memoryview_bytes":
        supplied = np.frombuffer(memoryview(owner.tobytes()), dtype=np.float64)
    else:
        supplied = np.frombuffer(np.array([1000.0, 1000.5, 1001.0]).tobytes(), dtype=np.float64)[
            ::2
        ]
    captured = SpectralIrradiance(wavenumber_grid(supplied), np.ones(2))
    assert not np.shares_memory(supplied, captured.wavenumbers_cm1)
    if kind in {"writable", "readonly_owner", "readonly_view"}:
        owner.setflags(write=True)
        owner[0] = 999.0
    elif kind == "memoryview_mutable":
        np.frombuffer(mutable_buffer, dtype=np.float64)[0] = 999.0
    np.testing.assert_array_equal(captured.wavenumbers_cm1, [1000.0, 1001.0])
    with pytest.raises(ValueError):
        captured.wavenumbers_cm1.setflags(write=True)


def test_contiguous_bytes_backed_subgrid_keeps_exact_original_coordinates() -> None:
    values = np.frombuffer(np.array([999.0, 1000.0, 1001.0, 1002.0]).tobytes(), dtype=np.float64)
    grid = values.reshape(2, 2).reshape(-1)[1:3]
    captured = SpectralIrradiance(wavenumber_grid(grid), np.ones(2))
    assert captured.wavenumbers_cm1 is grid
    np.testing.assert_array_equal(captured.wavenumbers_cm1, [1000.0, 1001.0])


@pytest.mark.parametrize("role", ["absorption", "rayleigh", "aerosol"])
def test_preparation_rejects_equal_shape_optics_on_a_different_source_grid(role: str) -> None:
    atmosphere, source, table = inputs()
    grid = source.irradiance.wavenumbers_cm1
    prepared_table = cross_sections(grid, table)
    gas = evaluate_table_absorption(atmosphere, prepared_table)
    total = compose_absorption(atmosphere, source, (gas,))
    optical_roles: dict[str, Any] = {
        "absorption": total,
        "rayleigh": RayleighComponent(wavenumber_grid(grid), np.full((2, 2), 0.01)),
        "aerosol": AerosolComponent(
            wavenumber_grid(grid),
            np.full((2, 2), 0.03),
            np.full((2, 2), 0.02),
            np.zeros((2, 2)),
        ),
    }
    shifted = replace(
        source, irradiance=replace(source.irradiance, grid=wavenumber_grid(grid + 0.25))
    )
    for function in (compose_atmospheric_optics, validate_optical_inputs):
        with pytest.raises(ValueError, match="axis and layer"):
            function(atmosphere, shifted, **{role: optical_roles[role]})
    with pytest.raises(ValueError, match="axis and layer"):
        compose_absorption(atmosphere, shifted, (gas,))
    assert prepared_table.wavenumbers_cm1 is grid
    # Reevaluate the table on the new source support explicitly, rather than relabel depths.
    new_gas = evaluate_table_absorption(
        atmosphere, cross_sections(shifted.irradiance.wavenumbers_cm1, table)
    )
    new_total = compose_absorption(atmosphere, shifted, (new_gas,))
    assert not np.array_equal(new_total.optical_depth, total.optical_depth)
    assert (
        compose_atmospheric_optics(atmosphere, shifted, absorption=new_total).wavenumbers_cm1
        is shifted.irradiance.wavenumbers_cm1
    )


@pytest.mark.parametrize("max_order", [0, 1, 2])
def test_transfer_rejects_reuse_even_if_scene_and_source_move_to_same_new_grid(
    max_order: int,
) -> None:
    atmosphere, source, table = inputs()
    grid = source.irradiance.wavenumbers_cm1
    gas = evaluate_table_absorption(atmosphere, cross_sections(grid, table))
    total = compose_absorption(atmosphere, source, (gas,))
    optics = compose_atmospheric_optics(atmosphere, source, absorption=total)
    shifted = replace(
        source, irradiance=replace(source.irradiance, grid=wavenumber_grid(grid + 0.25))
    )
    operation = partial(transfer.solar.reflected_radiance, max_order=max_order)
    with pytest.raises(ValueError, match="source and optical spectral axes"):
        operation(scene(atmosphere, shifted), source=shifted, optics=optics)
    with pytest.raises(ValueError, match="axis and layer"):
        replace(optics, grid=wavenumber_grid(shifted.irradiance.wavenumbers_cm1))
    with pytest.raises(ValueError, match="axis and layer"):
        AtmosphericOptics(wavenumber_grid(shifted.irradiance.wavenumbers_cm1), total, None, None)


@pytest.mark.parametrize("max_order", [0, 1, 2])
def test_source_properties_and_equal_coordinate_copies_do_not_invalidate_optics(
    max_order: int,
) -> None:
    atmosphere, source, table = inputs()
    grid = source.irradiance.wavenumbers_cm1
    gas = evaluate_table_absorption(atmosphere, cross_sections(grid, table))
    optics = compose_atmospheric_optics(
        atmosphere, source, absorption=compose_absorption(atmosphere, source, (gas,))
    )
    independent = replace(
        source, irradiance=replace(source.irradiance, grid=wavenumber_grid(grid.copy()))
    )
    assert independent.irradiance.wavenumbers_cm1 is not grid
    operation = partial(transfer.solar.reflected_radiance, max_order=max_order)
    material = scene(atmosphere, source)
    baseline = operation(material, source=source, optics=optics)
    independent_result = operation(material, source=independent, optics=optics)
    assert independent_result.grid == baseline.grid
    assert independent_result.geometry == baseline.geometry
    np.testing.assert_array_equal(independent_result.values, baseline.values)
    doubled = replace(
        independent,
        irradiance=replace(independent.irradiance, values=2 * source.irradiance.values),
    )
    np.testing.assert_allclose(
        operation(material, source=doubled, optics=optics).values, 2 * baseline.values, rtol=1e-14
    )
    redirected = replace(doubled, zenith_deg=40.0, relative_azimuth_deg=120.0)
    fresh = compose_atmospheric_optics(atmosphere, redirected, absorption=optics.absorption)
    reused_result = operation(material, source=redirected, optics=optics)
    fresh_result = operation(material, source=redirected, optics=fresh)
    assert reused_result.grid == fresh_result.grid
    assert reused_result.geometry == fresh_result.geometry
    np.testing.assert_array_equal(reused_result.values, fresh_result.values)
    assert optics.wavenumbers_cm1 is grid


@pytest.mark.parametrize("max_order", [0, 1, 2])
def test_surface_support_is_checked_at_transfer_not_optical_preparation(max_order: int) -> None:
    atmosphere, source, _ = inputs()
    optics = compose_atmospheric_optics(atmosphere, source)
    material = scene(atmosphere, source)
    changed = replace(
        material,
        surface=replace(material.surface, grid=wavenumber_grid(optics.wavenumbers_cm1 + 0.25)),
    )
    with pytest.raises(ValueError, match="scene spectral axes"):
        partial(transfer.solar.reflected_radiance, max_order=max_order)(
            changed, source=source, optics=optics
        )
    assert not hasattr(optics, "reference_atmosphere")
    with pytest.raises(ValueError, match="stale"):
        validate_atmospheric_optics(
            replace(material, atmosphere=replace(atmosphere)),
            source,
            optics,
            reference_atmosphere=atmosphere,
        )
