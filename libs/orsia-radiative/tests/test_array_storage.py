"""Array storage, ownership and identity comparison across radiative records."""

from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import pytest
from orsia_radiative import (
    Absorber,
    AerosolComponent,
    AerosolOptics,
    Atmosphere,
    CrossSectionField,
    LayerProfileBasis,
    Line,
    LineSet,
    ObservationGeometry,
    PartitionFunctionTable,
    RadiativeScene,
    RayleighComponent,
    ScatteringProperties,
    SolarIrradianceSpectrum,
    SpectralIrradiance,
    SurfaceState,
    WaterContinuumReference,
    compose_absorption,
    compose_atmospheric_optics,
    cross_sections,
    evaluate_profile,
    evaluate_table_absorption,
    solar_source,
    transfer,
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
        np.full(2, 1e28),
        (Absorber("X", np.array([1e25, 2e25]), "dry_air"),),
    )
    return (
        RadiativeScene(
            atmosphere,
            SurfaceState(wavenumber_grid(grid), np.array([0.2, 0.3])),
            ObservationGeometry(20.0),
        ),
        SpectralIrradiance(wavenumber_grid(grid), np.array([8.0, 10.0])),
        RayleighComponent(wavenumber_grid(grid), np.array([[0.02, 0.03], [0.01, 0.02]])),
        AerosolComponent(
            wavenumber_grid(grid),
            np.full((2, 2), 0.03),
            np.full((2, 2), 0.01),
            np.full((2, 2), 0.2),
        ),
    )


def records() -> tuple[Any, ...]:
    """Every public record whose homogeneous numerical storage was migrated."""
    scene, irradiance, rayleigh, aerosol = case()
    atmosphere = scene.atmosphere
    source = solar_source(irradiance, direction=(30.0, 70.0))
    grid = irradiance.wavenumbers_cm1
    table = CrossSectionField(
        "X",
        "1",
        "cross_section",
        wavenumber_grid(grid),
        atmosphere.temperature_k,
        atmosphere.pressure_pa,
        np.array([[1e-27, 2e-27], [2e-27, 3e-27]]),
    )
    prepared_table = cross_sections(grid, table)
    absorption = evaluate_table_absorption(atmosphere, prepared_table)
    basis = LayerProfileBasis(
        "K",
        atmosphere.temperature_k,
        ("offset", "lapse"),
        np.array([[1.0, 1.0], [2.0, -2.0]]),
        minimum=1.0,
    )
    lines = LineSet(
        "X",
        "1",
        (Line(1000.0, 1e-26, 0.0, 0.05, 0.1, 0.7),),
        np.ones(2),
        (250.0, 310.0),
        (80000.0, 100000.0),
    )
    properties = ScatteringProperties(
        wavenumber_grid(grid),
        rayleigh.optical_depth,
        np.tile([1.0, 0.0, 0.1], (2, 2, 1)),
        "source_untruncated",
    )
    aerosol_optics = AerosolOptics(
        wavenumber_grid(grid),
        aerosol.optical_depth,
        aerosol.scattering_depth / aerosol.optical_depth,
        aerosol.asymmetry,
    )
    optics = compose_atmospheric_optics(
        scene.atmosphere,
        source,
        absorption=compose_absorption(scene.atmosphere, source, (absorption,)),
        rayleigh=rayleigh,
        aerosol=aerosol,
    )
    result = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    return (
        irradiance,
        result,
        atmosphere.absorbers[0],
        atmosphere,
        basis,
        scene.surface,
        lines,
        table,
        absorption,
        optics.absorption,
        prepared_table,
        properties,
        aerosol_optics,
        rayleigh,
        aerosol,
        optics,
    )


def table_records() -> tuple[Any, ...]:
    """Additional native tables formerly using the combined validation/copy helper."""
    return (
        SolarIrradianceSpectrum(
            np.array([500.0, 600.0]), np.ones(2), np.full(2, 0.1), np.full(2, 0.5)
        ),
        PartitionFunctionTable(1, 1, np.array([250.0, 300.0]), np.array([1.0, 2.0])),
        WaterContinuumReference(
            np.array([-10.0, 0.0, 10.0, 20.0]),
            np.full(4, 1e-25),
            np.full(4, 1e-26),
            np.ones(4),
            101325.0,
            296.0,
            (250.0, 310.0),
        ),
    )


@pytest.mark.parametrize(
    "record", records() + table_records(), ids=lambda record: type(record).__name__
)
@pytest.mark.parametrize("layout", ["strided", "fortran", "readonly"])
def test_constructor_copies_each_retained_field_once(record: Any, layout: str, monkeypatch) -> None:
    inputs = {}
    for field_info in fields(record):
        value = getattr(record, field_info.name)
        if not isinstance(value, np.ndarray):
            continue
        if layout == "strided":
            owner = np.empty((*value.shape[:-1], 2 * value.shape[-1]))
            supplied = owner[..., ::2]
            supplied[:] = value
        else:
            supplied = np.array(value, order="F" if layout == "fortran" else "C")
            if layout == "readonly":
                supplied.setflags(write=False)
        inputs[field_info.name] = supplied

    copied = []
    numpy_copy = np.copy

    def count_copy(values, *args, **kwargs):
        copied.append(id(values))
        return numpy_copy(values, *args, **kwargs)

    monkeypatch.setattr(np, "copy", count_copy)
    captured = replace(record, **inputs)
    assert sorted(copied) == sorted(id(value) for value in inputs.values())
    for name, supplied in inputs.items():
        retained = getattr(captured, name)
        np.testing.assert_array_equal(retained, supplied)
        assert retained.dtype == supplied.dtype and retained.shape == supplied.shape
        assert retained.flags.writeable and not np.shares_memory(retained, supplied)
        retained.flat[0] += 1
        np.testing.assert_array_equal(supplied, getattr(record, name))
        supplied.setflags(write=True)
        supplied.fill(0)
        assert retained.flat[0] == getattr(record, name).flat[0] + 1


@pytest.mark.parametrize("record", records(), ids=lambda record: type(record).__name__)
def test_numerical_fields_own_writable_noncontiguous_input_copies(record: Any) -> None:
    inputs = {}
    for field_info in fields(record):
        value = getattr(record, field_info.name)
        if isinstance(value, np.ndarray):
            # Supply a real strided view with a separate mutable owner.
            shape = (*value.shape[:-1], 2 * value.shape[-1])
            owner = np.empty(shape, dtype=np.float64)
            strided = owner[..., ::2]
            strided[:] = value
            assert not strided.flags.c_contiguous
            inputs[field_info.name] = strided
    captured = replace(record, **inputs)
    for name, supplied in inputs.items():
        retained = getattr(captured, name)
        assert isinstance(retained, np.ndarray)
        assert retained.dtype == np.float64 and retained.shape == supplied.shape
        assert retained.flags.c_contiguous
        assert not np.shares_memory(retained, supplied)
        np.testing.assert_array_equal(retained, supplied)
        supplied.fill(0.0)
        np.testing.assert_array_equal(retained, getattr(record, name))
        for view in (retained, retained.view(), retained.reshape(-1)):
            assert view.flags.writeable
            original = retained.flat[0]
            view.flat[0] = original + 1
            assert retained.flat[0] == original + 1
            assert getattr(record, name).flat[0] == original
            view.flat[0] = original
        writable = retained.copy()
        assert writable.flags.writeable and not np.shares_memory(writable, retained)
        np.testing.assert_array_equal(retained, getattr(record, name))


@pytest.mark.parametrize(
    "record", records() + table_records(), ids=lambda record: type(record).__name__
)
@pytest.mark.parametrize("violation", ["nan", "infinity", "rank", "scalar", "empty", "empty_last"])
def test_numerical_fields_reject_invalid_finiteness_and_rank(
    record: Any,
    violation: str,
) -> None:
    for field_info in fields(record):
        value = getattr(record, field_info.name)
        if not isinstance(value, np.ndarray):
            continue
        invalid = value.copy()
        if violation == "nan":
            invalid.flat[0] = np.nan
        elif violation == "infinity":
            invalid.flat[0] = np.inf
        elif violation == "rank":
            invalid = invalid[..., None]
        elif violation == "scalar":
            invalid = np.array(invalid.flat[0])
        elif violation == "empty_last":
            invalid = invalid[..., :0]
        else:
            invalid = invalid[:0]
        with pytest.raises(ValueError):
            replace(record, **{field_info.name: invalid})


@pytest.mark.parametrize(
    "record", records() + table_records(), ids=lambda record: type(record).__name__
)
def test_array_records_use_identity_comparison_and_remain_unhashable(record: Any) -> None:
    alias = record
    copied = replace(record)
    assert record == alias
    assert record != copied
    assert copied != record
    assert record != object()
    with pytest.raises(TypeError):
        hash(record)


def test_scene_and_source_use_identity_comparison_and_remain_unhashable() -> None:
    scene, irradiance, _, _ = case()
    source = solar_source(irradiance, direction=(30.0, 70.0))
    for record in (scene, source):
        alias = record
        assert record == alias
        assert record != replace(record)
        with pytest.raises(TypeError):
            hash(record)
    assert scene != replace(scene, atmosphere=replace(scene.atmosphere))
    assert source != replace(source, irradiance=replace(irradiance))


@pytest.mark.parametrize("temperature_offset", [0.0, 1.0])
def test_optics_retain_no_atmosphere_dependency(temperature_offset: float) -> None:
    scene, irradiance, rayleigh, aerosol = case()
    source = solar_source(irradiance, direction=(30.0, 70.0))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    recomposed = compose_atmospheric_optics(
        replace(
            scene.atmosphere,
            temperature_k=scene.atmosphere.temperature_k + temperature_offset,
        ),
        source,
        rayleigh=rayleigh,
        aerosol=aerosol,
    )
    assert optics.grid == recomposed.grid
    assert optics.absorption is recomposed.absorption is None
    assert optics.rayleigh is recomposed.rayleigh is rayleigh
    assert optics.aerosol is recomposed.aerosol is aerosol
    assert not hasattr(optics, "reference_atmosphere")


@pytest.mark.parametrize(
    "record",
    [
        record
        for record in records()
        if isinstance(record, (Atmosphere, SurfaceState, LayerProfileBasis))
    ],
    ids=lambda record: type(record).__name__,
)
def test_domain_validation_does_not_rescan_retained_fields(record: Any, monkeypatch) -> None:
    checked = []
    numpy_isfinite = np.isfinite

    def count_finiteness(values, *args, **kwargs):
        checked.append(id(values))
        return numpy_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", count_finiteness)
    replace(record)
    for field_info in fields(record):
        value = getattr(record, field_info.name)
        if isinstance(value, np.ndarray):
            assert checked.count(id(value)) == 1


def test_profile_evaluation_is_a_fresh_writable_array_with_direction_axis_first() -> None:
    baseline = np.array([10.0, 20.0, 30.0])
    directions = np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
    coefficients = np.array([2.0, -0.5])
    basis = LayerProfileBasis("K", baseline, ("a", "b"), directions)
    result = evaluate_profile(basis, coefficients)
    np.testing.assert_array_equal(result, [13.0, 24.0, 35.0])
    assert result.shape == (3,) and result.dtype == np.float64 and result.flags.writeable
    for input_array in (baseline, directions, coefficients, basis.baseline, basis.directions):
        assert not np.shares_memory(result, input_array)
    result.fill(0.0)
    np.testing.assert_array_equal(evaluate_profile(basis, coefficients), [13.0, 24.0, 35.0])
    np.testing.assert_array_equal(coefficients, [2.0, -0.5])
    for invalid in (
        np.array([1.0]),
        np.array([[1.0, 2.0]]),
        np.array([np.nan, 1.0]),
    ):
        with pytest.raises(ValueError, match="coefficients"):
            evaluate_profile(basis, cast(Any, invalid))


def test_rayleigh_has_fixed_phase_and_hg_requires_same_support_asymmetry() -> None:
    _, irradiance, rayleigh, aerosol = case()
    assert not hasattr(rayleigh, "asymmetry")
    assert not hasattr(rayleigh, "phase")
    for invalid in (np.zeros((2, 1)), np.full((2, 2), 0.81)):
        with pytest.raises(ValueError):
            replace(aerosol, asymmetry=invalid)
    with pytest.raises(ValueError, match="nonnegative"):
        RayleighComponent(wavenumber_grid(irradiance.wavenumbers_cm1), -np.ones((2, 2)))
