from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import pytest
from orsia_radiative import (
    Absorber,
    Absorption,
    AbsorptionComponent,
    Atmosphere,
    AtmosphericOptics,
    CrossSectionField,
    ObservationGeometry,
    RadiativeScene,
    SpectralDomain,
    SpectralGrid,
    SpectralIrradiance,
    SurfaceState,
    atmospheric_optics,
    compose_absorption,
    compose_absorption_on_grid,
    compose_atmospheric_optics,
    convert_spectral_grid,
    cross_sections,
    evaluate_table_absorption,
    solar_source,
    transfer,
    wavenumber_grid,
)


def inputs() -> tuple[Atmosphere, CrossSectionField]:
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
    table = CrossSectionField(
        "X",
        "1",
        "cross_section",
        wavenumber_grid(np.array([999.0, 1003.0])),
        atmosphere.temperature_k,
        atmosphere.pressure_pa,
        np.array([[1e-27, 5e-27], [2e-27, 6e-27]]),
    )
    return atmosphere, table


def test_domain_and_grid_validate_explicit_support_without_inference() -> None:
    domain = SpectralDomain("wavelength_nm", ((900.0, 1100.0), (1900.0, 2100.0)))
    grid = SpectralGrid("wavelength_nm", np.array([1000.0, 2000.0]), domain)
    assert domain.contains(grid.points)
    assert grid.domain is domain
    assert wavenumber_grid(np.array([1000.0, 1001.0])).domain is None
    with pytest.raises(ValueError, match="outside"):
        SpectralGrid("wavelength_nm", np.array([1200.0]), domain)


def test_domain_rejects_empty_overlapping_or_nonpositive_intervals() -> None:
    for intervals in ((), ((0.0, 1.0),), ((2.0, 1.0),), ((1.0, 2.0), (2.0, 3.0))):
        with pytest.raises(ValueError, match="spectral"):
            SpectralDomain("wavenumber_cm1", intervals)


def test_grid_snapshots_mutable_and_strided_points() -> None:
    owner = np.array([1000.0, 0.0, 1001.0, 0.0])
    grid = wavenumber_grid(owner[::2])
    owner[:] = 7.0
    np.testing.assert_array_equal(grid.points, [1000.0, 1001.0])
    with pytest.raises(ValueError):
        grid.points.flags.writeable = True
    with pytest.raises(ValueError, match="strictly increasing"):
        SpectralGrid("wavenumber_cm1", np.array([1001.0, 1000.0]))


def test_exact_point_compatibility_is_distinct_from_domain_equality() -> None:
    points = np.array([1000.0, 1001.0])
    first = SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((999.0, 1002.0),))
    )
    second = SpectralGrid("wavenumber_cm1", points.copy())
    assert first.same_points(second)
    assert first != second
    assert not first.same_points(SpectralGrid("wavelength_nm", points))


def test_coordinate_conversion_reverses_points_and_intervals_only() -> None:
    grid = SpectralGrid(
        "wavelength_nm",
        np.array([1000.0, 2000.0]),
        SpectralDomain("wavelength_nm", ((900.0, 1100.0), (1900.0, 2100.0))),
    )
    converted = convert_spectral_grid(grid)
    np.testing.assert_array_equal(converted.points, [5000.0, 10000.0])
    assert converted.coordinate == "wavenumber_cm1"
    assert convert_spectral_grid(converted).points.tolist() == grid.points.tolist()
    assert convert_spectral_grid(SpectralGrid("wavenumber_cm1", np.array([1000.0]))).domain is None


def test_wavelength_grid_must_be_converted_before_radiative_use() -> None:
    wavelength_grid = SpectralGrid("wavelength_nm", np.array([1000.0, 1001.0]))
    with pytest.raises(ValueError, match="convert explicitly"):
        wavenumber_grid(wavelength_grid)
    with pytest.raises(ValueError, match="coordinate"):
        SpectralDomain(cast(Any, "frequency_hz"), ((1.0, 2.0),))
    with pytest.raises(ValueError, match="coordinate/unit"):
        SpectralGrid(
            "wavenumber_cm1",
            np.array([1000.0]),
            SpectralDomain("wavelength_nm", ((999.0, 1002.0),)),
        )


def test_one_grid_is_retained_through_spectroscopy_composition_and_transfer() -> None:
    atmosphere, table = inputs()
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.array([1000.0, 1001.0]),
        SpectralDomain("wavenumber_cm1", ((999.0, 1000.0), (1001.0, 1003.0))),
    )
    irradiance = SpectralIrradiance(grid, np.ones(2))
    source = solar_source(irradiance, direction=(30.0, 70.0))
    prepared = cross_sections(grid, table)
    component = evaluate_table_absorption(atmosphere, prepared)
    total = compose_absorption(atmosphere, source, (component,))
    optics = compose_atmospheric_optics(atmosphere, source, absorption=total)
    surface = SurfaceState(grid, reflectance=np.full(2, 0.2))
    scene = RadiativeScene(atmosphere, surface, ObservationGeometry(20.0))
    result = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    for record in (irradiance, prepared, component, total, optics, surface, result):
        assert record.grid is grid
        assert record.wavenumbers_cm1 is grid.points
        assert replace(record).grid is grid
        assert "grid" in {field.name for field in fields(record)}
        assert "wavenumbers_cm1" not in {field.name for field in fields(record)}
    assert compose_absorption_on_grid(atmosphere, grid, (component,)).grid is grid
    assert atmospheric_optics(atmosphere, grid, absorption=total).grid is grid
    with pytest.raises(ValueError, match="axis and layer"):
        compose_absorption_on_grid(atmosphere, wavenumber_grid(grid.points + 0.25), (component,))
    thermal_surface = SurfaceState(grid, emissivity=np.ones(2), temperature_k=290.0)
    thermal_scene = replace(scene, surface=thermal_surface)
    assert transfer.thermal.schwarzschild_radiance(thermal_scene, optics=optics).grid is grid


def test_radiative_records_reject_wavelength_units_at_construction() -> None:
    atmosphere, _ = inputs()
    grid = SpectralGrid("wavelength_nm", np.array([1000.0, 1001.0]))
    constructors = (
        lambda: SpectralIrradiance(grid, np.ones(2)),
        lambda: SurfaceState(grid, reflectance=np.ones(2)),
        lambda: Absorption(grid, np.zeros((2, 2))),
        lambda: AbsorptionComponent(grid, "X", "1", "cross_section", np.zeros((2, 2))),
        lambda: AtmosphericOptics(grid, None, None, None),
    )
    for construct in constructors:
        with pytest.raises(ValueError, match="convert explicitly"):
            construct()


def test_native_table_domain_does_not_authorize_interpolation_across_gaps() -> None:
    _, table = inputs()
    domain = SpectralDomain("wavenumber_cm1", ((999.0, 1000.0), (1002.0, 1003.0)))
    table = replace(table, grid=wavenumber_grid(table.wavenumbers_cm1, domain=domain))
    with pytest.raises(ValueError, match="outside declared"):
        cross_sections(wavenumber_grid(np.array([1001.0])), table)
    with pytest.raises(ValueError, match="crosses.*gap"):
        cross_sections(wavenumber_grid(np.array([999.5])), table)
    prepared = cross_sections(table.grid, table)
    np.testing.assert_array_equal(
        prepared.cross_sections_m2_per_molecule, table.cross_sections_m2_per_molecule
    )
