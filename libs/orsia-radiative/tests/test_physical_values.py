"""Physical-value fields and explicit, caller-owned freshness contracts."""

from dataclasses import asdict, fields, replace
from typing import Any

import numpy as np
import pytest
from orsia_radiative import (
    Absorption,
    AbsorptionComponent,
    AerosolOptics,
    AtmosphericOptics,
    ScatteringProperties,
    compose_absorption,
    compose_absorption_on_grid,
    compose_atmospheric_optics,
    solar_source,
    table_absorption,
    transfer,
    validate_atmospheric_optics,
    wavenumber_grid,
)
from orsia_radiative.optics.atmosphere import validate_scene_optics
from test_composable_transfer import case, field


@pytest.fixture
def physical_values() -> dict[str, Any]:
    scene, irradiance, rayleigh, aerosol = case()
    grid = irradiance.grid
    component = AbsorptionComponent(grid, "X", "1", "cross_section", np.ones((2, 2)))
    absorption = Absorption(grid, component.optical_depth)
    return {
        "irradiance": irradiance,
        "scene": scene,
        "component": component,
        "absorption": absorption,
        "rayleigh": rayleigh,
        "aerosol": aerosol,
        "moments": ScatteringProperties(
            grid,
            rayleigh.optical_depth,
            np.tile([1.0, 0.0, 0.1], (2, 2, 1)),
            "source_untruncated",
        ),
        "albedo": AerosolOptics(
            grid,
            aerosol.optical_depth,
            aerosol.scattering_depth / aerosol.optical_depth,
            aerosol.asymmetry,
        ),
        "optics": AtmosphericOptics(grid, absorption, rayleigh, aerosol),
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("irradiance", ("grid", "values")),
        ("scene", ("atmosphere", "surface", "geometry")),
        ("absorption", ("grid", "optical_depth")),
        (
            "component",
            (
                "grid",
                "species",
                "isotopologue",
                "kind",
                "optical_depth",
                "line_wing_convention",
            ),
        ),
        ("rayleigh", ("grid", "optical_depth")),
        ("aerosol", ("grid", "optical_depth", "scattering_depth", "asymmetry")),
        ("moments", ("grid", "optical_depth", "phase_moments", "truncation")),
        ("albedo", ("grid", "optical_depth", "single_scattering_albedo", "asymmetry")),
        ("optics", ("grid", "absorption", "rayleigh", "aerosol")),
    ],
)
def test_physical_fields_and_serialized_keys(
    physical_values: dict[str, Any], name: str, expected: tuple[str, ...]
) -> None:
    record = physical_values[name]
    assert tuple(f.name for f in fields(record)) == expected
    assert tuple(asdict(record)) == expected
    for obsolete in ("source_id", "input_source_ids", "reference_atmosphere"):
        assert not hasattr(record, obsolete)
        with pytest.raises(TypeError, match=obsolete):
            replace(record, **{obsolete: "removed"})
    np.testing.assert_equal(asdict(record), asdict(replace(record)))


def test_cross_section_inputs_also_exclude_provenance() -> None:
    scene, irradiance, _, _ = case()
    table = field(scene)
    assert not hasattr(table, "source_id")
    obsolete_fields: dict[str, Any] = {"source_id": "another-artifact"}
    with pytest.raises(TypeError, match="source_id"):
        replace(table, **obsolete_fields)
    np.testing.assert_equal(
        asdict(table_absorption(scene.atmosphere, irradiance.grid, table)),
        asdict(table_absorption(scene.atmosphere, irradiance.grid, replace(table))),
    )


def test_reference_sequence_is_explicit_and_must_align() -> None:
    scene, irradiance, _, _ = case()
    source = solar_source(irradiance, direction=(30.0, 70.0))
    component = table_absorption(scene.atmosphere, irradiance.grid, field(scene))
    for references in ((), (scene.atmosphere, scene.atmosphere)):
        with pytest.raises(ValueError, match="align"):
            compose_absorption(
                scene.atmosphere, source, (component,), reference_atmospheres=references
            )
    checked = compose_absorption_on_grid(
        scene.atmosphere,
        irradiance.grid,
        (component,),
        reference_atmospheres=(scene.atmosphere,),
    )
    unchecked = compose_absorption(scene.atmosphere, source, (component,))
    assert checked.grid == unchecked.grid
    np.testing.assert_array_equal(checked.optical_depth, unchecked.optical_depth)
    assert compose_absorption(
        scene.atmosphere, source, (), reference_atmospheres=()
    ).optical_depth.shape == (2, 2)


def test_optical_freshness_can_be_checked_without_storing_the_reference() -> None:
    scene, irradiance, rayleigh, aerosol = case()
    source = solar_source(irradiance, direction=(30.0, 70.0))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    validate_atmospheric_optics(scene, source, optics, reference_atmosphere=scene.atmosphere)
    validate_scene_optics(scene, optics, reference_atmosphere=scene.atmosphere)
    copied = replace(scene, atmosphere=replace(scene.atmosphere))
    with pytest.raises(ValueError, match="stale"):
        validate_scene_optics(copied, optics, reference_atmosphere=scene.atmosphere)
    # Without a reference, validators assert only support, not generating state.
    validate_scene_optics(copied, optics)
    copied_result = transfer.solar.reflected_radiance(
        copied, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    original_result = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    assert copied_result.grid == original_result.grid
    assert copied_result.geometry == original_result.geometry
    np.testing.assert_array_equal(copied_result.values, original_result.values)


@pytest.mark.parametrize("role", ["absorption", "rayleigh", "aerosol"])
def test_direct_optical_values_reject_mismatched_scene_layers(
    physical_values: dict[str, Any], role: str
) -> None:
    scene = physical_values["scene"]
    values = physical_values[role]
    changes = {"optical_depth": values.optical_depth[:1]}
    if role == "aerosol":
        changes.update(scattering_depth=values.scattering_depth[:1], asymmetry=values.asymmetry[:1])
    short = replace(values, **changes)
    roles = {"absorption": None, "rayleigh": None, "aerosol": None, role: short}
    optics = AtmosphericOptics(wavenumber_grid(scene.surface.wavenumbers_cm1), **roles)
    with pytest.raises(ValueError, match="layer"):
        validate_scene_optics(scene, optics)
