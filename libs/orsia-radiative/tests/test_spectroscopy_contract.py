"""Support and identity checks for supplied spectroscopy and prepared tables."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from orsia_radiative import (
    AbsorptionComponent,
    CrossSectionField,
    Line,
    LineSet,
    cross_sections,
    wavenumber_grid,
)


def table() -> CrossSectionField:
    return CrossSectionField(
        "X",
        "1",
        "cross_section",
        wavenumber_grid(np.array([1000.0, 1001.0])),
        np.array([290.0]),
        np.array([95000.0]),
        np.array([[1e-27, 2e-27]]),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"species": ""}, "species"),
        ({"kind": "unknown"}, "supported kind"),
        ({"isotopologue": None}, "isotopologue"),
        ({"grid": wavenumber_grid(np.array([1000.0]))}, "at least two"),
        ({"temperature_k": np.empty(0)}, "shape"),
        ({"pressure_pa": np.array([95000.0, 90000.0])}, "shape"),
        ({"cross_sections_m2_per_molecule": np.empty((0, 2))}, "shape"),
        ({"cross_sections_m2_per_molecule": np.array([[1e-27]])}, "shape"),
        ({"cross_sections_m2_per_molecule": np.array([[-1e-27, 2e-27]])}, "nonnegative"),
    ],
)
def test_invalid_cross_section_contract(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(table(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"species": ""}, "species"),
        ({"isotopologue": ""}, "isotopologue"),
        ({"kind": "unknown"}, "unsupported"),
        ({"optical_depth": np.empty((0, 2))}, "shape"),
        ({"optical_depth": np.array([[0.1]])}, "shape"),
    ],
)
def test_invalid_absorption_contract(changes: dict[str, Any], message: str) -> None:
    absorption = AbsorptionComponent(
        wavenumber_grid(table().wavenumbers_cm1),
        "X",
        "1",
        "cross_section",
        np.array([[0.1, 0.2]]),
    )
    with pytest.raises(ValueError, match=message):
        replace(absorption, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"species": ""},
        {"lines": ()},
        {"partition_ratios": np.empty(0)},
        {"temperature_range_k": (310.0, 250.0)},
        {"pressure_range_pa": (100000.0, 80000.0)},
    ],
)
def test_invalid_line_set_contract(changes: dict[str, Any]) -> None:
    lines = LineSet(
        "X",
        "1",
        (Line(1000.0, 1e-26, 0.0, 0.05, 0.1, 0.7),),
        np.ones(1),
        (250.0, 310.0),
        (80000.0, 100000.0),
    )
    with pytest.raises(ValueError):
        replace(lines, **changes)


def test_line_shift_must_be_finite() -> None:
    with pytest.raises(ValueError, match="shift"):
        Line(1000.0, 1e-26, 0.0, 0.05, 0.1, 0.7, pressure_shift_ref_cm1=float("inf"))


def test_signed_temperature_exponent() -> None:
    line = Line(1000.0, 1e-26, 0.0, 0.05, 0.1, -0.2)
    assert line.temperature_exponent == -0.2
    assert (296.0 / 250.0) ** line.temperature_exponent > 0
    for exponent in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="exponent"):
            replace(line, temperature_exponent=exponent)


@pytest.mark.parametrize("rows", [(), ((1e-27,),), ((-1e-27, 2e-27),)])
def test_prepared_cross_section_shape_and_sign(rows: tuple[tuple[float, ...], ...]) -> None:
    inputs = table()
    prepared = cross_sections(inputs.wavenumbers_cm1, inputs)
    with pytest.raises(ValueError):
        replace(prepared, cross_sections_m2_per_molecule=np.asarray(rows, dtype=np.float64))


def test_table_and_prepared_identity_and_owned_grids() -> None:
    inputs = table()
    assert inputs != table()
    np.testing.assert_equal(asdict(inputs), asdict(table()))
    changed = replace(
        inputs, cross_sections_m2_per_molecule=inputs.cross_sections_m2_per_molecule * 2
    )
    np.testing.assert_array_equal(
        changed.cross_sections_m2_per_molecule, 2 * inputs.cross_sections_m2_per_molecule
    )
    grid = inputs.wavenumbers_cm1.copy()
    prepared = cross_sections(grid, inputs)
    grid[0] = 900.0
    independently_prepared = cross_sections(inputs.wavenumbers_cm1, table())
    assert prepared != independently_prepared
    np.testing.assert_equal(asdict(prepared), asdict(independently_prepared))
    changed_prepared = replace(prepared, cross_sections_m2_per_molecule=np.array([[2e-27, 3e-27]]))
    np.testing.assert_array_equal(changed_prepared.cross_sections_m2_per_molecule, [[2e-27, 3e-27]])
    np.testing.assert_array_equal(prepared.cross_sections_m2_per_molecule, [[1e-27, 2e-27]])
    assert prepared.wavenumbers_cm1[0] == 1000.0
    assert not prepared.wavenumbers_cm1.flags.writeable


@pytest.mark.parametrize("values", [(1e-20, 1e-40), (1e-40, 1e-20), (1e-20, 0.0)])
def test_cross_section_interpolation_preserves_high_dynamic_range(
    values: tuple[float, float],
) -> None:
    inputs = replace(table(), cross_sections_m2_per_molecule=np.array([values]))
    nodes = inputs.wavenumbers_cm1
    exact = cross_sections(nodes, inputs)
    np.testing.assert_array_equal(exact.cross_sections_m2_per_molecule, [values])
    grid = np.array([nodes[0], 1000.25, 1000.5, 1000.75, nodes[-1]])
    prepared = cross_sections(grid, inputs)
    expected = np.array(
        [
            [
                values[0],
                0.75 * values[0] + 0.25 * values[1],
                0.5 * values[0] + 0.5 * values[1],
                0.25 * values[0] + 0.75 * values[1],
                values[1],
            ]
        ]
    )
    np.testing.assert_allclose(
        prepared.cross_sections_m2_per_molecule, expected, rtol=1e-15, atol=0
    )
