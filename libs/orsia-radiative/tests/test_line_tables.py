"""Pure physical table/materialisation invariants and ownership, without any data package."""

from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest
from orsia_radiative import (
    EmptyLineSelection,
    Line,
    LineDatabase,
    PartitionFunctionTable,
    line_set,
)
from test_optical import atmosphere


def database() -> LineDatabase:
    return LineDatabase((Line(1000.0, 1e-25, 0.0, 0.1, 0.2, 0.7, 0.5),), 1, 1, (900.0, 1100.0))


def table() -> PartitionFunctionTable:
    return PartitionFunctionTable(
        1, 1, np.array([250.0, 296.0, 320.0]), np.array([125.0, 148.0, 160.0])
    )


def materialize(lines=None, partitions=None, **options: Any):
    arguments: dict[str, Any] = dict(
        species="X",
        molecular_mass_kg=1e-26,
        pressure_range_pa=(80000.0, 101325.0),
        wing_cutoff_cm1=25.0,
    )
    arguments.update(options)
    return line_set(
        database() if lines is None else lines,
        table() if partitions is None else partitions,
        atmosphere(),
        np.array([999.0, 1001.0]),
        **arguments,
    )


def test_partition_ownership_and_no_provenance() -> None:
    knots = np.array([250.0, 320.0])
    values = np.array([125.0, 160.0])
    result = PartitionFunctionTable(1, 1, knots, values)
    knots.fill(1)
    values.fill(2)
    np.testing.assert_array_equal(result.temperatures_k, [250, 320])
    np.testing.assert_array_equal(result.partition_sums, [125, 160])
    for array in (result.temperatures_k, result.partition_sums):
        assert array.flags.writeable
    for record in (database(), table(), materialize()):
        assert "source_id" not in {field.name for field in fields(record)}


@pytest.mark.parametrize(
    "changes",
    [
        {"molecule_id": 0},
        {"isotopologue_id": 0},
        {"coverage_cm1": (np.nan, 1100)},
        {"coverage_cm1": (1100, 900)},
        {"coverage_cm1": (900, 999)},
    ],
)
def test_invalid_database(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(database(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"molecule_id": 0},
        {"isotopologue_id": 0},
        {"temperatures_k": np.array([250.0, 250.0, 320.0])},
        {"partition_sums": np.array([125.0, -1.0, 160.0])},
    ],
)
def test_invalid_partition_table(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(table(), **changes)


@pytest.mark.parametrize(
    "options",
    [
        {"molecular_mass_kg": 0.0},
        {"wing_cutoff_cm1": np.nan},
        {"wing_cutoff_cm1": 0.0},
        {"pressure_range_pa": (90000.0, 100000.0)},
        {"pressure_range_pa": (100000.0, 80000.0)},
        {"species": "H2O"},
        {"subtract_lorentz_pedestal": True},
    ],
)
def test_invalid_materialisation(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        materialize(**options)


def test_isotope_mismatch_and_pressure_shift_margin() -> None:
    with pytest.raises(ValueError, match="same isotope"):
        materialize(partitions=replace(table(), isotopologue_id=2))
    # Grid upper 1001 + cutoff 25 + maximal pressure shift 0.5 = 1026.5.
    lines = replace(database(), coverage_cm1=(900.0, 1026.4))
    with pytest.raises(ValueError, match="margin"):
        materialize(lines=lines)
    assert materialize(lines=replace(lines, coverage_cm1=(900.0, 1026.5))).lines == lines.lines
    with pytest.raises(EmptyLineSelection):
        materialize(lines=replace(database(), lines=()))
