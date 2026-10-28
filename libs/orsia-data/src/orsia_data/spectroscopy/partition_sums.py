"""Load partition-function samples for a selected local isotopologue."""

import csv
from pathlib import Path

import numpy as np
from orsia_radiative import PartitionFunctionTable


def read_partition_table(
    path: Path,
    *,
    molecule_id: int,
    isotopologue_id: int,
) -> PartitionFunctionTable:
    """Read dimensionless Q and temperature knots [K] for one local isotope.

    Columns: molecule_id,isotopologue_id,temperature_k,partition_sum.
    Return independent writable float64 arrays; no interpolation or extrapolation.

    Args:
        path: Existing local partition-function CSV with the declared columns.
        molecule_id: Numeric molecule identifier for rows to select.
        isotopologue_id: Numeric isotope identifier for rows to select.

    Returns:
        Partition table with increasing temperatures in K and dimensionless
        partition sums for the selected molecule/isotopologue.

    Raises:
        ValueError: Header is unsupported or selected temperatures or sums
            are insufficient, invalid or out of order.
        OSError: The local CSV cannot be read.
    """
    rows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "molecule_id",
            "isotopologue_id",
            "temperature_k",
            "partition_sum",
        ]:
            raise ValueError("unsupported partition CSV columns")
        for row in reader:
            if (
                int(row["molecule_id"]) == molecule_id
                and int(row["isotopologue_id"]) == isotopologue_id
            ):
                rows.append((float(row["temperature_k"]), float(row["partition_sum"])))
    values = np.asarray(rows, dtype=np.float64)
    if (
        len(rows) < 2
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or np.any(np.diff(values[:, 0]) <= 0)
    ):
        raise ValueError("partition table lacks positive increasing knots/sums for isotope")
    return PartitionFunctionTable(molecule_id, isotopologue_id, values[:, 0], values[:, 1])
