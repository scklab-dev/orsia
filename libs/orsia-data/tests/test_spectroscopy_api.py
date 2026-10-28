"""Reader exports and independent partition-table ownership."""

from pathlib import Path

import numpy as np
import orsia_data
from orsia_data import spectroscopy
from orsia_data.spectroscopy import continuum, cross_sections, lines, partition_sums


def test_exports() -> None:
    for module, name in (
        (lines, "read_aer_lines"),
        (lines, "read_hitran_lines"),
        (partition_sums, "read_partition_table"),
        (continuum, "read_mt_ckd_water"),
        (cross_sections, "read_cross_section_table"),
    ):
        assert getattr(orsia_data, name) is getattr(spectroscopy, name) is getattr(module, name)
    for name in ("SpectroscopyArtifact", "SolarArtifact", "line_set", "read_manifest"):
        assert not hasattr(orsia_data, name)


def test_partition_table_owned_storage(tmp_path: Path) -> None:
    path = tmp_path / "q.csv"
    path.write_text(
        "molecule_id,isotopologue_id,temperature_k,partition_sum\n"
        "1,1,200,100\n1,1,296,148\n1,1,350,175\n4,1,200,80\n4,1,350,140\n"
    )
    table = spectroscopy.read_partition_table(path, molecule_id=1, isotopologue_id=1)
    np.testing.assert_array_equal(table.temperatures_k, [200, 296, 350])
    np.testing.assert_array_equal(table.partition_sums, [100, 148, 175])
    assert not np.shares_memory(table.temperatures_k, table.partition_sums)
    fresh = spectroscopy.read_partition_table(path, molecule_id=1, isotopologue_id=1)
    assert not np.shares_memory(table.partition_sums, fresh.partition_sums)
    assert table.partition_sums.flags.writeable
    table.partition_sums[0] = 0
    assert fresh.partition_sums[0] == 100
