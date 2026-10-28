"""Synthetic format/materialisation contracts, not authentic spectroscopy."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from orsia_data import (
    read_aer_lines,
    read_cross_section_table,
    read_hitran_lines,
    read_mt_ckd_water,
    read_partition_table,
)
from orsia_radiative import (
    Absorber,
    Atmosphere,
    EmptyLineSelection,
    evaluate_partition_function,
    line_absorption,
    line_set,
    table_absorption,
    water_continuum_absorption,
)
from scipy.constants import atomic_mass
from scipy.io import netcdf_file


def record(molecule: int = 1, isotope: str = "1", centre: float = 950.0, flag: int = 0) -> str:
    prefix = (
        f"{molecule:2d}{isotope}{centre:12.6f}{1e-25:10.3E}{1.0:10.3E}"
        f"{0.1:5.3f}{0.2:5.3f}{0.0:10.3f}{0.7:4.2f}{0.0:8.5f}"
    )
    assert len(prefix) == 67
    return prefix.ljust(98) + f"{flag:2d}" + " " * 60 + "\n"


def atmosphere() -> Atmosphere:
    return Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([296.0, 280.0]),
        np.array([2e28, 1e28]),
        np.array([1.9e28, 0.95e28]),
        np.array([1e27, 5e26]),
        (Absorber("H2O", np.array([1e27, 5e26]), "dry_air"),),
    )


def partitions(root: Path):
    path = root / "q.csv"
    path.write_text(
        "molecule_id,isotopologue_id,temperature_k,partition_sum\n"
        "1,1,200,100\n1,1,296,148\n1,1,350,175\n"
    )
    return read_partition_table(path, molecule_id=1, isotopologue_id=1)


def materialize(lines, table, **options):
    arguments = dict(
        species="H2O",
        molecular_mass_kg=18 * atomic_mass,
        pressure_range_pa=(80000.0, 101325.0),
        wing_cutoff_cm1=25.0,
        subtract_lorentz_pedestal=True,
    )
    arguments.update(options)
    return line_set(lines, table, atmosphere(), np.array([949.0, 950.0, 951.0]), **arguments)


@pytest.mark.parametrize("reader", [read_aer_lines, read_hitran_lines])
def test_offline_materialisation_and_reference_strength(tmp_path: Path, reader) -> None:
    path = tmp_path / "lines"
    path.write_text(record() + record(4) + record(3))
    lines = reader(path, molecule_id=1, isotopologue_id=1, interval_cm1=(800.0, 1100.0))
    q = partitions(tmp_path)
    inputs = materialize(lines, q)
    np.testing.assert_allclose(inputs.partition_ratios, [1.0, 296 / 280])
    assert inputs.lines[0].strength_ref_cm_per_molecule == 1e-25
    assert inputs.isotopologue == "1" and inputs.subtract_lorentz_pedestal
    np.testing.assert_equal(asdict(materialize(lines, q)), asdict(inputs))
    result = line_absorption(atmosphere(), np.array([949.0, 950.0, 951.0]), inputs)
    assert np.all(np.isfinite(result.optical_depth)) and np.all(result.optical_depth > 0)
    for molecule in (4, 3):
        assert (
            len(
                reader(
                    path, molecule_id=molecule, isotopologue_id=1, interval_cm1=(949.0, 951.0)
                ).lines
            )
            == 1
        )


def test_empty_missing_invalid_and_coverage(tmp_path: Path) -> None:
    path = tmp_path / "lines"
    path.write_text(record(centre=850.0))
    lines = read_aer_lines(path, molecule_id=1, isotopologue_id=1, interval_cm1=(800.0, 1100.0))
    q = partitions(tmp_path)
    assert (
        read_aer_lines(path, molecule_id=1, isotopologue_id=2, interval_cm1=(900.0, 1000.0)).lines
        == ()
    )
    with pytest.raises(EmptyLineSelection):
        materialize(lines, q)
    with pytest.raises(ValueError, match="margin"):
        materialize(replace(lines, lines=(), coverage_cm1=(925.0, 975.0)), q)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_aer_lines(path, molecule_id=1, isotopologue_id=1, interval_cm1=(900.0, 1000.0))


def test_aer_headers_coupling_and_isotope_encoding(tmp_path: Path) -> None:
    path = tmp_path / "lines"
    coefficients = " 1" + (f"{0.0:13.5E}{0.0:11.3E}" * 4) + " 1\n"
    path.write_text(
        "> Copyright © AER\n%%%%%%%%\n\n"
        + record(flag=-1)
        + coefficients
        + record(4)
        + record(isotope="0")
        + record(isotope="A")
    )
    options = dict(molecule_id=1, isotopologue_id=1, interval_cm1=(900.0, 1000.0))
    with pytest.raises(ValueError, match="coupling"):
        read_aer_lines(path, **options)
    assert len(read_aer_lines(path, coupling="ignore", **options).lines) == 1
    for isotope in (10, 11):
        assert (
            len(
                read_aer_lines(
                    path, molecule_id=1, isotopologue_id=isotope, interval_cm1=(900.0, 1000.0)
                ).lines
            )
            == 1
        )
    for text in (
        record(flag=-1),
        record(flag=-1) + "bad\n",
        "%%%%%%%\n",
        record(isotope=" "),
        record(molecule=0),
        record().replace("1.000E-25", "      nan"),
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_aer_lines(path, coupling="ignore", **options)
    path.write_text(record())
    with pytest.raises(ValueError, match="policy"):
        invalid_policy: Any = "bad"
        read_aer_lines(path, coupling=invalid_policy, **options)


def test_partition_support_and_invalid_tables(tmp_path: Path) -> None:
    q = partitions(tmp_path)
    np.testing.assert_allclose(evaluate_partition_function(q, np.array([280.0, 296.0])), [140, 148])
    for temperatures in (
        np.array([180.0]),
        np.array([400.0]),
        np.empty(0),
        np.array([[280.0]]),
        np.array([np.nan]),
    ):
        with pytest.raises(ValueError, match="support"):
            invalid_temperatures: Any = temperatures
            evaluate_partition_function(q, invalid_temperatures)
    path = tmp_path / "q.csv"
    with pytest.raises(ValueError, match="isotope"):
        read_partition_table(path, molecule_id=1, isotopologue_id=2)
    for text in (
        "wrong,header\n",
        "molecule_id,isotopologue_id,temperature_k,partition_sum\n1,1,296,148\n1,1,296,150\n",
        "molecule_id,isotopologue_id,temperature_k,partition_sum\n1,1,200,-1\n1,1,300,100\n",
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_partition_table(path, molecule_id=1, isotopologue_id=1)


def test_normalized_cross_sections(tmp_path: Path) -> None:
    path = tmp_path / "sections.npz"
    np.savez(
        path,
        wavenumbers_cm1=np.array([900.0, 950.0, 1000.0]),
        temperature_k=np.array([296.0, 280.0]),
        pressure_pa=np.array([95000.0, 85000.0]),
        cross_sections_m2_per_molecule=np.full((2, 3), 1e-27),
    )
    field = read_cross_section_table(path, species="H2O", isotopologue="1", kind="cross_section")
    assert field.cross_sections_m2_per_molecule.flags.writeable
    result = table_absorption(atmosphere(), np.array([925.0, 950.0]), field)
    np.testing.assert_array_equal(result.optical_depth, [[1.0, 1.0], [0.5, 0.5]])
    np.savez(path, wrong=np.ones(2))
    with pytest.raises(ValueError, match="arrays"):
        read_cross_section_table(path, species="H2O", isotopologue="1", kind="cross_section")


def test_classic_mt_ckd_reading_and_absorption(tmp_path: Path) -> None:
    path = tmp_path / "water.nc"
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("wavenumber", 7)
        dataset.createDimension("scalar", 1)
        for name, values in {
            "wavenumbers": np.arange(800.0, 1101.0, 50.0),
            "self_absco_ref": np.full(7, 1e-24),
            "for_absco_ref": np.full(7, 2e-24),
            "self_texp": np.full(7, 3.0),
        }.items():
            dataset.createVariable(name, "d", ("wavenumber",))[:] = values
        for name, value in (("ref_press", 1013.0), ("ref_temp", 296.0)):
            dataset.createVariable(name, "d", ("scalar",))[:] = value
    parsed = read_mt_ckd_water(path, temperature_range_k=(200.0, 350.0))
    assert parsed.reference_pressure_pa == 101300.0
    assert parsed.self_absco_ref.flags.writeable
    assert np.all(
        water_continuum_absorption(atmosphere(), np.array([950.0]), parsed).optical_depth > 0
    )
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="classic"):
        read_mt_ckd_water(path, temperature_range_k=(200.0, 350.0))
    with netcdf_file(path, "w"):
        pass
    with pytest.raises(ValueError, match="variables"):
        read_mt_ckd_water(path, temperature_range_k=(200.0, 350.0))
