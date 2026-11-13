"""Offline collection integrity, native measurement decoding and grouped acquisition."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from orsia_data import (
    acquire_hitran_references,
    import_spectroscopy_collection,
    read_cross_section_measurements,
    read_line_spectroscopy_collection,
    read_spectroscopy_collection,
)
from orsia_radiative import (
    CrossSectionMeasurements,
    CrossSectionSpectrum,
    Line,
    LineSpectroscopy,
    PartitionFunctionTable,
    SpectralLineTable,
)


def references():
    table = PartitionFunctionTable(
        2, 1, np.array([200.0, 296.0, 350.0]), np.array([100.0, 148.0, 175.0])
    )
    line = Line(1000, 1e-23, 100, 0.07, 0.1, 0.7)
    ref = LineSpectroscopy(
        "CO2",
        (SpectralLineTable((line,), 2, 1, (970, 1030)), SpectralLineTable((), 2, 1, (1970, 2030))),
        table,
        7.3e-26,
    )
    spectrum = CrossSectionSpectrum(
        np.array([319.0, 320.0, 325.0, 326.0]),
        np.ones(4) * 1e-24,
        250.0,
        None,
        ((319.0, 320.0), (325.0, 326.0)),
        "0.2 nm",
    )
    return ref, CrossSectionMeasurements("O3", "all", (spectrum,))


def store(tmp_path):
    ref, measured = references()
    original = tmp_path / "original.txt"
    original.write_text("authenticity is outside this synthetic fixture")
    destination = tmp_path / "collection"
    path = import_spectroscopy_collection(
        destination,
        lines=(ref,),
        measurements=(measured,),
        source="synthetic",
        edition=None,
        acquired_at_utc=None,
        partition_source=None,
        source_files=(original,),
    )
    return destination, path


def test_collection_round_trip_offline_and_isotope_identity(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hapi", None)
    destination, path = store(tmp_path)
    collection = read_spectroscopy_collection(destination)
    ref = collection.lines[("CO2", 1)]
    assert tuple(d.coverage_cm1 for d in ref.databases) == ((970, 1030), (1970, 2030))
    assert len(ref.databases[1].lines) == 0
    np.testing.assert_array_equal(ref.partition_table.partition_sums, [100, 148, 175])
    assert ref.partition_table.partition_sums.flags.writeable is False
    measured = collection.measurements[("O3", "all")].spectra[0]
    assert measured.pressure_pa is None and measured.support_nm == ((319, 320), (325, 326))
    assert measured.resolution == "0.2 nm"
    assert read_line_spectroscopy_collection(destination)[("CO2", 1)].species == "CO2"
    with pytest.raises(TypeError):
        cast(Any, collection.lines)[("CO2", 1)] = ref
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        import_spectroscopy_collection(
            destination, source="x", edition=None, acquired_at_utc=None, partition_source=None
        )
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["acquired_at_utc"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "units",
        "hash",
        "path",
        "missing_hash",
        "identity",
        "measured_identity",
        "duplicate",
        "convention",
    ],
)
def test_collection_corruption(tmp_path, mutation):
    destination, path = store(tmp_path)
    manifest = json.loads(path.read_text())
    if mutation == "schema":
        manifest["schema"] = "future"
    elif mutation == "units":
        manifest["units"]["mass"] = "u"
    elif mutation == "hash":
        (destination / "lines-0.npz").write_bytes(b"corrupt")
    elif mutation == "path":
        manifest["sha256"]["../original.txt"] = "fake"
    elif mutation == "missing_hash":
        del manifest["sha256"]["lines-0.npz"]
    elif mutation == "identity":
        manifest["lines"][0]["molecule_id"] = 1
    elif mutation == "measured_identity":
        manifest["measurements"][0]["species"] = "NO2"
    elif mutation == "duplicate":
        manifest["lines"].append(manifest["lines"][0])
    else:
        manifest["lines"][0]["intensity_convention"] = "unweighted"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_spectroscopy_collection(destination)


def test_import_failure_cleanup_and_dates(tmp_path):
    ref, measured = references()
    for stamp in ("2026-01-01", "2026-01-01T01:00:00+01:00"):
        with pytest.raises(ValueError, match="UTC"):
            import_spectroscopy_collection(
                tmp_path / "invalid",
                source="x",
                edition=None,
                acquired_at_utc=stamp,
                partition_source=None,
            )
    for lines, measurements in (((ref, ref), ()), ((), (measured, measured))):
        with pytest.raises(ValueError, match="duplicate"):
            import_spectroscopy_collection(
                tmp_path / "duplicate",
                lines=lines,
                measurements=measurements,
                source="x",
                edition=None,
                acquired_at_utc=None,
                partition_source=None,
            )
    with pytest.raises(FileNotFoundError):
        import_spectroscopy_collection(
            tmp_path / "failed",
            lines=(ref,),
            source="x",
            edition="current",
            acquired_at_utc="2026-01-01T00:00:00+00:00",
            partition_source="unknown",
            source_files=(tmp_path / "missing",),
        )
    assert not (tmp_path / "failed").exists()


def native(path, temperature=203, header="vacuum nm; cm2/molec"):
    path.write_text(
        f"{header}\nTemperature: {temperature}K\n"
        "310.0 -1.0E-20\n318.0 1.0E-20\n319.0 2.0E-20\n"
        "320.0 3.0E-20\n325.0 4.0E-20\n326.0 5.0E-20\n330.0 -2.0E-20\n"
    )


def read_native(paths, **changes: Any):
    return read_cross_section_measurements(
        tuple(paths),
        **cast(
            dict[str, Any],
            dict(
                species="SO2",
                isotopologue="all",
                format="bogumil",
                interval_nm=(320, 325),
                **changes,
            ),
        ),
    )


def test_native_reader_retains_per_spectrum_axes_and_trims_negative_tails(tmp_path):
    first, second = tmp_path / "a.dat", tmp_path / "b.dat"
    native(first, 203)
    native(second, 293)
    second.write_text(second.read_text().replace("319.0", "319.1"))
    ref = read_native((second, first))
    assert [s.temperature_k for s in ref.spectra] == [203, 293]
    np.testing.assert_allclose(
        ref.spectra[0].cross_sections_m2_per_molecule, [2e-24, 3e-24, 4e-24], atol=0
    )
    assert ref.spectra[0].support_nm != ref.spectra[1].support_nm
    assert ref.spectra[0].pressure_pa is None


@pytest.mark.parametrize(
    "invalid",
    ["header", "rows", "axis", "columns", "temperature", "negative", "range", "format", "empty"],
)
def test_native_reader_rejects_invalid_retained_support(tmp_path, invalid):
    path = tmp_path / "a.dat"
    native(path)
    changes: dict[str, Any] = {}
    if invalid == "header":
        native(path, header="air nm; cm2/molec")
    elif invalid == "rows":
        path.write_text("vacuum nm cm2/molec\n")
    elif invalid == "axis":
        path.write_text(path.read_text().replace("319.0", "318.0"))
    elif invalid == "columns":
        path.write_text(path.read_text().replace("320.0 3.0E-20", "320.0 3.0E-20 1.0E-20"))
    elif invalid == "temperature":
        path.write_text(path.read_text().replace("Temperature:", "Temp:"))
    elif invalid == "negative":
        path.write_text(path.read_text().replace("320.0 3.0E-20", "320.0 -3.0E-20"))
    elif invalid == "range":
        changes["interval_nm"] = (300, 325)
    elif invalid == "format":
        changes["format"] = "unknown"
    else:
        changes["interval_nm"] = (325, 320)
    with pytest.raises(ValueError):
        read_cross_section_measurements(
            (path,),
            species="SO2",
            isotopologue="all",
            format=changes.get("format", "bogumil"),
            interval_nm=changes.get("interval_nm", (320, 325)),
        )


def test_native_species_units_pressure_and_concatenation(tmp_path):
    first, second = tmp_path / "so2.dat", tmp_path / "no2.dat"
    native(first, 203, header="SO2 Absorption Cross Sections\nvacuum nm; cm2/molec")
    native(second, 223, header="NO2 Absorption Cross Sections\nvacuum nm; cm2/molec")
    with pytest.raises(ValueError, match="species"):
        read_native((first, second))
    raw = first.read_text()
    first.write_text(raw.replace("vacuum nm", "wavelength (vacuum) Angstrom"))
    with pytest.raises(ValueError, match="vacuum nm"):
        read_native((first,))
    first.write_text(raw.replace("vacuum nm", "wavelength (vacuum) nm\nTotal pressure: 100 mbar"))
    ref = read_native((first,))
    assert ref.spectra[0].pressure_pa == 10000
    first.write_text(raw.replace("vacuum nm", "wavelength (vacuum) nm\nTotal pressure: 150 mbar"))
    assert read_native((first,)).spectra[0].pressure_pa == 15000
    first.write_text(
        raw.replace("vacuum nm", "1. column title: wavelength (vacuum)\n1. column unit: Angstrom")
    )
    with pytest.raises(ValueError, match="vacuum nm"):
        read_native((first,))
    first.write_text(
        raw.replace(
            "vacuum nm", "vacuum nm\nNote: Spectral Regions are cut and concatenated at 322.0 nm."
        )
    )
    ref = read_cross_section_measurements(
        (first,), species="SO2", isotopologue="all", format="bogumil", interval_nm=(319, 326)
    )
    assert len(ref.spectra[0].support_nm) == 2
    first.write_text(
        raw.replace(
            "vacuum nm", "vacuum nm\nNote: Spectral Regions are cut and concatenated at 319.5 nm."
        )
    )
    with pytest.raises(ValueError, match="two samples"):
        read_native((first,))


def test_serdyuchenko_columns(tmp_path):
    path = tmp_path / "o3.dat"
    axis = np.array([318.0, 319.0, 320.0, 325.0, 326.0])
    np.savetxt(
        path,
        np.column_stack([axis, np.tile(np.arange(11.0, 0.0, -1.0), (5, 1)) * 1e-20]),
        header="vacuum nm cm^2/molecule\n" + " ".join(f"@{t}K" for t in range(293, 192, -10)),
    )
    ref = read_cross_section_measurements(
        (path,), species="O3", isotopologue="all", format="serdyuchenko", interval_nm=(320, 325)
    )
    assert [s.temperature_k for s in ref.spectra] == list(range(193, 294, 10))
    np.testing.assert_allclose(ref.spectra[0].cross_sections_m2_per_molecule, 1e-24, atol=0)
    path.write_text(path.read_text().replace("@293K", "@294K"))
    with pytest.raises(ValueError, match="temperature columns"):
        read_cross_section_measurements(
            (path,), species="O3", isotopologue="all", format="serdyuchenko", interval_nm=(320, 325)
        )


def fake_hapi(monkeypatch, *, fail=False):
    state: dict[str, Any] = {"calls": [], "directory": None}

    def db_begin(directory):
        state["directory"] = Path(directory)

    def fetch(name, molecule, isotope, lo, hi):
        state["calls"].append((molecule, isotope, lo, hi))
        if fail:
            raise RuntimeError("synthetic failed transfer")
        record = (
            f"{molecule:2d}{isotope}{(lo + hi) / 2:12.6f}{1e-23:10.3E}{1.0:10.3E}"
            f"{0.07:5.3f}{0.1:5.3f}{100.0:10.3f}{0.7:4.2f}{0.0:8.5f}"
        ).ljust(160)
        (state["directory"] / "lines.data").write_text(record + "\n")
        (state["directory"] / "lines.header").write_text("synthetic")

    def partition_sum(molecule, isotope, temperature, *, version):
        assert version == 2025
        return temperature * 2

    module = SimpleNamespace(
        db_begin=db_begin,
        fetch=fetch,
        partitionSum=partition_sum,
        molecularMass=lambda m, i: 44.0,
        moleculeName=lambda m: {1: "H2O", 2: "CO2"}[m],
        HAPI_VERSION="synthetic",
    )
    monkeypatch.setitem(sys.modules, "hapi", module)
    return state


def acquire(destination, **changes):
    options: dict[str, Any] = dict(
        selections={
            ("CO2", 2, 1): ((970.0, 1030.0), (1970.0, 2030.0)),
            ("H2O", 1, 1): ((970.0, 1030.0),),
        },
        partition_temperatures_k=np.array([200.0, 296.0, 350.0]),
        tips_version=2025,
        edition="current",
        directory=destination,
    )
    return acquire_hitran_references(**{**options, **changes})


def test_grouped_acquisition_and_transactional_failure(tmp_path, monkeypatch):
    state = fake_hapi(monkeypatch)
    destination = tmp_path / "acquired"
    manifest = json.loads(acquire(destination).read_text())
    assert len(state["calls"]) == 3
    assert manifest["edition"] == "current" and "TIPS-2025" in manifest["partition_source"]
    refs = read_line_spectroscopy_collection(destination)
    assert len(refs) == 2 and len(refs[("CO2", 1)].databases) == 2
    np.testing.assert_array_equal(refs[("CO2", 1)].partition_table.partition_sums, [400, 592, 700])
    with pytest.raises(FileExistsError):
        acquire(destination)
    fake_hapi(monkeypatch, fail=True)
    with pytest.raises(RuntimeError) as error:
        acquire(tmp_path / "failed")
    assert "CO2" in error.value.__notes__[0]
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".hitran-*"))


@pytest.mark.parametrize(
    "changes",
    [
        {"edition": "2024"},
        {"selections": {}},
        {"partition_temperatures_k": np.array([300.0, 350.0])},
        {"selections": {("CO2", 2, 1): ((970.0, 1030.0),), ("CO2", 1, 1): ((970.0, 1030.0),)}},
        {"selections": {("CO2", 1, 1): ((970.0, 1030.0),)}},
        {"selections": {("CO2", 2, 1): ((970.0, 1030.0), (1000.0, 1040.0))}},
    ],
)
def test_grouped_acquisition_validation_before_transfer(tmp_path, monkeypatch, changes):
    state = fake_hapi(monkeypatch)
    with pytest.raises(ValueError):
        acquire(tmp_path / "invalid", **changes)
    assert not state["calls"]
    assert not (tmp_path / "invalid").exists()
