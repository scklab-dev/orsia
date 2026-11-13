"""Import retained line/measurement spectroscopy and native water-continuum inputs.

This explicit offline preparation preserves the original acquisition manifest.
The line/measurement collection retains the notebook's disjoint intervals; native
MT_CKD coefficients use a separate hash receipt. No data are downloaded.
"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from orsia_data import (
    import_spectroscopy_collection,
    read_cross_section_measurements,
    read_hitran_lines,
    read_mt_ckd_water,
    read_partition_table,
    read_spectroscopy_collection,
)
from orsia_data.acquisition import file_sha256
from orsia_radiative import LineSpectroscopy
from scipy.constants import atomic_mass

from notebooks.support.five_band_scenario import load_scenario
from notebooks.support.notebook_five_band_support import verify_water_continuum

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Verify references or import retained native files, without acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--water-continuum", type=Path, help="Retained MT_CKD 4.3 classic-NetCDF file to import"
    )
    args = parser.parse_args()
    scenario, _, _ = load_scenario(ROOT / "notebooks/data/five-band.json")
    prepare_water_continuum(scenario.water_continuum_manifest, native=args.water_continuum)
    prepare_collection(scenario)


def prepare_water_continuum(receipt_path: Path, *, native: Path | None) -> None:
    """Retain native water coefficients with a separate offline-import receipt.

    Args:
        receipt_path: New receipt location beside the retained native file.
        native: Local MT_CKD 4.3 source for first import; None verifies an existing receipt.

    Raises:
        FileNotFoundError: A first import lacks a native source.
        FileExistsError: A new import would overwrite an unreceipted native file.
        ValueError: The reference cannot be decoded or a retained checksum differs.
    """
    if receipt_path.exists():
        retained, _ = verify_water_continuum(receipt_path)
        if native is not None and file_sha256(native) != file_sha256(retained):
            raise ValueError("supplied water continuum differs from retained reference")
        read_mt_ckd_water(retained, temperature_range_k=(150.0, 350.0))
        print(f"Verified water continuum: {receipt_path}")
        return
    if native is None:
        raise FileNotFoundError("First import requires --water-continuum /path/to/MT_CKD-4.3.nc")
    read_mt_ckd_water(native, temperature_range_k=(150.0, 350.0))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    retained = receipt_path.parent / "absco-ref_wv-mt-ckd.nc"
    with native.open("rb") as source, retained.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    receipt = dict(
        edition="4.3",
        file=retained.name,
        sha256=file_sha256(retained),
        imported_at_utc=datetime.now(UTC).isoformat(),
        provenance="Caller-supplied retained MT_CKD 4.3 bytes; acquisition date unrecorded",
    )
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Imported water continuum offline: {receipt_path}")


def prepare_collection(scenario):
    """Import or verify the scenario's retained line and measurement collection."""
    destination = scenario.spectroscopy_collection
    if destination.exists():
        read_spectroscopy_collection(destination)
        print(f"Verified spectroscopy collection: {destination}")
        return
    lines = []
    for gas in scenario.inputs.spectroscopy:
        intervals = sorted(
            (
                1e7 / scenario.bands[name].high_nm - scenario.line_margin_cm1,
                1e7 / scenario.bands[name].low_nm + scenario.line_margin_cm1,
            )
            for name in gas.included_cases
        )
        databases = tuple(
            read_hitran_lines(
                gas.lines,
                molecule_id=gas.molecule_id,
                isotopologue_id=gas.isotopologue_id,
                interval_cm1=interval,
            )
            for interval in intervals
        )
        partition = read_partition_table(
            gas.partition, molecule_id=gas.molecule_id, isotopologue_id=gas.isotopologue_id
        )
        lines.append(LineSpectroscopy(gas.species, databases, partition, gas.mass_u * atomic_mass))
    measurements = tuple(
        read_cross_section_measurements(
            tuple(entry.files),
            species=entry.species,
            isotopologue="all",
            format=entry.format,
            interval_nm=(319.0, 326.0),
            resolution=entry.model_extra.get("resolution") if entry.model_extra else None,
        )
        for entry in scenario.inputs.cross_sections
    )
    manifest = json.loads(scenario.manifest.read_text())
    origins = {
        (entry["species"], str(entry["isotopologue_id"])): dict(
            source=entry.get("source"),
            edition="current",
            partition_source=None,
            acquired_at_utc=manifest.get("acquired_at_utc"),
        )
        for entry in manifest["spectroscopy"]
    }
    origins.update(
        {
            (entry["species"], "all"): dict(
                source=entry.get("source"),
                edition="1.0" if entry["format"] == "bogumil" else None,
                acquired_at_utc=manifest.get("acquired_at_utc"),
            )
            for entry in manifest["cross_sections"]
        }
    )
    source_files = [scenario.manifest]
    source_files.extend(
        path for gas in scenario.inputs.spectroscopy for path in (gas.lines, gas.partition)
    )
    source_files.extend(path for entry in scenario.inputs.cross_sections for path in entry.files)
    receipt = import_spectroscopy_collection(
        destination,
        lines=tuple(lines),
        measurements=measurements,
        source="Retained five-band inputs; original receipt copied alongside",
        edition=None,
        acquired_at_utc=manifest.get("acquired_at_utc"),
        partition_source=None,
        source_files=tuple(dict.fromkeys(source_files)),
        reference_provenance=origins,
    )
    print(f"Imported offline spectroscopy: {receipt}")


if __name__ == "__main__":
    main()
