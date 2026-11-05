"""Import the retained authentic five-band spectroscopy into one offline collection.

This explicit preparation reads the original manifest and never downloads.
The notebook's band selections define disjoint intervals; the reusable library
receives those intervals as caller declarations. Original inputs remain unchanged.
"""

import json
from pathlib import Path

from orsia_data import (
    import_spectroscopy_collection,
    read_cross_section_measurements,
    read_hitran_lines,
    read_partition_table,
    read_spectroscopy_collection,
)
from orsia_radiative import LineSpectroscopy
from scipy.constants import atomic_mass

from notebooks.support.five_band_scenario import load_scenario

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Verify existing references or import them to the scenario's new destination."""
    scenario, _, _ = load_scenario(ROOT / "notebooks/data/five-band.json")
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
