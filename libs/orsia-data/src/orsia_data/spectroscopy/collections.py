"""Offline, hash-verified spectroscopy collections with explicit source provenance."""

import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
from orsia_radiative.spectroscopy.line_tables import PartitionFunctionTable, SpectralLineTable
from orsia_radiative.spectroscopy.lines import Line
from orsia_radiative.spectroscopy.measurements import CrossSectionMeasurements, CrossSectionSpectrum
from orsia_radiative.spectroscopy.references import LineSpectroscopy

from orsia_data.acquisition import file_sha256


@dataclass(frozen=True)
class SpectroscopyCollection:
    """Named isotope references and laboratory measurements loaded from one receipt.

    Mapping keys include species and isotope to avoid ambiguous species selection.
    Numerical references own read-only snapshots. Mapping containers are read-only.
    Source provenance remains in the on-disk manifest, owned by Data.

    Attributes:
        lines: References keyed by (species, molecule-local isotope number).
        measurements: Measurements keyed by (species, isotope label).
    """

    lines: Mapping[tuple[str, int], LineSpectroscopy]
    measurements: Mapping[tuple[str, str], CrossSectionMeasurements]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", MappingProxyType(dict(self.lines)))
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))


def import_spectroscopy_collection(
    directory: Path,
    *,
    lines: tuple[LineSpectroscopy, ...] = (),
    measurements: tuple[CrossSectionMeasurements, ...] = (),
    source: str,
    edition: str | None,
    acquired_at_utc: str | None,
    partition_source: str | None,
    source_files: tuple[Path, ...] = (),
    reference_provenance: Mapping[tuple[str, str], Mapping[str, str | None]] | None = None,
) -> Path:
    """Store reusable spectroscopy and source bytes in a new collection directory.

    Normalize to pickle-free NPZ with identity and units in a versioned manifest.
    Preserve exact disjoint support and per-spectrum native axes. Acquisition date,
    edition and partition source are caller-supplied facts; None records unknown.
    Import time is recorded separately. Failures remove the exclusively owned new
    directory. Existing destinations remain unchanged. Callers serialize writes.

    Args:
        directory: New exclusively owned destination, with parents created if needed.
        lines: Isotope-specific numerical line references.
        measurements: Species/isotope laboratory reference collections.
        source: Source description retained verbatim in the receipt.
        edition: Known source edition, 'current' for an unpinned service, or None.
        acquired_at_utc: Original timezone-aware acquisition timestamp, or None.
        partition_source: Known Q(T) implementation/version description, or None.
        source_files: Original local files to copy and hash alongside normalized values.
        reference_provenance: Optional source facts keyed by (species, isotope text).
            These fields supplement collection-wide provenance for mixed-source imports.

    Returns:
        Path of the new manifest.json receipt.

    Raises:
        FileExistsError: Destination already exists.
        OSError: Local creation, copy or serialization fails.
        ValueError: Identity is duplicated, timestamp lacks UTC or serialization is invalid.
    """
    if acquired_at_utc is not None:
        stamp = datetime.fromisoformat(acquired_at_utc)
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("acquisition timestamp must specify UTC")
    _unique_keys(lines, measurements)
    origins = {} if reference_provenance is None else reference_provenance
    directory.mkdir(parents=True, exist_ok=False)
    try:
        entries = []
        for index, reference in enumerate(lines):
            table = reference.partition_table
            filename = f"lines-{index}.npz"
            arrays = dict(
                molecule_id=np.array(table.molecule_id),
                isotopologue_id=np.array(table.isotopologue_id),
                species=np.array(reference.species),
                molecular_mass_kg=np.array(reference.molecular_mass_kg),
                temperatures_k=table.temperatures_k,
                partition_sums=table.partition_sums,
            )
            intervals = []
            for segment, database in enumerate(reference.databases):
                arrays[f"transitions_{segment}"] = np.array(
                    [list(asdict(line).values()) for line in database.lines], dtype=np.float64
                ).reshape((-1, 7))
                intervals.append(database.coverage_cm1)
            np.savez(directory / filename, **arrays)
            entries.append(
                dict(
                    file=filename,
                    species=reference.species,
                    molecule_id=table.molecule_id,
                    isotopologue_id=table.isotopologue_id,
                    molecular_mass_kg=reference.molecular_mass_kg,
                    intervals_cm1=intervals,
                    intensity_convention=reference.intensity_convention,
                    provenance=dict(
                        origins.get((reference.species, str(table.isotopologue_id)), {})
                    ),
                )
            )
        measured = []
        for index, reference in enumerate(measurements):
            filename = f"measurements-{index}.npz"
            arrays = dict(
                species=np.array(reference.species), isotopologue=np.array(reference.isotopologue)
            )
            spectra = []
            for row, spectrum in enumerate(reference.spectra):
                arrays[f"wavelength_nm_{row}"] = spectrum.wavelength_nm
                arrays[f"cross_sections_m2_per_molecule_{row}"] = (
                    spectrum.cross_sections_m2_per_molecule
                )
                spectra.append(
                    dict(
                        temperature_k=spectrum.temperature_k,
                        pressure_pa=spectrum.pressure_pa,
                        support_nm=spectrum.support_nm,
                        resolution=spectrum.resolution,
                    )
                )
            np.savez(directory / filename, **arrays)
            measured.append(
                dict(
                    file=filename,
                    species=reference.species,
                    isotopologue=reference.isotopologue,
                    spectra=spectra,
                    provenance=dict(origins.get((reference.species, reference.isotopologue), {})),
                )
            )
        originals = []
        for index, path in enumerate(source_files):
            filename = f"source-{index}{path.suffix}"
            shutil.copyfile(path, directory / filename)
            originals.append(dict(file=filename, original_name=path.name))
        files = [entry["file"] for entry in [*entries, *measured, *originals]]
        manifest = dict(
            schema="orsia-spectroscopy-1",
            source=source,
            edition=edition,
            acquired_at_utc=acquired_at_utc,
            imported_at_utc=datetime.now(UTC).isoformat(),
            partition_source=partition_source,
            units=dict(
                line_centres="cm-1",
                line_strength="cm/molecule",
                mass="kg/molecule",
                temperature="K",
                pressure="Pa",
                wavelength="vacuum_nm",
                cross_section="m2/molecule",
            ),
            lines=entries,
            measurements=measured,
            source_files=originals,
            sha256={name: file_sha256(directory / name) for name in files},
        )
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return path
    except BaseException:
        shutil.rmtree(directory)
        raise


def read_spectroscopy_collection(directory: Path) -> SpectroscopyCollection:
    """Read one local spectroscopy collection, verifying all file hashes and identities.

    Reading is offline and has no writes or provider imports. Hashes establish
    integrity relative to the local manifest, which remains a caller-trusted
    provenance declaration. They do not authenticate the original provider.

    Args:
        directory: Existing collection directory containing manifest.json.

    Returns:
        Read-only isotope-keyed mappings of independent numerical references.

    Raises:
        OSError: Manifest or a declared file cannot be read.
        ValueError: Schema, hash, file identity or scientific values are invalid.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "orsia-spectroscopy-1":
        raise ValueError("unsupported spectroscopy collection schema")
    expected_units = dict(
        line_centres="cm-1",
        line_strength="cm/molecule",
        mass="kg/molecule",
        temperature="K",
        pressure="Pa",
        wavelength="vacuum_nm",
        cross_section="m2/molecule",
    )
    if manifest["units"] != expected_units:
        raise ValueError("unsupported spectroscopy collection units")
    hashes = manifest["sha256"]
    for name, digest in hashes.items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError(f"spectroscopy file hash/path mismatch: {name}")
    lines = []
    measurements = []
    for entry in [*manifest["lines"], *manifest["measurements"], *manifest["source_files"]]:
        if entry["file"] not in hashes:
            raise ValueError("every collection file requires a hash")
    for entry in manifest["lines"]:
        with np.load(directory / entry["file"], allow_pickle=False) as arrays:
            for key in ("species", "molecule_id", "isotopologue_id", "molecular_mass_kg"):
                if arrays[key].item() != entry[key]:
                    raise ValueError(f"line reference identity mismatch: {key}")
            molecule, isotope = entry["molecule_id"], entry["isotopologue_id"]
            databases = tuple(
                SpectralLineTable(
                    tuple(Line(*row) for row in arrays[f"transitions_{index}"]),
                    molecule,
                    isotope,
                    tuple(interval),
                )
                for index, interval in enumerate(entry["intervals_cm1"])
            )
            lines.append(
                LineSpectroscopy(
                    entry["species"],
                    databases,
                    PartitionFunctionTable(
                        molecule, isotope, arrays["temperatures_k"], arrays["partition_sums"]
                    ),
                    entry["molecular_mass_kg"],
                    entry["intensity_convention"],
                )
            )
    for entry in manifest["measurements"]:
        with np.load(directory / entry["file"], allow_pickle=False) as arrays:
            if (
                arrays["species"].item() != entry["species"]
                or arrays["isotopologue"].item() != entry["isotopologue"]
            ):
                raise ValueError("measurement reference identity mismatch")
            spectra = tuple(
                CrossSectionSpectrum(
                    arrays[f"wavelength_nm_{index}"],
                    arrays[f"cross_sections_m2_per_molecule_{index}"],
                    spectrum["temperature_k"],
                    spectrum["pressure_pa"],
                    tuple(tuple(interval) for interval in spectrum["support_nm"]),
                    spectrum["resolution"],
                )
                for index, spectrum in enumerate(entry["spectra"])
            )
            measurements.append(
                CrossSectionMeasurements(entry["species"], entry["isotopologue"], spectra)
            )
    _unique_keys(tuple(lines), tuple(measurements))
    return SpectroscopyCollection(
        {(r.species, r.partition_table.isotopologue_id): r for r in lines},
        {(r.species, r.isotopologue): r for r in measurements},
    )


def read_line_spectroscopy_collection(
    directory: Path,
) -> Mapping[tuple[str, int], LineSpectroscopy]:
    """Read all isotope line references from one verified local collection.

    Args:
        directory: Existing collection directory, also permitted to contain measurements.

    Returns:
        Read-only (species, local-isotope)-keyed numerical references.

    Raises:
        OSError: A declared file cannot be read.
        ValueError: The collection schema, hashes, identities or scientific values are invalid.
    """
    return read_spectroscopy_collection(directory).lines


def _unique_keys(
    lines: tuple[LineSpectroscopy, ...], measurements: tuple[CrossSectionMeasurements, ...]
) -> None:
    line_keys = [(r.species, r.partition_table.isotopologue_id) for r in lines]
    measured_keys = [(r.species, r.isotopologue) for r in measurements]
    if len(set(line_keys)) != len(line_keys) or len(set(measured_keys)) != len(measured_keys):
        raise ValueError("duplicate species/isotope identity in spectroscopy collection")
