"""HITRAN acquisition through HAPI; execution policy belongs to the caller."""

import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralLineTable
from orsia_radiative.spectroscopy.line_tables import PartitionFunctionTable
from orsia_radiative.spectroscopy.references import LineSpectroscopy
from scipy.constants import atomic_mass

from orsia_data.spectroscopy.collections import import_spectroscopy_collection
from orsia_data.spectroscopy.lines import read_hitran_lines


def acquire_hitran_references(
    *,
    selections: Mapping[tuple[str, int, int], tuple[tuple[float, float], ...]],
    partition_temperatures_k: NDArray[np.float64],
    tips_version: int,
    edition: Literal["current"],
    directory: Path,
) -> Path:
    """Acquire all requested isotope/interval references in one explicit operation.

    HAPI is imported lazily. Each key is (species, molecule ID, local isotope ID);
    intervals must be ordered and disjoint. Partition samples use an explicit TIPS
    version and scalar partitionSum calls. Store native source hashes, HAPI version,
    isotope masses and the current service selection. No frozen HITRAN edition is
    inferred. Failure removes temporary inputs and leaves no destination. Existing
    destinations are preserved. Upstream exceptions identify the failing operation
    through an added exception note. Caller owns HAPI serialization and cancellation:
    HAPI mutates global cache state and is unsafe for concurrent acquisition.

    Args:
        selections: Nonempty species/isotope selections and exact line-centre intervals, cm-1.
        partition_temperatures_k: Positive increasing Q(T) knots in K, bracketing 296 K.
        tips_version: Explicit HAPI TIPS version, such as 2025 for HAPI 1.3.
        edition: Explicit 'current' service selection.
        directory: New local collection directory; existing directories are preserved.

    Returns:
        Manifest path for one offline reusable collection.

    Raises:
        FileExistsError: The destination exists.
        ImportError: Optional HAPI is unavailable.
        ValueError: Identity, intervals, partition support or native outputs are invalid.
        OSError: Local files cannot be created or read.
    """
    if directory.exists():
        raise FileExistsError(directory)
    if edition != "current" or not selections:
        raise ValueError("nonempty selections and explicit current service are required")
    keys = [(species, isotope) for species, _, isotope in selections]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate species/isotope identity in acquisition selections")
    knots = partition_temperatures_k
    for (species, molecule, isotope), intervals in selections.items():
        table = PartitionFunctionTable(molecule, isotope, knots, np.ones_like(knots))
        LineSpectroscopy(
            species,
            tuple(SpectralLineTable((), molecule, isotope, interval) for interval in intervals),
            table,
            atomic_mass,
        )
    if knots[0] > 296 or knots[-1] < 296:
        raise ValueError("partition knots must bracket the 296 K reference temperature")
    import hapi  # type: ignore[import-not-found]

    for species, molecule, isotope in selections:
        if hapi.moleculeName(molecule) != species:
            raise ValueError("requested species conflicts with HAPI molecule identity")
        # Resolve isotope validity before any transfer.
        hapi.molecularMass(molecule, isotope)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".hitran-") as temporary:
        references = []
        files = []
        operation = "initialization"
        try:
            for index, ((species, molecule, isotope), intervals) in enumerate(selections.items()):
                databases = []
                for segment, interval in enumerate(intervals):
                    operation = (
                        f"HITRAN {species} molecule={molecule} "
                        f"isotope={isotope} interval={interval}"
                    )
                    native = Path(temporary) / f"{index}-{segment}"
                    databases.append(
                        fetch_hitran_lines(
                            molecule_id=molecule,
                            isotopologue_id=isotope,
                            interval_cm1=interval,
                            edition=edition,
                            directory=native,
                        )
                    )
                    files.extend((native / "lines.data", native / "lines.header"))
                operation = f"TIPS-{tips_version} {species} molecule={molecule} isotope={isotope}"
                sums = np.array(
                    [
                        hapi.partitionSum(molecule, isotope, float(t), version=tips_version)
                        for t in knots
                    ],
                    dtype=np.float64,
                )
                references.append(
                    LineSpectroscopy(
                        species,
                        tuple(databases),
                        PartitionFunctionTable(molecule, isotope, knots, sums),
                        float(hapi.molecularMass(molecule, isotope)) * atomic_mass,
                    )
                )
            operation = "collection serialization"
            return import_spectroscopy_collection(
                directory,
                lines=tuple(references),
                source=f"HITRAN HAPI {hapi.HAPI_VERSION}",
                edition=edition,
                acquired_at_utc=datetime.now(UTC).isoformat(),
                partition_source=f"HAPI {hapi.HAPI_VERSION}; TIPS-{tips_version}; sampled Q(T)",
                source_files=tuple(files),
            )
        except Exception as error:
            error.add_note(f"Grouped spectroscopy acquisition failed during {operation}")
            raise


def fetch_hitran_lines(
    *,
    molecule_id: int,
    isotopologue_id: int,
    interval_cm1: tuple[float, float],
    edition: Literal["current"],
    directory: Path,
) -> SpectralLineTable:
    """Use HAPI fetch(M, I, min, max), retaining native lines.data/lines.header.

    One HITRAN molecule/local isotopologue per call; combine multiple returned
    databases explicitly. Interval is inclusive line-centre coverage [cm-1],
    not later wing coverage. Intensities retain HITRAN abundance weighting.
    Only explicit ``edition="current"`` is supported: HAPI fetch has no edition
    selector, so this does not claim to pin HITRAN2020/2024 or an archival release.

    Requires the optional ``orsia-data[hitran]`` extra. Imports never load HAPI.
    HAPI is imported on acquisition and runs synchronously in the caller's
    process/thread. Its db_begin/fetch operations mutate shared module-wide
    database/cache state, which is not restored on success or failure. Separate
    directories do not make concurrent HAPI calls safe, including calls outside
    Orsia. The caller owns serialization, isolation and cancellation policy;
    Orsia supplies no lock, executor or total deadline. HAPI transfer has no byte
    bound here. Missing HAPI raises ImportError; upstream exceptions propagate.
    A successful empty table is valid. Disabled GIL is not thread-safety evidence.

    Directory must not exist and is exclusively owned for this call. Failures
    remove it, including partial HAPI files. Do not modify it concurrently. The
    local reader remains independently usable without HAPI or network access.
    No partition evaluation, line materialisation, registry or manifest is added.

    Args:
        molecule_id: Positive HITRAN molecule identifier passed to HAPI.
        isotopologue_id: Local isotopologue identifier passed to HAPI.
        interval_cm1: Inclusive requested line-centre range in cm^-1.
        edition: Explicit ``"current"`` service selection.
        directory: New directory for the native HAPI header and data files.

    Returns:
        Sorted line database for the selected identity and declared coverage.

    Raises:
        FileExistsError: If the destination directory already exists.
        ImportError: If optional HAPI support is unavailable after directory creation.
        ValueError: If the selection, native output or parsed records are invalid.
    """
    if edition != "current":
        raise ValueError("HAPI fetch supports only the explicit current service")
    # Validate physical support before acquiring resources or contacting HAPI.
    SpectralLineTable((), molecule_id, isotopologue_id, interval_cm1)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        import hapi  # type: ignore[import-not-found]

        hapi.db_begin(str(directory.resolve()))
        hapi.fetch("lines", molecule_id, isotopologue_id, *interval_cm1)
        # A data file alone is not evidence of a successful HAPI fetch.
        if not (directory / "lines.header").is_file():
            raise ValueError("HAPI acquisition did not produce its native header")
        path = directory / "lines.data"
        database = read_hitran_lines(
            path,
            molecule_id=molecule_id,
            isotopologue_id=isotopologue_id,
            interval_cm1=interval_cm1,
        )
        with path.open(encoding="ascii") as stream:
            row_count = sum(bool(row.strip()) for row in stream)
        if row_count != len(database.lines):
            raise ValueError("HAPI returned lines outside the requested identity or interval")
        return database
    except BaseException:
        shutil.rmtree(directory)
        raise
