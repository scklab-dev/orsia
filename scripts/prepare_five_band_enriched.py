"""Explicit public acquisition for the enriched five-window demonstration.

Authentic bytes and receipts live under outputs/. The historical baseline
manifest and its inputs retain their existing identities.
"""

import argparse
import csv
import hashlib
import importlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from orsia_data import fetch_hitran_lines
from orsia_data.acquisition import download_file, file_sha256
from orsia_data.services.mipas import download_mipas_profile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs/five-band-enriched-inputs"
OZONE = (
    "https://zenodo.org/api/records/5793207/files/SerdyuchenkoGorshelev5digits_latest.dat/content"
)
BREMEN = "https://www.iup.uni-bremen.de/gruppen/molspec/net/scia"
TEMPERATURES = [203, 223, 243, 273, 293]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    destination = DESTINATION
    baseline = ROOT / "outputs/five-band-inputs/manifest.json"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        receipt = json.loads(manifest_path.read_text())
        for name, digest in receipt["sha256"].items():
            if file_sha256(destination / name) != digest:
                raise ValueError(f"input checksum mismatch: {name}")
        print(f"Verified retained enriched inputs: {manifest_path}")
        return
    if not args.download:
        raise FileNotFoundError("Run prepare-five-band-enriched --download for public acquisition")
    destination.mkdir(parents=True, exist_ok=True)
    # Reuse the explicitly selected ERA5, solar and baseline spectroscopy bytes.
    manifest = json.loads(baseline.read_text())
    for key in ("era5", "solar_csv"):
        manifest[key] = "../five-band-inputs/" + manifest[key]
    for entry in manifest["spectroscopy"]:
        for key in ("lines", "partition"):
            entry[key] = "../five-band-inputs/" + entry[key]
    native = destination / "native"
    native.mkdir(exist_ok=True)
    ozone = native / "O3.dat"
    _acquire(OZONE, ozone)
    if hashlib.md5(ozone.read_bytes()).hexdigest() != "c2b00d277776d5388a66cdc463cd6b2e":
        raise ValueError("O3 bytes differ from Zenodo 5793207 provider MD5")
    xs = [
        dict(
            species="O3",
            format="serdyuchenko",
            files=["native/O3.dat"],
            temperatures_k=list(range(193, 294, 10)),
            source=OZONE,
            resolution="1 cm-1 HWHM in 290-350 nm; native 0.01 nm samples",
        )
    ]
    for species in ("SO2", "NO2"):
        files = []
        for temperature in TEMPERATURES:
            filename = f"{species}_{temperature}K.dat"
            _acquire(f"{BREMEN}/{filename}", native / filename)
            files.append(f"native/{filename}")
        xs.append(
            dict(
                species=species,
                format="bogumil",
                files=files,
                temperatures_k=TEMPERATURES,
                source=BREMEN,
                resolution="0.21 nm FWHM near 320-325 nm",
            )
        )
    for entry in xs:
        entry.update(
            included_cases=["UV"],
            spectral_coordinate="vacuum_wavelength_nm",
            temperature_policy="clip",
            pressure_policy="fixed laboratory spectrum",
        )
    manifest["cross_sections"] = xs
    profile = native / "equatorial_day.atm"
    digest = "4f366cfc5de2bfde21a82ab27c5c1a3ad9cd2e25ee3296bb5d6109980196a9f6"
    if not profile.exists():
        retained = ROOT / "outputs/mipas-2001-validation/equatorial_day.atm"
        if retained.is_file() and file_sha256(retained) == digest:
            shutil.copyfile(retained, profile)
        else:
            download_mipas_profile(
                edition="2001",
                profile="equatorial_day",
                destination=profile,
                expected_sha256=digest,
            )
    if file_sha256(profile) != digest:
        raise ValueError("MIPAS profile checksum mismatch")
    manifest["mipas"] = dict(
        file="native/equatorial_day.atm",
        profile="equatorial_day",
        edition="2001",
        species=["O3", "SO2", "NO2", "NH3"],
        source="https://eodg.atm.ox.ac.uk/RFM/atm/equ.atm",
        basis="total_air_mole",
    )
    # Exact mass and molecule identifiers follow the HAPI isotopologue table.
    hapi = importlib.import_module("hapi")

    for species, molecule in (("O3", 3), ("NH3", 11)):
        directory = destination / f"{species}-LWIR-margin26"
        if not directory.exists():
            fetch_hitran_lines(
                molecule_id=molecule,
                isotopologue_id=1,
                interval_cm1=(1e7 / 10050 - 26, 1e7 / 9950 + 26),
                edition="current",
                directory=directory,
            )
        partition = destination / f"{species}-Q.csv"
        if not partition.exists():
            with partition.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["molecule_id", "isotopologue_id", "temperature_k", "partition_sum"]
                )
                for temperature in range(70, 401):
                    writer.writerow(
                        [molecule, 1, temperature, hapi.partitionSum(molecule, 1, temperature)]
                    )
        manifest["spectroscopy"].append(
            dict(
                species=species,
                molecule_id=molecule,
                isotopologue_id=1,
                mass_u=hapi.molecularMass(molecule, 1),
                lines=f"{directory.name}/lines.data",
                partition=partition.name,
                pressure_range_pa=[100, 105000],
                edition="HITRAN public HAPI endpoint; bundled TIPS",
                source="HAPI fetch / partitionSum",
                included_cases=["LWIR"],
                abundance_source="mipas",
            )
        )
    files = [manifest["era5"], manifest["solar_csv"], "native/equatorial_day.atm"]
    files += [e[k] for e in manifest["spectroscopy"] for k in ("lines", "partition")]
    files += [f for e in xs for f in e["files"]]
    manifest["sha256"] = {f: file_sha256(destination / f) for f in files}
    manifest["acquired_at_utc"] = datetime.now(UTC).isoformat()
    manifest["baseline_manifest_sha256"] = file_sha256(baseline)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Enriched manifest: {manifest_path}")


def _acquire(url: str, path: Path):
    if not path.exists():
        print(f"Acquiring {path.name}", flush=True)
        download_file(url, path, timeout_s=120)


if __name__ == "__main__":
    main()
