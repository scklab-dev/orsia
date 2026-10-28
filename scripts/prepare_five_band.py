"""Acquire authentic five-band inputs explicitly; replay offline thereafter."""

import csv
import importlib
import json
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr
from dotenv import dotenv_values
from orsia_data import download_era5, fetch_hitran_lines, fetch_solar_spectrum

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/five-band-inputs"


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    credentials = dotenv_values(ROOT / ".env")
    request: dict[str, object] = dict(
        product_type=["reanalysis"],
        variable=["temperature", "specific_humidity"],
        year=["2024"],
        month=["01"],
        day=["15"],
        time=["12:00"],
        pressure_level=[
            "1",
            "5",
            "10",
            "20",
            "50",
            "100",
            "200",
            "300",
            "500",
            "700",
            "850",
            "1000",
        ],
        area=[1, -1, -1, 1],
        data_format="netcdf",
        download_format="unarchived",
    )
    raw = BASE / "era5-native.nc"
    if not raw.exists():
        url = credentials.get("CDSAPI_URL")
        key = credentials.get("CDSAPI_KEY")
        if not url or not key:
            raise ValueError("Set CDSAPI_URL and CDSAPI_KEY in the repository-root .env")
        client = cdsapi.Client(
            url=url,
            key=key,
            quiet=True,
            debug=False,
            timeout=120,
            retry_max=2,
        )
        print("Acquiring ERA5 (2024-01-15 12:00 UTC, equatorial ocean)", flush=True)
        download_era5(
            product="reanalysis-era5-pressure-levels",
            request=request,
            retrieve=client.retrieve,
            destination=raw,
        )
    prepared = BASE / "era5-pa.nc"
    rebuild = not prepared.exists()
    if not rebuild:
        with xr.open_dataset(prepared, decode_times=False, engine="scipy") as existing:
            rebuild = existing.time.attrs.get("calendar") != "standard"
    if rebuild:
        with xr.open_dataset(raw, engine="h5netcdf") as native:
            data = native[["t", "q"]].load()
        # Explicit provider adaptation, not a change to the Orsia reader.
        data = data.rename({"valid_time": "time", "pressure_level": "pressure"})
        data = data.assign_coords(pressure=data.pressure.values * 100.0)
        data.pressure.attrs["units"] = "Pa"
        data.q.attrs["units"] = "kg kg-1"
        # Dates in this modern demonstration are identical in the Gregorian
        # and proleptic Gregorian calendars; serialize the reader's convention.
        data.time.encoding = {"calendar": "standard", "units": "seconds since 1970-01-01"}
        temporary = prepared.with_suffix(".tmp.nc")
        data.to_netcdf(temporary, engine="scipy")
        temporary.replace(prepared)
    solar_directory = BASE / "solar"
    if not solar_directory.exists():
        print("Acquiring LISIRD HSRS", flush=True)
        fetch_solar_spectrum(
            product="tsis1_hsrs_p025nm", edition="current", directory=solar_directory
        )
    species = [
        ("H2O", 1, 18.010565, None),
        ("CO2", 2, 43.98983, 420e-6),
        ("CH4", 6, 16.0313, 1.9e-6),
        ("O2", 7, 31.98983, 0.2095),
    ]
    # Five disjoint windows acquired separately: avoid an enormous unused line interval.
    windows = [(320, 325), (758, 765), (1645, 1655), (4250, 4300), (9950, 10050)]
    interfaces = [100000, 85000, 70000, 50000, 30000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    entries = []
    for name, molecule, mass, fraction in species:
        merged = BASE / f"{name}-margin26.par"
        if not merged.exists():
            paths = []
            selected = {"H2O": range(5), "CO2": (2, 3, 4), "CH4": (2, 4), "O2": (1,)}[name]
            for index in selected:
                low, high = windows[index]
                directory = BASE / f"{name}-{index}-margin26"
                if not directory.exists():
                    print(f"Acquiring HITRAN {name}, window {index}", flush=True)
                    fetch_hitran_lines(
                        molecule_id=molecule,
                        isotopologue_id=1,
                        interval_cm1=(1e7 / high - 26, 1e7 / low + 26),
                        edition="current",
                        directory=directory,
                    )
                paths.append(directory / "lines.data")
            with merged.open("x") as output:
                for path in paths:
                    output.write(path.read_text())
        partition = BASE / f"{name}-Q.csv"
        if not partition.exists():
            hapi = importlib.import_module("hapi")

            with partition.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["molecule_id", "isotopologue_id", "temperature_k", "partition_sum"]
                )
                for temperature in np.arange(70.0, 401.0):
                    writer.writerow(
                        [molecule, 1, temperature, hapi.partitionSum(molecule, 1, temperature)]
                    )
        entry: dict[str, object] = dict(
            species=name,
            molecule_id=molecule,
            isotopologue_id=1,
            mass_u=mass,
            lines=merged.name,
            partition=partition.name,
            pressure_range_pa=[100, 105000],
            edition="HITRAN current; HAPI bundled TIPS",
            source="HAPI fetch / partitionSum",
            included_cases={
                "H2O": ["UV", "VNIR", "SWIR", "MWIR", "LWIR"],
                "CO2": ["SWIR", "MWIR", "LWIR"],
                "CH4": ["SWIR", "LWIR"],
                "O2": ["VNIR"],
            }[name],
        )
        if fraction is not None:
            entry["dry_air_mole_fraction"] = [fraction] * (len(interfaces) - 1)
        entries.append(entry)
    manifest = dict(
        era5=prepared.name,
        coordinates=["time", "pressure", "latitude", "longitude"],
        backend="classic",
        valid_time="2024-01-15T12:00:00+00:00",
        latitude_deg=0,
        longitude_deg=0,
        pressure_interfaces_pa=interfaces,
        solar_csv="solar/tsis1_hsrs_p025nm.csv",
        spectroscopy=entries,
        source="CDS ERA5 pressure levels; LISIRD TSIS1 HSRS current",
        edition="current",
        era5_request=request,
        adaptation="hPa to Pa; q units normalized; modern UTC dates encoded Gregorian",
        abundance_note="CO2/CH4 constant illustrative assumptions, not ERA5 measurements",
    )
    destination = BASE / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Offline manifest: {destination}")


if __name__ == "__main__":
    main()
