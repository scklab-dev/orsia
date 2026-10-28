"""Local input verification, timing and display for the five-band notebook.

Scientific preparation and complete Orsia solver calls remain in the notebook.
The persistent reference cache owns its identity and invalidation contract.
"""

import hashlib
import json
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from orsia_data.acquisition import file_sha256


def verify_inputs(path):
    """Verify every local manifest checksum and retain the original provenance."""
    manifest = json.loads(path.read_text())
    assert all(k in manifest for k in ("mipas", "cross_sections", "sha256"))
    base = path.parent

    def local(name):
        return (base / name).resolve()

    input_paths = [path, local(manifest["era5"]), local(manifest["solar_csv"])]
    input_paths += [
        local(s[key]) for s in manifest["spectroscopy"] for key in ("lines", "partition")
    ]
    input_paths += [local(manifest["mipas"]["file"])] + [
        local(f) for e in manifest["cross_sections"] for f in e["files"]
    ]
    for filename, digest in manifest["sha256"].items():
        assert file_sha256(local(filename)) == digest, f"Input checksum mismatch: {filename}"
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    print(f"Local inputs verified; {len(manifest['sha256'])} checksums matched.")
    return manifest, base, input_paths, provenance


def time_orsia(calculate, *, runs):
    """Time one complete first call and the declared warm repetitions."""
    start = perf_counter()
    value = calculate()
    first_seconds = perf_counter() - start
    seconds = []
    for _ in range(runs):
        start = perf_counter()
        value = calculate()
        seconds.append(perf_counter() - start)
    return value, dict(
        first_seconds=first_seconds, seconds=seconds, median_seconds=float(np.median(seconds))
    )


def show_atmosphere(temperature, humidity, atmosphere):
    """Display the retained layer-temperature and humidity profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(temperature, atmosphere.height_boundaries_m[:-1] / 1000)
    axes[0].set(xlabel="Temperature [K]", ylabel="Layer bottom altitude [km]")
    axes[1].plot(humidity, atmosphere.height_boundaries_m[:-1] / 1000)
    axes[1].set(xlabel="Specific humidity [kg/kg]")
    plt.show()


def show_comparison(name, grid, a, b):
    """Display radiance, signed residuals and pointwise relative residuals."""
    relative = np.divide(a - b, b, out=np.full_like(b, np.nan), where=b != 0)
    x = 1e7 / grid.points
    order = np.argsort(x)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for values, label in ((a, "Orsia"), (b, "SASKTRAN2 (32 cells, 32 streams)")):
        axes[0].plot(x[order], values[order], label=label)
    axes[0].set(title=name, ylabel="Radiance per cm-1")
    axes[0].legend()
    axes[1].plot(x[order], (a - b)[order], label="Orsia minus SASKTRAN2")
    axes[1].set(ylabel="Signed difference per cm-1")
    axes[1].legend(fontsize=8)
    axes[2].plot(x[order], 100 * relative[order])
    axes[2].set(xlabel="Wavelength [nm]", ylabel="Relative difference [%]")
    plt.tight_layout()
    plt.show()


def show_agreement(results):
    """Display the original maximum absolute and relative discrepancy table."""
    rows = [
        "| Band | Max absolute difference | Max relative difference [%] |",
        "| --- | ---: | ---: |",
    ]
    for name, record in results.items():
        a = np.asarray(record["orsia"])
        b = np.asarray(record["reference"][-1]["radiance"])
        relative = np.divide(a - b, b, out=np.full_like(b, np.nan), where=b != 0)
        rows.append(
            f"| {name} | {np.max(np.abs(a - b)):.6g} | {100 * np.nanmax(np.abs(relative)):.6g} |"
        )
    display(
        Markdown(
            "### Agreement summary\nAbsolute differences: W m⁻² sr⁻¹ (cm⁻¹)⁻¹. "
            "Relative differences: 100 × |Orsia − SASKTRAN2| / |SASKTRAN2|.\n\n" + "\n".join(rows)
        )
    )


def show_timing(results, runs):
    """Display current access times and provenance-labelled solver timings."""
    rows = [
        "| Band | Orsia current median [s] | SASKTRAN2 median [s] | "
        "Timing source | Current access [s] |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    dates = []
    for name, record in results.items():
        ot = record["orsia_timing"]
        sk = record["reference"][-1]
        st = sk["timing"]
        rows.append(
            f"| {name} | {ot['median_seconds']:.6g} | {st['median_seconds']:.6g} | "
            f"{sk['cache']['timing_provenance']} | {sk['cache']['access_wall_seconds']:.4g} |"
        )
        dates.append(f"{name}: {sk['computed_at_utc']}")
    display(
        Markdown(
            "\n".join(rows)
            + "\n\nSASKTRAN2 original computation dates (UTC): "
            + "; ".join(dates)
            + "."
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    fig.suptitle(f"Calculation time comparison: median of {runs} warm calls; min-max ranges")
    for ax, (name, record) in zip(axes.flat, results.items(), strict=False):
        sk = record["reference"][-1]
        timings = [record["orsia_timing"], sk["timing"]]
        sk_label = "SASKTRAN2\n" + sk["cache"]["timing_provenance"]
        orsia_label = (
            "Orsia\ncomplete solar call"
            if record["orsia_timing"]["scope"].startswith("solar.reflected_radiance;")
            else "Orsia\nSchwarzschild"
        )
        medians = np.array([t["median_seconds"] for t in timings])
        errors = np.array(
            [
                [t["median_seconds"] - min(t["seconds"]) for t in timings],
                [max(t["seconds"]) - t["median_seconds"] for t in timings],
            ]
        )
        bars = ax.bar([0, 1], medians, width=0.6, yerr=errors, capsize=3, color=["C0", "C1"])
        for bar, timing in zip(bars, timings, strict=True):
            ax.annotate(
                f"{timing['median_seconds']:.4g} s",
                (bar.get_x() + bar.get_width() / 2, max(timing["seconds"])),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=[orsia_label, sk_label],
            ylabel="Wall time [s]",
            ylim=(0, 1.3 * max(max(t["seconds"]) for t in timings)),
            title=f"{name}\nSASKTRAN2 / Orsia = {medians[1] / medians[0]:.3g}",
        )
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.95,
        "Zero-baseline linear axes.\nIndependent scale per band: compare\n"
        "heights within panels, seconds across panels.\nRatios use median seconds; <1 means\n"
        "SASKTRAN2 took less time.\n\nSolar: complete reflected_radiance\n"
        "with call-local geometry and scratch.\nSASKTRAN2: calculate_radiance.\n"
        "Thermal boundary adapter runs separately.\n\nDifferent solver settings;\n"
        "accuracy equivalence remains unmeasured.\nSASKTRAN2: 32 interior altitude nodes\n"
        "per physical layer / 32 streams.",
        va="top",
        transform=axes.flat[-1].transAxes,
    )
    plt.show()
