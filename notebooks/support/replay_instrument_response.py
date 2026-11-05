"""Replay current instrument cells on the full run's retained spectral scene.

Atmospheric evaluation belongs to the notebook. Replay writes separate arrays
and receipts and leaves its canonical saved displays unchanged.
"""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = {"atmosphere", "absorption-h2o", "absorption-co2", "absorption-ch4", "radiance"}


def instrument_cells(root: Path) -> list[dict]:
    """Return ordered current code cells excluding atmospheric preparation."""
    notebook = json.loads((root / "notebooks/era5_hitran_instrument_responses.ipynb").read_text())
    return [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and cell["id"] not in PREPARATION
    ]


def execute_instrument_cells(
    root: Path, output: Path, radiance: Radiance, *, checks=True, archive=True, provenance=None
) -> dict:
    """Execute literal notebook cells for replay or bounded synthetic regression.

    The supplied radiance replaces atmospheric preparation only. Tests can omit
    slow refinement and archival cells explicitly. Main scientific calls remain
    in the notebook. Cell exceptions propagate with their notebook ID.
    """
    output.mkdir(parents=True, exist_ok=True)
    scenario_path = root / "notebooks/data/gesat-toulouse-2025.json"
    namespace: dict = {}
    for cell in instrument_cells(root):
        name = cell["id"]
        if (name == "refinement" and not checks) or (name == "archive" and not archive):
            continue
        if name == "scene":
            namespace.update(
                radiance=radiance,
                OUT=output,
                scenario_path=scenario_path,
                scenario=json.loads(scenario_path.read_text()),
                provenance={} if provenance is None else provenance,
                spectroscopy_path=root / "outputs/spectroscopy/five-band",
            )
        start = time.perf_counter()
        print(f"Executing {name}", flush=True)
        exec(compile("".join(cell["source"]), f"instrument-notebook:{name}", "exec"), namespace)
        print(f"Completed {name} in {time.perf_counter() - start:.3f} s", flush=True)
    return namespace


def compare_arrays(reference: dict, replay: dict) -> dict:
    """Check finite agreement and exact NaN/positive/negative infinity masks."""
    assert reference.keys() == replay.keys(), "archive schema changed; execute the notebook first"
    comparisons: dict[str, dict[str, bool | float]] = {}
    for key, expected in reference.items():
        actual = replay[key]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
        if expected.dtype.kind in "USbiu":
            np.testing.assert_array_equal(actual, expected)
            comparisons[key] = {"exact": True}
        else:
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
            finite = np.isfinite(expected)
            scale = float(np.max(abs(expected[finite]))) if finite.any() else 0.0
            delta = float(np.max(abs(actual[finite] - expected[finite]))) if finite.any() else 0.0
            assert delta <= 3e-11 * scale, (key, delta, scale)
            comparisons[key] = {
                "max_abs": delta,
                "max_abs_over_peak": delta / scale if scale else 0.0,
            }
    return comparisons


def main():
    """Replay seeded selected acquisitions and compare with the current full run."""
    start = time.perf_counter()
    baseline = ROOT / "outputs/instrument-responses"
    output = ROOT / "outputs/instrument-response-replay"
    evidence = json.loads((baseline / "evidence.json").read_text())
    hashes = {
        cell["id"]: hashlib.sha256("".join(cell["source"]).encode()).hexdigest()
        for cell in instrument_cells(ROOT)
    }
    assert all(evidence["code_cell_hashes"][name] == digest for name, digest in hashes.items()), (
        "instrument sources changed; execute the notebook before replay"
    )
    with np.load(baseline / "spectral-scene.npz") as source:
        radiance = Radiance(
            wavenumber_grid(source["wavenumber_cm1"]), source["radiance"], ObservationGeometry(0.0)
        )
    namespace = execute_instrument_cells(
        ROOT, output, radiance, provenance=evidence["input_hashes"]
    )
    with np.load(baseline / "chains.npz") as original, np.load(output / "chains.npz") as replay:
        comparisons = compare_arrays(dict(original), dict(replay))
    receipt = dict(
        scope="instrument-only replay, same saved radiance, current literal cells, seeded noisy DN",
        executed_code_cell_hashes=hashes,
        comparisons=comparisons,
        arrays=len(comparisons),
        max_relative_delta=max(item.get("max_abs_over_peak", 0) for item in comparisons.values()),
        baseline_sha256=hashlib.sha256((baseline / "chains.npz").read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256((output / "chains.npz").read_bytes()).hexdigest(),
        runtime=namespace["runtime_settings"](),
        wall_s=time.perf_counter() - start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        // (1024 if sys.platform == "darwin" else 1),
    )
    (output / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in receipt.items() if key != "comparisons"}, indent=2)
    )


if __name__ == "__main__":
    main()
