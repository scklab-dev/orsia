"""Replay instrument response against locally executed scene/operator arrays.

Use the user's own instrument execution as the reference; atmospheric data and
provider evaluation belong to the full notebook execution. Write independent
replay NPZ files and an explicit array comparison receipt under ignored outputs/.
"""

import hashlib
import json
import os
import resource
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ground-response-integration"
REFERENCE = ROOT / "outputs/instrument-responses"


def main():
    """Execute current instrument cells against locally generated spectral inputs."""
    start = time.perf_counter()
    required = [
        REFERENCE / name
        for name in ("operators.npz", "scene-images.npz", "channels.json", "evidence.json")
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run execute-instrument-responses first; missing local reference: {', '.join(missing)}"
        )
    old_operators = np.load(REFERENCE / "operators.npz")
    old_images = np.load(REFERENCE / "scene-images.npz")
    radiance = Radiance(
        wavenumber_grid(old_operators["wavenumber_cm1"]),
        old_images["source_spectra"][0],
        ObservationGeometry(0.0),
    )
    after = OUT / "after"
    after.mkdir(parents=True, exist_ok=True)
    notebook_path = ROOT / "notebooks/era5_hitran_instrument_responses.ipynb"
    current_notebook = json.loads(notebook_path.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in current_notebook["cells"]}
    namespace: dict = {}
    exec(compile(cells["imports"], f"{notebook_path}:imports", "exec"), namespace)
    exec(compile(cells["scan-example"], f"{notebook_path}:scan-example", "exec"), namespace)
    exec(compile(cells["scene-example"], f"{notebook_path}:scene-example", "exec"), namespace)
    settings = namespace["comparison_settings"](
        json.loads((ROOT / "notebooks/data/gesat-toulouse-2025.json").read_text())
    )
    runtime = dict(
        python=sys.version,
        executable=sys.executable,
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        gil_enabled=sys._is_gil_enabled(),
        affinity=sorted(os.sched_getaffinity(0)),
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        loaded_thread_pools=threadpool_info(),
        thread_pools={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )
    assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 14)
    assert runtime["free_threaded"] and not runtime["gil_enabled"]
    assert runtime["nice"] >= 10
    assert all(pool["num_threads"] == 1 for pool in runtime["loaded_thread_pools"])
    assert set(runtime["affinity"]) <= set(range(8)) and set(runtime["thread_pools"].values()) == {
        "1"
    }
    # Trace the notebook's native raster timing boundary.
    scan_calls = {"along_track_raster_reads": 0}
    native_reads = namespace["native_reads"]
    native_reads.__globals__["along_track_raster_reads"] = _count_calls(
        native_reads.__globals__["along_track_raster_reads"],
        "along_track_raster_reads",
        scan_calls,
    )
    namespace.update(
        radiance=radiance,
        points=radiance.grid.points,
        settings=settings,
        OUT=after,
    )
    # Trace the public response boundaries where the notebook calls them.
    api_calls = dict(gaussian_along_track_response=0, response_electrons=0)
    for name in (
        "gaussian_along_track_response",
        "response_electrons",
    ):
        namespace[name] = _count_calls(namespace[name], name, api_calls)
    response_stages = ("physical-response", "operators-images")
    print("executing notebook physical response and archive on retained scene", flush=True)
    for name in response_stages:
        exec(compile(cells[name], f"{notebook_path}:{name}", "exec"), namespace)
    layouts = namespace["layouts"]
    templates = sum(layout.read_count for layout in layouts.values())
    assert api_calls == dict(
        gaussian_along_track_response=len(layouts),
        response_electrons=len(layouts),
    )
    assert templates > 0
    assert scan_calls == {"along_track_raster_reads": len(layouts)}
    operators = dict(np.load(after / "operators.npz"))
    images = dict(np.load(after / "scene-images.npz"))
    channel_receipts = json.loads((after / "channels.json").read_text())
    diagnostics = {
        family: dict(reconstruction=value.diagnostics, detector=value.detector)
        for family, value in namespace["results"].items()
    }
    old_channels = json.loads((REFERENCE / "channels.json").read_text())
    old_channel_arrays = {
        f"{family}/{name}": np.array(value)
        for family, fields in old_channels.items()
        for name, value in fields.items()
    }
    channel_arrays = {
        f"{family}/{name}": np.array(value)
        for family, fields in channel_receipts.items()
        for name, value in fields.items()
    }
    comparisons: dict = {}
    for name, before, current in (
        ("operators", dict(old_operators), operators),
        ("scene-images", dict(old_images), images),
        ("channels", old_channel_arrays, channel_arrays),
    ):
        assert set(before) == set(current)
        for key, value in current.items():
            previous = before[key]
            assert value.shape == previous.shape and value.dtype == previous.dtype
            if value.dtype.kind in "USiu":
                np.testing.assert_array_equal(value, previous)
                comparisons[name + "/" + key] = dict(exact=True)
            else:
                # Signed inverse weights can have near-zero entries. Use global
                # scale for the absolute tolerance while recording the raw delta.
                delta = float(np.max(np.abs(value - previous)))
                scale = float(np.max(np.abs(previous)))
                tolerance = 3e-11 * scale
                assert delta <= tolerance, (name, key, delta, tolerance)
                comparisons[name + "/" + key] = dict(
                    max_abs=delta,
                    max_abs_over_peak=delta / scale if scale else 0,
                    tolerance_abs=tolerance,
                    rtol_global=3e-11,
                )
    sources = [
        "scripts/replay_instrument_response.py",
        "notebooks/support/notebook_ground_pixel.py",
        "notebooks/support/notebook_instrument_responses.py",
        "notebooks/support/notebook_instrument_support.py",
        "libs/orsia-instruments/src/orsia_instruments/acquisition.py",
        "libs/orsia-instruments/src/orsia_instruments/detector.py",
        "libs/orsia-instruments/src/orsia_instruments/optics/pixels.py",
        "libs/orsia-uncertainty/src/orsia_uncertainty/covariance.py",
        "libs/orsia-instruments/src/orsia_instruments/integration.py",
        "libs/orsia-instruments/src/orsia_instruments/imaging/response.py",
        "notebooks/era5_hitran_instrument_responses.ipynb",
    ]
    saved_output_sha256 = {
        cell["id"]: hashlib.sha256(json.dumps(cell["outputs"], sort_keys=True).encode()).hexdigest()
        for cell in current_notebook["cells"]
        if cell["cell_type"] == "code" and cell["outputs"]
    }
    receipt = dict(
        scope=(
            "offline instrument response replay of current notebook cells "
            "from retained canonical spectral scene"
        ),
        executed_notebook_cells={
            name: hashlib.sha256(cells[name].encode()).hexdigest()
            for name in ("imports", "scan-example", "scene-example", *response_stages)
        },
        response_api_calls=api_calls,
        native_read_api_calls=scan_calls,
        comparisons=comparisons,
        comparison_summary={
            "arrays": len(comparisons),
            "maximum_global_relative_delta": max(
                item.get("max_abs_over_peak", 0) for item in comparisons.values()
            ),
            "maximum_native_electron_absolute_delta": max(
                item["max_abs"]
                for name, item in comparisons.items()
                if name.endswith("_native_electrons")
            ),
            "maximum_physical_kernel_global_relative_delta": max(
                item["max_abs_over_peak"]
                for name, item in comparisons.items()
                if name.endswith("_K")
            ),
            "native_dn_exact": all(
                item["exact"] for name, item in comparisons.items() if name.endswith("_native_dn")
            ),
            "saved_notebook_output_cells": len(saved_output_sha256),
        },
        diagnostics=diagnostics,
        runtime=runtime,
        saved_notebook_output_sha256=saved_output_sha256,
        wall_s=time.perf_counter() - start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources
        },
        reference_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required
        },
    )
    evidence_path = REFERENCE / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    scenario = ROOT / "notebooks/data/gesat-toulouse-2025.json"
    assert evidence["scenario_sha256"] == hashlib.sha256(scenario.read_bytes()).hexdigest()
    comparison_path = OUT / "comparison.json"
    # Preserve a previous local comparison when its inputs or sources change.
    if comparison_path.exists():
        old_bytes = comparison_path.read_bytes()
        digest = hashlib.sha256(old_bytes).hexdigest()
        predecessor = OUT / f"comparison.{digest[:12]}.json"
        if predecessor.exists():
            assert predecessor.read_bytes() == old_bytes, "previous comparison changed"
        else:
            predecessor.write_bytes(old_bytes)
        receipt["previous_replay"] = {
            "receipt": str(predecessor.relative_to(ROOT)),
            "receipt_sha256": digest,
        }
    comparison_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "scope",
                    "executed_notebook_cells",
                    "response_api_calls",
                    "native_read_api_calls",
                    "comparison_summary",
                    "wall_s",
                    "max_rss_kib",
                    "runtime",
                )
            },
            indent=2,
        )
    )


def _count_calls(function, name, counts):
    def counted(*args, **kwargs):
        counts[name] += 1
        return function(*args, **kwargs)

    return counted


def _retained_notebook_outputs(previous_notebook, current_notebook):
    """Check archived metadata and outputs for shared IDs, regardless of cell order."""
    previous_cells = previous_notebook["cells"]
    current_cells = current_notebook["cells"]
    previous_by_id = {cell["id"]: cell for cell in previous_cells}
    current_by_id = {cell["id"]: cell for cell in current_cells}
    assert len(previous_by_id) == len(previous_cells)
    assert len(current_by_id) == len(current_cells)
    assert {"imports", "inputs", "operators-images"} <= previous_by_id.keys() & current_by_id.keys()
    retained_outputs = {}
    checked_nonempty_outputs = 0
    for previous_cell in previous_cells:
        name = previous_cell["id"]
        if name not in current_by_id:
            continue
        current_cell = current_by_id[name]
        assert current_cell["cell_type"] == previous_cell["cell_type"]
        assert current_cell["metadata"] == previous_cell["metadata"]
        if current_cell["cell_type"] == "code":
            assert current_cell["outputs"] == previous_cell["outputs"]
            assert current_cell["execution_count"] == previous_cell["execution_count"]
            checked_nonempty_outputs += bool(previous_cell["outputs"])
            retained_outputs[name] = hashlib.sha256(
                json.dumps(current_cell["outputs"], sort_keys=True).encode()
            ).hexdigest()
    assert checked_nonempty_outputs > 0
    missing_ids = [cell["id"] for cell in previous_cells if cell["id"] not in current_by_id]
    return retained_outputs, missing_ids


if __name__ == "__main__":
    main()
