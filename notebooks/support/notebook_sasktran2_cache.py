"""Persistent complete SASKTRAN2 references for the five-band notebook.

JSON entries hold owned numerical lists in the worker's per-cm-1 convention.
The caller owns cache location and reuse policy; numerical solvers stay separate.
"""

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import numpy as np

SCHEMA = 1


def reference(
    shared: dict,
    cells: int,
    streams: int,
    *,
    runs: int,
    interpreter: Path,
    worker: Path,
    cache_directory: Path,
    mode: str = "reuse",
) -> dict:
    """Reuse, refresh or bypass a complete reference (mode: reuse/refresh/off).

    Snapshot all shared arrays before hashing and solving. Timings in ``timing``
    and ``worker_wall_seconds`` describe the original computation; ``cache``
    records this access, including identity-probe and load wall times separately.
    Every access probes the isolated environment; a hit skips the solve worker.
    Concurrent writers may compute the same key, then atomically replace it.
    """
    if mode not in {"reuse", "refresh", "off"}:
        raise ValueError("cache mode must be reuse, refresh or off")
    if cells < 2 or streams < 4 or streams % 2 or runs < 1:
        raise ValueError("cells >= 2, even streams >= 4 and runs >= 1 required")
    start = perf_counter()
    arrays = {name: np.array(value, copy=True) for name, value in shared.items()}
    for value in arrays.values():
        if value.dtype.kind not in "bifu" or not np.all(np.isfinite(value)):
            raise ValueError("shared inputs must be finite numeric arrays or scalars")
    identity_start = perf_counter()
    backend = _run_json([str(interpreter), str(worker), "--identity"], timeout=120)
    identity_seconds = perf_counter() - identity_start
    signature = {
        "schema": SCHEMA,
        "inputs": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
            for name, value in sorted(arrays.items())
        },
        "cells": cells,
        "streams": streams,
        "runs": runs,
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "cache_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backend": backend,
    }
    key = _digest(signature)
    destination = cache_directory / f"{key}.json"
    status = mode
    record = None
    load_start = perf_counter()
    if mode == "reuse":
        try:
            entry = json.loads(destination.read_text())
            if entry["signature"] != signature or entry["key"] != key:
                raise ValueError("cache signature mismatch")
            if entry["record_sha256"] != _digest(entry["record"]):
                raise ValueError("cache record checksum mismatch")
            _validate(entry["record"], arrays, cells, streams, runs, backend)
            record = entry["record"]
            status = "hit"
        except FileNotFoundError:
            status = "miss"
        except OSError, ValueError, KeyError, TypeError, OverflowError, AssertionError:
            status = "invalid"
    load_seconds = perf_counter() - load_start if mode == "reuse" else 0.0
    if record is None:
        worker_start = perf_counter()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared.npz"
            np.savez(path, allow_pickle=False, **arrays)
            record = _run_json(
                [
                    str(interpreter),
                    str(worker),
                    str(path),
                    "--cells",
                    str(cells),
                    "--streams",
                    str(streams),
                    "--runs",
                    str(runs),
                ],
                timeout=300 * (runs + 1),
            )
        record["worker_wall_seconds"] = perf_counter() - worker_start
        record["computed_at_utc"] = datetime.now(UTC).isoformat()
        _validate(record, arrays, cells, streams, runs, backend)
        if mode != "off":
            entry = {
                "signature": signature,
                "key": key,
                "record": record,
                "record_sha256": _digest(record),
            }
            _write_atomic(destination, entry)
    record["cache"] = {
        "status": status,
        "key": key,
        "path": str(destination) if mode != "off" else None,
        "identity_wall_seconds": identity_seconds,
        "load_wall_seconds": load_seconds,
        "access_wall_seconds": perf_counter() - start,
        "timing_provenance": "historical" if status == "hit" else "fresh",
    }
    return record


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _run_json(command: list[str], *, timeout: int) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return json.loads(result.stdout)


def _validate(record: dict, arrays: dict, cells: int, streams: int, runs: int, backend: dict):
    """Validate the persisted boundary and the shared-input LOS invariant."""
    size = arrays["wavenumbers_cm1"].size
    for name in ("radiance", "los_optical_depth"):
        vector = np.asarray(record[name], dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid {name}")
    thermal = bool(arrays["thermal"])
    if thermal:
        vector = np.asarray(record["reflected_boundary_radiance"], dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError("invalid reflected boundary")
    elif record["reflected_boundary_radiance"] is not None:
        raise ValueError("solar boundary must be null")
    coupling = "explicit_lambertian_boundary" if thermal else "native_solar"
    if record["thermal_surface_coupling"] != coupling:
        raise ValueError("surface coupling mismatch")
    if record["cells"] != cells or record["streams"] != streams:
        raise ValueError("solver settings mismatch")
    for name in ("python", "sasktran2", "gil_enabled"):
        if record[name] != backend[name]:
            raise ValueError(f"backend {name} mismatch")
    timing = record["timing"]
    samples = np.asarray(timing["seconds"], dtype=float)
    if samples.shape != (runs,) or not np.all(np.isfinite(samples)) or np.any(samples <= 0):
        raise ValueError("invalid timing samples")
    if timing["median_seconds"] != float(np.median(samples)) or not timing["scope"]:
        raise ValueError("invalid timing summary")
    for value in (timing["first_seconds"], record["worker_wall_seconds"]):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("invalid wall time")
    if datetime.fromisoformat(record["computed_at_utc"]).utcoffset() is None:
        raise ValueError("computation time requires a timezone")
    expected_depth = (arrays["absorption_depth"] + arrays["rayleigh_depth"]).sum(axis=0)
    np.testing.assert_allclose(
        record["los_optical_depth"], expected_depth / arrays["muv"], rtol=1e-4, atol=1e-10
    )


def _write_atomic(destination: Path, entry: dict):
    content = json.dumps(entry, sort_keys=True, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=destination.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
