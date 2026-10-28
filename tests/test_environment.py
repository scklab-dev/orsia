"""Environment checks, not scientific qualification."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "check_runtime.py"


def test_native_imports_preserve_free_threading() -> None:
    result = subprocess.run(
        [sys.executable, str(PROBE)], check=True, capture_output=True, text=True, cwd=ROOT
    )
    report = json.loads(result.stdout)
    assert report["free_threaded_build"] is True
    assert report["gil_enabled"] is False
    assert set(report["imports"]) == {"numpy", "scipy"}


def test_enabled_gil_is_rejected() -> None:
    result = subprocess.run(
        [sys.executable, "-X", "gil=1", str(PROBE)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode != 0
    assert "GIL disabled is required" in result.stderr


def test_forced_disabled_gil_is_not_compatibility_evidence() -> None:
    result = subprocess.run(
        [sys.executable, "-X", "gil=0", str(PROBE)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode != 0
    assert "Do not force the GIL off" in result.stderr
