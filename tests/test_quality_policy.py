"""Regression checks for module naming and the library coverage gate."""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from coverage import CoverageData

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
LIBRARIES = CONFIG["tool"]["uv"]["workspace"]["members"]


def test_library_modules_use_responsibility_names() -> None:
    for library in LIBRARIES:
        modules = list((ROOT / library / "src").rglob("*.py"))
        assert modules
        assert not [p for p in modules if p.name.startswith("_") and p.name != "__init__.py"]


def test_shared_transfer_operations_do_not_import_private_helpers_or_composition() -> None:
    directory = ROOT / "libs/orsia-radiative/src/orsia_radiative/transfer"
    for name in ("formal", "orders"):
        tree = ast.parse((directory / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "solar"
                assert not [alias.name for alias in node.names if alias.name.startswith("_")]
                assert "solar" not in {alias.name for alias in node.names}


def test_coverage_configuration_includes_all_library_sources_without_omissions() -> None:
    coverage = CONFIG["tool"]["coverage"]
    assert set(coverage["run"]["source"]) == {f"{library}/src" for library in LIBRARIES}
    assert coverage["run"]["branch"] is True
    assert coverage["report"]["fail_under"] == 90
    assert "omit" not in coverage["run"] and "omit" not in coverage["report"]
    assert "exclude_lines" not in coverage["report"] and "exclude_also" not in coverage["report"]
    assert "--cov-config=pyproject.toml" in CONFIG["tool"]["pytest"]["ini_options"]["addopts"]
    recipes = (ROOT / "justfile").read_text(encoding="utf-8")
    for library in LIBRARIES:
        assert f"coverage report --include='{library}/src/*'" in recipes


@pytest.mark.parametrize("library", LIBRARIES)
def test_coverage_gate_rejects_a_low_library_hidden_by_the_aggregate(
    tmp_path: Path, library: str
) -> None:
    # Synthesise execution data, not library tests: a well-covered large module
    # makes the total pass while the selected library must still fail its gate.
    sources: dict[str, str] = {}
    for member in LIBRARIES:
        directory = tmp_path / member / "src" / "example"
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
        module = directory / "model.py"
        module.write_text("value = 1\n" * (10 if member == library else 200), encoding="utf-8")
        sources[member] = str(module.relative_to(tmp_path))
    data_file = tmp_path / ".coverage"
    data = CoverageData(basename=str(data_file))
    data.add_arcs(
        {
            path: [(-1, 1), (1, -1)]
            if member == library
            else [(-1, 1), *((i, i + 1) for i in range(1, 200)), (200, -1)]
            for member, path in sources.items()
        }
    )
    data.write()
    command = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        f"--rcfile={ROOT / 'pyproject.toml'}",
        f"--data-file={data_file}",
    ]
    aggregate = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert aggregate.returncode == 0, aggregate.stdout + aggregate.stderr
    selected = subprocess.run(
        [*command, f"--include={library}/src/*"], cwd=tmp_path, capture_output=True, text=True
    )
    assert selected.returncode == 2, selected.stdout + selected.stderr
    assert "less than fail-under=90" in selected.stdout


def test_coverage_native_import_preserves_free_threading() -> None:
    assert not sys._is_gil_enabled()
