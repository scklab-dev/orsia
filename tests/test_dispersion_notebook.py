"""Check the self-contained dispersion tutorial and its retained displays."""

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/gaussian_puff_dispersion.ipynb"


def test_dispersion_notebook_schema_and_code():
    notebook = cast(nbformat.NotebookNode, nbformat.read(NOTEBOOK, as_version=4))
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK.name}:{cell.id}", "exec")


def test_dispersion_notebook_runs_with_synthetic_inputs():
    """Run the actual cells headlessly without changing the pytest process backend."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=NOTEBOOK.parent.parent,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_dispersion_notebook_retains_successful_execution_and_figures():
    notebook = cast(nbformat.NotebookNode, nbformat.read(NOTEBOOK, as_version=4))
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    images = []
    for cell in code_cells:
        for output in cell.outputs:
            assert output.output_type != "error"
            if "image/png" in output.get("data", {}):
                images.append(base64.b64decode(output.data["image/png"]))
    assert len(images) == 3
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in images)


def _assert_notebook_execution():
    """Check cell execution and numerical assertions in the isolated Agg process."""
    import matplotlib as mpl
    import numpy as np

    assert mpl.get_backend().lower() == "agg"
    notebook = cast(nbformat.NotebookNode, nbformat.read(NOTEBOOK, as_version=4))
    namespace = {}
    images = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{NOTEBOOK.name}:{cell.id}", "exec"), namespace)
            if cell.id == "imports":
                namespace["display"] = images.append
    assert len(images) == 3
    assert all(image.data.startswith(b"\x89PNG\r\n\x1a\n") for image in images)
    assert len(namespace["final_state"].masses_kg) == 18
    assert namespace["column_kg_m2"].shape == (141, 201)
    assert np.all(namespace["column_kg_m2"] >= 0.0)
    assert namespace["map_relative_error"] < 1e-5
    np.testing.assert_allclose(namespace["map_mass_kg"], 3.6, rtol=1e-5)


if __name__ == "__main__":
    _assert_notebook_execution()
