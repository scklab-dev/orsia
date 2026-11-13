"""Check the spatial-scene tutorial's actual cells and retained displays."""

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/spatial_scene.ipynb"


def test_spatial_scene_notebook_schema_and_code():
    notebook = cast(nbformat.NotebookNode, nbformat.read(NOTEBOOK, as_version=4))
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK.name}:{cell.id}", "exec")


def test_spatial_scene_notebook_runs_with_synthetic_inputs():
    """Execute cells in a separate bounded Agg process without modifying outputs."""
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=NOTEBOOK.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_spatial_scene_notebook_retains_successful_execution_and_figures():
    notebook = cast(nbformat.NotebookNode, nbformat.read(NOTEBOOK, as_version=4))
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    images = []
    for cell in code_cells:
        for output in cell.outputs:
            assert output.output_type != "error"
            if "image/png" in output.get("data", {}):
                images.append(base64.b64decode(output.data["image/png"]))
    assert len(images) == 7
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in images)


def _assert_notebook_execution():
    """Check displayed figures and values from the unmodified notebook cells."""
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
    assert len(images) == 7
    assert all(image.data.startswith(b"\x89PNG\r\n\x1a\n") for image in images)
    assert namespace["small_values"].shape == (2, 5)
    assert namespace["small_tangent"].shape == (2, 5, 2)
    assert namespace["covered"].shape == (8,)
    assert namespace["covered"].all()
    assert np.isfinite(namespace["image"]).all()
    assert (namespace["image"] > 0).all()
    assert namespace["jacobian"].shape == (2, 2, 2, 2)
    assert len(namespace["fd_errors"]) == 2
    assert max(namespace["fd_errors"]) < 2e-7


if __name__ == "__main__":
    _assert_notebook_execution()
