"""Build an archived dispersion package and exercise an isolated installed copy."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "dispersion-validation" / "distribution"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if env.get("PYTHON_GIL") == "0":
        raise RuntimeError("Remove forced-off PYTHON_GIL before import qualification")
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MPLBACKEND="Agg",
        MAKEFLAGS="-j2",
        CARGO_BUILD_JOBS="2",
        UV_CONCURRENT_BUILDS="2",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
    )

    def run(*args: str, cwd: Path = root) -> None:
        print("RUN", *args, flush=True)
        subprocess.run(args, cwd=cwd, env=env, check=True, timeout=180)

    packages = ("orsia-geometry", "orsia-dispersion")
    for name in packages:
        run("uv", "build", "--package", name, "--sdist", "--out-dir", str(output))
    with tempfile.TemporaryDirectory(prefix="isolated-", dir=output) as directory:
        stage = Path(directory)
        for name in packages:
            archive = output / f"{name.replace('-', '_')}-0.1.0.tar.gz"
            copied = stage / archive.name
            shutil.copy2(archive, copied)
            run(
                "uv",
                "build",
                str(copied),
                "--wheel",
                "--no-sources",
                "--out-dir",
                str(output),
                cwd=stage,
            )
        wheel = output / "orsia_dispersion-0.1.0-py3-none-any.whl"
        geometry_wheel = output / "orsia_geometry-0.1.0-py3-none-any.whl"
        with ZipFile(wheel) as packaged:
            names = packaged.namelist()
            for module in (
                "forcing",
                "stationary",
                "particles",
                "sampling",
                "evaluation",
                "emissions",
                "puffs/population",
                "puffs/simulation",
                "puffs/evaluation",
                "puffs/transport",
                "puffs/kernels",
            ):
                assert f"orsia_dispersion/{module}.py" in names
            for module in ("puffs", "transport", "simulation", "splitting"):
                assert f"orsia_dispersion/{module}.py" not in names
            source_root = root / "libs/orsia-dispersion/src"
            for path in (source_root / "orsia_dispersion").rglob("*.py"):
                assert packaged.read(path.relative_to(source_root).as_posix()) == path.read_bytes()
        venv = stage / "venv"
        run("uv", "venv", str(venv), "--python", sys.executable, cwd=stage)
        python = venv / "bin" / "python"
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            str(geometry_wheel),
            str(wheel),
            cwd=stage,
        )
        smoke = """
import importlib.metadata
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy
import orsia_dispersion as dispersion
from orsia_geometry import LocalCartesianFrame, PointLattice, RectangularCells, LayeredCells
assert "site-packages" in dispersion.__file__, dispersion.__file__
assert importlib.util.find_spec("satkit") is None
assert not sys._is_gil_enabled()
requirements = importlib.metadata.requires("orsia-dispersion")
assert requirements is not None and len(requirements) == 3
assert all(
    requirement.startswith(("numpy", "scipy", "orsia-geometry")) for requirement in requirements
)
assert not any(name.startswith(("orsia_radiative", "orsia_instruments")) for name in sys.modules)
frame = LocalCartesianFrame("installed test", "synthetic altitude")
points = PointLattice.regular((1,1), (60.,60.), (0.,0.), frame)
support = LayeredCells(
    RectangularCells.around_points(points, (60.,60.)),
    np.array([0.,1.,3.,20.]), np.zeros((1,1)),
)
state = dispersion.GaussianPuffs(
    0., np.zeros(1), np.array([3.]), np.array([[0.,0.,2.]]),
    np.array([np.diag([4.,9.,1.])]),
)
field = dispersion.GaussianPuffField(state, "reflecting")
moments = dispersion.pixel_layer_moments(field, support)
assert moments.support is support
np.testing.assert_allclose(moments.columns_kg_m2.sum()*3600, 3., rtol=1e-14)
expected = dispersion.concentration(field, np.array([[0.,0.,2.]]))
with ThreadPoolExecutor(max_workers=2) as executor:
    actual = list(executor.map(
        lambda _: dispersion.concentration(field, np.array([[0.,0.,2.]])), range(4)
    ))
for value in actual:
    np.testing.assert_array_equal(value, expected)
path = dispersion.puff_path_columns(
    state, np.array([0.,0.,0.]), np.array([0.,0.,20.]), boundary="reflecting",
)
columns = dispersion.layer_columns(field, np.zeros((1,2)), support.height_edges_m)
np.testing.assert_allclose(path, columns.sum(), rtol=1e-12)
assert not sys._is_gil_enabled()
print("PASS isolated archived Dispersion and Geometry, complete support, mass/path invariants, "
      "bounded concurrency and no SatKit")
"""
        run(str(python), "-I", "-c", smoke, cwd=stage)


if __name__ == "__main__":
    main()
