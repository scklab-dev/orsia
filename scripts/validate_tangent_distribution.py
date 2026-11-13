"""Build and exercise archived tangent packages with workspace sources disabled."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    packages = (
        "orsia-numerics",
        "orsia-geometry",
        "orsia-uncertainty",
        "orsia-radiative",
        "orsia-instruments",
        "orsia-inverse",
    )
    output = root / "outputs" / "tangent-distribution"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHON_GIL", None)
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

    for name in packages:
        run("uv", "build", "--package", name, "--sdist", "--out-dir", str(output))
    # Only archived distribution contents and installed build dependencies are
    # used below. No workspace sources, editable installs or checkout PYTHONPATH.
    with tempfile.TemporaryDirectory(prefix="isolated-", dir=output) as directory:
        stage = Path(directory)
        wheels = stage / "wheels"
        wheels.mkdir()
        for name in (package.replace("-", "_") for package in packages):
            archive = output / f"{name}-0.1.0.tar.gz"
            copied = stage / archive.name
            shutil.copy2(archive, copied)
            run(
                "uv",
                "build",
                str(copied),
                "--wheel",
                "--no-sources",
                "--find-links",
                str(wheels),
                "--out-dir",
                str(wheels),
                cwd=stage,
            )
        numerics_wheel = next(wheels.glob("orsia_numerics-*.whl"))
        with ZipFile(numerics_wheel) as archive:
            members = archive.namelist()
            assert "orsia_numerics/tangents.py" in members
            assert "orsia_numerics/operators.py" in members
            assert "orsia_numerics/autodiff.py" not in members
            assert "orsia_numerics/linearization.py" not in members
            assert "orsia_numerics/tangent.py" not in members
            assert "orsia_numerics/include/orsia_numerics/tangent.h" in members
        with ZipFile(next(wheels.glob("orsia_radiative-*.whl"))) as archive:
            stub = archive.read("orsia_radiative/optics/absorption_native.pyi").decode()
            assert "fixed_cross_section_" not in stub
            for name in (
                "cross_section_optical_depth",
                "cross_section_optical_depth_tangent",
            ):
                assert f"def {name}(" in stub
            assert not any("adjoint" in name for name in archive.namelist())
            assert "adjoint" not in stub
            assert (
                "adjoint"
                not in archive.read("orsia_radiative/transfer/native/transfer_native.pyi").decode()
            )
        venv = stage / "venv"
        run("uv", "venv", str(venv), "--python", sys.executable, cwd=stage)
        python = venv / "bin" / "python"
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--find-links",
            str(wheels),
            *(f"{name}==0.1.0" for name in packages),
            cwd=stage,
        )
        smoke = """
import sys
from pathlib import Path
import numpy as np
import orsia_numerics
import orsia_radiative
from orsia_numerics import tangents
from orsia_numerics.tangents import directional_derivative, jacobian
from orsia_radiative.optics import absorption, absorption_native, cross_sections
from orsia_radiative.optics.cross_sections import (
    cross_section_optical_depth,
    cross_section_optical_depth_tangent,
)
from orsia_radiative.transfer.native import transfer_native
from orsia_radiative.transfer import solar, thermal
for module in (orsia_radiative, absorption_native, cross_sections, transfer_native, solar, thermal):
    assert not any("adjoint" in name for name in dir(module))
assert "site-packages" in orsia_numerics.__file__
assert "site-packages" in orsia_radiative.__file__
assert Path(orsia_numerics.get_include(), "orsia_numerics/tangent.h").is_file()
assert not hasattr(tangents, "dense_jacobian")
assert "dense_jacobian" not in tangents.__all__
def square(x, d):
    return x*x, 2*x[:,None]*d
x = np.array([2.0, 3.0])
v, j = jacobian(function=square, x=x, block_size=1)
np.testing.assert_array_equal(v, [4.0, 9.0])
np.testing.assert_array_equal(j, np.diag([4.0, 6.0]))
v, d = directional_derivative(function=square, x=x, direction=np.array([1.0, -2.0]))
np.testing.assert_array_equal(d, [4.0, -12.0])
v, d = directional_derivative(square, x, np.eye(2))
np.testing.assert_array_equal(d, j)
v, d = directional_derivative(square, x, np.empty((2, 0)))
assert d.shape == (2, 0)
tau, dtau = cross_section_optical_depth_tangent(np.ones((1,2)), np.ones(1), np.ones((1,3)))
for module in (orsia_radiative, absorption, absorption_native, cross_sections):
    assert not any(name.startswith("fixed_cross_section_") for name in dir(module))
for name in (
    "cross_section_optical_depth",
    "cross_section_optical_depth_tangent",
):
    assert getattr(orsia_radiative, name) is globals()[name]
    assert callable(getattr(absorption_native, name))
assert orsia_radiative.cross_section_optical_depth is cross_section_optical_depth
np.testing.assert_array_equal(cross_section_optical_depth(np.ones((1,2)), np.ones(1)), tau)
v, d = transfer_native.reflected_radiance_tangent(
    tau, dtau, np.ones(2), np.full(2, .5), np.zeros((2,3)), 0., 0.
)
assert d.shape == (2,3) and np.all(d < 0)
assert not sys._is_gil_enabled()
print("Installed Numerics, absorption, solar and thermal imports: GIL disabled")
from dataclasses import replace
from orsia_radiative import (
    Absorption,
    AerosolComponent,
    LayerAtmosphere,
    ObservationGeometry,
    ColumnScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from orsia_radiative.transfer.orders import scattering_orders_tangent
assert "site-packages" in sys.modules[scattering_orders_tangent.__module__].__file__
grid = wavenumber_grid(np.array([1000., 1001.]))
atmosphere = LayerAtmosphere(
    np.array([0., 1000.]), np.array([100000., 90000.]), np.array([95000.]),
    np.array([285.]), np.array([2e29]), np.array([1.9e29]), np.array([1e28]),
)
scene = ColumnScene(atmosphere, SurfaceState(grid, reflectance=np.full(2, .3)),
    ObservationGeometry(20.))
source = solar_source(SpectralIrradiance(grid, np.ones(2)), direction=(30., 20.))
optics = atmospheric_optics(atmosphere, grid,
    absorption=Absorption(grid, np.full((1,2), .04)),
    rayleigh=RayleighComponent(grid, np.full((1,2), .02)),
    aerosol=AerosolComponent(grid, np.full((1,2), .07), np.full((1,2), .04),
        np.full((1,2), .3)),
)
controls = dict(max_order=3, angular_order=3, layer_subdivisions=3, delta_m_moments=4)
seeds = dict(optical_depth_tangent=np.zeros((1,2,1)),
    reflectance_tangent=np.zeros((2,1)), aerosol_asymmetry_tangent=np.ones((1,2,1)))
value, derivative = solar.reflected_radiance_tangent(
    scene, source=source, optics=optics, **seeds, **controls)
rows, drows = scattering_orders_tangent(scene, source=source, optics=optics, **seeds, **controls)
np.testing.assert_array_equal(rows.sum(axis=0), value.values)
np.testing.assert_array_equal(drows.sum(axis=0), derivative)
trials = []
for sign in (1, -1):
    perturbed = replace(optics, aerosol=replace(optics.aerosol,
        asymmetry=optics.aerosol.asymmetry+sign*1e-5))
    trials.append(solar.reflected_radiance(scene, source=source, optics=perturbed,
        **controls).values)
np.testing.assert_allclose((trials[0]-trials[1])/2e-5, derivative[:,0], rtol=1e-8)
_, empty = solar.reflected_radiance_tangent(scene, source=source, optics=optics,
    optical_depth_tangent=np.empty((1,2,0)), reflectance_tangent=np.empty((2,0)), **controls)
assert empty.shape == (2,0)
assert derivative.flags.owndata and not sys._is_gil_enabled()
print("Installed SOS/Delta-M tangent: order sum, finite difference and empty directions passed")
"""
        smoke += """
from orsia_numerics.operators import linear_operator, materialize_operator
from orsia_inverse import solve_map
from orsia_uncertainty import factor_covariance
import orsia_instruments, orsia_inverse
assert "site-packages" in orsia_instruments.__file__
assert "site-packages" in orsia_inverse.__file__
op = linear_operator((2,2), lambda v: 2*v, lambda w: 2*w)
np.testing.assert_array_equal(materialize_operator(op, max_bytes=1000), np.eye(2)*2)
estimate = solve_map(
    lambda q: 2*q, lambda q: np.eye(2)*2, np.array([2., 4.]), np.zeros(2),
    observation_factor=factor_covariance(np.ones(2), method="cholesky"),
)
assert estimate.termination.success
np.testing.assert_allclose(estimate.state, [1., 2.])
assert not sys._is_gil_enabled()
print("Installed instrument, inverse and native imports: GIL disabled")
"""
        run(str(python), "-I", "-c", smoke, cwd=stage)
        example = stage / "thermal_tangents.py"
        shutil.copy2(root / "scripts" / example.name, example)
        run(str(python), "-I", str(example), cwd=stage)
        for wheel in wheels.glob("*.whl"):
            shutil.copy2(wheel, output / wheel.name)
    print("Archived build and installed smoke checks passed; distributions:", output)


if __name__ == "__main__":
    main()
