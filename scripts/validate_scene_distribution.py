"""Build archived scene dependencies and check isolated installed composition offline."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "scene-validation" / "distribution"
    output.mkdir(parents=True, exist_ok=True)
    packages = (
        "orsia-numerics",
        "orsia-geometry",
        "orsia-uncertainty",
        "orsia-radiative",
        "orsia-instruments",
        "orsia-dispersion",
        "orsia-scene",
        "orsia-data",
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if env.get("PYTHON_GIL") == "0":
        raise RuntimeError("Remove forced-off PYTHON_GIL before native import validation")
    env.update(
        UV_OFFLINE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        CARGO_BUILD_JOBS="2",
        MAKEFLAGS="-j2",
        UV_CONCURRENT_BUILDS="2",
        MPLBACKEND="Agg",
    )

    def run(*args: str) -> None:
        print("RUN", *args, flush=True)
        subprocess.run(args, cwd=root, env=env, check=True, timeout=300)

    for name in packages:
        run("uv", "build", "--package", name, "--sdist", "--out-dir", str(output))
    with tempfile.TemporaryDirectory(prefix="isolated-", dir=output) as directory:
        stage = Path(directory)
        wheels = stage / "wheels"
        wheels.mkdir()
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
                "--find-links",
                str(wheels),
                "--out-dir",
                str(wheels),
            )
        venv = stage / "venv"
        run("uv", "venv", str(venv), "--python", sys.executable)
        python = venv / "bin" / "python"
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--find-links",
            str(wheels),
            str(next(wheels.glob("orsia_scene-*.whl"))),
        )
        readme = (root / "libs/orsia-scene/README.md").read_text()
        blocks = re.findall(r"^```python\n(.*?)^```", readme, re.M | re.S)
        assert blocks, "Scene README must contain an executable example"
        direct_guide = (root / "libs/orsia-scene/docs/direct-solar.md").read_text()
        direct_blocks = re.findall(r"^```python\n(.*?)^```", direct_guide, re.M | re.S)
        assert direct_blocks, "Physical cell-scene guide must contain an executable example"
        filter_guide = (root / "libs/orsia-instruments/docs/wavelength-response.md").read_text()
        filter_blocks = re.findall(r"^```python\n(.*?)^```", filter_guide, re.M | re.S)
        assert filter_blocks, "Filter response guide must contain an executable example"
        terrain_guide = (root / "libs/orsia-geometry/docs/terrain.md").read_text()
        terrain_blocks = re.findall(r"^```python\n(.*?)^```", terrain_guide, re.M | re.S)
        assert terrain_blocks, "Terrain guide must contain executable interpolation examples"
        standalone = f"""
import inspect, sys
from importlib.util import find_spec
import orsia_scene
import orsia_geometry
from orsia_scene.radiance import surface_radiance
assert find_spec('orsia_scene.observation') is None
assert not hasattr(orsia_scene, 'radiance_for_observation')
assert not hasattr(orsia_scene, 'observation_callbacks')
assert find_spec('orsia_scene.queries') is None
assert find_spec('orsia_scene.extinction') is None
assert not hasattr(orsia_scene, 'SceneRadiance')
assert orsia_scene.DirectSolarScene.__module__ == 'orsia_scene.radiance'
for name in ('surface_points', 'intersect_terrain'):
    assert name in orsia_geometry.__all__
    assert callable(getattr(orsia_geometry, name))
assert orsia_scene.surface_radiance is surface_radiance
assert 'surface_radiance' in orsia_scene.__all__
assert 'reflectance_tangent' not in inspect.signature(surface_radiance).parameters
from orsia_radiative.optics import absorption_native
from orsia_radiative.transfer.native import transfer_native
assert find_spec('orsia_instruments') is None
assert find_spec('satkit') is None
from orsia_geometry import LocalCartesianFrame, PointLattice, RectangularCells, LayeredCells
from orsia_dispersion import GaussianPuffs, GaussianPuffField, pixel_layer_moments
from orsia_scene import gas_layer_columns
import numpy as np
from orsia_radiative import (
    Radiance, SpectralIrradiance, ObservationGeometry, SpectralGrid,
    convert_spectral_grid, density_per_nm_to_per_cm1,
)
nm = SpectralGrid("wavelength_nm", np.array([500., 1000.]))
for spectrum in (
    SpectralIrradiance(nm, np.ones(2)),
    Radiance(nm, np.ones(2), ObservationGeometry(0.)),
):
    np.testing.assert_array_equal(spectrum.values, [1., 1.])
    try:
        spectrum.wavenumbers_cm1
    except ValueError:
        pass
    else:
        raise AssertionError("nm spectrum was silently relabelled cm-1")
np.testing.assert_array_equal(convert_spectral_grid(nm).points, [10000., 20000.])
converted = density_per_nm_to_per_cm1(nm.points, np.ones((3,2,4)), axis=1)
np.testing.assert_allclose(converted[:,0], .1)
np.testing.assert_allclose(converted[:,1], .025)
frame = LocalCartesianFrame("installed scene", "synthetic altitude")
points = PointLattice.regular((1,1), (60.,60.), (0.,0.), frame)
cells = LayeredCells(
    RectangularCells.around_points(points, (60.,60.)),
    np.array([0.,1.,3.,20.]), np.zeros((1,1)),
)
state = GaussianPuffs(
    0., np.zeros(1), np.array([3.]), np.array([[0.,0.,2.]]),
    np.array([np.diag([4.,9.,1.])]),
)
mass = pixel_layer_moments(GaussianPuffField(state, "reflecting"), cells)
gas = gas_layer_columns(mass, species="CH4", molar_mass_kg_mol=0.016)
assert gas.support is mass.support is cells
np.testing.assert_allclose(gas.columns_molecules_m2 * 0.016 / 6.02214076e23, mass.columns_kg_m2)
assert not gas.columns_molecules_m2.flags.writeable
from orsia_dispersion import path_columns
from orsia_scene.projection import project_layer_columns_by_centroid
from orsia_radiative.transfer.extinction import (
    exposure_mean_radiance,
    exposure_mean_radiance_tangent,
    linearize_exposure_mean_radiance,
)
assert find_spec('orsia_radiative.absorption_response') is None
assert find_spec('orsia_radiative.coaddition') is None
field = GaussianPuffField(state, "reflecting", kernel_cutoff_radius=3.)
starts, ends = np.zeros((1,3)), np.array([[0.,0.,20.]])
columns = orsia_scene.puff_molecular_columns(
    starts, ends, np.zeros(1, dtype=np.int64), height_edges_m=np.array([0.,20.]),
    background_columns_molecules_m2=np.zeros((1,1)), plumes=(field,),
    species_indices=np.zeros(1, dtype=np.int64), molar_masses_kg_mol=np.array([.016]),
    scales=np.ones(1),
)
np.testing.assert_allclose(columns[:,0], path_columns(field, starts, ends)*6.02214076e23/.016)
amounts = np.ones((3,4,1))
projected = project_layer_columns_by_centroid(
    amounts, amounts, (1.,0.), (1.,1.), continuation="finite",
)
np.testing.assert_array_equal(projected[0], 0)
np.testing.assert_array_equal(projected[1:], 1)
source, depth = np.array([1.,2.]), np.array([.1,.2])
masses, weights = np.array([2.,4.]), np.ones(2)
expected = source*np.exp(-masses[:,None]*depth).mean(axis=0)
np.testing.assert_allclose(exposure_mean_radiance(source, depth, masses, weights), expected)
reference, intercept, effective = linearize_exposure_mean_radiance(source, depth, masses, weights)
np.testing.assert_allclose(np.exp(intercept-effective*depth), reference)
value, seed = exposure_mean_radiance_tangent(
    source, depth, masses, weights, source_radiance_tangent=np.zeros((2,1)),
    optical_depth_tangent=np.ones((2,1)),
)
np.testing.assert_allclose(seed[:,0], -value*effective)
for module in (orsia_scene, orsia_geometry, absorption_native, transfer_native):
    assert 'site-packages' in module.__file__, module.__file__
for code in {blocks!r}:
    exec(code)
direct_namespace = {{}}
for code in {direct_blocks!r}:
    exec(code, direct_namespace)
terrain_namespace = {{}}
for code in {terrain_blocks!r}:
    exec(code, terrain_namespace)
for mode in ('independent_column', 'heterogeneous_direct_solar'):
    physical['mode'] = mode
    direct = surface_radiance(xy, view, columns=columns, **physical)
    value, tangent = surface_radiance_tangent(
        xy, view, columns=columns_with_tangent,
        reflectance_tangent=np.ones((1, 2, 1)), **physical)
    np.testing.assert_array_equal(direct, value)
    assert direct.shape == (1, 2) and tangent.shape == (1, 2, 1)
assert not sys._is_gil_enabled()
print('PASS installed callable terrain/interpolation and Scene direct/tangent APIs; '
      'affine modes without Instruments; '
      'GIL disabled', sys.version)
"""
        run(str(python), "-I", "-c", standalone)
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--find-links",
            str(wheels),
            str(next(wheels.glob("orsia_instruments-*.whl"))),
            str(next(wheels.glob("orsia_data-*.whl"))),
        )
        example = stage / "spatial_scene.py"
        shutil.copy2(root / "notebooks" / "support" / "spatial_scene.py", example)
        smoke = f"""
import runpy, sys
from importlib.metadata import requires
from importlib.util import find_spec
from pathlib import Path
import numpy as np
import scipy
import orsia_geometry, orsia_radiative, orsia_instruments, orsia_dispersion, orsia_scene, orsia_data
from orsia_radiative.optics import absorption_native
from orsia_radiative.transfer.native import transfer_native
for module in (orsia_geometry, orsia_radiative, orsia_instruments,
               orsia_dispersion, orsia_scene, orsia_data, absorption_native, transfer_native):
    assert 'site-packages' in module.__file__, module.__file__
    print('INSTALLED', module.__name__, module.__file__)
assert not sys._is_gil_enabled()
assert not any(req.startswith('orsia-instruments') for req in requires('orsia-scene'))
for obsolete in ('orsia_instruments.scene', 'orsia_instruments.imaging.response.instrument',
                 'orsia_instruments.imaging.cells', 'orsia_instruments.spectral_response',
                 'orsia_instruments.photoelectrons', 'orsia_instruments.charge',
                 'orsia_instruments.quadrature', 'orsia_instruments.los',
                 'orsia_instruments.radiometry', 'orsia_radiative.scene',
                 'orsia_radiative.bands', 'orsia_numerics.autodiff', 'orsia_scene.composition'):
    try:
        spec = find_spec(obsolete)
    except ModuleNotFoundError as error:
        assert obsolete.startswith(error.name)
    else:
        assert spec is None
from orsia_geometry import LocalCartesianFrame, PointLattice
from orsia_radiative import SpectralGrid, SpectralDomain
from orsia_instruments import (
    GridElectronResponse,
    SurfaceRadianceSamples,
    AngularRadianceSamples,
    ReadSchedule,
    response_electrons,
    sample_radiance,
)
frame = LocalCartesianFrame('installed response', 'synthetic surface')
lattice = PointLattice(np.array([0.,1.,2.]), np.array([0.,1.]), frame)
spectral = SpectralGrid('wavenumber_cm1', np.array([1000.,2000.]),
    SpectralDomain('wavenumber_cm1', ((1000.,2000.),)))
reads = ReadSchedule(('r',), ('p',), ('s',), ('direct',), np.array([[0.,1.]]))
response = GridElectronResponse(reads, spectral, lattice,
    np.array([[0.,2.,0.,1.]]), np.ones((1,2,3,2)))
samples = SurfaceRadianceSamples(lattice, spectral, np.full((2,3,2), 2.))
assert response.lattice is samples.lattice is lattice
np.testing.assert_array_equal(response_electrons(response, samples).total, [24.])
angular = AngularRadianceSamples(spectral, np.array([0.,1.]),
    np.array([-.1,.1]), np.array([-.1,.1]), np.ones((2,2,2,2)))
np.testing.assert_array_equal(
    sample_radiance(angular, np.array([.5]), np.zeros((1,2)), slice(None)), [[1.,1.]],
)
from orsia_data.spectroscopy.cross_sections import read_cross_section_table
local_table = Path({str(stage / "synthetic-cross-sections.npz")!r})
np.savez(local_table, wavenumbers_cm1=np.array([1000.,2000.]),
    temperature_k=np.array([280.]), pressure_pa=np.array([95000.]),
    cross_sections_m2_per_molecule=np.array([[1e-23,2e-23]]))
table = read_cross_section_table(local_table, species='CH4', isotopologue='all',
    kind='cross_section', height_edges_m=np.array([0.,100.]))
assert table.species == 'CH4' and table.grid.coordinate == 'wavenumber_cm1'
np.testing.assert_array_equal(table.cross_sections_m2_per_molecule, [[1e-23,2e-23]])
filter_namespace = {{}}
for code in {filter_blocks!r}:
    exec(code, filter_namespace)
namespace = runpy.run_path({str(example)!r})
for mode in ('independent_column', 'heterogeneous_direct_solar'):
    value, tangent = namespace['detector_prediction'](
        mode=mode, inputs=namespace['synthetic_inputs'](), angular_order=3, spectral_count=9)
    assert value.shape == (2,2,2) and tangent.shape == (2,2,2,2)
    assert np.isfinite(value).all() and np.isfinite(tangent).all()
    assert (tangent[...,0] < 0).all() and (tangent[...,1] > 0).all()
    blocked, blocked_tangent = namespace['detector_prediction'](
        mode=mode, inputs=namespace['synthetic_inputs'](), angular_order=3,
        spectral_count=9, spectral_block_size=2)
    np.testing.assert_allclose(blocked, value, rtol=2e-14)
    np.testing.assert_allclose(blocked_tangent, tangent, rtol=2e-14)
assert not sys._is_gil_enabled()
print('PASS installed archived scene composition, native imports, GIL disabled', sys.version)
"""
        run(str(python), "-I", "-c", smoke)
        for wheel in wheels.glob("*.whl"):
            shutil.copy2(wheel, output / wheel.name)
    print("PASS isolated installed scene validation; archives and wheels:", output)


if __name__ == "__main__":
    main()
