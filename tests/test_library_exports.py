"""Public library roots expose their own functions, not dependency facades."""

import importlib
import inspect
import subprocess
import sys

import pytest


def test_common_dispersion_contract_removes_historical_reconstruction():
    dispersion = importlib.import_module("orsia_dispersion")
    for name in ("stationary_mass_column", "stationary_vertical_weights"):
        assert not hasattr(dispersion, name)
        assert name not in dispersion.__all__
    stationary = importlib.import_module("orsia_dispersion.stationary")
    assert stationary.StationaryGaussianPlume is dispersion.StationaryGaussianPlume
    for name in ("stationary_mass_column", "stationary_vertical_weights"):
        assert not hasattr(stationary, name)
    assert importlib.util.find_spec("orsia_scene.extinction") is None
    scene = importlib.import_module("orsia_scene")
    assert not hasattr(scene, "SceneRadiance")
    assert not hasattr(scene, "RenderedScene")


def test_dispersion_specialized_puff_operations_use_the_puffs_package():
    for module in ("fields", "integration", "kernels", "simulation", "splitting", "transport"):
        assert importlib.util.find_spec(f"orsia_dispersion.{module}") is None
    puffs = importlib.import_module("orsia_dispersion.puffs")
    assert puffs.__path__


@pytest.mark.parametrize(
    "package_name",
    [
        "orsia_data",
        "orsia_dispersion",
        "orsia_geometry",
        "orsia_instruments",
        "orsia_inverse",
        "orsia_numerics",
        "orsia_radiative",
        "orsia_scene",
        "orsia_uncertainty",
    ],
)
def test_root_functions_belong_to_the_library(package_name):
    package = importlib.import_module(package_name)
    for name, value in vars(package).items():
        if not name.startswith("_") and inspect.isroutine(value):
            assert value.__module__.split(".")[0] == package_name, name


def test_scene_flat_physical_contract_removes_observation_adapter():
    package = importlib.import_module("orsia_scene")
    assert not hasattr(package, "radiance_for_observation")
    assert importlib.util.find_spec("orsia_scene.observation") is None
    assert importlib.util.find_spec("orsia_scene.queries") is None
    assert set(package.__all__) == {
        "SpatialOpticalDepth",
        "GasLayerColumns",
        "gas_layer_columns",
        "DirectSolarScene",
        "puff_molecular_columns",
        "surface_radiance",
        "surface_radiance_tangent",
    }
    assert (
        inspect.signature(package.puff_molecular_columns).parameters["scale_tangent"].default
        is None
    )


def test_geometry_propagator_exports_and_trajectory_contract():
    package = importlib.import_module("orsia_geometry")
    orbits = importlib.import_module("orsia_geometry.orbits")
    los = importlib.import_module("orsia_geometry.los")
    for name in ("NumericalPropagator", "SGP4Propagator"):
        assert name in package.__all__
        assert getattr(package, name) is getattr(orbits, name)
        assert getattr(package, name).__name__ == name
    for name in (
        "classical_trajectory",
        "sgp4_trajectory",
        "numerical_propagator",
        "sgp4_propagator",
    ):
        assert name not in package.__all__
        assert not hasattr(package, name)
        assert not hasattr(orbits, name)
    assert "Trajectory" in package.__all__
    assert package.Trajectory is los.Trajectory
    assert "compute_lines_of_sight" in package.__all__
    assert package.compute_lines_of_sight is los.compute_lines_of_sight
    assert package.compute_lines_of_sight.__name__ == "compute_lines_of_sight"
    assert not hasattr(package, "observe")
    assert not hasattr(los, "observe")
    output = inspect.signature(package.compute_lines_of_sight).parameters["output_frame"]
    assert output.default == "GCRS" and output.kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(package.compute_lines_of_sight).parameters["trajectory"].annotation
        is package.Trajectory
    )


@pytest.mark.parametrize(
    "module,names",
    [
        ("orsia_geometry.surfaces", "wgs84_surface_projection wgs84_surface_projection_from_ray"),
        (
            "orsia_instruments.response.grid",
            "gaussian_cross_track_response GaussianCrossTrackResponse "
            "GaussianCrossTrackAcquisitionResponse",
        ),
    ],
)
def test_removed_constructors_have_no_aliases(module, names):
    owner = importlib.import_module(module)
    root = importlib.import_module(module.split(".")[0])
    for name in names.split():
        assert not hasattr(owner, name)
        assert not hasattr(root, name)
        assert name not in root.__all__


def test_geometry_public_signatures_and_type_hints_are_native_free_and_lazy():
    code = """
import importlib.util
import inspect
import sys
from typing import get_type_hints
import orsia_geometry as geometry
from orsia_geometry import frames, orbits, times

assert times.__all__ == []
assert importlib.util.find_spec('orsia_geometry.time_scales') is None
for module in (geometry, frames, orbits, times):
    assert not hasattr(module, 'satkit_times')
    assert not hasattr(module, 'native_instants')
    for name, value in vars(module).items():
        if name.startswith('_') or not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        if not getattr(value, '__module__', '').startswith('orsia_geometry'):
            continue
        signature = inspect.signature(value)
        hints = get_type_hints(value)
        assert 'satkit' not in str(signature), (module.__name__, name, signature)
        assert 'satkit' not in str(hints), (module.__name__, name, hints)
assert 'satkit' not in sys.modules, 'API introspection loaded a native dependency'
parameters = inspect.signature(geometry.NumericalPropagator).parameters
assert tuple(parameters) == (
    'elements', 'epoch', 'validity_s', 'gravity_degree', 'gravity_order',
    'sun_gravity', 'moon_gravity', 'absolute_tolerance', 'relative_tolerance',
)
for name, default in (
    ('gravity_degree', 2), ('gravity_order', 0), ('sun_gravity', False),
    ('moon_gravity', False), ('absolute_tolerance', 1e-4), ('relative_tolerance', 1e-11),
):
    assert parameters[name].default == default
    assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
hints = get_type_hints(geometry.NumericalPropagator.__init__)
for name, annotation in (
    ('gravity_degree', int), ('gravity_order', int), ('sun_gravity', bool),
    ('moon_gravity', bool), ('absolute_tolerance', float), ('relative_tolerance', float),
):
    assert hints[name] is annotation
assert hints['return'] is type(None)
for constructor in (
    geometry.NumericalPropagator,
    geometry.SGP4Propagator,
    geometry.WGS84SurfaceProjection,
):
    for method in (constructor.__init__, constructor.__call__):
        assert 'satkit' not in str(get_type_hints(method))
ray_return = get_type_hints(geometry.WGS84SurfaceProjection.from_ray)['return']
assert ray_return is geometry.WGS84SurfaceProjection
assert 'satkit' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module,names",
    [
        (
            "orsia_dispersion.particles",
            "ParticleReleases MassParticles ParticleDiagnostics ParticleMassField",
        ),
        ("orsia_dispersion.particles", "BoundedParticleDispersion"),
        ("orsia_dispersion.stationary", "StationaryGaussianPlume"),
        ("orsia_dispersion.sampling", "DispersionModel MassField sample_fields"),
        ("orsia_dispersion.evaluation", "LayerMassMoments concentration layer_columns"),
        ("orsia_dispersion.puffs.population", "GaussianPuffs PuffReleases"),
        (
            "orsia_dispersion.puffs.simulation",
            "GaussianPuffDispersion GaussianPuffField PuffDiagnostics",
        ),
        ("orsia_geometry.orbits", "NumericalPropagator SGP4Propagator"),
        ("orsia_geometry.surfaces", "WGS84SurfaceProjection"),
        ("orsia_geometry.frames", "frame_rotations transform_rays transform_states"),
        ("orsia_geometry.los", "AttitudeLaw Trajectory compute_lines_of_sight"),
        (
            "orsia_geometry.attitude",
            "axis_rotations instrument_tracking platform_tracking "
            "attitude_interval_speeds radial_nadir",
        ),
        ("orsia_data.surface.sentinel2", "read_sentinel2_l2a read_sentinel2_metadata"),
        ("orsia_radiative.irradiance", "SpectralIrradiance"),
        ("orsia_radiative.radiance", "Radiance"),
        ("orsia_scene.gas", "puff_molecular_columns"),
        ("orsia_scene.radiance", "surface_radiance surface_radiance_tangent"),
    ],
)
def test_root_exports_have_canonical_module_owners(module, names):
    canonical = importlib.import_module(module)
    package = importlib.import_module(module.split(".")[0])
    for name in names.split():
        value = getattr(canonical, name)
        assert name in package.__all__
        assert getattr(package, name) is value
        assert value.__module__ == module
