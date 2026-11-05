"""Public library roots expose their own functions, not dependency facades."""

import importlib
import inspect
import subprocess
import sys
from annotationlib import Format

import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "orsia_data",
        "orsia_geometry",
        "orsia_instruments",
        "orsia_inverse",
        "orsia_numerics",
        "orsia_radiative",
        "orsia_uncertainty",
    ],
)
def test_root_functions_belong_to_the_library(package_name):
    package = importlib.import_module(package_name)
    for name, value in vars(package).items():
        if not name.startswith("_") and inspect.isroutine(value):
            assert value.__module__.split(".")[0] == package_name, name


def test_geometry_propagator_exports_and_trajectory_contract():
    package = importlib.import_module("orsia_geometry")
    orbits = importlib.import_module("orsia_geometry.orbits")
    los = importlib.import_module("orsia_geometry.los")
    for name in ("numerical_propagator", "sgp4_propagator"):
        assert name in package.__all__
        assert getattr(package, name) is getattr(orbits, name)
        assert getattr(package, name).__name__ == name
    for name in ("classical_trajectory", "sgp4_trajectory"):
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
    for name in ("numerical_propagator", "sgp4_propagator"):
        signature = inspect.signature(getattr(package, name), annotation_format=Format.STRING)
        assert signature.return_annotation == "Trajectory"


def test_geometry_public_signatures_and_type_hints_are_native_free_and_lazy():
    code = """
import inspect
import sys
from typing import get_type_hints
import orsia_geometry as geometry
from orsia_geometry import frames, orbits, time_scales

assert time_scales.__all__ == []
for module in (geometry, frames, orbits):
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
parameters = inspect.signature(geometry.numerical_propagator).parameters
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
hints = get_type_hints(geometry.numerical_propagator)
for name, annotation in (
    ('gravity_degree', int), ('gravity_order', int), ('sun_gravity', bool),
    ('moon_gravity', bool), ('absolute_tolerance', float), ('relative_tolerance', float),
):
    assert hints[name] is annotation
assert hints['return'] is geometry.Trajectory
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module,names",
    [
        ("orsia_geometry.frames", "frame_rotations transform_rays transform_states"),
        ("orsia_geometry.los", "AttitudeLaw Trajectory compute_lines_of_sight"),
        (
            "orsia_geometry.attitude",
            "axis_rotations instrument_tracking platform_tracking pointing_rates radial_nadir",
        ),
        ("orsia_data.surface.sentinel2", "read_sentinel2_l2a read_sentinel2_metadata"),
        ("orsia_radiative.irradiance", "SpectralIrradiance"),
        ("orsia_radiative.radiance", "Radiance"),
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
