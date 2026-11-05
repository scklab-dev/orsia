"""WGS84 reference shape, fixed surface signatures and the deferred-import boundary."""

import inspect
import subprocess
import sys

import numpy as np
import pytest
from orsia_geometry import ellipsoid_intersections, surface_visibility
from orsia_geometry.constants import (
    WGS84_FLATTENING,
    WGS84_INVERSE_FLATTENING,
    WGS84_SEMIAXES_M,
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMINOR_AXIS_M,
)


def test_wgs84_definitions_and_derived_shape():
    # Independent reference literals keep a mistaken defining value detectable.
    assert WGS84_SEMIMAJOR_AXIS_M == 6378137.0
    assert WGS84_INVERSE_FLATTENING == 298.257223563
    assert WGS84_FLATTENING == 1 / 298.257223563
    assert WGS84_SEMIMINOR_AXIS_M == pytest.approx(6356752.314245179, abs=1e-9, rel=0)
    assert isinstance(WGS84_SEMIAXES_M, tuple)
    assert WGS84_SEMIAXES_M == (
        WGS84_SEMIMAJOR_AXIS_M,
        WGS84_SEMIMAJOR_AXIS_M,
        WGS84_SEMIMINOR_AXIS_M,
    )


def test_wgs84_surface_signatures_are_fixed():
    assert tuple(inspect.signature(ellipsoid_intersections).parameters) == (
        "origins_m",
        "directions",
    )
    assert tuple(inspect.signature(surface_visibility).parameters) == ("observers_m", "targets_m")


def test_surface_core_keeps_satkit_and_scipy_imports_deferred():
    code = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'satkit', 'scipy'}:
        raise AssertionError('surface core imported a deferred or physical-constants backend')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import numpy as np
from orsia_geometry import ellipsoid_intersections, surface_visibility
from orsia_geometry.constants import WGS84_SEMIMAJOR_AXIS_M
origin = np.array([WGS84_SEMIMAJOR_AXIS_M + 700000., 0., 0.])
hit, point = ellipsoid_intersections(origin, np.array([-1., 0., 0.]))
assert hit and surface_visibility(origin, point)
assert 'satkit' not in sys.modules and 'scipy' not in sys.modules
assert not sys._is_gil_enabled()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_surface_constants_do_not_expose_shared_mutable_storage():
    # A caller can construct a writable array without changing later intersections.
    axes = np.array(WGS84_SEMIAXES_M)
    axes[:] = 1
    origin = np.array([WGS84_SEMIMAJOR_AXIS_M + 700000.0, 0.0, 0.0])
    hit, point = ellipsoid_intersections(origin, np.array([-1.0, 0.0, 0.0]))
    assert hit
    np.testing.assert_allclose(point, [WGS84_SEMIMAJOR_AXIS_M, 0, 0], atol=1e-9, rtol=0)
