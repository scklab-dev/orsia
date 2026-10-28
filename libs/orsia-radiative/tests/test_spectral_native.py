"""Full native SOS against frozen Python and independent physical ray tests."""

import gc
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib import import_module
from types import ModuleType
from unittest.mock import patch

import numpy as np
import orders_reference as reference
import pytest
from orsia_radiative import AerosolComponent
from orsia_radiative.transfer import orders
from orsia_radiative.transfer.native import propagation_native
from test_solar_accumulation import _inputs
from test_solar_boundary import _sweep_reference


def mixed_inputs(layers=5):
    scene, source, optics = _inputs("zero_point", 0.4, layers=layers)
    g = np.resize(np.array([0.8, -0.8, 0.0, 0.4, 0.8]), (layers, 3))
    scattering = np.full((layers, 3), 0.02)
    scattering[:, 1] = 0
    scattering[1] = 0
    aerosol = AerosolComponent(optics.grid, np.full((layers, 3), 0.03), scattering, g)
    return scene, source, replace(optics, aerosol=aerosol)


@pytest.mark.parametrize("backend", ["portable", "blas"])
@pytest.mark.parametrize("moments", [None, 4, 6])
@pytest.mark.parametrize("angular", [3, 9])
@pytest.mark.parametrize("maximum", [0, 1, 4])
def test_mixed_general_path(backend, moments, angular, maximum):
    scene, source, optics = mixed_inputs()
    controls = dict(
        max_order=maximum, angular_order=angular, layer_subdivisions=2, delta_m_moments=moments
    )
    with patch.object(reference, "_accumulate_orders", _sweep_reference):
        expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    kernel = (
        propagation_native.spectral_orders
        if backend == "portable"
        else propagation_native.bind_spectral_blas(import_module("scipy.linalg.cython_blas"))
    )
    with patch.object(orders, "_native_spectral_orders", kernel):
        actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)
    assert (actual[1:, 1] == 0).all()


@pytest.mark.parametrize("backend", ["portable", "blas"])
@pytest.mark.parametrize("moments", [None, 4])
@pytest.mark.parametrize("maximum", [0, 1, 3])
@pytest.mark.parametrize(
    ("zenith", "view_zenith", "azimuth", "depth_scale"),
    [
        pytest.param(89.99999999999989, 20.0, 20.0, 1e-12, id="reported-incident"),
        pytest.param(89.99999999999989, 23.0, 117.0, 1e-15, id="incident-direct"),
        pytest.param(37.0, 89.99999999999989, 236.6, 1e-15, id="observer"),
        pytest.param(89.99999999999989, 89.99999999999989, 123.4, 1e-15, id="both"),
        pytest.param(float(np.nextafter(90.0, 0.0)), 23.0, 117.0, 1e-16, id="last-incident"),
        pytest.param(37.0, float(np.nextafter(90.0, 0.0)), 117.0, 1e-16, id="last-observer"),
        pytest.param(37.0, 53.0, 236.6, 1.0, id="ordinary"),
        pytest.param(0.0, 0.0, 0.0, 1.0, id="normal"),
    ],
)
def test_angular_conversion_matches_original_recurrence(
    backend, moments, maximum, zenith, view_zenith, azimuth, depth_scale
):
    # Thin columns retain nonzero direct radiance at tiny positive cosines.
    # The reference executes the original Python recurrence and math.radians.
    scene, source, optics = mixed_inputs()
    source = replace(source, zenith_deg=zenith, relative_azimuth_deg=azimuth)
    scene = replace(scene, geometry=replace(scene.geometry, view_zenith_deg=view_zenith))
    for component in (optics.absorption, optics.rayleigh, optics.aerosol):
        assert component is not None
        component.optical_depth[:] *= depth_scale
    assert optics.aerosol is not None
    optics.aerosol.scattering_depth[:] *= depth_scale
    controls = dict(
        max_order=maximum, angular_order=3, layer_subdivisions=2, delta_m_moments=moments
    )
    expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    kernel = (
        propagation_native.spectral_orders
        if backend == "portable"
        else propagation_native.bind_spectral_blas(import_module("scipy.linalg.cython_blas"))
    )
    with patch.object(orders, "_native_spectral_orders", kernel):
        actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)
    if depth_scale <= 1e-15:
        assert (expected[0] > 0).all()
    if maximum:
        assert (expected[1:, (0, 2)] > 0).all()
        assert (actual[1:, 1] == 0).all()


@pytest.mark.parametrize("layout", ["F", "strided", "reversed", "unaligned"])
def test_borrowed_layouts_preserved(layout):
    scene, source, optics = mixed_inputs()
    controls = dict(max_order=4, angular_order=4, layer_subdivisions=3, delta_m_moments=6)
    expected = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    arrays = []
    for owner, name in (
        (optics.absorption, "optical_depth"),
        (optics.rayleigh, "optical_depth"),
        (optics.aerosol, "optical_depth"),
        (optics.aerosol, "scattering_depth"),
        (optics.aerosol, "asymmetry"),
        (source.irradiance, "values"),
        (scene.surface, "reflectance"),
    ):
        original = getattr(owner, name)
        if layout == "F":
            array = np.asfortranarray(original)
        elif layout == "unaligned":
            array = np.ndarray(
                original.shape, dtype=np.float64, buffer=bytearray(original.nbytes + 1), offset=1
            )
            array[:] = original
        else:
            storage = np.zeros(tuple(2 * n for n in original.shape))
            array = storage[
                tuple(slice(None, None, -2 if layout == "reversed" else 2) for _ in original.shape)
            ]
            array[:] = original
        array.flags.writeable = False
        object.__setattr__(owner, name, array)
        arrays.append(array)
    snapshots = [a.tobytes() for a in arrays]
    actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_array_equal(actual, expected)
    assert [a.tobytes() for a in arrays] == snapshots
    assert all(not np.shares_memory(actual, a) for a in arrays)


def test_concurrent_general_calls_and_one_native_boundary():
    scene, source, optics = mixed_inputs()
    controls = dict(max_order=6, angular_order=4, layer_subdivisions=3, delta_m_moments=6)

    def calculate(_):
        return orders.scattering_orders(scene, source=source, optics=optics, **controls)

    expected = calculate(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(calculate, range(16)))
    assert not sys._is_gil_enabled()
    for result in results:
        np.testing.assert_array_equal(result, expected)
        assert result.flags.owndata and result.flags.writeable
        assert not np.shares_memory(result, expected)
    results[0][:] = -1
    np.testing.assert_array_equal(results[1], expected)
    with patch.object(
        orders, "_native_spectral_orders", wraps=orders._native_spectral_orders
    ) as calls:
        calculate(0)
    assert calls.call_count == 1


def native_arguments():
    scene, source, optics = mixed_inputs()
    assert optics.absorption is not None and optics.rayleigh is not None
    assert optics.aerosol is not None
    directions, weights = orders.angular_grid(3)
    return [
        (
            optics.absorption.optical_depth,
            optics.rayleigh.optical_depth,
            optics.aerosol.optical_depth,
            optics.aerosol.scattering_depth,
            optics.aerosol.asymmetry,
        ),
        source.irradiance.values,
        scene.surface.reflectance,
        directions,
        weights,
        5,
        8,
        3,
        6,
        source.zenith_deg,
        scene.geometry.view_zenith_deg,
        source.relative_azimuth_deg,
    ]


@pytest.mark.parametrize(
    "failure", ["dtype", "rank", "shape", "aerosol", "geometry", "size", "directions", "weights"]
)
def test_private_boundary_rejections(failure):
    args = native_arguments()
    if failure == "dtype":
        args[1] = args[1].astype(np.float32)
    elif failure == "rank":
        args[1] = args[1][:, None]
    elif failure == "shape":
        args[2] = args[2][:-1]
    elif failure == "aerosol":
        args[0] = (*args[0][:4], None)
    elif failure == "geometry":
        args[9] = 90.0
    elif failure == "size":
        args[6] = sys.maxsize
    elif failure == "directions":
        args[3] = args[3].copy()
        args[3][0, 2] = 0
    else:
        args[4] = args[4].copy()
        args[4][0] = np.nan
    with pytest.raises((ValueError, OverflowError)):
        propagation_native.spectral_orders(*args)


@pytest.mark.parametrize("failure", [None, "shape", "detached-weight"])
def test_exports_and_blas_owner_lifetimes(failure):
    class Exporter:
        def __init__(self, a):
            self.array = a
            self.acquired = self.released = 0

        def __buffer__(self, flags):
            self.acquired += 1
            return memoryview(self.array)

        def __release_buffer__(self, view):
            self.released += 1

    args = native_arguments()
    wrappers = [Exporter(a) for a in (*args[0], *args[1:5])]
    args[:5] = [tuple(wrappers[:5]), *wrappers[5:]]
    if failure:
        if failure == "shape":
            wrappers[-1].array = wrappers[-1].array[:-1]
        else:
            wrappers[-1].array[0] = np.nan
        with pytest.raises(ValueError):
            propagation_native.spectral_orders(*args)
    else:
        owner = ModuleType("blas_owner")
        owner.__dict__["__pyx_capi__"] = dict(
            import_module("scipy.linalg.cython_blas").__pyx_capi__
        )
        lifetime = weakref.ref(owner)
        kernel = propagation_native.bind_spectral_blas(owner)
        del owner
        gc.collect()
        assert lifetime() is not None
        kernel(*args)
        del kernel
        gc.collect()
        assert lifetime() is None
    assert all(w.acquired == w.released == 1 for w in wrappers[5:])
    assert all(w.acquired == w.released == (0 if failure else 1) for w in wrappers[:5])


@pytest.mark.parametrize("angular", [17, 20])
def test_phase_cache_eviction_and_oversized_operator(angular):
    # Three distinct HG phases exceed the 16 MiB retention budget; at order 20
    # each dense operator exceeds the budget by itself.
    scene, source, optics = mixed_inputs(layers=3)
    controls = dict(max_order=2, angular_order=angular, layer_subdivisions=1, delta_m_moments=4)
    expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)


@pytest.mark.parametrize("moments", [None, 4])
def test_direct_and_inactive_spectrum(moments):
    scene, source, optics = mixed_inputs()
    controls = dict(max_order=0, delta_m_moments=moments)
    expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    assert optics.rayleigh is not None and optics.aerosol is not None
    optics.rayleigh.optical_depth.fill(0)
    optics.aerosol.scattering_depth.fill(0)
    inactive = orders.scattering_orders(scene, source=source, optics=optics, max_order=4)
    assert np.isfinite(inactive).all() and (inactive[1:] == 0).all()


def test_native_rejects_nonfinite_accumulated_radiance():
    scene, source, optics = mixed_inputs()
    source.irradiance.values.fill(1e308)
    assert optics.aerosol is not None
    optics.aerosol.optical_depth.fill(1e307)
    optics.aerosol.scattering_depth.fill(1e307)
    with pytest.raises(ValueError, match="scattering order radiance must be finite"):
        orders.scattering_orders(scene, source=source, optics=optics, max_order=2)
